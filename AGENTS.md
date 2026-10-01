# AGENTS.md — Codex / Agent 在本仓库的工作约束

## ⚠️ 沙箱无外网（必须遵守，否则任务会卡死）

`codex exec --sandbox workspace-write` **没有网络访问**。以下操作会因 DNS 解析失败而卡住：

```
❌ pip install <任何包>        # error sending request for url (https://pypi.org/...)
❌ npm install / npm ci
❌ curl / wget 任何外部地址
❌ 试图访问 api.worldquantbrain.com
```

**本仓库的所有依赖必须使用已装好的环境，不要尝试安装任何东西。**

## 正确的测试执行路径

宿主 Python 缺 pandas/numpy/fastapi 等依赖，**不要在宿主跑测试**。使用运行中的生产容器：

```bash
# 容器 quantgpt 已有完整依赖（pandas / numpy / fastapi / mcp）
# 把改动的文件同步进容器再跑
cp quantgpt/wq_validator.py quantgpt/wq_operator_catalog.py \
   quantgpt/expression_parser.py quantgpt/mcp_server.py  /tmp/sync_new/

docker cp /tmp/sync_new/. quantgpt:/app/quantgpt/
docker cp tests/ quantgpt:/app/tests/

# 在容器内跑（工作目录必须是 /app，否则 import 不到 quantgpt 包）
docker exec quantgpt sh -lc 'cd /app && python -m pytest tests/ -v'
```

若容器内缺 pytest，用纯 python 脚本断言，不要 install：

```bash
docker exec quantgpt sh -lc 'cd /app && python your_check_script.py'
```

## 零误杀回归（最高优先级验收）

```bash
# 基线：115 条真实在 WQ BRAIN 跑通过的表达式
docker cp tests/fixtures/wq_known_good_expressions.json quantgpt:/app/tests/fixtures/
docker exec quantgpt sh -lc 'cd /app && python tests/regress_known_good.py'
```

**判定标准：115 条中 error 数必须为 0。**
任何 `status="error"` 都是回归缺陷——这些表达式全部在 WQ 生产环境真实跑通过。

## WQ 字段/算子目录

两个缓存文件已预置在容器命名卷中（`/app/data/`，重启不丢）：

```
/app/data/wq_field_catalog.json     4367 项字段
/app/data/wq_operator_catalog.json  官方 66 个算子（GET /operators）
```

两者都由 `quantgpt/wq_field_catalog.py` / `quantgpt/wq_operator_catalog.py`
管理，TTL 24h、落盘原子写、查询路径零网络。

**注意**：目录写盘后，运行中的 MCP 进程**不会**自动重读
（`_ensure_loaded()` 只在进程内首次调用时读盘），必须 `docker restart quantgpt`
才生效。测试时用 `docker exec` 单进程读取则无需重启。

目录**不可用**时校验降级为 warning（不阻断提交），见 `AGENTS.md` 教训 4。

## 历史教训（勿重蹈）

1. **`/data-fields` 必须携带完整 simulation settings**，仅传 `limit`/`dataset`
   会得到 `400 ["Invalid query"]`。
2. **`pyproject.toml` 必须锁 `mcp>=1.0,<2.0`**——2.x 移除了 `FastMCP`，
   容器会启动即崩。
3. **`docker restart` 不会换镜像**，重建容器才会用上新镜像。
4. **本地 `_WQ_OPERATORS` 白名单曾导致 25.7% 误杀**（漏 `signed_power` /
   `ts_zscore`，误杀 Sharpe 2.29 的 ACTIVE 因子 `9qWZ9G2x`）。算子校验必须以
   官方 `GET /operators` 为唯一权威，不得依赖手写白名单。现已由
   `quantgpt/wq_operator_catalog.py` 实现，目录不可用时**降级为 warning**，
   绝不阻断。
5. **`_LOCAL_ONLY_OPERATORS` 不是"WQ 不支持的算子"黑名单**。它只表示"本地
   先实现了，方便 pandas 回测"，其中 `ts_zscore` / `indneutralize` / `clip`
   是**官方合法算子**。拿它当黑名单是 25.7% 误杀的根因。无条件硬拒只能用
   `_LOCAL_ONLY_UNSUPPORTED`（= 前者减去官方确认存在的）。
6. **官方算子的参数个数要按官方 `definition` 校准**，不要凭印象写。
   `ts_regression` 官方是 3~5 个参数（`(y, x, d)` 合法），`group_mean` 支持
   3 个参数 `(x, weight, group)`。写窄了同样是误杀。
7. 沙箱内 `pip install` 失败会让 Codex 静默卡死，日志停在无意义的
   `ReasoningSummaryDelta` 报错上——看到这行先检查是不是在装包。

## 零误杀回归：改动算子/目录后必跑（含完整同步步骤）

```bash
mkdir -p /tmp/sync_new
cp quantgpt/wq_operator_catalog.py quantgpt/expression_parser.py \
   quantgpt/wq_validator.py quantgpt/mcp_server.py /tmp/sync_new/

docker cp /tmp/sync_new/. quantgpt:/app/quantgpt/
docker cp tests/ quantgpt:/app/tests/

# 1) 零误杀基线（覆盖“目录不可用”的降级路径）
docker exec quantgpt sh -lc 'cd /app && python tests/regress_known_good.py'
# 2) 细粒度用例矩阵（scope / 合规 / 黑名单 / 目录降级 / 两目录均可用）
docker exec quantgpt sh -lc 'cd /app && python -m pytest tests/test_wq_operator_catalog.py -v'
```

**判定标准：115 条中 error 数必须为 0**（脚本退出码 0）。
