# GOAL — QuantGPT WQ 表达式校验闸门（Operator / Data Field / Variable 白名单）

## 背景与根因（已由实测确认，勿再推翻）

上游 `Miasyster/QuantGPT` 自 2026-05-20 停更。生产部署暴露一个致命缺陷：

**`validate_expression` 对三类语义错误全部静默放行**，导致 subagent 提交必然失败的表达式，
白白浪费 WQ BRAIN 1~7 分钟模拟时间（占 cron 失败率的主要成因）。

已实测的四类放行（`mode="wq"` 均返回 OK，但 WQ 官方拒绝）：

| 错误类型 | WQ 官方报错原文 | validate_expression 现状 |
|---|---|---|
| 算子不存在 | `Attempted to use inaccessible or unknown operator "where"` | **OK 放行** |
| 变量不存在 | `Attempted to use unknown variable "anl4_fs_detail_estimates_basic_v4_nd_eps_std"` | **OK 放行** |
| 数据字段非法 | `Invalid data field gross_profit` | **OK 放行** |
| 完全编造字段 | — | **OK 放行** |
| 括号不配对 | — | ERROR（唯一有效） |

### 根因代码定位（已验证行号）

1. **`quantgpt/expression_parser.py:131`** — `_WQ_OPERATORS` 里 `'where'` 是**伪算子**。
   它在 `expression_parser.py:725` 被实现为 `pandas.Series.where()` 的三元选择，
   是 pandas 惯用法被误当成 WQ 算子。WQ 官方对应算子是 `trade_when`（语义不同：
   WQ 的 `trade_when` 条件不满足时**继承上一期持仓**，pandas 的 `where` 是逐元素切换）。
   **上游从未有过这个算子**，属遗留 bug。

2. **`quantgpt/expression_parser.py:876-881`** — 未知字段无条件透传：
   ```python
   if col_name in ALL_FUNDAMENTAL_NAMES:
       pass                                    # 放行
   else:
       logger.warning(f"WQ 模式：未知字段 '{col_name}'，将透传给 WQ BRAIN 校验")
       def _wq_unknown_field_stub(df, _c=col_name): ...
       return _wq_unknown_field_stub            # ★ 任何未知字段都返回合法 stub
   ```
   这是**有意设计**（WQ 有数万 data field，服务端不可能全知道），
   但代价是 `validate_expression` 永远返回 OK。

3. **`quantgpt/expression_parser.py:506-507`** — 白名单外的算子只 `logger.warning` 后
   当作"远程算子"透传，同样不报错。

### 关键机制发现（决定了错误为何来得慢）

实测 `POST /simulations` 对含 `where` 的表达式返回 **HTTP 201（提交成功）**，
错误只在**轮询阶段**才暴露：
```
POST /simulations → 201
轮询 → Attempted to use inaccessible or unknown operator "where"
```
所以任何"先提交再看错误"的模式，代价都是 1~7 分钟。

---

## 改造目标

让 `validate_expression` 成为**真正的提交前闸门**：本地秒级拦截三类语义错误，
杜绝"提交后等 1~7 分钟才被拒"的浪费。

---

## 任务清单

### 任务 1：Operator 白名单治理

**文件**：`quantgpt/expression_parser.py`

1. 从 `_WQ_OPERATORS` 移除 `'where'`（第 131 行）。
   **保留 `'trade_when'`**——它是 WQ 官方真实算子。

2. `mode="wq"` 下，遇到白名单外算子**必须抛错**，不得透传。
   删除或改造 `expression_parser.py:506-507` 的 warning+透传分支。
   错误信息需给出**替代建议**，例如：
   ```
   WQ 模式下不支持算子 'where'；三元条件请用 trade_when(cond, enter, exit)
   ```

3. **`mode="local"` 保持现状**——本地 A 股引擎可以继续用 pandas 语义，
   不要破坏 `run_backtest` / `score_factor` 等本地工具。`where` 仅从 WQ 白名单移除。

4. 提供**别名映射表**（`_WQ_REPLACEMENTS`，若已存在则扩充），至少覆盖：
   - `where` → `trade_when`
   - 其余常见 pandas 惯用法 → 对应 WQ 算子

### 任务 2：Data Field 白名单（真实字段表）

**新增文件**：`quantgpt/wq_field_catalog.py`

1. 实现从 WQ BRAIN 拉取真实 data field 的函数：
   `GET https://api.worldquantbrain.com/data-fields`（需已认证 session）。
   分页遍历，字段名取 `items[].id`，数据集取 `items[].dataset`。

2. 提供磁盘缓存：落盘到 `/app/data/wq_field_catalog.json`
   （容器已有 `/app/data` 挂载到宿主机 `/root/data/quantgpt/`）。
   **缓存必须带 TTL（默认 24h）**，避免每次校验都打 WQ。
   WQ 不可达时**降级使用过期缓存**，并在返回结果里标注 `stale: true`。

3. 提供查询接口：
   - `field_exists(name) -> bool`
   - `list_fields(dataset=None) -> list[str]`
   - `catalog_status() -> dict`（条目数 / 缓存时间 / 是否 stale）

4. **硬性要求**：首次拉取失败时**不得阻塞校验流程**，必须以"未知目录"模式运行，
   此时 `validate_expression` 对字段类错误返回**警告**而非 OK，
   并明确告知"字段目录不可用，无法校验字段合法性"。

### 任务 3：Variable 白名单（三类语义校验的第三类）

**背景**：FASTEXPR 里的 variable 分三类，来源完全不同，必须分别校验：

| 类别 | 例子 | 来源 | 校验方式 |
|---|---|---|---|
| **内置变量** | `vwap`, `returns`, `cap`, `adv20`, `adv60` | `_SPECIAL_VARS` + `adv{N}` 规则 | 白名单精确匹配 + 模式匹配 |
| **Price 列** | `open`,`high`,`low`,`close`,`volume`,`amount` | `_PRICE_COLUMNS` | 白名单精确匹配 |
| **Data field** | `gross_profit`, `assets`, `sales`, `est_eps`, `fnd6_*`, `anl4_*` | WQ data field | 任务 2 的目录 |

**文件**：`quantgpt/expression_parser.py`（`_SPECIAL_VARS` 约 295 行）

1. 明确区分 `_WQ_SPECIAL_VARS`（WQ 允许的内置变量）与本地专有的。
   现有代码已有 `_WQ_SPECIAL_VARS` 引用（838 行附近），核对其完整性。
   **注意 `cap` 与 `market_cap` 的映射**：A 股本地引擎叫 `market_cap`，
   WQ 侧叫 `cap`。这类映射错误是漏网之鱼，重点覆盖。

2. `adv{N}` 仅接受 `1 <= N <= 500`（与 `MAX_WINDOW` 一致），
   超出范围或非数字必须报错，不得静默。

3. 新增 `variable_category(name) -> str`，返回
   `"builtin" | "price" | "datafield" | "unknown"`，供校验器使用。

### 任务 4：改造 `validate_expression`

**文件**：`quantgpt/mcp_server.py:109-135`

1. 保持 `mode="local"` 行为不变（纯语法 + 本地可执行性）。

2. `mode="wq"` 必须执行**四道校验**，任一不过即返回 ERROR：
   - **① 括号/语法**：保持现有检查
   - **② Operator**：每个函数名必须在 `_WQ_OPERATORS` 内，且必须有本地实现
   - **③ Variable/Field**：按任务 3 的分类逐一校验；
     data field 需在任务 2 的目录中（目录 stale 或不可用时降级为警告）
   - **④ 量纲**：分母为常数、窗口长度合法、`adv{N}` 范围、
     嵌套深度 ≤ `MAX_DEPTH`、表达式长度 ≤ `MAX_EXPRESSION_LENGTH`

3. **返回结构必须区分严重级别**，便于 subagent 决策：
   ```json
   {"status": "ok" | "warning" | "error",
    "level": "...",
    "message": "...",
    "errors": [{"kind": "operator|field|variable|syntax|dimension",
                "name": "...", "hint": "替代方案"}],
    "warnings": [...]}
   ```
   - `error` = 必然被 WQ 拒绝，**禁止提交**
   - `warning` = 可能失败，建议先小样本验证（如字段目录不可用时的未知字段）

4. **向后兼容**：现有返回是纯字符串（如 `"OK: expression is valid for WQ BRAIN submission"`）。
   必须保持该字符串**仍然出现在返回中**，避免破坏依赖字符串匹配的现有调用方
   （cron prompt、subagent 手册、测试）。建议返回 JSON 字符串但 `message` 字段保持原文案。

5. **新增可选参数** `strict: bool = False`：
   - `strict=False`（默认）：`error` 级阻断，`warning` 级放行但标注
   - `strict=True`：任何 warning 也阻断

### 任务 5：新增快速预检工具

**文件**：`quantgpt/mcp_server.py`（新增 `@mcp.tool()`）

新增 `precheck_expression(expression: str, mode: str = "wq", strict: bool = True) -> str`：

- **纯本地校验，绝不调用 WQ API**（区别于 `wq_brain_batch_submit`）
- 返回**批量友好**结构：单个表达式的完整校验结果
- 目标耗时 < 100ms（表达式解析是纯 Python，无 IO）
- 供 subagent 在批量生成表达式后、提交前统一过闸

⚠️ 注意：字段目录读取若走磁盘缓存，必须是同步小文件读取，不得引入网络调用。

---

## 验收标准（必须全部通过，逐条实测）

### A. 回归：现有能力不退化

```bash
# 本地 A 股工具仍可用
validate_expression("rank(close - open)", "local")        → ok
validate_expression("rank(close +", "wq")                  → error(语法)
validate_expression("rank(close)", "wq")                   → ok
```

### B. 三类错误必须被拦截（核心验收）

对以下每个表达式，`validate_expression(mode="wq")` 必须返回 `status=error`：

| # | 表达式 | 期望错误 kind |
|---|---|---|
| 1 | `where(returns > 0, 1, -1)` | `operator` |
| 2 | `rank(anl4_fs_detail_estimates_basic_v4_nd_eps_std)` | `field` |
| 3 | `rank(gross_profit/assets)` | `field` |
| 4 | `rank(nonexistent_field_xyz)` | `field` |
| 5 | `ts_mean(unknown_macro_series, 20)` | `field` |

且每条错误必须给出**可操作的 hint**（如 `where` → 提示改用 `trade_when`）。

### C. 合规表达式不得被误杀

对以下**真实可用**表达式，必须返回 `status=ok`：

```
rank(ts_delta(close, 5) / ts_std_dev(returns, 20))
group_rank(ts_rank(turnover, 126), subindustry)
trade_when(volume > adv20, returns, -returns)
group_neutralize(rank(est_eps / close), subindustry)
group_zscore(winsorize(rank(returns), std=4), industry)
```
（其中 `turnover` / `est_eps` 需确认在 data field 目录内；若不在，
应返回 `warning` 而非 `error`——**误杀合规表达式比漏检更严重**。）

### D. 性能

`precheck_expression` 单次调用 < 100ms（本地无 IO）。

### E. 字段目录可用性

- WQ 可达时：目录条目数 > 1000，`stale=false`
- WQ 不可达时：降级 `stale=true`，校验降级为 warning，**不得抛异常**

---

## 硬性约束

1. **不得修改 `wq_brain_client.py` 的轮询与重试逻辑**——它工作正常，
   且服务端 6 分钟轮询上限与 MCP 420s 超时的关系已验证正确。
2. **不得修改 `mcp_server.py` 中其他 14 个工具的签名或行为**——
   现有 subagent 正在使用它们。
3. **不得引入新的第三方依赖**（`pyproject.toml` 不动）。全部用 stdlib
   （`urllib.request` / `json` / `pathlib`）。已确认可用依赖：
   pandas, numpy, requests, httpx。
4. **`mode="local"` 路径必须保持 100% 兼容**——本地 A 股回测
   （`run_backtest` / `score_factor` / `run_anti_overfit` /
   `run_rolling_validation` / `diagnose_factor` / `compute_factor_values`）
   全部依赖它。
5. 代码风格遵循上游：`ruff` 配置，类型注解齐全，中文 docstring。
6. **所有新增代码必须有中文注释解释"为什么"**，尤其要写明
   "为什么服务端无法预知 WQ 全部算子/字段"这一设计约束，
   避免后人再次把它改回静默放行。

---

## 交付物

1. 修改后的 `quantgpt/expression_parser.py`
2. 新增 `quantgpt/wq_field_catalog.py`
3. 修改后的 `quantgpt/mcp_server.py`（仅动 `validate_expression` + 新增 `precheck_expression`）
4. `tests/test_expression_validation.py` —— 覆盖验收标准 B 与 C 全部用例
5. 简明改动说明（改了什么 / 为什么 / 已知局限）

---

## 执行方式

```bash
cd /root/.hermes/submodule/quantgpt
codex exec --sandbox workspace-write "/goal 读取根目录 GOAL_WQ_VALIDATION.md 并完整实施"
```

配置要求：`~/.codex/config.toml` 中 `approval_policy = "never"`（已固化）。