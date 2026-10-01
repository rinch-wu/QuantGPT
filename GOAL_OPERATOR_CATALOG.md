# GOAL：修正 WQ 算子白名单 —— 以官方 /operators 目录为唯一权威

## 一、问题（生产实测数据，非推测）

误杀回归测试：**187 条真实在 WQ BRAIN 跑通过的表达式 → 48 条被本地校验器误杀（误杀率 25.7%）**。

误杀明细（全部为官方合法算子，仅因本地白名单遗漏而拦截）：

| 算子 | 被误杀次数 | WQ 官方确认 |
|---|---|---|
| `signed_power` | 33 | ✅ 合法 |
| `ts_zscore` | 12 | ✅ 合法 |

**真实损失已经发生**：已入库 ACTIVE 因子 `9qWZ9G2x`（Sharpe 2.29，全库最高分）因含 `signed_power` 被本地校验判 error；`kqgq6VJP`（Sharpe 1.54）同样被误杀。

根因：`expression_parser._WQ_OPERATORS` 是硬编码的 61 个算子，已严重过时。

## 二、权威依据（本次实测取得）

### 2.1 `GET /operators` 可用 —— 这是唯一权威来源
```
GET https://api.worldquantbrain.com/operators  -> HTTP 200
返回 66 个算子，每个含：
{
  "name": "add",
  "category": "Arithmetic",
  "scope": ["REGULAR"],
  "definition": "add(x, y, filter = false), x + y",
  "description": "...",
  "documentation": "/operators/add",
  "level": "ALL"
}
```
**`scope` 字段可直接用于「variable 作用域不匹配」校验**（REGULAR / COMBO 等）。

### 2.2 真实跑通表达式的算子交集（零歧义证据）
从 187 条真实跑通表达式中提取全部函数调用名（27 个），与官方目录比对：

```
['abs', 'group_mean', 'group_neutralize', 'group_rank', 'group_zscore', 'log',
 'max', 'min', 'power', 'rank', 'sign', 'signed_power', 'sqrt', 'trade_when',
 'ts_backfill', 'ts_corr', 'ts_covariance', 'ts_decay_linear', 'ts_delay',
 'ts_delta', 'ts_mean', 'ts_rank', 'ts_regression', 'ts_std_dev', 'ts_sum',
 'ts_zscore', 'winsorize']
```
**27/27 全部命中官方目录，无一是字段，无任何歧义。**

## 三、本地白名单与官方目录的差异（必须处理的全部问题）

### 3.1 官方有、本地缺（35 个）→ 当前全部会被误杀
```
add, and, densify, divide, equal, greater, greater_equal, group_backfill,
group_scale, hump, if_else, inverse, is_nan, kth_element, less, less_equal,
multiply, not, not_equal, or, reverse, signed_power, subtract,
ts_arg_max, ts_arg_min, ts_count_nans, ts_covariance, ts_decay_linear,
ts_delay, ts_product, ts_quantile, ts_scale, ts_std_dev, ts_step, ts_zscore
```

### 3.2 本地有、官方无（30 个）→ 需分类处理，不得一律删除
```
# 命名分歧（本地别名，官方用另一写法，历史上真实跑通过）
decay_linear, product, sign_power, ts_cov, ts_shift,
ts_argmax, ts_argmin,            # 官方为 ts_arg_max / ts_arg_min
ts_decay_exp_window, ts_ir, humpdecay, indneutralize, pasteurize,
ts_max, ts_min, ts_std, ts_skewness, ts_kurtosis,
vector_neut, group_vector_neut,
# COMBO 作用域专用（官方 scope=COMBO，不出现在 REGULAR 白名单）
vec_choose, vec_count, vec_ir, vec_kurtosis, vec_max, vec_min, vec_norm,
vec_percentage, vec_range, vec_skewness, vec_stddev
```

**要求**：
- 命名分歧类：保留为**兼容别名**，但不得视为非法；建议在错误提示中给出官方写法（如 `ts_argmin` → `ts_arg_min`）。
- `vec_*`：按 `scope` 归类为 COMBO 专用，REGULAR 表达式中使用应给出明确错误（提示改用 REGULAR 算子），而非笼统的「算子不存在」。
- `where`：**当前已正确排除**，改造后必须继续排除（官方目录中无 `where`）。

## 四、改造要求

### 4.1 新增官方算子目录同步（`quantgpt/wq_operator_catalog.py`）
- `GET /operators`，认证后拉取，缓存 24h（与 `wq_field_catalog` 一致的 TTL / stale 降级 / 命名卷落盘 `/app/data/wq_operator_catalog.json`）。
- 返回结构：`{name, category, scope[], definition, description}`。
- 提供 `list_operators()` / `operator_info(name)` / `is_known(name)` / `scope_of(name)`。
- **降级策略**：目录不可用时，**operator 校验必须降级为 warning，绝不阻断**。理由：算子白名单一旦过时就是误杀源，宁可放过也不可误杀（与 field 校验同一原则）。
  - 但仍保留一个**极小的、官方目录确认存在的核心集**作为兜底白名单（至少覆盖 2.2 节列出的 27 个 + 3.1 节全部 35 个），目录不可用时用它。
  - **关键**：`where` 不在兜底集内，目录不可用时 `where` 仍应被拦下（它是 pandas 语义，从未是 WQ 算子）。可用显式黑名单 `{"where"}` 保证。

### 4.2 `wq_validator` 改为官方目录优先
判定顺序：
1. 官方目录可用 → 以目录为唯一权威（附带 scope 判定）。
2. 目录不可用 → 用 4.1 的兜底核心集（官方确认过的全集）。
3. 两种情况下 `where` 均拦截。

### 4.3 scope 校验（variable 作用域）
- 表达式解析需知道顶层 `type`（REGULAR / COMBO）。`wq_brain_submit` 传的是 REGULAR。
- 若使用了 `scope` 中不含 `REGULAR` 的算子（如 `vec_*`），报 error 并说明该算子属 COMBO 作用域。
- `group_*` 族：第二个参数是分组字段（`subindustry`/`industry`/`market`/`sector`），不得当作 data field 报「字段不存在」——**当前日志中已出现 `未知字段 'subindustry'` 的误判，必须修复**。

### 4.4 参数解析 bug（已在生产日志中暴露，必须修）
当前解析器把以下内容误判为「未知字段」并透传：
- `std=3`、`std=4`、`std=2.5`（命名参数）
- `subindustry`、`industry`（group 函数的分组参数）
- 空字符串 `''`

这些必须从「字段」判定中排除。命名参数 `name=value` 在函数调用内不应被视为字段标识符。

### 4.5 兼容性铁律
- `mode="local"` 行为**不得改变**（本地 pandas 语义保留 `where`）。
- `validate_expression` 原有返回结构（`status`/`level`/`message`/`errors[]`/`warnings[]`/`details`/`submit_allowed`）保持不变。
- 不新增必需依赖。

## 五、验收标准（必须全部通过）

### 5.1 零误杀（最高优先级）
```
187 条真实跑通表达式 → 100% 不得出现 status="error"
```
唯一例外：若某条确实含 `where`（历史上不该存在，需先核实）。

### 5.2 不误拦真实错误
```
where(returns > 0, 1, -1)                      → error [operator]
vector_neut(returns, ts_delay(returns,1))       → error [operator]（COMBO 作用域）
rank(gross_profit/assets)                      → error [field]
rank(nonexistent_field_xyz)                    → error [field]
rank(close +)                                  → error [syntax]
```

### 5.3 合规表达式
```
signed_power(returns, 2)                        → ok
ts_zscore(returns, 60)                         → ok
group_rank(ts_rank(est_eps, 126), subindustry) → ok
group_neutralize(rank(est_eps/close), subindustry) → ok
group_zscore(winsorize(rank(returns), std=4), industry) → ok
trade_when(volume>adv20, returns, -returns)    → ok
rank(close-open)  mode=local                   → ok
```

### 5.4 目录降级
模拟目录不可用：`where` 仍必须 error（显式黑名单保证）；已知 27 个核心算子仍 ok。

## 六、交付要求
- 测试文件放入 `tests/`，包含零误杀回归（可从 WQ 拉取或内嵌样本）。
- 不部署，由主 Agent 验收后决定。