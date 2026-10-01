"""WQ BRAIN 提交前四道校验闸门（任务 4）。

为什么需要这一层
----------------
实测：``POST /simulations`` 对含非法算子/字段的表达式返回 **HTTP 201（提交成功）**，
错误只在**轮询阶段**才暴露。也就是说"先提交再看错误"的模式，代价恒为 1~7 分钟
模拟时间，这是 cron 失败率的主要成因。本模块把服务端校验前移到本地，秒级拦截。

四道闸门
--------
1. **语法**：括号配对（唯一原本就有效的检查）。
2. **Operator**：算子必须存在于 :func:`wq_operators`，且能本地解析。
3. **Variable/Field**：按 :func:`variable_category` 分类——内置变量与分组字段
   用内置白名单，data field 才查 :mod:`quantgpt.wq_field_catalog` 的真实目录。
4. **量纲**：表达式长度、嵌套深度、adv{N} 范围、常数与价格列做加减等。

严重级别契约
------------
- ``error``   —— 必然被 WQ 拒绝，**禁止提交**（白名单外的算子、已确认不在目录的字段）。
- ``warning`` —— 可能失败，建议小样本验证（**目录不可用**时的未知字段）。

⚠️ 关键设计约束：**误杀合规表达式比漏检更严重**。WQ data field 有数万条且服务端
持续新增，所以目录不可用时绝不能把"查不到"当成"不存在"——那会让整个 cron 在
目录过期期间全线拒绝所有表达式。宁可放行并标注，让服务端去发现新增字段。

``mode="local"`` 路径完全不受本模块影响：A 股本地引擎继续用 pandas 语义
（含 ``where``），行为保持 100% 兼容。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .expression_parser import (
    ExpressionParser,
    _LOCAL_ONLY_COLUMNS,
    _WQ_REPLACEMENTS,
    _WQ_UNIT_PATTERNS,
    parse_expression,
    variable_category,
    wq_operators,
)

# 算子别名（delta→ts_delta、ts_std_dev→ts_std 等）。必须在白名单比对**之前**归一，
# 否则 `ts_std_dev` 这类合法别名会被误判成"WQ 不存在的算子"——这是最典型的误杀。
_OPERATOR_ALIASES = ExpressionParser._OPERATOR_ALIASES

_MSG_WQ_OK = "OK: expression is valid for WQ BRAIN submission"
_MSG_LOCAL_OK = "OK: expression is valid"

# FASTEXPR 关键字/字面量：不是变量，校验时必须排除，否则会被当成未知字段误杀。
_EXPR_KEYWORDS = {"if", "else", "and", "or", "not", "true", "false"}

# 函数调用名：后面紧跟 '(' 的是算子，不是变量。
_FUNC_TOKEN_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")

# 标识符（支持 IndClass.industry 这类带点前缀的分组字段）。
# 为什么不直接用 extract_components：它会把点号拆成 IndClass + industry 两个 token，
# 导致 'indclass' 前缀匹配失效，把合规的分组字段误判成非法 data field。
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")


def extract_names(expression: str) -> tuple[set[str], set[str]]:
    """抽出表达式中的算子名与变量名（**保留点号**的分组字段前缀）。

    两个必须处理的细节，否则合规表达式会被误杀：
    - **named argument**：`winsorize(x, std=4)` 里的 `std` 是参数名不是 data field，
      否则会凭空多出一条"Invalid data field std"。
    - **算子别名**：`ts_std_dev` 是 `ts_std` 的别名，必须先归一再比对白名单。

    Returns:
        ``(operators, variables)``，两者均已转小写且算子已归一。
    """
    raw_ops = {m.group(1).lower() for m in _FUNC_TOKEN_RE.finditer(expression)}
    operators = {_OPERATOR_ALIASES.get(op, op) for op in raw_ops}

    named_args = {
        m.group(1).lower()
        for m in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)", expression)
    }

    variables: set[str] = set()
    for match in _IDENT_RE.finditer(expression):
        # 紧跟 '(' 的是算子调用，不是变量
        if expression[match.end():match.end() + 1] == "(":
            continue
        token = match.group(0).lower()
        if token in _EXPR_KEYWORDS or token in raw_ops or token in named_args:
            continue
        variables.add(token)
    return operators, variables


@dataclass
class Issue:
    """单条校验问题。``kind`` 用于 subagent 按类别聚合处理。"""

    kind: str  # operator | field | variable | syntax | dimension
    name: str
    message: str
    hint: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "name": self.name, "message": self.message, "hint": self.hint}


@dataclass
class ValidationResult:
    """校验结果。``status`` 驱动 subagent 的提交决策。"""

    status: str  # ok | warning | error
    message: str
    mode: str
    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        """是否禁止提交。"""
        return self.status == "error"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "level": self.status,  # 兼容 goal 文档里的双字段约定
            "message": self.message,
            "mode": self.mode,
            "submit_allowed": self.status != "error",
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "details": self.details,
        }

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


_DUMMY_DF: Any = None


def _validation_dummy() -> Any:
    """构造校验用的 dummy DataFrame（惰性 + 缓存）。

    只在需要**实际执行**表达式时使用（本地模式 / execute=True）。
    放在这里而不是 mcp_server，是为了避免 wq_validator → mcp_server 的反向依赖：
    mcp_server 已经 import wq_validator，再反向引用会形成循环导入，
    并且把纯校验逻辑绑死在 MCP 运行时上。
    """
    global _DUMMY_DF
    if _DUMMY_DF is None:
        import pandas as pd

        from .fundamental_data import ALL_FUNDAMENTAL_NAMES

        _DUMMY_DF = pd.DataFrame({
            "open": [1.0, 2.0, 3.0], "high": [1.1, 2.1, 3.1],
            "low": [0.9, 1.9, 2.9], "close": [1.0, 2.0, 3.0],
            "volume": [100, 200, 300], "amount": [100, 400, 900],
            "pct_change": [0, 100, 50],
            "trade_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            **{name: [1.0, 1.1, 1.2] for name in ALL_FUNDAMENTAL_NAMES},
        })
    return _DUMMY_DF


def _check_balanced_parens(expression: str) -> Issue | None:
    """闸门①：括号配对。这是上游唯一真正生效的检查，必须保留。"""
    depth = 0
    for i, ch in enumerate(expression):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return Issue(
                    "syntax",
                    ")",
                    f"括号不平衡：位置 {i} 处多余的右括号 ')'",
                )
    if depth > 0:
        return Issue("syntax", "(", f"括号不平衡：缺少 {depth} 个右括号 ')'")
    return None


def _check_length(expression: str) -> list[Issue]:
    """闸门④：表达式长度。WQ 服务端有长度上限，超长必然被拒。"""
    if len(expression) > ExpressionParser.MAX_EXPRESSION_LENGTH:
        return [
            Issue(
                "dimension",
                f"length={len(expression)}",
                f"表达式过长（{len(expression)} 字符，上限 {ExpressionParser.MAX_EXPRESSION_LENGTH}）",
            )
        ]
    return []


def _check_depth(expression: str) -> list[Issue]:
    """闸门④：最大嵌套深度。用解析器自身的限制做静态检查。"""
    depth = 0
    max_depth = 0
    for ch in expression:
        if ch == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ")":
            depth -= 1
    if max_depth > ExpressionParser.MAX_DEPTH:
        return [
            Issue(
                "dimension",
                f"depth={max_depth}",
                f"嵌套过深（{max_depth} 层，上限 {ExpressionParser.MAX_DEPTH}）",
            )
        ]
    return []


def _check_adv_ranges(expression: str) -> list[Issue]:
    """闸门④：adv{N} 必须落在 1~MAX_WINDOW，adv 后缀非数字也要报错。

    WQ 只提供有限几个预置 adv{n}，越界或拼写错误都会被服务端拒绝。
    """
    issues: list[Issue] = []
    # 注意不能写成 \badv(...)：'adv' 与 '20' 之间不存在 \b 边界，
    # 正则会退化成只匹配 'adv' 本身，把合法的 adv20 误报成"非法的 adv{N}"。
    for match in re.finditer(r"(?<![A-Za-z0-9_])adv([A-Za-z0-9_]*)", expression, flags=re.IGNORECASE):
        digits = match.group(1)
        token = match.group(0)
        if not digits.isdigit():
            issues.append(
                Issue(
                    "dimension",
                    token,
                    f"变量 '{token}' 不是合法的 adv{{N}}",
                    f"N 必须是 1~{ExpressionParser.MAX_WINDOW} 的整数，如 adv20 / adv60",
                )
            )
            continue
        if not 1 <= int(digits) <= ExpressionParser.MAX_WINDOW:
            issues.append(
                Issue(
                    "dimension",
                    token,
                    f"adv 窗口越界：{token}（合法范围 1~{ExpressionParser.MAX_WINDOW}）",
                    f"改用范围内的窗口，如 adv{min(int(digits), ExpressionParser.MAX_WINDOW)}",
                )
            )
    return issues


def _check_units(expression: str) -> list[Issue]:
    """闸门④：量纲。复用解析器里 WQ 专用的量纲正则。"""
    normalized = re.sub(r"\s+", " ", expression.lower())
    issues: list[Issue] = []
    for pattern, message in _WQ_UNIT_PATTERNS:
        if pattern.search(normalized):
            issues.append(Issue("dimension", "unit", message))
    return issues


def _collect_names(expression: str) -> tuple[set[str], set[str]]:
    """兼容包装：委托给 :func:`extract_names`。"""
    return extract_names(expression)


def _validate_variables(expression: str) -> tuple[list[Issue], list[Issue], dict[str, Any]]:
    """闸门③：变量/字段分类校验。

    data field 走真实目录；目录不可用时降级为 **warning**（不是 error）——
    这是"宁可漏检也不误杀"的直接体现。
    """
    from . import wq_field_catalog

    status = wq_field_catalog.catalog_status()
    catalog_available = status["available"]
    errors: list[Issue] = []
    warnings: list[Issue] = []

    _, fields = _collect_names(expression)
    for raw_name in sorted(fields):
        name = raw_name.lower()

        # 必须先按**字面名**判掉本地专有列。variable_category() 会把 market_cap
        # 归一成 cap（这是它用于分类的正确行为），但校验要看的是表达式里真正
        # 写的那个 token —— WQ 服务端只认 cap，写 market_cap 就是 Invalid data field。
        if name in _LOCAL_ONLY_COLUMNS:
            hint = _WQ_REPLACEMENTS.get(name, "")
            errors.append(
                Issue(
                    "field",
                    name,
                    f"字段 '{name}' 不是 WQ data field（服务端会报 Invalid data field）",
                    hint or "改用 WQ 官方字段（price 列 / 内置变量 / data field 目录）",
                )
            )
            continue

        category = variable_category(name)

        if category == "unknown":
            # 归一后仍认不出：既不是内置/价格/分组字段，也不匹配任何已知前缀
            hint = _WQ_REPLACEMENTS.get(name, "")
            errors.append(
                Issue(
                    "field",
                    name,
                    f"未知字段 '{name}'：不是 WQ 内置变量、价格列、分组字段，"
                    f"目录中也无此 data field",
                    hint or "用 GET /data-fields 确认字段 id，或改用 close/volume/cap 等内置字段",
                )
            )
            continue

        if category == "datafield" and catalog_available:
            if not wq_field_catalog.field_exists(name):
                # 目录可用 → 查不到即确认非法（服务端会报 Invalid data field）
                errors.append(
                    Issue(
                        "field",
                        name,
                        f"字段 '{name}' 不在 WQ data field 目录中（服务端会报 "
                        f"Invalid data field）",
                        "用 list_fields() 或 GET /data-fields 检索正确字段名",
                    )
                )
        elif category == "datafield" and not catalog_available:
            # 目录不可用 → 降级警告，绝不当 error
            warnings.append(
                Issue(
                    "field",
                    name,
                    f"字段 '{name}' 无法校验：WQ 字段目录不可用"
                    f"（{status.get('last_error') or '首次拉取失败或缓存缺失'}）",
                    "目录恢复后会自动校验；现在可提交，但建议先小样本验证",
                )
            )

    details = {
        "catalog_available": catalog_available,
        "catalog_count": status["count"],
        "catalog_stale": status["stale"],
    }
    return errors, warnings, details


def validate_wq_expression(
    expression: str,
    strict: bool = False,
    execute: bool = False,
) -> ValidationResult:
    """执行 WQ 提交前的完整校验（闸门①~④）。

    Args:
        expression: FASTEXPR 表达式。
        strict: True 时任何 warning 也升级为 error（默认 False 只阻断 error）。
        execute: True 时额外用 dummy DataFrame 实跑一次，捕获运行期错误。
            wq 模式下大多数算子只能远程执行，故默认关闭。

    这是**纯本地**校验：除字段目录的磁盘缓存外无任何 IO/网络，
    目标耗时 <100ms（见 acceptance D）。
    """
    expression = (expression or "").strip()
    errors: list[Issue] = []
    warnings: list[Issue] = []
    details: dict[str, Any] = {}

    if not expression:
        return ValidationResult(
            status="error",
            message="ERROR: 表达式为空",
            mode="wq",
            errors=[Issue("syntax", "", "表达式为空")],
        )

    # 闸门①：括号
    paren_issue = _check_balanced_parens(expression)
    if paren_issue:
        return ValidationResult(
            status="error",
            message=f"ERROR: {paren_issue.message}",
            mode="wq",
            errors=[paren_issue],
        )

    # 闸门④：维度类静态检查
    for issue in [*_check_length(expression), *_check_depth(expression),
                  *_check_adv_ranges(expression), *_check_units(expression)]:
        (errors if issue.kind == "dimension" else warnings).append(issue)

    # 闸门②：算子。ExpressionParser 在 mode="wq" 下对白名单外算子硬报错，
    # 这里先做一遍**集合**层面的预检，以便给出比解析器更友好的 hint。
    allowed_ops = wq_operators()
    ops, _ = _collect_names(expression)  # ops 已做别名归一
    for op in sorted(ops):
        if op in allowed_ops:
            continue
        hint = _WQ_REPLACEMENTS.get(op, "")
        if hint:
            msg = f"WQ 模式下不存在算子 '{op}'，替代方案：{hint}"
        else:
            msg = (
                f"WQ 模式下不存在算子 '{op}'（服务端会报 "
                f"unknown operator）"
            )
        errors.append(Issue("operator", op, msg, hint))

    # 闸门③：变量/字段
    var_errors, var_warnings, var_details = _validate_variables(expression)
    errors.extend(var_errors)
    warnings.extend(var_warnings)
    details.update(var_details)

    # 闸门②续：完整解析（含算子参数个数、窗口合法性、远程算子参数校验）。
    # 解析器对所有 ValueError 都抛，这里转成 error 级 issue。
    if not errors:
        try:
            parse_expression(expression, mode="wq")
        except ValueError as exc:
            errors.append(Issue("operator" if "算子" in str(exc) else "syntax",
                                "", str(exc)))
        except Exception as exc:  # noqa: BLE001 — 解析异常一律降级为 error，不外泄
            errors.append(Issue("syntax", "", f"解析失败：{exc}"))

    if execute and not errors:
        try:
            parse_expression(expression, mode="wq")(_validation_dummy())
        except RuntimeError:
            # WQ 字段/算子无本地数据，属预期（需远程执行），不算错误
            pass
        except Exception as exc:  # noqa: BLE001
            errors.append(Issue("syntax", "", f"本地执行失败：{exc}"))

    status = _resolve_status(errors, warnings, strict)
    message = _build_message(status, errors, warnings)
    details["checked_operators"] = sorted(ops)
    return ValidationResult(status, message, "wq", errors, warnings, details)


def validate_local_expression(expression: str) -> ValidationResult:
    """本地模式校验（**行为保持原样**，仅统一返回结构）。

    本地 A 股引擎继续支持全部 pandas 算子（含 where），
    run_backtest / score_factor 等工具全部依赖这条路径。
    """
    expression = (expression or "").strip()
    if not expression:
        return ValidationResult(
            status="error", message="ERROR: 表达式为空", mode="local",
            errors=[Issue("syntax", "", "表达式为空")],
        )

    paren_issue = _check_balanced_parens(expression)
    if paren_issue:
        return ValidationResult(
            status="error", message=f"ERROR: {paren_issue.message}",
            mode="local", errors=[paren_issue],
        )

    try:
        func = parse_expression(expression, mode="local")
        func(_validation_dummy())
    except Exception as exc:  # noqa: BLE001 — 对齐上游：任何异常都转为 ERROR 文案
        return ValidationResult(
            status="error", message=f"ERROR: {exc}", mode="local",
            errors=[Issue("syntax", "", str(exc))],
        )
    return ValidationResult(status="ok", message=_MSG_LOCAL_OK, mode="local")


def _resolve_status(errors: list[Issue], warnings: list[Issue], strict: bool) -> str:
    """错误 → error；strict 下 warning 也升级为 error。"""
    if errors:
        return "error"
    if warnings:
        return "error" if strict else "warning"
    return "ok"


def _build_message(status: str, errors: list[Issue], warnings: list[Issue]) -> str:
    """构造 message。

    ⚠️ 向后兼容硬要求：`"OK: expression is valid for WQ BRAIN submission"` 与
    `"OK: expression is valid"` 两个字符串必须原样出现在返回里，
    现有 cron prompt / subagent 手册 / 测试都靠它们做字符串匹配。
    """
    if status == "ok":
        return _MSG_WQ_OK
    if status == "warning":
        parts = [f"WARNING: {w.message}" for w in warnings]
        parts.append("可通过校验，但建议先小样本验证")
        return " | ".join(parts)
    parts = [f"ERROR: {e.message}" for e in errors]
    if warnings:
        parts.extend(f"WARNING: {w.message}" for w in warnings)
    return " | ".join(parts)


def validate_expression(expression: str, mode: str = "local", strict: bool = False) -> ValidationResult:
    """统一入口：按 mode 分派到 wq / local 校验。"""
    if mode == "wq":
        return validate_wq_expression(expression, strict=strict)
    if mode == "local":
        return validate_local_expression(expression)
    return ValidationResult(
        status="error",
        message=f"ERROR: 未知模式 {mode!r}，支持 'wq' 或 'local'",
        mode=mode,
        errors=[Issue("syntax", "", f"未知模式 {mode!r}")],
    )


def precheck(expression: str, mode: str = "wq", strict: bool = True) -> ValidationResult:
    """批量预检入口。

    与 :func:`validate_wq_expression` 的区别：strict 默认 **True**。
    subagent 批量生成表达式后过这道闸时，应取最严口径——否则"目录不可用"
    的 warning 会让一批未知字段表达式被原样提交，重蹈 1~7 分钟等待的覆辙。
    """
    return validate_expression(expression, mode=mode, strict=strict)


__all__ = [
    "Issue",
    "ValidationResult",
    "precheck",
    "validate_expression",
    "validate_local_expression",
    "validate_wq_expression",
]
