"""官方算子目录改造的验收测试（GOAL_OPERATOR_CATALOG.md）。

背景
----
手写算子白名单上线后，**187 条真实在 WQ BRAIN 跑通过的表达式被误杀 48 条**
（误杀率 25.7%）：``signed_power`` 33 次、``ts_zscore`` 12 次。
已入库的 ACTIVE 因子 ``9qWZ9G2x``（Sharpe 2.29）因此被判 error。

本文件锁住四条铁律：
1. **零误杀** —— 已知跑通过的表达式一条都不能出现 status="error"；
2. **不误拦真错误** —— where / COMBO 作用域 / 非法字段 / 语法残缺必须拦下；
3. **合规放行** —— 官方合法算子（含以前被误杀的）必须 ok；
4. **降级不误杀** —— 目录不可用时降级为 warning，且 ``where`` 仍被拦下。

兼容性铁律（goal 4.5）：``mode="local"`` 行为不变，``validate_expression``
返回结构不变。
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pytest

from quantgpt import wq_operator_catalog
from quantgpt.expression_parser import parse_expression, wq_operators
from quantgpt.wq_validator import (
    extract_names,
    precheck,
    validate_expression,
)

FIXTURE = Path(__file__).parent / "fixtures" / "wq_known_good_expressions.json"


# ------------------------------------------------------------------ fixtures


@pytest.fixture(autouse=True)
def _isolated_catalog(monkeypatch, tmp_path):
    """每个测试都从"无目录"状态起步，避免互相污染。"""
    monkeypatch.setenv(
        "QUANTGPT_WQ_OPERATOR_CATALOG_PATH",
        str(tmp_path / "operator_catalog_absent.json"),
    )
    wq_operator_catalog.reset_for_tests()
    yield
    wq_operator_catalog.reset_for_tests()


# 官方 `GET /operators` 实测返回的算子（scope 原样保留）。
# 与 tests/test_expression_validation.py 的目录保持一致，含 REGULAR / COMBO。
_CATALOG_OPERATORS: dict[str, tuple[list[str], str]] = {
    # Arithmetic / Logical
    "add": (["REGULAR"], "Arithmetic"), "and": (["REGULAR"], "Logical"),
    "divide": (["REGULAR"], "Arithmetic"), "equal": (["REGULAR"], "Logical"),
    "greater": (["REGULAR"], "Logical"), "greater_equal": (["REGULAR"], "Logical"),
    "if_else": (["REGULAR"], "Logical"), "inverse": (["REGULAR"], "Arithmetic"),
    "is_nan": (["REGULAR"], "Logical"), "less": (["REGULAR"], "Logical"),
    "less_equal": (["REGULAR"], "Logical"), "multiply": (["REGULAR"], "Arithmetic"),
    "not": (["REGULAR"], "Logical"), "not_equal": (["REGULAR"], "Logical"),
    "or": (["REGULAR"], "Logical"), "reverse": (["REGULAR"], "Arithmetic"),
    "signed_power": (["REGULAR"], "Power"), "subtract": (["REGULAR"], "Arithmetic"),
    # Group / Other
    "densify": (["REGULAR"], "Group"), "group_backfill": (["REGULAR"], "Group"),
    "group_scale": (["REGULAR"], "Group"), "hump": (["REGULAR"], "Other"),
    "kth_element": (["REGULAR"], "Other"),
    # 27 个真实在 WQ BRAIN 跑通过的算子
    "abs": (["REGULAR"], "Arithmetic"), "group_mean": (["REGULAR"], "Group"),
    "group_neutralize": (["REGULAR"], "Group"), "group_rank": (["REGULAR"], "Group"),
    "group_zscore": (["REGULAR"], "Group"), "log": (["REGULAR"], "Arithmetic"),
    "max": (["REGULAR"], "Arithmetic"), "min": (["REGULAR"], "Arithmetic"),
    "power": (["REGULAR"], "Power"), "rank": (["REGULAR"], "Cross Sectional"),
    "sign": (["REGULAR"], "Arithmetic"), "sqrt": (["REGULAR"], "Arithmetic"),
    "trade_when": (["REGULAR"], "Other"), "winsorize": (["REGULAR"], "Other"),
    # Time Series（ts_delay / ts_delta / ts_mean / ts_rank / ts_sum / ts_zscore
    # 的官方 scope 是 ["REGULAR", "COMBO"]，向量表达式同样使用时序算子）
    "ts_backfill": (["REGULAR"], "Time Series"),
    "ts_corr": (["REGULAR"], "Time Series"),
    "ts_covariance": (["REGULAR"], "Time Series"),
    "ts_decay_linear": (["REGULAR"], "Time Series"),
    "ts_regression": (["REGULAR"], "Time Series"),
    "ts_std_dev": (["REGULAR"], "Time Series"),
    "ts_arg_max": (["REGULAR"], "Time Series"),
    "ts_arg_min": (["REGULAR"], "Time Series"),
    "ts_count_nans": (["REGULAR"], "Time Series"),
    "ts_product": (["REGULAR"], "Time Series"),
    "ts_quantile": (["REGULAR"], "Time Series"),
    "ts_scale": (["REGULAR"], "Time Series"),
    "ts_step": (["REGULAR"], "Time Series"),
    "ts_delay": (["REGULAR", "COMBO"], "Time Series"),
    "ts_delta": (["REGULAR", "COMBO"], "Time Series"),
    "ts_mean": (["REGULAR", "COMBO"], "Time Series"),
    "ts_rank": (["REGULAR", "COMBO"], "Time Series"),
    "ts_sum": (["REGULAR", "COMBO"], "Time Series"),
    "ts_zscore": (["REGULAR", "COMBO"], "Time Series"),
    # Combination（COMBO 专用）
    "vector_neut": (["COMBO"], "Combination"),
    "group_vector_neut": (["COMBO"], "Combination"),
    "vec_avg": (["COMBO"], "Combination"),
    "vec_count": (["COMBO"], "Combination"),
    "vec_ir": (["COMBO"], "Combination"),
    "vec_kurtosis": (["COMBO"], "Combination"),
    "vec_max": (["COMBO"], "Combination"),
    "vec_min": (["COMBO"], "Combination"),
    "vec_norm": (["COMBO"], "Combination"),
    "vec_percentage": (["COMBO"], "Combination"),
    "vec_range": (["COMBO"], "Combination"),
    "vec_skewness": (["COMBO"], "Combination"),
    "vec_stddev": (["COMBO"], "Combination"),
    "vec_sum": (["COMBO"], "Combination"),
}


@pytest.fixture
def operator_catalog(tmp_path, monkeypatch):
    """写入一份官方算子目录缓存，让算子校验走"目录可用"路径。"""
    path = tmp_path / "wq_operator_catalog.json"
    path.write_text(
        json.dumps({
            "fetched_at": time.time(),
            "count": len(_CATALOG_OPERATORS),
            "operators": [
                {"name": name, "category": category, "scope": scope,
                 "definition": "", "description": ""}
                for name, (scope, category) in _CATALOG_OPERATORS.items()
            ],
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("QUANTGPT_WQ_OPERATOR_CATALOG_PATH", str(path))
    wq_operator_catalog.reset_for_tests()
    yield path
    wq_operator_catalog.reset_for_tests()


def _load_known_good() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


# ------------------------------------------------- 验收标准 5.1 零误杀（最高优先级）


class TestZeroFalseKills:
    """5.1：187 条真实跑通过表达式（此处内嵌 115 条样本）→ 0 误杀。

    这是本次改造的**最高优先级**验收项。任何一条出现 status="error"
    都是回归缺陷——这些表达式全部在 WQ 生产环境真实跑通过。
    """

    def test_known_good_expressions_have_zero_errors(self):
        known_good = _load_known_good()
        assert known_good, "known-good 样本为空，回归测试失去意义"

        failures = []
        for item in known_good:
            result = validate_expression(item["expr"], mode="wq")
            if result.status == "error":
                failures.append(
                    f"[{item['id']}] {item['expr'][:90]}\n"
                    f"    -> {result.message[:200]}"
                )

        assert not failures, (
            f"{len(failures)}/{len(known_good)} 条真实跑通过的表达式被误杀：\n"
            + "\n".join(failures[:10])
        )

    def test_known_good_expressions_never_blocked(self):
        """误杀之外还要保证 submit_allowed 一致（不因 warning 被 precheck 误伤）。"""
        for item in _load_known_good():
            result = validate_expression(item["expr"], mode="wq")
            payload = result.to_dict()
            assert payload["submit_allowed"] is True, (
                f"[{item['id']}] 被判定不可提交：{payload['message'][:120]}"
            )

    def test_previously_false_killed_operators_now_pass(self):
        """回归点：`signed_power`（33 次）/ `ts_zscore`（12 次）曾被误杀。"""
        for expression in (
            "signed_power(returns, 2)",
            "ts_zscore(returns, 60)",
            "signed_power(ts_delta(close, 5), 1.5)",
            "group_rank(signed_power(returns, 2), subindustry)",
        ):
            result = validate_expression(expression, mode="wq")
            assert result.status != "error", f"{expression}: {result.message}"

    def test_known_good_contains_the_falsely_killed_operators(self):
        """确保样本真的覆盖了回归点，否则上面的测试是空转。"""
        blob = " ".join(item["expr"] for item in _load_known_good())
        assert "signed_power" in blob
        assert "ts_zscore" in blob


# ------------------------------------------------- 验收标准 5.2 不误拦真实错误


class TestRealErrorsStillBlocked:
    """5.2：确定性错误必须本地拦下（不靠服务端 1~7 分钟等待）。"""

    def test_where_rejected_as_operator(self):
        result = validate_expression("where(returns > 0, 1, -1)", mode="wq")
        assert result.status == "error"
        assert "operator" in [e.kind for e in result.errors]
        issue = next(e for e in result.errors if e.kind == "operator")
        assert issue.name == "where"
        assert "trade_when" in issue.hint  # 必须给出官方替代写法

    @pytest.mark.parametrize(
        "expression",
        [
            "vector_neut(returns, ts_delay(returns, 1))",
            "vec_max(returns, ts_delay(returns, 1))",
            "vec_count(returns)",
            "group_vector_neut(returns, adv20)",
        ],
    )
    def test_combo_scope_operators_rejected_in_regular(self, expression):
        """COMBO 作用域算子写进 REGULAR 表达式必须报 error，并说明作用域。"""
        result = validate_expression(expression, mode="wq")
        assert result.status == "error", result.message
        issue = next(e for e in result.errors if e.kind == "operator")
        assert "COMBO" in issue.message, issue.message

    def test_combo_scope_operators_allowed_in_combo_expression(self):
        """作用域校验不能反过来误杀：COMBO 表达式用 COMBO 算子必须放行。"""
        result = validate_expression(
            "vec_max(returns, ts_delay(returns, 1))",
            mode="wq",
            expression_type="COMBO",
        )
        assert result.status != "error", result.message

    def test_syntax_dangling_operator_rejected(self):
        """`rank(close +)` 曾在 wq 模式下被解析成合法表达式，直到服务端才报错。"""
        result = validate_expression("rank(close +)", mode="wq")
        assert result.status == "error"
        assert "syntax" in [e.kind for e in result.errors]

    @pytest.mark.parametrize(
        "expression",
        [
            "rank(close +)",
            "close -",
            "ts_sum(returns, 20) -",
            "rank(close)*(1 + )",
        ],
    )
    def test_dangling_operator_variants(self, expression):
        result = validate_expression(expression, mode="wq")
        assert result.status == "error", f"{expression}: {result.message}"

    @pytest.mark.parametrize(
        "expression",
        [
            "rank(-1)",
            "+1",
            "rank(-close)",
            "rank(close - open)",
            "-0.5 * rank(close)",
            "rank(close) - -1",
            "group_mean(returns, 1, market)",
        ],
    )
    def test_unary_signs_not_mistaken_for_dangling(self, expression):
        """一元正负号是合法字面量，绝不能被残缺运算符检测误杀。"""
        result = validate_expression(expression, mode="wq")
        assert not any(e.kind == "syntax" for e in result.errors), result.message


# ------------------------------------------------- 验收标准 5.3 合规表达式


class TestCompliantExpressionsPass:
    """5.3：goal 列出的合规表达式必须全部 ok。"""

    @pytest.mark.parametrize(
        "expression",
        [
            "signed_power(returns, 2)",
            "ts_zscore(returns, 60)",
            "group_rank(ts_rank(est_eps, 126), subindustry)",
            "group_neutralize(rank(est_eps/close), subindustry)",
            "group_zscore(winsorize(rank(returns), std=4), industry)",
            "trade_when(volume>adv20, returns, -returns)",
        ],
    )
    def test_compliant_wq_expressions(self, expression):
        result = validate_expression(expression, mode="wq")
        assert result.status != "error", f"{expression}: {result.message}"

    def test_local_mode_where_still_works(self):
        """兼容性铁律 4.5：`where` 只对 WQ 失效，本地 pandas 语义保持。"""
        result = validate_expression("rank(close-open)", mode="local")
        assert result.status == "ok"
        assert "OK: expression is valid" in result.message
        assert callable(
            parse_expression("where(close > open, 1, -1)", mode="local")
        )

    def test_local_mode_tanh_still_works(self):
        assert callable(parse_expression("tanh(close)", mode="local"))

    def test_return_structure_unchanged(self):
        """兼容性铁律 4.5：返回结构不得变动。"""
        payload = validate_expression("rank(close)", mode="wq").to_dict()
        for key in (
            "status",
            "level",
            "message",
            "errors",
            "warnings",
            "details",
            "submit_allowed",
        ):
            assert key in payload, f"缺少返回字段 {key}"
        assert payload["status"] == payload["level"]
        assert payload["submit_allowed"] is True

    def test_where_absent_from_wq_operators(self):
        assert "where" not in wq_operators()
        assert "trade_when" in wq_operators()


# ------------------------------------------------- goal 3.2 兼容别名


class TestLegacyAliasesNotIllegal:
    """goal 3.2：本地有、官方无的写法必须保留为**兼容别名**，不得判非法。"""

    @pytest.mark.parametrize(
        ("alias", "official"),
        [
            ("ts_argmin", "ts_arg_min"),
            ("ts_argmax", "ts_arg_max"),
            ("ts_shift", "ts_delay"),
            ("sign_power", "signed_power"),
            ("humpdecay", "hump"),
        ],
    )
    def test_alias_not_blocked(self, alias, official, operator_catalog):
        """别名在目录可用时：warning + 提示官方写法，**不得** error。"""
        assert wq_operator_catalog.resolve_alias(alias) == official
        result = validate_expression(f"{alias}(returns, 20)", mode="wq")
        assert result.status != "error", f"{alias}: {result.message}"
        assert any(official in w.message for w in result.warnings)

    @pytest.mark.parametrize("alias", ["ts_argmin", "ts_shift", "sign_power"])
    def test_alias_not_blocked_without_catalog(self, alias):
        """目录不可用时别名同样不得被误杀。"""
        result = validate_expression(f"{alias}(returns, 20)", mode="wq")
        assert result.status != "error", f"{alias}: {result.message}"

    @pytest.mark.parametrize(
        ("expression", "should_warn"),
        [
            # 用户写的是官方名 → 不该被提醒"改写成官方写法"（自相矛盾）
            ("ts_arg_max(returns, 20)", False),
            ("ts_delay(returns, 5)", False),
            ("ts_std_dev(returns, 20)", False),
            ("rank(ts_delta(close, 5) / ts_std_dev(returns, 20))", False),
            # 用户写的是历史写法 → 应该提示官方写法
            ("ts_argmin(returns, 20)", True),
            ("ts_shift(returns, 5)", True),
            ("sign_power(returns, 2)", True),
            ("humpdecay(returns, 0.01)", True),
        ],
    )
    def test_hint_only_for_legacy_spellings(
        self, expression, should_warn, operator_catalog
    ):
        """``extract_names`` 会把官方名与历史写法归一到同一个键。

        提示必须回看原文判断：写官方名不该被提醒改写，写历史写法才提醒。
        弄反了就会对合规表达式产生莫名其妙的 warning。
        """
        result = validate_expression(expression, mode="wq")
        assert result.status != "error", f"{expression}: {result.message}"
        alias_warnings = [
            w for w in result.warnings
            if "不是官方算子名" in w.message
        ]
        assert bool(alias_warnings) is should_warn, (
            f"{expression}: alias_warnings={alias_warnings}"
        )

    def test_alias_scope_follows_official_name(self):
        """别名的 scope 必须跟随官方写法。

        否则 COMBO 表达式里的历史写法会被 scope 校验误杀——与本次改造
        要消灭的误杀同源（`ts_shift` → `ts_delay`，官方 scope 含 COMBO）。
        """
        assert wq_operator_catalog.scope_of("ts_shift") == wq_operator_catalog.scope_of(
            "ts_delay"
        )
        result = validate_expression(
            "ts_shift(returns, 5)", mode="wq", expression_type="COMBO"
        )
        assert not any(e.kind == "operator" for e in result.errors), result.message

    def test_combo_shared_operators_available_in_combo(self):
        """官方 scope = [REGULAR, COMBO] 的算子在 COMBO 表达式里必须可用。"""
        for operator in ("ts_delay", "ts_delta", "ts_mean", "ts_rank", "ts_sum"):
            scope = wq_operator_catalog.scope_of(operator)
            assert "COMBO" in scope, f"{operator} 兜底 scope 应含 COMBO：{scope}"


# ------------------------------------------------- 4.4 参数解析修复


class TestFieldExtractionFixes:
    """4.4：命名参数 / 分组字段不得被误判成 data field。

    这些误判在生产日志里已经出现过 ``未知字段 'std=4'`` /
    ``未知字段 'subindustry'``，属必须修的缺陷。
    """

    @pytest.mark.parametrize(
        ("expression", "forbidden"),
        [
            ("group_zscore(winsorize(rank(returns), std=4), industry)", "std"),
            ("winsorize(rank(returns), std=2.5)", "std"),
            ("rank(group_rank(returns, subindustry))", "std"),
            ("group_rank(ts_rank(est_eps, 126), subindustry)", "est_eps=126"),
        ],
    )
    def test_named_args_not_treated_as_fields(self, expression, forbidden):
        _, variables = extract_names(expression)
        assert not any("=" in v for v in variables), variables
        assert forbidden not in variables

    @pytest.mark.parametrize(
        "group_field", ["subindustry", "industry", "sector", "market"]
    )
    def test_group_fields_recognised_as_group_not_datafield(self, group_field):
        from quantgpt.expression_parser import variable_category

        assert variable_category(group_field) == "group"
        _, variables = extract_names(f"group_rank(returns, {group_field})")
        assert group_field in variables  # 不该从变量集合里消失

    def test_group_expression_has_no_field_errors(self):
        """分组字段不是 data field：`group_rank(x, subindustry)` 不应报字段错误。"""
        result = validate_expression(
            "group_rank(ts_rank(est_eps, 126), subindustry)", mode="wq"
        )
        assert not any(e.kind == "field" for e in result.errors), result.message

    def test_named_arg_expression_has_no_field_errors(self):
        result = validate_expression(
            "group_zscore(winsorize(rank(returns), std=4), industry)", mode="wq"
        )
        assert not any(e.kind == "field" for e in result.errors), result.message

    def test_empty_string_literal_not_treated_as_field(self):
        _, variables = extract_names("group_rank(returns, '')")
        assert "" not in variables


# ------------------------------------------------- 验收标准 5.4 目录降级


class TestCatalogDegradation:
    """5.4：目录不可用时降级为 warning，绝不阻断（误杀比漏检严重得多）。"""

    def test_where_still_rejected_without_catalog(self):
        """显式黑名单保证：`where` 在任何目录状态下都被拦下。"""
        result = validate_expression("where(returns > 0, 1, -1)", mode="wq")
        assert result.status == "error"
        assert "operator" in [e.kind for e in result.errors]

    @pytest.mark.parametrize(
        "operator",
        [
            # 27 个真实跑通过的算子（goal 2.2）
            "abs", "group_mean", "group_neutralize", "group_rank",
            "group_zscore", "log", "max", "min", "power", "rank", "sign",
            "signed_power", "sqrt", "trade_when", "ts_backfill", "ts_corr",
            "ts_covariance", "ts_decay_linear", "ts_delay", "ts_delta",
            "ts_mean", "ts_rank", "ts_regression", "ts_std_dev", "ts_sum",
            "ts_zscore", "winsorize",
        ],
    )
    def test_core_operators_pass_without_catalog(self, operator):
        assert operator in wq_operator_catalog.fallback_operators()
        assert operator in wq_operators()

    def test_known_good_zero_errors_without_catalog(self):
        """目录不可用时也必须 0 误杀（这是整个改造存在的理由）。"""
        failures = [
            f"[{i['id']}] {validate_expression(i['expr'], mode='wq').message[:160]}"
            for i in _load_known_good()
            if validate_expression(i["expr"], mode="wq").status == "error"
        ]
        assert not failures, "\n".join(failures[:10])

    def test_unknown_operator_degrades_to_warning_not_error(self):
        """目录不可用 + 算子不在兜底集 → warning，不阻断。"""
        result = validate_expression("ts_some_future_op(close, 20)", mode="wq")
        assert result.status == "warning"
        assert any(w.kind == "operator" for w in result.warnings)

    def test_local_only_operators_rejected_even_without_catalog(self):
        """本地专有算子（tanh/rsi/...）在任何目录状态下都必然非法。

        这不是"目录查不到"，而是本地语义定义，所以不参与降级。
        """
        for operator in ("tanh", "sigmoid", "ema", "rsi"):
            result = validate_expression(f"{operator}(close)", mode="wq")
            assert result.status == "error", f"{operator}: {result.message}"

    def test_degradation_details_reported(self):
        """降级必须自我标注，让 subagent 知道这条没被真正校验过。"""
        details = validate_expression("rank(close)", mode="wq").details
        assert details["operator_catalog_available"] is False
        assert details["operator_catalog_stale"] is True
        assert details["expression_type"] == "REGULAR"

    def test_precheck_strict_still_blocks_unknown_ops(self):
        """strict 口径下 warning 升级为 error（批量提交场景的最严闸门）。"""
        result = precheck("ts_some_future_op(close, 20)", mode="wq")
        assert result.status == "error"


# ------------------------------------- 5.1 追加：两个目录都可用（生产真实配置）


class TestWithFieldCatalogAvailable:
    """生产环境的真实状态是**字段目录 + 算子目录都可用**（4367 字段 / 66 算子）。

    5.1 的零误杀必须在这个状态下也成立：算子改造如果只是"把误杀从算子挪到
    字段"，等于没修。这里用样本里真实出现的全部 data field 构造一份完整目录，
    复现生产配置。
    """

    @pytest.fixture
    def _full_catalog(self, tmp_path, monkeypatch):
        from quantgpt import wq_field_catalog
        from quantgpt.expression_parser import variable_category

        fields = set()
        for item in _load_known_good():
            _, variables = extract_names(item["expr"])
            fields.update(v for v in variables if variable_category(v) == "datafield")

        path = tmp_path / "full_field_catalog.json"
        path.write_text(
            json.dumps({
                "fetched_at": time.time(),
                "count": len(fields),
                "fields": sorted(fields),
                "datasets": {"fixture": sorted(fields)},
            }),
            encoding="utf-8",
        )
        monkeypatch.setenv("QUANTGPT_WQ_CATALOG_PATH", str(path))
        wq_field_catalog.reset_for_tests()
        yield path
        wq_field_catalog.reset_for_tests()

    def test_zero_errors_with_both_catalogs_available(self, _full_catalog):
        known_good = _load_known_good()
        failures = [
            f"[{i['id']}] {i['expr'][:80]} -> {validate_expression(i['expr'], mode='wq').message[:160]}"
            for i in known_good
            if validate_expression(i["expr"], mode="wq").status == "error"
        ]
        assert not failures, (
            f"{len(failures)}/{len(known_good)} 条被误杀（两个目录均可用）:\n"
            + "\n".join(failures[:10])
        )

    def test_no_field_errors_for_known_good(self, _full_catalog):
        """算子改造不能把误杀转移成字段误报。"""
        for item in _load_known_good():
            result = validate_expression(item["expr"], mode="wq")
            assert not any(e.kind == "field" for e in result.errors), (
                f"[{item['id']}] 字段误报：{result.message[:160]}"
            )

    def test_group_params_still_not_fields_with_catalog(self, _full_catalog):
        """分组字段不是 data field——目录可用时也不能被判非法字段。"""
        result = validate_expression(
            "group_rank(ts_rank(est_eps, 126), subindustry)", mode="wq"
        )
        assert not any(e.kind == "field" for e in result.errors), result.message


# ------------------------------------------------- 目录模块自身


class TestOperatorCatalogModule:
    def test_fallback_contains_all_27_core_operators(self):
        fallback = wq_operator_catalog.fallback_operators()
        for name in ("signed_power", "ts_zscore", "trade_when", "group_rank"):
            assert name in fallback

    def test_fallback_excludes_where(self):
        assert "where" not in wq_operator_catalog.fallback_operators()
        assert wq_operator_catalog.is_blacklisted("where")

    def test_fallback_marks_combo_scope(self):
        assert wq_operator_catalog.scope_of("vec_max") == ["COMBO"]
        assert wq_operator_catalog.scope_of("vector_neut") == ["COMBO"]
        assert wq_operator_catalog.scope_of("rank") == ["REGULAR"]

    def test_unknown_scope_returns_empty(self):
        """scope 未知 → 空列表，调用方必须跳过 scope 校验而非报错。"""
        assert wq_operator_catalog.scope_of("no_such_operator_xyz") == []

    def test_status_shape(self):
        status = wq_operator_catalog.catalog_status()
        for key in (
            "available", "stale", "count", "fetched_at",
            "age_seconds", "ttl_seconds", "path", "last_error",
        ):
            assert key in status

    def test_cache_roundtrip(self, tmp_path, monkeypatch):
        """落盘 → 重载：算子目录必须能像字段目录一样跨进程复用。"""
        path = tmp_path / "cached_operators.json"
        monkeypatch.setenv("QUANTGPT_WQ_OPERATOR_CATALOG_PATH", str(path))
        path.write_text(
            json.dumps({
                "fetched_at": time.time(),
                "count": 2,
                "operators": [
                    {"name": "signed_power", "category": "Power",
                     "scope": ["REGULAR"], "definition": "", "description": ""},
                    {"name": "vec_max", "category": "Combination",
                     "scope": ["COMBO"], "definition": "", "description": ""},
                ],
            }),
            encoding="utf-8",
        )
        wq_operator_catalog.reset_for_tests()

        assert wq_operator_catalog.catalog_status()["available"] is True
        assert wq_operator_catalog.is_known("signed_power") is True
        assert wq_operator_catalog.scope_of("vec_max") == ["COMBO"]
        info = wq_operator_catalog.operator_info("signed_power")
        assert info["category"] == "Power"

    def test_corrupt_cache_degrades_safely(self, tmp_path, monkeypatch):
        """缓存损坏绝不能抛异常——必须降级到兜底集。"""
        path = tmp_path / "corrupt.json"
        path.write_text("{not json at all", encoding="utf-8")
        monkeypatch.setenv("QUANTGPT_WQ_OPERATOR_CATALOG_PATH", str(path))
        wq_operator_catalog.reset_for_tests()

        assert wq_operator_catalog.catalog_status()["available"] is False
        assert "signed_power" in wq_operator_catalog.catalog_operators()
        result = validate_expression("signed_power(returns, 2)", mode="wq")
        assert result.status != "error"

    def test_legacy_aliases_resolve_to_official_names(self):
        assert wq_operator_catalog.resolve_alias("ts_argmin") == "ts_arg_min"
        assert wq_operator_catalog.resolve_alias("ts_argmax") == "ts_arg_max"
        assert wq_operator_catalog.resolve_alias("ts_delay") is None

    def test_refresh_returns_false_without_network(self, monkeypatch):
        """refresh 失败必须返回 False 而**不是**抛异常（硬性要求）。"""
        monkeypatch.setenv("WQ_BRAIN_EMAIL", "")
        monkeypatch.setenv("WQ_BRAIN_PASSWORD", "")
        assert wq_operator_catalog.refresh() is False
        # 降级后仍可用
        assert "signed_power" in wq_operator_catalog.catalog_operators()


# ------------------------------------------------- 兼容性 / 性能


class TestBackwardCompatibility:
    def test_operator_set_tracks_catalog(self, tmp_path, monkeypatch):
        """目录恢复后算子集合必须同步更新，不能冻结在 import 时刻。"""
        from quantgpt import expression_parser

        before = set(expression_parser.wq_operators())
        assert "brand_new_operator" not in before

        path = tmp_path / "grew.json"
        path.write_text(
            json.dumps({
                "fetched_at": time.time(),
                "count": 1,
                "operators": [
                    {"name": "brand_new_operator", "category": "Other",
                     "scope": ["REGULAR"], "definition": "", "description": ""},
                ],
            }),
            encoding="utf-8",
        )
        monkeypatch.setenv("QUANTGPT_WQ_OPERATOR_CATALOG_PATH", str(path))
        wq_operator_catalog.reset_for_tests()

        after = set(expression_parser.wq_operators())
        assert "brand_new_operator" in after
        assert "brand_new_operator" not in before

    def test_precheck_performance_under_100ms(self):
        """验收标准 D：precheck_expression <100ms（纯本地，无网络）。"""
        expressions = [i["expr"] for i in _load_known_good()][:30]
        start = time.perf_counter()
        for expression in expressions:
            precheck(expression, mode="wq")
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 100 * len(expressions), (
            f"{len(expressions)} 条耗时 {elapsed_ms:.0f}ms，超出 <100ms/条 指标"
        )


# ------------------------------------------------- 数据完整性守卫


def test_fixture_is_wq_valid_subset():
    """守卫：样本里不应出现 `where` / COMBO 算子（它们从来不是 REGULAR 算子）。"""
    blob = " ".join(item["expr"] for item in _load_known_good())
    assert not re.search(r"\bwhere\s*\(", blob, re.IGNORECASE)
    assert not re.search(r"\bvec_[a-z_]+\s*\(", blob, re.IGNORECASE)
    assert not re.search(r"\bvector_neut\s*\(", blob, re.IGNORECASE)
