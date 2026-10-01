"""WQ 表达式提交前校验闸门测试。

覆盖 GOAL_WQ_VALIDATION.md 的验收标准：
- A 回归：本地 A 股工具不退化
- B 核心：三类语义错误（算子/字段）必须本地拦截
- C 合规表达式不得误杀（**误杀比漏检更严重**）
- D 性能：precheck <100ms
- E 字段目录可用性 / 降级行为
"""

import json
import time

import pytest

from quantgpt import wq_field_catalog
from quantgpt.expression_parser import (
    ExpressionParser,
    parse_expression,
    variable_category,
    wq_operators,
)
from quantgpt.wq_validator import (
    extract_names,
    precheck,
    validate_expression,
)

# 验收标准 C 里的真实可用字段。完整目录需要联网拉取，这里只放被测表达式
# 实际引用到的字段 + 常见字段，保证"查得到"路径被覆盖。
_CATALOG_FIELDS = {
    # 价格 / 内置
    "close", "open", "high", "low", "volume", "vwap", "returns", "cap",
    # 基本面
    "assets", "sales", "revenue", "earnings", "net_income", "equity",
    "book_value", "cash_flow", "capex", "dividends", "inventory",
    # 分析师预期
    "est_eps", "est_revenue",
    # 其它数据集
    "turnover", "turnover_rate", "shares_outstanding", "mdf_oey",
    "implied_volatility", "short_interest", "snt_buzz",
}


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    """写入一份内存字段目录缓存，让字段校验走"目录可用"路径。"""
    path = tmp_path / "wq_field_catalog.json"
    path.write_text(
        json.dumps(
            {
                "fetched_at": time.time(),
                "count": len(_CATALOG_FIELDS),
                "fields": sorted(_CATALOG_FIELDS),
                "datasets": {"fundamental": sorted(_CATALOG_FIELDS)},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("QUANTGPT_WQ_CATALOG_PATH", str(path))
    wq_field_catalog.reset_for_tests()
    yield path
    wq_field_catalog.reset_for_tests()


@pytest.fixture
def no_catalog(tmp_path, monkeypatch):
    """字段目录缺失：模拟 WQ 不可达 / 首次拉取失败。"""
    monkeypatch.setenv(
        "QUANTGPT_WQ_CATALOG_PATH", str(tmp_path / "does_not_exist.json")
    )
    wq_field_catalog.reset_for_tests()
    yield
    wq_field_catalog.reset_for_tests()


# ---------------------------------------------------------------- 验收标准 A


class TestRegression:
    """A：现有能力不退化。"""

    def test_local_mode_still_works(self):
        result = validate_expression("rank(close - open)", mode="local")
        assert result.status == "ok"
        # 兼容旧调用方的字符串必须原样保留
        assert "OK: expression is valid" in result.message

    def test_local_mode_keeps_pandas_where(self):
        """`where` 只从 WQ 白名单移除，本地 A 股引擎仍须可用。"""
        result = validate_expression("where(close > open, 1, -1)", mode="local")
        assert result.status == "ok"
        assert callable(parse_expression("where(close > open, 1, -1)", mode="local"))

    def test_wq_syntax_error_still_blocked(self):
        result = validate_expression("rank(close +", mode="wq")
        assert result.status == "error"
        assert result.errors[0].kind == "syntax"

    def test_wq_valid_expression(self):
        result = validate_expression("rank(close)", mode="wq")
        assert result.status == "ok"
        assert "OK: expression is valid for WQ BRAIN submission" in result.message

    def test_backward_compatible_ok_string(self):
        """goal 要求：返回中必须仍能找到该原文案。"""
        assert "OK: expression is valid for WQ BRAIN submission" in (
            validate_expression("rank(close)", mode="wq").message
        )
        assert "OK: expression is valid" in validate_expression(
            "rank(close)", mode="local"
        ).message


# ---------------------------------------------------------------- 验收标准 B


class TestThreeErrorClassesBlocked:
    """B（核心）：三类语义错误必须返回 status=error。"""

    @pytest.mark.parametrize(
        ("expression", "expected_kind"),
        [
            ("where(returns > 0, 1, -1)", "operator"),
            ("rank(anl4_fs_detail_estimates_basic_v4_nd_eps_std)", "field"),
            ("rank(gross_profit/assets)", "field"),
            ("rank(nonexistent_field_xyz)", "field"),
            ("ts_mean(unknown_macro_series, 20)", "field"),
        ],
    )
    def test_error_is_blocked(self, catalog, expression, expected_kind):
        result = validate_expression(expression, mode="wq")
        assert result.status == "error", result.message
        assert result.blocked
        assert expected_kind in [e.kind for e in result.errors]

    def test_where_gives_actionable_hint(self, catalog):
        """`where` 必须给出可操作的替代建议（trade_when）。"""
        result = validate_expression("where(returns > 0, 1, -1)", mode="wq")
        operator_errors = [e for e in result.errors if e.kind == "operator"]
        assert operator_errors, result.message
        issue = operator_errors[0]
        assert issue.name == "where"
        assert "trade_when" in issue.hint
        assert "trade_when" in issue.message

    def test_invalid_field_error_message_mentions_wq(self, catalog):
        result = validate_expression("rank(gross_profit/assets)", mode="wq")
        combined = " ".join(e.message for e in result.errors)
        assert "gross_profit" in combined
        assert "data field" in combined.lower()

    def test_all_rejected_expressions_have_hints(self, catalog):
        """B 表里每条错误都要有可操作 hint。"""
        expressions = [
            "where(returns > 0, 1, -1)",
            "rank(anl4_fs_detail_estimates_basic_v4_nd_eps_std)",
            "rank(gross_profit/assets)",
            "rank(nonexistent_field_xyz)",
            "ts_mean(unknown_macro_series, 20)",
        ]
        for expression in expressions:
            result = validate_expression(expression, mode="wq")
            assert result.status == "error", expression
            assert any(e.hint for e in result.errors), (
                f"{expression} 的错误缺少可操作 hint"
            )

    def test_wq_where_rejected_but_local_allowed(self, catalog):
        with pytest.raises(ValueError, match="trade_when|WQ 模式下"):
            parse_expression("where(returns > 0, 1, -1)", mode="wq")
        # 本地照常可用
        assert callable(parse_expression("where(close > open, 1, -1)", mode="local"))

    def test_where_not_in_wq_operator_whitelist(self):
        assert "where" not in wq_operators()
        assert "trade_when" in wq_operators()

    @pytest.mark.parametrize("operator", ["tanh", "sigmoid", "ema", "rsi", "clip"])
    def test_local_only_operators_rejected(self, catalog, operator):
        result = validate_expression(f"{operator}(close)", mode="wq")
        assert result.status == "error"
        assert any(e.kind == "operator" for e in result.errors)

    def test_typo_operator_rejected(self, catalog):
        """白名单外算子不再透传——拼写错误必须本地拦住。"""
        result = validate_expression("ts_mena(close, 20)", mode="wq")
        assert result.status == "error"
        assert any(e.name == "ts_mena" and e.kind == "operator" for e in result.errors)


# ---------------------------------------------------------------- 验收标准 C


class TestNoFalsePositives:
    """C：合规表达式不得被误杀。误杀合规表达式比漏检更严重。"""

    @pytest.mark.parametrize(
        "expression",
        [
            "rank(ts_delta(close, 5) / ts_std_dev(returns, 20))",
            "group_rank(ts_rank(turnover, 126), subindustry)",
            "trade_when(volume > adv20, returns, -returns)",
            "group_neutralize(rank(est_eps / close), subindustry)",
            "group_zscore(winsorize(rank(returns), std=4), industry)",
        ],
    )
    def test_compliant_expression_is_ok(self, catalog, expression):
        result = validate_expression(expression, mode="wq")
        assert result.status == "ok", (
            f"误杀合规表达式 {expression!r}: {result.message}"
        )
        assert not result.blocked

    def test_turnover_not_in_catalog_returns_warning_not_error(self, no_catalog):
        """字段目录不可用时，未知字段必须是 warning 而非 error。"""
        result = validate_expression(
            "group_rank(ts_rank(turnover, 126), subindustry)", mode="wq"
        )
        assert result.status == "warning"
        assert result.warnings
        assert any(w.kind == "field" for w in result.warnings)
        assert not result.blocked

    def test_aliases_are_normalized_before_whitelist_check(self):
        """`ts_std_dev` 是 `ts_std` 的别名，比对白名单前必须先归一。"""
        operators, _ = extract_names("rank(ts_std_dev(returns, 20))")
        assert "ts_std" in operators
        assert "ts_std_dev" not in operators

    def test_named_argument_is_not_treated_as_field(self, catalog):
        """`winsorize(x, std=4)` 的 std 是参数名，不是 data field。"""
        result = validate_expression(
            "group_zscore(winsorize(rank(returns), std=4), industry)", mode="wq"
        )
        assert result.status == "ok", result.message

    def test_group_fields_not_treated_as_datafield(self, catalog):
        """分组字段在 WQ 是独立 namespace，不查 data field 目录。"""
        for name in ("industry", "subindustry", "sector", "market"):
            assert variable_category(name) == "group"

    def test_dotted_group_field_prefix_preserved(self, catalog):
        result = validate_expression(
            "group_neutralize(rank(close), IndClass.industry)", mode="wq"
        )
        assert result.status == "ok", result.message

    def test_market_cap_rejected_with_cap_hint(self, catalog):
        """`market_cap` 在 WQ 侧不存在，必须报错并提示改用 cap。"""
        result = validate_expression("rank(market_cap)", mode="wq")
        assert result.status == "error"
        assert "cap" in " ".join(e.message + e.hint for e in result.errors)

    def test_cap_is_valid_builtin(self, catalog):
        result = validate_expression("rank(cap)", mode="wq")
        assert result.status == "ok", result.message

    def test_trade_when_accepts_expression_hold_value(self, catalog):
        """trade_when 第三参数允许表达式（官方标准写法 -returns）。"""
        assert callable(
            parse_expression("trade_when(volume > adv20, returns, -returns)", mode="wq")
        )
        assert callable(
            parse_expression("trade_when(volume > adv20, returns, 0)", mode="wq")
        )


# ------------------------------------------------------------ 量纲 / 闸门④


class TestDimensionChecks:
    """闸门④：量纲、窗口、深度、长度。"""

    def test_adv_window_out_of_range(self, catalog):
        result = validate_expression("rank(volume / adv9999)", mode="wq")
        assert result.status == "error"
        assert any(e.kind == "dimension" for e in result.errors)

    def test_adv_valid_range_ok(self, catalog):
        for n in (1, 20, 500):
            result = validate_expression(f"rank(volume / adv{n})", mode="wq")
            assert result.status == "ok", f"adv{n} 应合法: {result.message}"

    def test_adv_non_numeric_rejected(self, catalog):
        result = validate_expression("rank(volume / advx)", mode="wq")
        assert result.status == "error"

    def test_expression_too_long(self, catalog):
        result = validate_expression("rank(" + "close + " * 300 + "close)", mode="wq")
        assert result.status == "error"
        assert any(e.kind == "dimension" for e in result.errors)

    def test_nesting_too_deep(self, catalog):
        expr = "close"
        for _ in range(ExpressionParser.MAX_DEPTH + 5):
            expr = f"abs({expr})"
        result = validate_expression(expr, mode="wq")
        assert result.status == "error"

    def test_unit_violation_detected(self, catalog):
        """WQ 专用量纲规则仍生效（不得把常数与价格列相加）。"""
        result = validate_expression("rank(close + 0.0001)", mode="wq")
        assert result.status == "error"


# ------------------------------------------------------- 严重级别 / strict


class TestSeverityLevels:
    """返回结构必须区分 ok / warning / error 三级。"""

    def test_result_schema(self, catalog):
        payload = validate_expression("rank(close)", mode="wq").to_dict()
        for key in ("status", "level", "message", "errors", "warnings"):
            assert key in payload, f"返回结构缺少 {key}"
        assert payload["status"] == payload["level"]

    def test_error_entry_schema(self, catalog):
        payload = validate_expression("where(close > open, 1, -1)", mode="wq").to_dict()
        assert payload["errors"]
        for entry in payload["errors"]:
            assert set(("kind", "name", "message", "hint")) <= set(entry)

    def test_strict_false_allows_warning(self, no_catalog):
        result = validate_expression("rank(some_unknown_field)", mode="wq")
        assert result.status == "warning"
        assert not result.blocked

    def test_strict_true_blocks_warning(self, no_catalog):
        result = validate_expression(
            "rank(some_unknown_field)", mode="wq", strict=True
        )
        assert result.status == "error"
        assert result.blocked

    def test_precheck_defaults_to_strict(self, no_catalog):
        """批量预检必须取最严口径，否则整批未知字段会被原样提交。"""
        assert precheck("rank(some_unknown_field)").status == "error"
        assert validate_expression(
            "rank(some_unknown_field)", mode="wq"
        ).status == "warning"

    def test_json_roundtrip(self, catalog):
        payload = json.loads(validate_expression("rank(close)", mode="wq").to_json())
        assert payload["status"] == "ok"

    def test_unknown_mode_rejected(self):
        result = validate_expression("rank(close)", mode="bogus")
        assert result.status == "error"


# ------------------------------------------------------------ 字段目录 E


class TestFieldCatalog:
    """E：字段目录可用性与降级。"""

    def test_catalog_status_fields(self, catalog):
        status = wq_field_catalog.catalog_status()
        assert status["available"] is True
        assert status["stale"] is False
        assert status["count"] == len(_CATALOG_FIELDS)

    def test_field_exists(self, catalog):
        assert wq_field_catalog.field_exists("close") is True
        assert wq_field_catalog.field_exists("gross_profit") is False

    def test_list_fields_and_datasets(self, catalog):
        fields = wq_field_catalog.list_fields()
        assert "close" in fields
        assert len(wq_field_catalog.list_fields("fundamental")) == len(_CATALOG_FIELDS)
        assert wq_field_catalog.list_fields("nope") == []
        assert wq_field_catalog.known_datasets() == ["fundamental"]

    def test_unavailable_catalog_does_not_raise(self, no_catalog):
        """硬性要求：目录不可用不得抛异常。"""
        status = wq_field_catalog.catalog_status()
        assert status["available"] is False
        assert status["stale"] is True
        assert wq_field_catalog.field_exists("close") is False
        assert wq_field_catalog.list_fields() == []

    def test_validation_degrades_to_warning_without_catalog(self, no_catalog):
        result = validate_expression("rank(gross_profit/assets)", mode="wq")
        assert result.status == "warning"
        assert any("目录不可用" in w.message for w in result.warnings)

    def test_warning_message_explains_catalog_unavailable(self, no_catalog):
        result = validate_expression("rank(gross_profit)", mode="wq")
        joined = " ".join(w.message for w in result.warnings)
        assert "无法校验" in joined
        assert "字段目录不可用" in joined

    def test_refresh_failure_degrades_to_stale_cache(self, catalog, monkeypatch):
        """WQ 不可达时降级使用过期缓存，不抛异常。"""
        # 造一份过期缓存
        catalog.write_text(
            json.dumps(
                {
                    "fetched_at": time.time() - 10 * 24 * 3600,
                    "count": 2,
                    "fields": ["assets", "sales"],
                    "datasets": {},
                }
            ),
            encoding="utf-8",
        )
        wq_field_catalog.reset_for_tests()

        # 模拟拉取失败
        def _boom(*args, **kwargs):
            raise OSError("network down")

        monkeypatch.setattr(wq_field_catalog, "_fetch_all", _boom)
        assert wq_field_catalog.refresh() is False

        status = wq_field_catalog.catalog_status()
        assert status["count"] == 2, "失败后应继续使用过期缓存而非丢弃"
        assert status["stale"] is True
        assert wq_field_catalog.field_exists("assets") is True

    def test_corrupt_cache_does_not_raise(self, tmp_path, monkeypatch):
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        monkeypatch.setenv("QUANTGPT_WQ_CATALOG_PATH", str(path))
        wq_field_catalog.reset_for_tests()
        assert wq_field_catalog.catalog_status()["available"] is False


# ------------------------------------------------------------------ 性能 D


class TestPerformance:
    """D：precheck_expression 单次调用 <100ms（纯本地，无网络 IO）。"""

    def test_precheck_under_100ms(self, catalog):
        expressions = [
            "rank(ts_delta(close, 5) / ts_std_dev(returns, 20))",
            "group_rank(ts_rank(turnover, 126), subindustry)",
            "group_neutralize(rank(est_eps / close), subindustry)",
        ]
        precheck(expressions[0])  # 预热（首次会读磁盘缓存）
        start = time.perf_counter()
        for expression in expressions:
            precheck(expression)
        elapsed_ms = (time.perf_counter() - start) / len(expressions) * 1000
        assert elapsed_ms < 100, f"precheck 耗时 {elapsed_ms:.1f}ms，超过 100ms"

    def test_precheck_never_calls_network(self, catalog, monkeypatch):
        """precheck 必须纯本地：任何网络调用都要让测试失败。"""
        import urllib.request

        def _no_network(*args, **kwargs):
            raise AssertionError("precheck 不应发起网络请求")

        monkeypatch.setattr(urllib.request, "urlopen", _no_network)
        monkeypatch.setattr(wq_field_catalog, "_fetch_all", _no_network)
        assert precheck("rank(close)").status == "ok"
        assert precheck("where(close > open, 1, -1)").status == "error"


# ------------------------------------------------------------ 变量分类任务3


class TestVariableCategory:
    """任务 3：variable_category 分类。"""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("vwap", "builtin"),
            ("returns", "builtin"),
            ("cap", "builtin"),
            ("adv20", "builtin"),
            ("close", "price"),
            ("open", "price"),
            ("volume", "price"),
            ("industry", "group"),
            ("subindustry", "group"),
            ("IndClass.industry", "group"),
            ("earnings", "datafield"),
            ("est_eps", "datafield"),
            ("mdf_oey", "datafield"),
        ],
    )
    def test_categories(self, name, expected):
        assert variable_category(name) == expected

    def test_case_insensitive(self):
        assert variable_category("CLOSE") == "price"
        assert variable_category("Adv20") == "builtin"

    def test_market_cap_alias_maps_to_cap(self):
        """本地 market_cap → WQ cap，这是最容易漏的映射错。"""
        assert variable_category("market_cap") == "builtin"

    def test_unknown_returns_unknown(self):
        assert variable_category("") == "unknown"


# ------------------------------------------------- mcp_server 工具层（任务4/5）


class TestMcpToolWiring:
    """mcp_server 的两个工具必须委托到 wq_validator，且不破坏其它工具。

    这里用 AST 静态检查而非 import mcp_server —— 后者会拉起 MCP/SQLAlchemy
    全套依赖，而本模块要能在最小环境下独立跑通。
    """

    def _mcp_source(self) -> str:
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "quantgpt" / "mcp_server.py"
        return path.read_text(encoding="utf-8")

    def test_both_tools_registered(self):
        import ast

        tree = ast.parse(self._mcp_source())
        funcs = {
            n.name: n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
            and any(
                isinstance(d, ast.Call)
                and getattr(getattr(d.func, "attr", None), "__str__", lambda: "")() == "tool"
                or getattr(d.func, "id", "") == "tool"
                for d in n.decorator_list
            )
        }
        assert "validate_expression" in funcs, "validate_expression 必须仍是 MCP 工具"
        assert "precheck_expression" in funcs, "precheck_expression 必须是新增的 MCP 工具"

    def test_validate_expression_has_strict_param(self):
        import ast

        tree = ast.parse(self._mcp_source())
        fn = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "validate_expression"
        )
        defaults = [ast.unparse(d) for d in fn.args.defaults]
        assert any("False" in d for d in defaults), "strict 默认必须是 False"
        assert [a.arg for a in fn.args.args][-1] == "strict"

    def test_precheck_signature_matches_spec(self):
        import ast

        tree = ast.parse(self._mcp_source())
        fn = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "precheck_expression"
        )
        args = [a.arg for a in fn.args.args]
        assert args == ["expression", "mode", "strict"], args
        # mode 默认 "wq"，strict 默认 True
        defaults = [ast.literal_eval(d) for d in fn.args.defaults]
        assert defaults == ["wq", True], defaults

    def test_tools_delegate_to_validator(self):
        """两个工具都必须走 wq_validator，不能内联重复实现校验逻辑。"""
        source = self._mcp_source()
        assert "from .wq_validator import precheck as precheck_result" in source
        assert (
            "from .wq_validator import validate_expression as validate_expression_result"
            in source
        )

    def test_other_tool_signatures_untouched(self):
        """硬性约束：其余 14 个工具的签名不得被改动。"""
        import ast

        tree = ast.parse(self._mcp_source())
        expected = {
            "run_backtest": 8,
            "score_factor": 5,
        }
        tools = {
            n.name: n
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(
                getattr(d.func, "attr", "") == "tool" or getattr(d.func, "id", "") == "tool"
                for d in n.decorator_list
            )
        }
        assert len(tools) >= 16, f"工具数量异常：{sorted(tools)}"
        for name, min_args in expected.items():
            if name in tools:
                args = tools[name].args.args
                assert len(args) >= min_args, f"{name} 参数数量被改动"

    def test_wq_brain_client_untouched(self):
        """硬性约束：wq_brain_client 的轮询与重试逻辑不得修改。"""
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "quantgpt" / "wq_brain_client.py"
        src = path.read_text(encoding="utf-8")
        assert "_POLL_MAX_ATTEMPTS = 36" in src
        assert "_POLL_INTERVAL = 10" in src
        assert "_CONCURRENT_BACKOFF = 30" in src
