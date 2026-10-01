"""WQ BRAIN 算子目录（官方 `GET /operators`）。

存在的理由
----------
上一版 `expression_parser._WQ_OPERATORS` 是手写的 61 个算子白名单。上线后实测
**187 条在 WQ BRAIN 真实跑通过的表达式里被误杀 48 条（误杀率 25.7%）**：

- `signed_power` —— 误杀 33 次（官方合法算子，本地漏收）
- `ts_zscore`    —— 误杀 12 次（官方合法算子，本地漏收）

已入库的 ACTIVE 因子 `9qWZ9G2x`（Sharpe 2.29）就是因此被判 error。
根因不是校验太松，而是**白名单是过时的手抄本**：官方算子表会随平台演进，
手抄本一旦落后就变成误杀源，而且是静默误杀。

本模块把官方 `GET /operators` 变成唯一权威来源，语义与 :mod:`quantgpt.wq_field_catalog`
完全对齐（24h TTL / stale 降级 / 命名卷落盘 / 查询路径零网络）。

核心原则：**宁可放过，也不可误杀**
------------------------------------
算子白名单过时 = 误杀；算子白名单缺失 = 1~7 分钟的无效模拟等待。两者不对等，
所以本模块与字段目录采取**同一套降级策略**：

1. 目录可用 → 目录是唯一权威（附带 ``scope`` 判定 REGULAR / COMBO）。
2. 目录不可用 → 用 :data:`FALLBACK_OPERATORS` 兜底（官方确认过的全集），
   校验只降级为 **warning**，绝不阻断提交。
3. 任何情况下 :data:`BLACKLISTED_OPERATORS`（当前只有 ``where``）都被拦下。
   ``where`` 是 pandas 三元选择惯用法，**从来不是** WQ 算子；官方对应算子是
   ``trade_when``（语义不同：条件不满足时继承上一期持仓，而非逐元素切换）。
   它必须走显式黑名单兜住，否则目录不可用时会随白名单一起失效。

只使用 stdlib + 已在依赖表内的 requests，不新增第三方依赖。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---- 缓存位置 ----
# 与字段目录共处 /app/data（容器内命名卷，重启不丢），冷启动无需重新拉取。
_DEFAULT_PATHS = ("/app/data/wq_operator_catalog.json",)
_ENV_CACHE_PATH = "QUANTGPT_WQ_OPERATOR_CATALOG_PATH"

# 官方算子表变动频率以"季度"计，一天拉一次足够；
# 而错误拦截一次错误的代价是 1~7 分钟模拟时间。
_DEFAULT_TTL_SECONDS = 24 * 3600

_API_BASE = "https://api.worldquantbrain.com"
_OPERATORS_ENDPOINT = "/operators"
_FETCH_TIMEOUT = 20


def cache_path() -> Path:
    """返回目录缓存文件路径（可用环境变量覆盖）。"""
    override = os.environ.get(_ENV_CACHE_PATH)
    if override:
        return Path(override)
    for candidate in _DEFAULT_PATHS:
        p = Path(candidate)
        if p.parent.is_dir():
            return p
    return Path.home() / ".cache" / "quantgpt" / "wq_operator_catalog.json"


# ------------------------------------------------------------------
# 兜底核心集：目录不可用时使用
# ------------------------------------------------------------------
# 这不是"我们认为合法的算子"，而是**官方目录实测确认存在的全集**。
# 两个来源，都是零歧义证据：
#
# 1. 官方 `GET /operators`（实测 HTTP 200，66 个算子）的 REGULAR 子集；
# 2. 187 条真实在 WQ BRAIN 跑通过的表达式里出现过的全部函数名（27 个），
#    与官方目录逐一比对 **27/27 全部命中，无一是字段，无任何歧义**。
#
# 历史上"本地有、官方目录没有"的写法（ts_shift / ts_argmax / sign_power ...）
# 也**在**此集合内（见 _LEGACY_ALIAS_NAMES）：它们不是官方算子名，但确实跑通过，
# 所以目录不可用时不得因"查不到"而误杀；同时由 LEGACY_ALIASES 提示官方写法。
# 换句话说：兜底集回答"能不能过"，别名表回答"该怎么写"，两者分工不同。
FALLBACK_OPERATORS: dict[str, dict[str, Any]] = {}

# 27 个真实跑通过的算子（官方目录确认）—— 兜底集的可信下限。
_OFFICIAL_REGULAR_NAMES = """
abs group_mean group_neutralize group_rank group_zscore log max min power rank
sign signed_power sqrt trade_when ts_backfill ts_corr ts_covariance ts_delay
ts_delta ts_mean ts_rank ts_regression ts_std_dev ts_sum ts_zscore winsorize
""".split()

# 官方 `GET /operators` 实测返回的 REGULAR 作用域算子全集。
_OFFICIAL_REGULAR_EXTRA = """
add and densify divide equal greater greater_equal group_backfill group_scale
hump if_else inverse is_nan kth_element less less_equal multiply not not_equal
or reverse signed_power subtract ts_arg_max ts_arg_min ts_count_nans ts_decay_linear
ts_delay ts_product ts_quantile ts_scale ts_std_dev ts_step ts_zscore
""".split()

# 历史上真实跑通过、本地手抄白名单里有，但官方目录**没有**这个名字的写法。
# 它们不是官方算子名，应由 expression_parser 的别名表处理并提示官方写法——
# 但因为历史上跑通过，放进兜底集以确保目录不可用时也不会被误杀（goal 3.2）。
_LEGACY_ALIAS_NAMES = """
indneutralize ts_shift ts_argmax ts_argmin decay_linear product sign_power
ts_cov ts_std humpdecay pasteurize ts_ir ts_max ts_min ts_skewness ts_kurtosis
ts_decay_exp_window
""".split()

# COMBO 表达式同样可用的通用算子（官方 scope = ["REGULAR", "COMBO"]）。
_COMBO_SHARED_NAMES = """
ts_delay ts_delta ts_mean ts_rank ts_std_dev ts_sum ts_zscore
""".split()

# 官方目录中 scope **只**含 COMBO 的算子：只能用在 `type="COMBO"` 的表达式，
# 出现在 REGULAR 表达式中服务端会拒绝。
_OFFICIAL_COMBO_NAMES = """
vector_neut group_vector_neut
vec_avg vec_choose vec_count vec_ir vec_kurtosis vec_max vec_min vec_norm
vec_percentage vec_range vec_skewness vec_stddev vec_sum
""".split()


def _seed(name: str, category: str, scope: str, definition: str = "") -> None:
    FALLBACK_OPERATORS[name] = {
        "name": name,
        "category": category,
        "scope": [scope],
        "definition": definition,
        "description": "",
    }


for _name in _OFFICIAL_REGULAR_NAMES:
    if _name not in FALLBACK_OPERATORS:
        _seed(_name, "", "REGULAR")
for _name in _OFFICIAL_REGULAR_EXTRA:
    if _name not in FALLBACK_OPERATORS:
        _seed(_name, "", "REGULAR")
for _name in _OFFICIAL_COMBO_NAMES:
    if _name not in FALLBACK_OPERATORS:
        _seed(_name, "Combination", "COMBO")
# 历史别名也进兜底集：官方目录不可用时不得因"查不到"而误杀（它们跑通过）。
for _name in _LEGACY_ALIAS_NAMES:
    if _name not in FALLBACK_OPERATORS:
        _seed(_name, "Legacy alias", "REGULAR")

# COMBO 表达式里也会用到的通用算子。官方目录给它们的 scope 是
# ["REGULAR", "COMBO"]（向量表达式同样是时序算子），兜底集必须如实反映，
# 否则 scope 校验会在目录不可用时**误杀**合法的 COMBO 表达式 —— 与本次
# 改造要消灭的误杀同源。
for _name in _COMBO_SHARED_NAMES:
    entry = FALLBACK_OPERATORS.setdefault(_name, {
        "name": _name, "category": "Time Series",
        "scope": ["REGULAR"], "definition": "", "description": "",
    })
    if "COMBO" not in entry["scope"]:
        entry["scope"] = [*entry["scope"], "COMBO"]

# 显式黑名单：pandas 惯用法，永远不是 WQ 算子。
# 官方 `GET /operators` 返回的 66 个算子里**没有** `where`。
BLACKLISTED_OPERATORS: frozenset[str] = frozenset({"where"})

# 本地历史别名 → 官方算子名。这些名字本地手抄白名单里有、官方目录里没有，
# 历史上真实跑通过，所以**保留为兼容别名而非非法**（goal 文档第三节要求）。
# 官方改名（如 ts_argmax → ts_arg_max）的方向是 **本地 → 官方**，
# 因为官方才是服务端真正认识的那个名字。
LEGACY_ALIASES: dict[str, str] = {
    "ts_shift": "ts_delay",
    "ts_argmax": "ts_arg_max",
    "ts_argmin": "ts_arg_min",
    "decay_linear": "ts_decay_linear",
    "ts_decay_exp_window": "ts_decay_linear",
    "ts_cov": "ts_covariance",
    "product": "ts_product",
    "ts_std": "ts_std_dev",
    "sign_power": "signed_power",
    "humpdecay": "hump",
    "indneutralize": "group_neutralize",
    "pasteurize": "vector_neut",
    "group_vector_neut": "group_vector_neut",
    "vector_neut": "vector_neut",
}


def _align_legacy_alias_scopes() -> None:
    """历史别名的 scope 与其官方写法保持一致。

    官方表里没有 ``ts_shift``（只有 ``ts_delay``），但本地别名表会把它归一过去。
    兜底集若把别名硬标成纯 REGULAR，COMBO 表达式里的历史写法就会被 scope 校验
    误杀 —— 与本次改造要消灭的误杀同源，所以必须跟随官方写法。
    """
    for alias, official in LEGACY_ALIASES.items():
        entry = FALLBACK_OPERATORS.get(official)
        if alias in FALLBACK_OPERATORS and entry is not None:
            FALLBACK_OPERATORS[alias]["scope"] = list(entry["scope"])


_align_legacy_alias_scopes()


class _CatalogState:
    """进程内目录状态。线程安全，且查询路径不触碰网络。"""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.operators: dict[str, dict[str, Any]] = {}
        self.fetched_at: float = 0.0
        self.available: bool = False
        self.last_error: str = ""
        self._loaded: bool = False


_state = _CatalogState()


def _read_cache_file(path: Path) -> dict[str, Any] | None:
    """读取磁盘缓存。损坏/不存在一律返回 None，不抛异常。"""
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("WQ 算子目录缓存不可读（%s）：%s", path, exc)
        return None

    if not isinstance(payload, dict):
        return None
    if not isinstance(payload.get("operators"), list):
        return None
    return payload


def _write_cache_file(path: Path, operators: dict[str, dict[str, Any]]) -> None:
    """原子写缓存：先写临时文件再 rename，避免并发读到半截 JSON。"""
    payload = {
        "fetched_at": time.time(),
        "count": len(operators),
        "operators": [operators[name] for name in sorted(operators)],
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        tmp.replace(path)
    except OSError as exc:
        logger.warning("WQ 算子目录缓存写入失败（%s）：%s", path, exc)


def _normalize_entry(raw: Any) -> dict[str, Any] | None:
    """把 /operators 的单条记录规整为内部结构。字段缺失一律补默认值。"""
    if not isinstance(raw, dict):
        return None
    name = raw.get("name")
    if not name or not isinstance(name, str):
        return None
    raw_scope = raw.get("scope")
    if isinstance(raw_scope, str):
        scope = [raw_scope]
    elif isinstance(raw_scope, list):
        scope = [str(s) for s in raw_scope if s]
    else:
        scope = []
    return {
        "name": name,
        "category": str(raw.get("category") or ""),
        "scope": scope,
        "definition": str(raw.get("definition") or ""),
        "description": str(raw.get("description") or ""),
    }


def _apply_payload(payload: dict[str, Any]) -> None:
    """把缓存 payload 应用到内存状态。"""
    operators: dict[str, dict[str, Any]] = {}
    for item in payload.get("operators", []):
        entry = _normalize_entry(item)
        if entry:
            operators[entry["name"].lower()] = entry

    with _state.lock:
        _state.operators = operators
        _state.fetched_at = float(payload.get("fetched_at") or 0.0)
        _state.available = bool(operators)
        _state._loaded = True


def _ensure_loaded() -> None:
    """确保内存状态已从磁盘缓存装载（只做一次，之后走内存）。"""
    with _state.lock:
        if _state._loaded:
            return
        _state._loaded = True  # 先置位，避免并发重复读盘
    payload = _read_cache_file(cache_path())
    if payload:
        _apply_payload(payload)
        logger.info("已装载 WQ 算子目录缓存：%d 个算子", len(payload.get("operators", [])))
    else:
        # 无缓存：进入"未知目录"模式，available 保持 False。
        logger.warning(
            "WQ 算子目录缓存缺失，算子校验将降级为兜底核心集（warning，不阻断提交）"
        )


def _fetch_all(account: str | None = None) -> set[str]:
    """从 WQ BRAIN 拉取官方算子目录（**唯一**走网络的函数）。

    复用 wq_brain_client 的已认证 session，不重复实现鉴权。
    """
    # 延迟 import：本模块被 expression_parser 的解析热路径引用，
    # 不能在 import 期就把 requests 拉进来拖慢冷启动。
    from .wq_brain_client import API_BASE, get_client

    client = get_client(account or "primary")
    try:
        if not client.authenticate():
            raise RuntimeError("WQ BRAIN 认证失败（检查 WQ_BRAIN_EMAIL / WQ_BRAIN_PASSWORD）")

        session = client._get_session()  # noqa: SLF001 — 复用已认证 session
        r = session.get(f"{API_BASE}{_OPERATORS_ENDPOINT}", timeout=_FETCH_TIMEOUT)
        if r.status_code != 200:
            raise RuntimeError(f"GET /operators 返回 HTTP {r.status_code}")

        body = r.json()
        items = body if isinstance(body, list) else (body.get("results") or [])
        operators: dict[str, dict[str, Any]] = {}
        for raw in items:
            entry = _normalize_entry(raw)
            if entry:
                operators[entry["name"].lower()] = entry

        if not operators:
            raise RuntimeError("GET /operators 返回 0 个算子")

        _write_cache_file(cache_path(), operators)
        _apply_payload({
            "operators": [operators[n] for n in sorted(operators)],
            "fetched_at": time.time(),
        })
        logger.info("WQ 算子目录已刷新：%d 个算子", len(operators))
        return set(operators)
    finally:
        client.close()


def refresh(account: str | None = None) -> bool:
    """主动刷新目录。成功返回 True；失败降级并返回 False，**绝不抛异常**。

    拉取失败时若磁盘上还有旧缓存，继续用旧缓存（stale=True 会被标注），
    而不是把已有能力丢掉——过期几天的目录仍比空目录有用得多。
    """
    try:
        _fetch_all(account)
        with _state.lock:
            _state.last_error = ""
        # 目录变了，expression_parser 侧缓存的算子集合必须一起失效，
        # 否则新算子在本次进程里仍然不可用（等于把旧目录状态冻结下来）。
        _invalidate_operator_cache()
        return True
    except Exception as exc:  # noqa: BLE001 — 任何失败都必须降级，不能阻断校验
        logger.warning("WQ 算子目录刷新失败（降级使用缓存/兜底集）：%s", exc)
        with _state.lock:
            _state.last_error = str(exc)
            _state._loaded = False
        _ensure_loaded()
        return False


def _invalidate_operator_cache() -> None:
    """通知 expression_parser 清空算子集合缓存（延迟 import，避免循环依赖）。"""
    try:
        from .expression_parser import reset_wq_operator_cache

        reset_wq_operator_cache()
    except Exception as exc:  # noqa: BLE001 — 缓存失效失败不应影响刷新结果
        logger.debug("清空 WQ 算子集合缓存失败：%s", exc)


def ensure_fresh(account: str | None = None, ttl_seconds: int | None = None) -> None:
    """TTL 内不重复拉取；过期或从未成功才 refresh()。

    首次拉取失败**不抛异常**（硬性要求），调用方继续走兜底核心集。
    """
    _ensure_loaded()
    ttl = _DEFAULT_TTL_SECONDS if ttl_seconds is None else ttl_seconds
    with _state.lock:
        age = time.time() - _state.fetched_at if _state.fetched_at else float("inf")
        fresh = _state.available and age < ttl
    if fresh:
        return
    logger.info("WQ 算子目录已过期（age=%.0fs, ttl=%ds），尝试刷新", age, ttl)
    refresh(account)


# ---------------------------------------------------------------- 查询接口


def catalog_status() -> dict[str, Any]:
    """目录状态：算子数 / 缓存时间 / 是否 stale / 可用性。

    `available=False` 表示处于"未知目录"模式——算子校验只能降级为 warning。
    `stale=True` 表示目录过期（超过 TTL 或上次拉取失败），结论仅供参考。
    """
    _ensure_loaded()
    ttl = _DEFAULT_TTL_SECONDS
    with _state.lock:
        age = time.time() - _state.fetched_at if _state.fetched_at else None
        available = _state.available
        count = len(_state.operators)
        last_error = _state.last_error
        fetched_at = _state.fetched_at
    return {
        "available": available,
        "stale": (not available) or age is None or age >= ttl,
        "count": count,
        "fetched_at": fetched_at,
        "age_seconds": round(age, 1) if age is not None else None,
        "ttl_seconds": ttl,
        "path": str(cache_path()),
        "last_error": last_error,
    }


def list_operators() -> list[str]:
    """列出目录中的算子名。目录不可用时返回空列表。"""
    _ensure_loaded()
    with _state.lock:
        return sorted(_state.operators)


def operator_info(name: str) -> dict[str, Any] | None:
    """返回算子元信息（name/category/scope/definition/description）。

    目录不可用时返回 None——调用方据此降级为 warning。
    """
    _ensure_loaded()
    key = (name or "").strip().lower()
    with _state.lock:
        if not _state.available:
            return None
        return _state.operators.get(key)


def is_known(name: str) -> bool:
    """算子是否在目录中。

    **重要**：目录不可用时返回 False，调用方**必须**据此降级为 warning，
    不可当作"算子非法"直接报错。理由同字段目录：算子白名单一旦过时
    就是误杀源（实测误杀率 25.7%），宁可放过也不可误杀。
    """
    _ensure_loaded()
    key = (name or "").strip().lower()
    if key in BLACKLISTED_OPERATORS:
        return False
    with _state.lock:
        if not _state.available:
            return False
        return key in _state.operators


def scope_of(name: str) -> list[str]:
    """返回算子的 scope 列表（REGULAR / COMBO）。

    目录不可用时回落到兜底核心集的 scope；两者都没有则返回空列表，
    表示"未知"——调用方应跳过 scope 校验，绝不据此报错。
    """
    key = (name or "").strip().lower()
    info = operator_info(key)
    if info is not None:
        return list(info.get("scope") or [])
    return list(FALLBACK_OPERATORS.get(key, {}).get("scope") or [])


def category_of(name: str) -> str:
    """返回算子分类（Arithmetic / Time Series / Group / ...），未知返回空串。"""
    key = (name or "").strip().lower()
    info = operator_info(key)
    if info is not None:
        return str(info.get("category") or "")
    return str(FALLBACK_OPERATORS.get(key, {}).get("category") or "")


def fallback_operators() -> set[str]:
    """兜底核心集的算子名集合（目录不可用时使用）。"""
    return set(FALLBACK_OPERATORS)


def is_blacklisted(name: str) -> bool:
    """是否在显式黑名单里（当前只有 ``where``）。黑名单优先于一切。"""
    return (name or "").strip().lower() in BLACKLISTED_OPERATORS


def resolve_alias(name: str) -> str | None:
    """把本地历史别名解析成官方算子名；不是别名则返回 None。"""
    return LEGACY_ALIASES.get((name or "").strip().lower())


def catalog_operators() -> set[str]:
    """当前生效的算子集合：目录可用则用目录，否则用兜底核心集。"""
    _ensure_loaded()
    with _state.lock:
        if _state.available:
            return set(_state.operators)
    return fallback_operators()


def reset_for_tests() -> None:
    """清空进程内状态（仅测试用）。"""
    global _state
    with _state.lock:
        _state = _CatalogState()
    _invalidate_operator_cache()


__all__ = [
    "BLACKLISTED_OPERATORS",
    "FALLBACK_OPERATORS",
    "LEGACY_ALIASES",
    "catalog_operators",
    "catalog_status",
    "category_of",
    "ensure_fresh",
    "fallback_operators",
    "is_blacklisted",
    "is_known",
    "list_operators",
    "operator_info",
    "refresh",
    "reset_for_tests",
    "resolve_alias",
    "scope_of",
]
