"""WQ BRAIN data field 目录（真实字段表）。

存在的理由
----------
`ExpressionParser` 在 mode="wq" 下无法预知 WQ BRAIN 的**全部** data field——
服务端有数万条（ fundamentals / analyst / model / news / options / macro ... ），
硬编码白名单必然漏。而上游为了"不误杀"，把未知字段一律 warning 后透传，
代价是 `validate_expression` 对字段类错误**永远返回 OK**，subagent 提交后
白等 1~7 分钟模拟才被服务端拒绝（`Invalid data field xxx`）。

本模块用**真实目录**补上这块：定期从 `GET /data-fields` 全量拉取并落盘缓存，
让本地校验能在**毫秒级**判断 `gross_profit` 这类字段是否真实存在。

设计约束（务必保留，勿改回静默放行）
--------------------------------------
1. **拉取失败绝不能阻塞校验**。WQ 不可达 / 未配置账号时，目录进入"未知"模式，
   此时 `field_exists()` 返回 False 但调用方必须把它降级为 **warning** 而非 error
   （见 wq_validator），因为服务端可能有我们没拉到的字段——误杀合规表达式比漏检更严重。
2. **降级必须自我标注**（`stale: true`），让 subagent 知道"这条没被真正校验过"。
3. **查询路径必须零网络**。`field_exists()` 只读内存/磁盘缓存，是同步小文件读取，
   以满足 `precheck_expression` 的 <100ms 硬指标。只有 `refresh()` 会走网络，
   且由显式调用或 cron 触发。

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
# 容器内 /app/data 已挂载到宿主机，进程重启后目录仍在，避免每次冷启动都重拉。
# 本地开发（无 /app 目录）自动退化到 ~/.cache/quantgpt/。
_DEFAULT_PATHS = ("/app/data/wq_field_catalog.json",)

# /data-fields 必须携带完整 simulation settings 才返回结果（实测：仅传 limit
# 会得到 400 ["Invalid query"]）。这里给出与 wq_brain_client.simulate 一致的
# 默认 settings，对齐项目内既有的 TOP3000 / SUBINDUSTRY 约定。
_DEFAULT_SETTINGS = {
    "instrumentType": "EQUITY",
    "region": "USA",
    "universe": "TOP3000",
    "delay": 1,
    "decay": 0,
    "neutralization": "SUBINDUSTRY",
    "truncation": 0.08,
}
_ENV_CACHE_PATH = "QUANTGPT_WQ_CATALOG_PATH"

# 默认 TTL 24h：WQ 的 data field 集合按季度级变动，一天拉一次足够，
# 而 1~7 分钟的模拟成本远高于一次几百 KB 的分页拉取。
_DEFAULT_TTL_SECONDS = 24 * 3600

_API_BASE = "https://api.worldquantbrain.com"
_DATA_FIELDS_ENDPOINT = "/data-fields"
_PAGE_SIZE = 50
_MAX_PAGES = 400  # 20000 条上限，防御性护栏
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
    return Path.home() / ".cache" / "quantgpt" / "wq_field_catalog.json"


class _CatalogState:
    """进程内目录状态。线程安全，且查询路径不触碰网络。"""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.fields: set[str] = set()
        self.by_dataset: dict[str, list[str]] = {}
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
        logger.warning("WQ 字段目录缓存不可读（%s）：%s", path, exc)
        return None

    if not isinstance(payload, dict):
        return None
    names = payload.get("fields")
    if not isinstance(names, list):
        return None
    return payload


def _write_cache_file(path: Path, fields: set[str], by_dataset: dict[str, list[str]]) -> None:
    """原子写缓存：先写临时文件再 rename，避免并发读到半截 JSON。"""
    payload = {
        "fetched_at": time.time(),
        "count": len(fields),
        "fields": sorted(fields),
        "datasets": {k: sorted(v) for k, v in sorted(by_dataset.items())},
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        tmp.replace(path)
    except OSError as exc:
        logger.warning("WQ 字段目录缓存写入失败（%s）：%s", path, exc)


def _apply_payload(payload: dict[str, Any]) -> None:
    """把缓存 payload 应用到内存状态。"""
    fields = {str(n) for n in payload.get("fields", []) if isinstance(n, str)}
    raw_datasets = payload.get("datasets")
    by_dataset: dict[str, list[str]] = {}
    if isinstance(raw_datasets, dict):
        for dataset, names in raw_datasets.items():
            if isinstance(names, list):
                by_dataset[str(dataset)] = [str(n) for n in names]

    with _state.lock:
        _state.fields = fields
        _state.by_dataset = by_dataset
        _state.fetched_at = float(payload.get("fetched_at") or 0.0)
        _state.available = bool(fields)
        _state._loaded = True


def _ensure_loaded() -> None:
    """确保内存状态已从磁盘缓存装载（只做一次，之后走内存）。

    这是查询路径（field_exists 等）唯一的 IO：小文件同步读取，进程内缓存，
    满足 precheck_expression <100ms 的要求。
    """
    with _state.lock:
        if _state._loaded:
            return
        _state._loaded = True  # 先置位，避免并发重复读盘
    payload = _read_cache_file(cache_path())
    if payload:
        _apply_payload(payload)
        logger.info("已装载 WQ 字段目录缓存：%d 条", len(payload.get("fields", [])))
    else:
        # 无缓存：进入"未知目录"模式，available 保持 False。
        logger.warning("WQ 字段目录缓存缺失，字段校验将降级为 warning")


def _fetch_all(account: str | None = None) -> set[str]:
    """从 WQ BRAIN 全量拉取 data field（**唯一**走网络的函数）。

    分页遍历 /data-fields，字段名取 items[].id，数据集取 items[].dataset。
    这里复用 wq_brain_client 的认证 session，不重复实现鉴权。
    """
    # 延迟 import：本模块被 expression_parser 的校验路径引用，
    # 不能在 import 期就把 requests 拉进来拖慢冷启动。
    from .wq_brain_client import API_BASE, get_client

    client = get_client(account or "primary")
    try:
        if not client.authenticate():
            raise RuntimeError("WQ BRAIN 认证失败（检查 WQ_BRAIN_EMAIL / WQ_BRAIN_PASSWORD）")

        fields: set[str] = set()
        by_dataset: dict[str, list[str]] = {}
        session = client._get_session()  # noqa: SLF001 — 复用已认证 session
        offset = 0

        for _ in range(_MAX_PAGES):
            params = {**_DEFAULT_SETTINGS, "limit": _PAGE_SIZE, "offset": offset}
            r = session.get(
                f"{API_BASE}{_DATA_FIELDS_ENDPOINT}",
                params=params,
                timeout=_FETCH_TIMEOUT,
            )
            if r.status_code != 200:
                raise RuntimeError(f"GET /data-fields 返回 HTTP {r.status_code}")

            body = r.json()
            items = body.get("results") or body.get("items") or []
            if not items:
                break

            for item in items:
                name = item.get("id")
                if not name or not isinstance(name, str):
                    continue
                fields.add(name)
                dataset = item.get("dataset")
                if isinstance(dataset, str) and dataset:
                    by_dataset.setdefault(dataset, []).append(name)

            # WQ 返回 count 表示总数；翻够就停，避免最后一次空请求
            total = body.get("count")
            offset += len(items)
            if total is not None and offset >= int(total):
                break

        if not fields:
            raise RuntimeError("GET /data-fields 返回 0 条字段")

        _write_cache_file(cache_path(), fields, by_dataset)
        _apply_payload({
            "fields": sorted(fields),
            "datasets": by_dataset,
            "fetched_at": time.time(),
        })
        logger.info("WQ 字段目录已刷新：%d 条 / %d 个数据集", len(fields), len(by_dataset))
        return fields
    finally:
        client.close()


def refresh(account: str | None = None) -> bool:
    """主动刷新目录。成功返回 True；失败降级并返回 False，**绝不抛异常**。

    拉取失败时若磁盘上还有旧缓存，继续用旧缓存（stale=True 会被标注），
    而不是把已有能力丢掉——过期 3 天的目录仍比空目录有用得多。
    """
    try:
        _fetch_all(account)
        with _state.lock:
            _state.last_error = ""
        return True
    except Exception as exc:  # noqa: BLE001 — 任何失败都必须降级，不能阻断校验
        logger.warning("WQ 字段目录刷新失败（降级使用缓存）：%s", exc)
        with _state.lock:
            _state.last_error = str(exc)
        # 失败后重新读一次磁盘：可能有别的进程/上一次运行写下的缓存
        with _state.lock:
            _state._loaded = False
        _ensure_loaded()
        return False


def ensure_fresh(account: str | None = None, ttl_seconds: int | None = None) -> None:
    """TTL 内不重复拉取；过期或从未成功才 refresh()。

    首次拉取失败**不抛异常**（硬性要求），调用方继续走"未知目录"降级模式。
    """
    _ensure_loaded()
    ttl = _DEFAULT_TTL_SECONDS if ttl_seconds is None else ttl_seconds
    with _state.lock:
        age = time.time() - _state.fetched_at if _state.fetched_at else float("inf")
        fresh = _state.available and age < ttl
    if fresh:
        return
    logger.info("WQ 字段目录已过期（age=%.0fs, ttl=%ds），尝试刷新", age, ttl)
    refresh(account)


# ---------------------------------------------------------------- 查询接口


def catalog_status() -> dict[str, Any]:
    """目录状态：条目数 / 缓存时间 / 是否 stale / 可用性。

    `available=False` 表示处于"未知目录"模式——字段校验只能降级为 warning。
    `stale=True` 表示目录过期（超过 TTL 或上次拉取失败），结论仅供参考。
    """
    _ensure_loaded()
    ttl = _DEFAULT_TTL_SECONDS
    with _state.lock:
        age = time.time() - _state.fetched_at if _state.fetched_at else None
        available = _state.available
        count = len(_state.fields)
        datasets = len(_state.by_dataset)
        last_error = _state.last_error
        fetched_at = _state.fetched_at
    return {
        "available": available,
        "stale": (not available) or age is None or age >= ttl,
        "count": count,
        "datasets": datasets,
        "fetched_at": fetched_at,
        "age_seconds": round(age, 1) if age is not None else None,
        "ttl_seconds": ttl,
        "path": str(cache_path()),
        "last_error": last_error,
    }


def field_exists(name: str) -> bool:
    """字段是否在目录中。

    **重要**：目录不可用时返回 False，调用方**必须**据此降级为 warning，
    不可当作"字段非法"直接报错——服务端可能有我们没拉到的字段。
    """
    _ensure_loaded()
    with _state.lock:
        if not _state.available:
            return False
        return name.lower() in _state.fields


def list_fields(dataset: str | None = None) -> list[str]:
    """列出目录中的字段名，可按数据集过滤。目录不可用时返回空列表。"""
    _ensure_loaded()
    with _state.lock:
        if dataset:
            return sorted(_state.by_dataset.get(dataset.lower(), []))
        return sorted(_state.fields)


def known_datasets() -> list[str]:
    """目录中已知的数据集名。"""
    _ensure_loaded()
    with _state.lock:
        return sorted(_state.by_dataset)


def reset_for_tests() -> None:
    """清空进程内状态（仅测试用）。"""
    global _state
    with _state.lock:
        _state = _CatalogState()
