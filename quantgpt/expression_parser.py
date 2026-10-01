"""Factor Expression Parser — QuantGPT
Copyright (c) 2026 Miasyster. Licensed under the MIT License.
https://github.com/Miasyster/QuantGPT

Parses simple factor expressions and returns callables
that operate on DataFrames.

Supported operations:
- rank(col)           : cross-sectional rank
- zscore(col)         : cross-sectional z-score standardization
- sign(col)           : sign function
- log(col)            : natural log
- abs(col)            : absolute value
- scale(col)          : standardize to [0, 1] range
- tanh(col)           : hyperbolic tangent
- sigmoid(col)        : logistic sigmoid (1/(1+exp(-x)))
- exp(col)            : exponential (capped to avoid overflow)
- sqrt(col)           : square root (negative values clipped to 0)
- ts_mean(col, N)     : rolling mean over N periods
- ts_std(col, N)      : rolling std over N periods
- ts_max(col, N)      : rolling max over N periods
- ts_min(col, N)      : rolling min over N periods
- ts_sum(col, N)      : rolling sum over N periods
- ts_shift(col, N)    : shift values by N periods (positive=lag, negative=lead)
- ts_delta(col, N)    : N-period change
- ts_rank(col, N)     : rolling percentile rank (returns 0~1, e.g. 0.8 means top 80%)
- ts_argmax(col, N)   : position of max in rolling window
- ts_argmin(col, N)   : position of min in rolling window
- ts_corr(col1, col2, N) : rolling correlation
- ts_cov(col1, col2, N)  : rolling covariance
- decay_linear(col, N) : linear decay weights over N periods
- product(col, N)     : rolling product over N periods
- power(base, exp)    : power operation (base ** exp)
- sign_power(base, exp) : sign(base) * (abs(base) ** exp)
- max(a, b)           : element-wise maximum
- min(a, b)           : element-wise minimum
- clip(expr, lo, hi)  : clip values to [lo, hi] range
- where(cond, t, f)   : conditional selection (t if cond else f) — **local mode only**;
                        WQ BRAIN 没有 where 算子，wq 模式请用 trade_when
- indneutralize(col, industry) : industry neutralization (placeholder)
- ts_av_diff(col, N)  : deviation from rolling mean (col - ts_mean(col, N))
- ts_zscore(col, N)   : rolling z-score ((col - ts_mean) / ts_std over N periods)
- trade_when(cond, alpha, hold_val) : conditional signal — use alpha when cond is true, else hold last value (initial=hold_val)
- group_rank(col, group) : cross-sectional rank within group (e.g., group_rank(close, industry))
- group_zscore(col, group) : cross-sectional z-score within group
- winsorize(x[, std])   : winsorize outliers (WQ BRAIN remote operator)
Technical indicators:
- ema(col, N)         : exponential moving average (span=N)
- sma(col, N)         : simple moving average (alias for ts_mean)
- wma(col, N)         : weighted moving average (linear decay)
- rsi(col, N)         : Relative Strength Index (0~100)
- macd(col, N)        : MACD histogram (fast=N/2, slow=N, signal=N/4)
- obv(col, N)         : On-Balance Volume rolling sum (simplified)
- atr(N)              : Average True Range (uses high/low/close columns)
- boll_upper(col, N)  : Bollinger Band upper (mean + 2*std)
- boll_lower(col, N)  : Bollinger Band lower (mean - 2*std)
- boll_mid(col, N)    : Bollinger Band middle (rolling mean)
- Arithmetic: +, -, *, /, ^
- Comparison: >, <, >=, <=, ==, !=
- Logical: and, or, &, |

Special variables:
- vwap                : volume-weighted average price
- adv{N}              : N-day average daily volume (e.g., adv20)
- returns             : daily returns
- cap                 : market capitalization
- day                 : day of month (1-31)
- weekday             : day of week (0=Monday, 4=Friday)
- month               : month (1-12)

Fundamental data variables (quarterly financials, aligned to daily via pubDate):
  Profitability:
  - roe              : 平均净资产收益率
  - np_margin        : 净利润率
  - gp_margin        : 毛利率
  - net_profit       : 净利润(元)
  - eps_ttm          : 每股收益(TTM)
  - revenue          : 营业收入(元)
  - total_share      : 总股本
  - float_share      : 流通股本
  Growth:
  - yoy_ni           : 净利润同比增长率
  - yoy_equity       : 净资产同比增长率
  - yoy_asset        : 总资产同比增长率
  - yoy_pni          : 归母净利润同比增长率
  Balance sheet:
  - current_ratio    : 流动比率
  - debt_ratio       : 资产负债率
  - equity_multiplier: 权益乘数
  Operations:
  - asset_turnover   : 总资产周转率
  - inv_turnover     : 存货周转率
  - dupont_roe       : 杜邦分析ROE
  - dupont_asset_turn: 杜邦资产周转率
  Cash flow:
  - cfo_to_np        : 经营现金流/净利润
  Valuation (derived from close + fundamental):
  - pe               : 市盈率 (close * total_share / net_profit)
  - pb               : 市净率
  - ps               : 市销率 (close * total_share / revenue)

Operator aliases (for Alpha101 compatibility):
- delta(col, N)       : alias for ts_delta
- delay(col, N)       : alias for ts_shift
- covariance(col1, col2, N) : alias for ts_cov
- correlation(col1, col2, N) : alias for ts_corr
- IndNeutralize(col, industry) : alias for indneutralize

Syntax extensions:
- Ternary operator: (condition ? true_value : false_value)
- Power operator: base ^ exponent (equivalent to power(base, exponent))
"""

import logging
import re
import threading
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# 官方算子目录提供的"本地尚未实现运行时"的算子（远程执行专用）。
# 目录不可用时 wq_operator_catalog 会回落到自己的兜底核心集，这里再并一层
# 本地实现已覆盖的算子，保证 mode="wq" 的解析在**任何目录状态下**都不会误杀。
_WQ_LEGACY_CORE = {
    'rank', 'zscore', 'scale', 'group_rank', 'group_zscore',
    'abs', 'sign', 'log', 'sqrt',
    'power', 'max', 'min',
    'ts_mean', 'ts_std', 'ts_max', 'ts_min', 'ts_sum',
    'ts_shift', 'ts_delta', 'ts_rank', 'ts_argmax', 'ts_argmin',
    'decay_linear', 'product', 'ts_av_diff',
    'ts_corr', 'ts_cov',
    'winsorize',
    # 注意：这里**不能**有 'where'。'where' 是 pandas.Series.where() 的三元选择惯用法，
    # 从来不是 WQ BRAIN 算子；WQ 官方对应算子是 trade_when（语义不同：条件不满足时
    # 继承上一期持仓，而非逐元素切换）。上游把它误收进白名单，导致本地校验放行、
    # 服务端报 unknown operator，subagent 白等 1~7 分钟模拟。
    'trade_when',
    'indneutralize',
    # 实测官方 GET /operators 确认存在、本地此前漏收，算子白名单过时导致
    # 187 条真实跑通表达式被误杀 48 条（signed_power 33 次 / ts_zscore 12 次），
    # 其中包括 Sharpe 2.29 的 ACTIVE 因子 9qWZ9G2x。
    'signed_power', 'ts_zscore',
}

# WQ 模式下**不再有任何"未知算子透传"通道**（见 _build_function 里的硬报错分支）。
# 为什么要改成硬报错：WQ BRAIN 的算子集合是官方固定且可枚举的（远小于 data field 数量），
# 服务端不存在"我们不知道的新算子"这种合法场景。此前的 warning + 透传分支等于把
# 拼写错误、pandas 惯用法误用全部放行，是 cron 失败率的主要成因。
# 代价是：若 WQ 未来新增算子，本地会误拒。缓解方式见 _WQ_REPLACEMENTS 的提示文案
# 与 wq_field_catalog 的同类降级机制——宁可让作者显式确认，也不要静默提交后等 1~7 分钟。

_LOCAL_ONLY_OPERATORS = {
    'tanh', 'sigmoid', 'exp', 'ts_zscore', 'clip',
    'ema', 'sma', 'wma', 'rsi', 'macd', 'obv', 'atr',
    'boll_upper', 'boll_lower', 'boll_mid', 'indneutralize',
}

# `_LOCAL_ONLY_OPERATORS` 里那些**官方目录确认存在**的算子。
#
# 历史教训：这张表曾被当作"WQ 不支持的算子"黑名单用，导致 `ts_zscore`
# （官方合法算子）被误杀 12 次。名字里的 "local_only" 只表示"本地先实现，
# 方便本地 pandas 回测"，**不**表示"不是 WQ 算子"——两者是不同的概念，
# 混为一谈就会重演 25.7% 的误杀事故。
# 只列**有实测证据**的：`GET /operators` 返回中存在，或真实跑通过的表达式里出现过。
# `clip` 不在其中——官方 66 个算子里没有它（WQ 用 `max(lo, min(hi, x))`），
# 所以它留在 _LOCAL_ONLY_UNSUPPORTED 里被硬拒，是正确行为。
_LOCAL_ONLY_BUT_OFFICIAL = frozenset({
    'ts_zscore',        # GET /operators 确认存在（曾被误杀 12 次）
    'indneutralize',    # 官方 indneutralize / group_neutralize 的历史写法
})

# 真正只在本地存在的算子：任何目录状态下都必然不是 WQ 算子。
# 只有这些才会被无条件硬拒（见 _build_function 与 wq_validator._validate_operators）。
_LOCAL_ONLY_UNSUPPORTED = frozenset(_LOCAL_ONLY_OPERATORS) - _LOCAL_ONLY_BUT_OFFICIAL

# WQ 价格/成交列。注意：**没有 market_cap** —— WQ 侧市值字段叫 cap，
# 写 market_cap 会被服务端判 Invalid data field。本地 A 股引擎才叫 market_cap，
# 这个映射错是历史漏网之鱼，务必不要再把 market_cap 加回这里。
_WQ_COLUMNS = {'open', 'high', 'low', 'close', 'volume', 'vwap', 'returns'}

# WQ 内置变量（服务端直接认识，不属于 data field，不在 data-fields 目录里）
_WQ_SPECIAL_VARS = {
    'vwap', 'returns', 'cap', 'adv20',
}

# 本地专有列名 → WQ 官方名称。任务 3 要求重点覆盖这类映射错。
_WQ_NAME_ALIASES = {
    'market_cap': 'cap',
    'float_market_cap': 'cap',
    'amount': 'vwap',
    'pct_change': 'returns',
    'turnover_rate': 'volume / adv20',
    'shares': 'cap / close',
}

_WQ_FUNDAMENTAL_FIELDS = {
    'earnings', 'ebit', 'ebitda', 'sales', 'revenue', 'equity', 'debt',
    'assets', 'liabilities', 'capex', 'cash_flow', 'book_value',
    'dividends', 'net_income', 'operating_income', 'free_cash_flow',
    'total_debt', 'cash_and_equivalents', 'accounts_receivable',
    'inventory', 'goodwill', 'intangibles', 'shares_outstanding',
    'enterprise_value', 'net_debt',
}

_WQ_ANALYST_FIELDS = {
    'est_eps', 'est_revenue', 'est_ebitda', 'est_growth',
    'price_target', 'recommendation',
    'fam_score', 'fam_growth', 'fam_value', 'fam_quality', 'fam_sentiment',
    'fam_roe_rank', 'fam_growth_rank', 'fam_value_rank',
}

_WQ_MDF_FIELDS = {
    'mdf_oey', 'mdf_gry', 'mdf_pbk', 'mdf_eg3', 'mdf_sg3',
    'mdf_ep', 'mdf_bp', 'mdf_sp', 'mdf_cfp', 'mdf_dp',
    'mdf_ey', 'mdf_roic', 'mdf_quality', 'mdf_leverage',
}

_WQ_NEWS_SENTIMENT_FIELDS = {
    'snt_buzz', 'snt_buzz_ret', 'snt_bullish', 'snt_bearish',
    'snt_sentiment', 'snt_volume', 'snt_score',
}

_WQ_OPTIONS_FIELDS = {
    'pcr_oi', 'pcr_oi_all', 'pcr_vol', 'pcr_vol_all',
    'implied_volatility', 'implied_volatility_call', 'implied_volatility_put',
    'implied_volatility_slope', 'implied_volatility_skew',
    'option_volume', 'open_interest',
}

_WQ_RELATIONSHIP_FIELDS = {
    'rel_num_cust', 'rel_num_supp', 'rel_ret_cust', 'rel_ret_supp',
    'rel_momentum', 'rel_volume',
    'short_interest', 'short_sale_cost', 'short_ratio',
    'insider_buy', 'insider_sell', 'institutional_ownership',
}

_WQ_EXTENDED_FIELDS = (
    _WQ_FUNDAMENTAL_FIELDS | _WQ_ANALYST_FIELDS | _WQ_MDF_FIELDS |
    _WQ_NEWS_SENTIMENT_FIELDS | _WQ_OPTIONS_FIELDS | _WQ_RELATIONSHIP_FIELDS
)

_WQ_NEWS_PREFIXES = ('nws12_', 'nws24_', 'nws_', 'snt_')
_WQ_GROUP_PREFIXES = ('indclass.', 'ind.', 'sector.', 'subindustry.')

# WQ 分组字段（classification namespace）。它们**不在** /data-fields 目录里，
# 是服务端单独的一组内置标识符；用目录去查会把所有 group_rank(x, industry)
# 误判为非法字段。验收标准 C 里的合规表达式大量依赖这一类。
_WQ_GROUP_FIELDS = {
    'market', 'sector', 'industry', 'subindustry', 'exchange',
    'country', 'currency', 'densification', 'market_coverage',
}

_LOCAL_ONLY_COLUMNS = {
    # 注意 market_cap 也在列：本地 A 股引擎叫 market_cap，WQ 侧叫 cap。
    # 不加进来它会落到"未知字段透传"分支，被静默放行（Invalid data field）。
    'amount', 'pct_change', 'market_cap', 'float_market_cap',
    'turnover_rate', 'shares',
}

_WQ_REMOTE_ONLY_OPS = {
    'vector_neut': (2, 2, 'vector_neut(alpha, risk_factor)'),
    'humpdecay': (2, 3, 'humpdecay(x, p) or humpdecay(x, p, relative)'),
    'days_from_last_change': (1, 1, 'days_from_last_change(x)'),
    'last_diff_value': (1, 2, 'last_diff_value(x) or last_diff_value(x, default)'),
    'group_neutralize': (2, 2, 'group_neutralize(x, group)'),
    'group_mean': (2, 3, 'group_mean(x, group) 或 group_mean(x, weight, group)'),
    'group_vector_neut': (2, 2, 'group_vector_neut(x, group)'),
    'ts_regression': (3, 5, 'ts_regression(y, x, d) 或 ts_regression(y, x, d, lag, rettype)'),
    'ts_decay_exp_window': (2, 3, 'ts_decay_exp_window(x, d, factor)'),
    'ts_ir': (2, 2, 'ts_ir(x, d)'),
    'ts_skewness': (2, 2, 'ts_skewness(x, d)'),
    'ts_kurtosis': (2, 2, 'ts_kurtosis(x, d)'),
    'ts_backfill': (2, 2, 'ts_backfill(x, d)'),
    'normalize': (1, 1, 'normalize(x)'),
    'winsorize': (1, 2, 'winsorize(x) or winsorize(x, std)'),
    'quantile': (1, 3, 'quantile(x) or quantile(x, driver, n)'),
    'pasteurize': (1, 1, 'pasteurize(x)'),
    'bucket': (2, 3, 'bucket(x, n) or bucket(x, range, n)'),
    'vec_avg': (1, None, 'vec_avg(x, ...)'),
    'vec_sum': (1, None, 'vec_sum(x, ...)'),
    'vec_max': (1, None, 'vec_max(x, ...)'),
    'vec_min': (1, None, 'vec_min(x, ...)'),
    'vec_count': (1, None, 'vec_count(x, ...)'),
    'vec_range': (1, None, 'vec_range(x, ...)'),
    'vec_stddev': (1, None, 'vec_stddev(x, ...)'),
    'vec_skewness': (1, None, 'vec_skewness(x, ...)'),
    'vec_kurtosis': (1, None, 'vec_kurtosis(x, ...)'),
    'vec_ir': (1, None, 'vec_ir(x, ...)'),
    'vec_norm': (1, None, 'vec_norm(x, ...)'),
    'vec_percentage': (2, None, 'vec_percentage(x, x1, ...)'),
    'vec_choose': (2, None, 'vec_choose(n, x1, ...)'),
}

_WQ_REMOTE_ONLY_OP_NAMES = set(_WQ_REMOTE_ONLY_OPS.keys())

# ``_build_function`` 中以字面量分支实现、本地可直接求值的算子/函数。
# 仅本地模式可用（mode="wq" 下多数由 _WQ_REMOTE_ONLY_OPS 拦截）。
_LOCAL_ONLY_FUNCTIONS = {
    'trade_when', 'group_rank', 'group_zscore',
    'atr', 'boll_upper', 'boll_lower', 'boll_mid', 'clip', 'where',
}


def _refresh_wq_operator_cache() -> frozenset[str]:
    """重新计算算子集合并写入缓存。返回缓存结果。"""
    global _WQ_OPERATORS_CACHE
    if _WQ_OPERATORS_CACHE is None:
        _WQ_OPERATORS_CACHE = frozenset(_wq_operator_names())
    return _WQ_OPERATORS_CACHE


def _wq_operator_names() -> set[str]:
    """mode="wq" 下当前允许的算子名集合（官方目录优先）。

    历史教训：这里曾经是一份手写的 61 项白名单，上线后误杀 25.7% 的真实合规
    表达式（`signed_power` / `ts_zscore` 等官方合法算子）。手抄本一旦落后于
    平台就是**静默误杀源**，所以权威只能是官方 `GET /operators`：

    1. 目录可用 → 目录 + 本地已实现的算子 + 远程执行算子；
    2. 目录不可用 → 目录模块的兜底核心集（官方确认过的全集）+ 同上两层。

    `where` 无论目录状态如何都**不会**出现在结果里（黑名单由目录模块保证）。
    """
    from . import wq_operator_catalog

    return (
        wq_operator_catalog.catalog_operators()
        | _WQ_LEGACY_CORE
        | _WQ_REMOTE_ONLY_OP_NAMES
    ) - set(wq_operator_catalog.BLACKLISTED_OPERATORS)


class _DynamicOperatorSet:
    """随官方算子目录刷新的算子集合代理。

    历史教训：``_WQ_OPERATORS`` 曾是一份手写白名单，上线后误杀 25.7% 的真实合规
    表达式（``signed_power`` 33 次 / ``ts_zscore`` 12 次）。手抄本一旦落后于平台
    就是**静默误杀源**，所以权威只能是官方 ``GET /operators``。

    这里用代理而不是模块级常量，是因为算子目录是**运行时**才装载的（进程启动时
    目录缓存可能还没落盘）。把集合做成常量并在 import 期求值，会把目录状态冻结
    在 import 时刻，目录恢复后仍然继续误杀——那正是我们要消灭的行为。
    """

    def _current(self) -> frozenset[str]:
        if _WQ_OPERATORS_CACHE is None:
            with _WQ_OPERATOR_CACHE_LOCK:
                if _WQ_OPERATORS_CACHE is None:
                    return _refresh_wq_operator_cache()
        return _WQ_OPERATORS_CACHE  # type: ignore[return-value]

    # ---- 集合语义（全部走当前目录状态，而非冻结快照）----

    def __contains__(self, item) -> bool:
        return str(item).lower() in self._current()

    def __iter__(self):
        return iter(sorted(self._current()))

    def __len__(self) -> int:
        return len(self._current())

    def __bool__(self) -> bool:
        return bool(self._current())

    def __eq__(self, other) -> bool:
        if isinstance(other, (set, frozenset)):
            return self._current() == frozenset(other)
        return NotImplemented

    def __ne__(self, other) -> bool:
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    def __hash__(self) -> int:
        return hash(self._current())

    def __repr__(self) -> str:
        return repr(sorted(self._current()))

    def __or__(self, other):
        return self._current() | frozenset(other)

    def __ror__(self, other):
        return frozenset(other) | self._current()

    def __and__(self, other):
        return self._current() & frozenset(other)

    def __rand__(self, other):
        return frozenset(other) & self._current()

    def __sub__(self, other):
        return self._current() - frozenset(other)

    def __rsub__(self, other):
        return frozenset(other) - self._current()

    def __xor__(self, other):
        return self._current() ^ frozenset(other)

    def __rxor__(self, other):
        return frozenset(other) ^ self._current()

    def __le__(self, other) -> bool:
        return self._current() <= frozenset(other)

    def __lt__(self, other) -> bool:
        return self._current() < frozenset(other)

    def __ge__(self, other) -> bool:
        return self._current() >= frozenset(other)

    def __gt__(self, other) -> bool:
        return self._current() > frozenset(other)

    def copy(self) -> frozenset[str]:
        return self._current()

    def isdisjoint(self, other) -> bool:
        return self._current().isdisjoint(other)


_WQ_OPERATORS_CACHE: frozenset[str] | None = None
_WQ_OPERATOR_CACHE_LOCK = threading.RLock()
_WQ_OPERATORS = _DynamicOperatorSet()


def reset_wq_operator_cache() -> None:
    """目录刷新后清空算子集合缓存（refresh / cron / 测试用）。

    不清的话进程会一直用"目录尚未装载"时算出的旧快照，等于把启动瞬间的
    目录状态冻结下来——目录恢复后仍然误杀。
    """
    global _WQ_OPERATORS_CACHE
    with _WQ_OPERATOR_CACHE_LOCK:
        _WQ_OPERATORS_CACHE = None


def wq_operators() -> set:
    """WQ 模式下允许的全部算子名（含仅远程可执行的 operator-typed 算子）。

    算子表以官方 ``GET /operators`` 为唯一权威（见 :mod:`quantgpt.wq_operator_catalog`）。
    目录不可用时回落到兜底核心集，校验降级为 warning 而不是误杀。
    """
    return set(_WQ_OPERATORS)


# 算子/变量别名 → WQ 官方替代方案。
# 用途有二：(1) mode="wq" 报错时给出可直接抄的替代写法；(2) variable_category()
# 识别出本地专有名字时给出迁移指引。这张表是**唯一**的知识来源，务必写全常见的
# pandas / numpy 惯用法——它们是 subagent 最常误用的东西（上游最典型的就是 'where'）。
_WQ_REPLACEMENTS = {
    # --- pandas/numpy 惯用法 → WQ 官方算子 ---
    'where': 'trade_when(cond, enter, exit)（条件不满足时继承上一期持仓，非逐元素切换）',
    'if_else': 'trade_when(cond, enter, exit)',
    'np.where': 'trade_when(cond, enter, exit)',
    'select': 'trade_when(cond, enter, exit)',
    'fillna': 'ts_backfill(x, d)',
    'ffill': 'ts_backfill(x, d)',
    'iloc': 'ts_shift(x, N)',
    'rolling': 'ts_mean / ts_std / ts_sum(x, N)',
    'pct_change': 'returns（WQ 已内置日收益）',
    'isnull': 'trade_when(cond, enter, exit)',
    'notnull': 'trade_when(cond, enter, exit)',
    'any': 'trade_when(cond, enter, exit)',
    'all': 'trade_when(cond, enter, exit)',
    'between': 'trade_when(cond, enter, exit)',
    'corr': 'ts_corr(x, y, d)',
    'cov': 'ts_cov(x, y, d)',
    'quantile': 'quantile(x, driver, n)（WQ 官方算子）',
    'std': 'ts_std(x, d)',
    'var': 'ts_std(x, d)',
    'mean': 'ts_mean(x, d)',
    'median': 'ts_median(x, d)',
    'abs_diff': 'abs(x - y)',
    'diff': 'ts_delta(x, d)',
    'shift': 'ts_shift(x, d)',
    'sort': 'rank(x)',
    'nlargest': 'ts_argmax(x, d)',
    'cumsum': 'ts_sum(x, d)',

    # --- 本地专有算子 → WQ 官方算子 ---
    'tanh': 'sign_power(x, 0.5) 或 x / (1 + abs(x))',
    'sigmoid': 'rank(x) 或 1 / (1 + power(2.718, -x))',
    'exp': 'power(2.718, x)',
    'log1p': 'log(1 + x)',
    'clip': 'max(lo, min(hi, x))',
    'ema': 'decay_linear(x, N)',
    'sma': 'ts_mean(x, N)',
    'wma': 'decay_linear(x, N)',
    'ts_zscore': '(x - ts_mean(x, N)) / ts_std(x, N)',
    'rsi': 'ts_rank(x, N)',
    'macd': 'ts_delta(ts_mean(x, N), N)',
    'obv': 'ts_sum(sign(ts_delta(close, 1)) * volume, N)',
    'atr': '(high - low) / close',
    'boll_upper': 'ts_mean(x, N) + 2 * ts_std(x, N)',
    'boll_lower': 'ts_mean(x, N) - 2 * ts_std(x, N)',
    'boll_mid': 'ts_mean(x, N)',
    'group_neutralize_ind': 'group_neutralize(x, group)',

    # --- 本地专有字段/变量 → WQ 官方字段 ---
    # 重点：本地 A 股引擎叫 market_cap，WQ 侧叫 cap —— 这类映射错是漏网之鱼。
    'market_cap': 'cap（WQ 侧市值字段名为 cap，不是 market_cap）',
    'amount': 'vwap (= amount/volume)',
    'pct_change': 'returns',
    'turnover_rate': 'volume / adv20',
    'float_market_cap': 'cap（WQ 无流通市值单独字段）',
    'shares': 'cap / close（WQ 无股本字段）',
    'trade_date': 'delay',
    'stock_code': '（WQ 无此概念，按标的自动区分）',
    'dividend_yield': 'dividends / cap',
}

_WQ_UNIT_PATTERNS = [
    (re.compile(r'(?:close|open|high|low|volume|vwap|market_cap|adv\d+)\s*[+\-]\s*\d+\.?\d*(?!\s*\*)'),
     "WQ 量纲错误：不得将常数与价格/量做加减。去掉 epsilon（如 +0.0001），WQ 内部处理零值"),
    (re.compile(r'close\s*/\s*ts_(?:delay|shift)\s*\(\s*close\s*,\s*\d+\s*\)\s*-\s*1'),
     "WQ 量纲错误：close/ts_delay(close,N)-1 → 改用 ts_delta(close,N)/ts_delay(close,N)"),
]


class ExpressionParser:
    """Parse factor expressions into callable functions.

    mode='wq'   — only WQ BRAIN compatible operators and columns (for submission)
    mode='local' — all operators and columns (default, for local research)
    """

    MAX_WINDOW = 500
    MAX_DEPTH = 100
    MAX_EXPRESSION_LENGTH = 1000

    def __init__(self, mode: str = "local"):
        if mode not in ("wq", "local"):
            raise ValueError(f"未知模式：{mode!r}，支持 'wq' 或 'local'")
        self.mode = mode

    # Pattern: func_name(args)
    _FUNC_PATTERN = re.compile(
        r'^(\w+)\((.+)\)$'
    )

    # Operator aliases for compatibility with Alpha101 and other factor libraries
    _OPERATOR_ALIASES = {
        'delta': 'ts_delta',
        'delay': 'ts_shift',
        'covariance': 'ts_cov',
        'correlation': 'ts_corr',
        'IndNeutralize': 'indneutralize',  # Alpha101 uses capital I
        'av_diff': 'ts_av_diff',
        'stddev': 'ts_std',
        'ts_decay_linear': 'decay_linear',
        'ts_product': 'product',
        # ---- 官方命名（本地手抄白名单里写成了旧名）----
        # 官方 `GET /operators` 里是 ts_std_dev / ts_delay / ts_covariance /
        # ts_arg_max / ts_arg_min，以下几条把历史写法归一到**官方**名字。
        'ts_std_dev': 'ts_std',
        'ts_delay': 'ts_shift',
        'ts_covariance': 'ts_cov',
        'ts_arg_max': 'ts_argmax',
        'ts_arg_min': 'ts_argmin',
    }

    # Special variable mappings (computed from DataFrame columns)
    _SPECIAL_VARS = {
        'vwap': lambda df: df['vwap'] if 'vwap' in df.columns else (df['amount'] / df['volume'].replace(0, np.nan) if 'amount' in df.columns else df['close']),
        'returns': lambda df: df.groupby('stock_code')['close'].pct_change() if 'stock_code' in df.columns else df['close'].pct_change(),
        'cap': lambda df: df.get('market_cap', df['close'] * df.get('shares', 1)),  # fallback if no market_cap
        'day': lambda df: pd.Series(df['trade_date'].dt.day, index=df.index, dtype=float),
        'weekday': lambda df: pd.Series(df['trade_date'].dt.weekday, index=df.index, dtype=float),  # 0=Mon, 4=Fri
        'month': lambda df: pd.Series(df['trade_date'].dt.month, index=df.index, dtype=float),
    }

    # Cross-sectional operators that need per-date grouping.
    # These are handled specially in _build_function() — they are NOT in _UNARY_OPS.
    _CROSS_SECTIONAL_OPS = {'rank', 'zscore'}

    # Supported unary functions (column -> Series)
    _UNARY_OPS = {
        'log': lambda s: np.log(s.clip(lower=1e-10)),
        'abs': lambda s: s.abs(),
        'sign': lambda s: np.sign(s),
        'scale': lambda s: (s - s.min()) / (s.max() - s.min() + 1e-10),  # normalize to [0, 1]
        'tanh': lambda s: np.tanh(s),
        'sigmoid': lambda s: 1.0 / (1.0 + np.exp(-s.clip(-500, 500))),
        'exp': lambda s: np.exp(s.clip(upper=500)),  # clip to avoid overflow
        'sqrt': lambda s: np.sqrt(s.clip(lower=0)),
    }

    # Technical indicator helpers (standalone functions, not lambdas)
    @staticmethod
    def _calc_rsi(s: "pd.Series", w: int) -> "pd.Series":
        delta = s.diff()
        gain = delta.clip(lower=0).rolling(w, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(w, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _calc_macd(s: "pd.Series", w: int) -> "pd.Series":
        # w is slow period; fast = w//2, signal = w//4 (min 2)
        fast = max(2, w // 2)
        signal = max(2, w // 4)
        ema_fast = s.ewm(span=fast, adjust=False).mean()
        ema_slow = s.ewm(span=w, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line - signal_line  # histogram

    @staticmethod
    def _calc_atr(df: "pd.DataFrame", w: int) -> "pd.Series":
        high = df.get('high', df['close'])
        low = df.get('low', df['close'])
        close_prev = df['close'].shift(1)
        tr = pd.concat([
            high - low,
            (high - close_prev).abs(),
            (low - close_prev).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(w, min_periods=1).mean()

    # Supported time-series functions (column, window -> Series)
    # When the DataFrame has a 'stock_code' column, these automatically
    # apply per-stock via groupby to avoid mixing different stocks' data.
    _TS_OPS = {
        'ts_mean': lambda s, w: s.rolling(w, min_periods=1).mean(),
        'ts_std': lambda s, w: s.rolling(w, min_periods=1).std(),
        'ts_max': lambda s, w: s.rolling(w, min_periods=1).max(),
        'ts_min': lambda s, w: s.rolling(w, min_periods=1).min(),
        'ts_sum': lambda s, w: s.rolling(w, min_periods=1).sum(),
        'ts_shift': lambda s, w: s.shift(w),
        'ts_delta': lambda s, w: s - s.shift(w),
        'ts_rank': lambda s, w: s.rolling(w, min_periods=1).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False),
        'ts_argmax': lambda s, w: s.rolling(w, min_periods=1).apply(lambda x: x.argmax(), raw=True),
        'ts_argmin': lambda s, w: s.rolling(w, min_periods=1).apply(lambda x: x.argmin(), raw=True),
        'decay_linear': lambda s, w: s.rolling(w, min_periods=1).apply(
            lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)) if len(x) > 0 else np.nan,
            raw=True
        ),
        'product': lambda s, w: s.rolling(w, min_periods=1).apply(lambda x: np.prod(x), raw=True),
        'ts_av_diff': lambda s, w: s - s.rolling(w, min_periods=1).mean(),
        'ts_zscore': lambda s, w: (s - s.rolling(w, min_periods=1).mean()) / (s.rolling(w, min_periods=1).std() + 1e-10),
        # Technical indicators
        'ema': lambda s, w: s.ewm(span=w, adjust=False).mean(),
        'sma': lambda s, w: s.rolling(w, min_periods=1).mean(),  # same as ts_mean
        'rsi': lambda s, w: ExpressionParser._calc_rsi(s, w),
        'macd': lambda s, w: ExpressionParser._calc_macd(s, w),
        'obv': lambda s, w: s.rolling(w, min_periods=1).sum(),  # simplified: rolling OBV sum
        'wma': lambda s, w: s.rolling(w, min_periods=1).apply(
            lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)) if len(x) > 0 else np.nan,
            raw=True
        ),  # weighted moving average (same as decay_linear)
    }

    @staticmethod
    def _apply_ts_op_per_stock(df, inner_fn, op, window):
        """Apply a time-series operation per-stock when DataFrame has stock_code."""
        s = inner_fn(df)
        if 'stock_code' in df.columns:
            return s.groupby(df['stock_code']).transform(lambda x: op(x, window))
        return op(s, window)

    # Supported dual-column time-series functions (col1, col2, window -> Series)
    _TS_DUAL_OPS = {
        'ts_corr': lambda s1, s2, w: s1.rolling(w, min_periods=1).corr(s2),
        'ts_cov': lambda s1, s2, w: s1.rolling(w, min_periods=1).cov(s2),
    }

    # Supported binary operations (base, exponent -> Series)
    _BINARY_OPS = {
        'power': lambda s, exp: s ** exp,
        'pow': lambda s, exp: s ** exp,  # alias
        'sign_power': lambda s, exp: np.sign(s) * (np.abs(s) ** exp),
        'max': lambda a, b: np.maximum(a, b),
        'min': lambda a, b: np.minimum(a, b),
    }

    # Industry neutralization (placeholder - requires industry data)
    _NEUTRALIZE_OPS = {
        'indneutralize': lambda s, industry: s - s.groupby(industry).transform('mean'),  # simple demeaning
    }

    def parse(self, expression: str, _depth: int = 0) -> Callable[[pd.DataFrame], pd.Series]:
        """Parse an expression string and return a callable.

        Args:
            expression: Factor expression, e.g. "rank(close/open)",
                        "ts_mean(volume, 20)"

        Returns:
            A callable that takes a DataFrame and returns a Series.
        """
        if _depth > self.MAX_DEPTH:
            raise ValueError(f"Expression nesting too deep (max {self.MAX_DEPTH})")

        expression = expression.strip()

        if len(expression) > self.MAX_EXPRESSION_LENGTH:
            raise ValueError(f"Expression too long (max {self.MAX_EXPRESSION_LENGTH} chars)")

        # Store depth for sub-calls
        self._depth = _depth

        # WQ mode: check unit-incompatible patterns at top level
        if self.mode == "wq" and _depth == 0:
            normalized = re.sub(r'\s+', ' ', expression.lower())
            for pattern, message in _WQ_UNIT_PATTERNS:
                if pattern.search(normalized):
                    raise ValueError(message)

        # Preprocess: convert C-style ternary operators to Python style
        if _depth == 0:
            expression = self._convert_ternary_operators(expression)

        logger.info(f"Parsing expression: {expression}")

        # Try to match a function call at the outermost level.
        func_match = self._match_function_call(expression)
        if func_match is not None:
            func_name, args_str, remainder = func_match
            if not remainder:
                return self._build_function(func_name, args_str)

        # Otherwise treat as arithmetic column expression
        return self._build_arithmetic(expression)

    @staticmethod
    def _match_function_call(expression: str) -> tuple | None:
        """Match a function call at the start of expression.

        Returns (func_name, args_str, remainder) or None.
        remainder is the part after the closing paren (stripped).
        """
        m = re.match(r'^(\w+)\(', expression)
        if not m:
            return None

        func_name = m.group(1).lower()
        start = m.end() - 1  # index of '('
        depth = 0
        for i in range(start, len(expression)):
            if expression[i] == '(':
                depth += 1
            elif expression[i] == ')':
                depth -= 1
                if depth == 0:
                    args_str = expression[start + 1:i]
                    remainder = expression[i + 1:].strip()
                    return (func_name, args_str, remainder)
        return None

    def _sub_parse(self, expr: str) -> Callable[[pd.DataFrame], pd.Series]:
        """Parse a sub-expression, incrementing depth."""
        return self.parse(expr, self._depth + 1)

    def _is_locally_implemented(self, func_name: str) -> bool:
        """``_build_function`` 里是否有该算子的本地运行时实现。

        必须与 ``_build_function`` 的分派链保持一致：任何漏掉的算子都会落到末尾
        ``raise ValueError("Unknown function")``，把官方合法表达式判死。
        新增本地实现时记得同步这里。
        """
        return (
            func_name in self._UNARY_OPS
            or func_name in self._TS_OPS
            or func_name in self._TS_DUAL_OPS
            or func_name in self._BINARY_OPS
            or func_name in self._CROSS_SECTIONAL_OPS
            or func_name in self._NEUTRALIZE_OPS
            or func_name in _WQ_REMOTE_ONLY_OP_NAMES
            or func_name in _LOCAL_ONLY_FUNCTIONS
        )

    @staticmethod
    def _operator_catalog_can_decide(func_name: str) -> bool:
        """官方目录能否对该算子下"不存在"的结论。

        - 目录可用 → 能（目录是唯一权威）
        - 目录不可用 → 算子落在兜底核心集之外，**不能**判定。
          此时放行 + warning，而不是把目录故障变成误杀。
        """
        from . import wq_operator_catalog

        if wq_operator_catalog.catalog_status()["available"]:
            return True
        return func_name in wq_operator_catalog.fallback_operators()

    def _build_remote_only_stub(
        self, func_name: str, args_str: str
    ) -> Callable[[pd.DataFrame], pd.Series]:
        """官方目录里存在、但本地没有运行时实现的算子 → 远程执行占位。

        解析子表达式只为捕获括号/嵌套错误；真正求值由 WQ BRAIN 完成。
        """
        for part in self._split_top_level(args_str):
            if part.strip():
                self._sub_parse(part.strip())

        def _wq_remote_stub(df, _name=func_name):
            raise RuntimeError(
                f"算子 '{_name}' 仅支持 WQ BRAIN 远程执行，不可本地计算"
            )
        return _wq_remote_stub

    def _validate_window(self, window: int, func_name: str) -> int:
        """Validate rolling window size."""
        if window < 1:
            raise ValueError(f"{func_name}: window must be >= 1, got {window}")
        if window > self.MAX_WINDOW:
            raise ValueError(f"{func_name}: window too large (max {self.MAX_WINDOW}), got {window}")
        return window

    def _build_function(
        self, func_name: str, args_str: str
    ) -> Callable[[pd.DataFrame], pd.Series]:
        """Build a callable for a named function."""

        # Apply operator aliases (e.g., delta -> ts_delta, delay -> ts_shift)
        func_name = self._OPERATOR_ALIASES.get(func_name, func_name)

        if self.mode == "wq" and func_name in _WQ_OPERATORS:
            # 官方目录确认存在、但本地没有运行时实现的算子（如 signed_power /
            # ts_zscore / ts_regression / group_backfill ...）。
            # 这里只做**结构**校验（括号、嵌套），求值交给 WQ BRAIN 远程执行。
            # 绝不能落到末尾的 "Unknown function" 分支——那会把合规表达式判死。
            if not self._is_locally_implemented(func_name):
                return self._build_remote_only_stub(func_name, args_str)

        if self.mode == "wq":
            # 官方目录可用时以目录为唯一权威；目录不可用时用兜底核心集，
            # 仍不可判定则**放行**（降级为 warning 由 wq_validator 负责标注）——
            # 算子白名单过时就是误杀源，宁可放过也不可误杀（见 wq_operator_catalog）。
            from . import wq_operator_catalog

            if wq_operator_catalog.is_blacklisted(func_name):
                hint = _WQ_REPLACEMENTS.get(func_name, "")
                hint_msg = f"，替代方案：{hint}" if hint else ""
                raise ValueError(
                    f"WQ 模式下不存在算子 '{func_name}'（pandas 惯用法，"
                    f"从来不是 WQ BRAIN 算子）{hint_msg}"
                )
            if func_name not in _WQ_OPERATORS:
                # 本地专有算子（tanh / rsi / sigmoid ...）在任何目录状态下都
                # **必然**不是 WQ 算子——这是本地语义定义，不是目录查不到。
                # 所以它们不参与降级：拒绝对所有目录状态都成立。
                if func_name in _LOCAL_ONLY_UNSUPPORTED:
                    hint = _WQ_REPLACEMENTS.get(func_name, "")
                    hint_msg = f"，替代方案：{hint}" if hint else ""
                    raise ValueError(f"WQ 模式下不支持算子 '{func_name}'{hint_msg}")
                # 历史别名（ts_argmin / sign_power / humpdecay ...）：官方目录里
                # 没有这个名字，但历史上真实跑通过 —— 不得判死，放行并提示官方写法。
                official = wq_operator_catalog.resolve_alias(func_name)
                if official:
                    logger.info(
                        "WQ 模式：'%s' 为历史别名，官方写法是 '%s'",
                        func_name, official,
                    )
                    return self._build_remote_only_stub(func_name, args_str)
                if self._operator_catalog_can_decide(func_name):
                    hint = _WQ_REPLACEMENTS.get(func_name, "")
                    hint_msg = f"，替代方案：{hint}" if hint else ""
                    raise ValueError(
                        f"WQ 模式下不存在算子 '{func_name}'（WQ BRAIN 会报 "
                        f"unknown operator）{hint_msg}"
                    )
                # 目录不可用且不在兜底核心集内：放行，交给服务端判定。
                return self._build_remote_only_stub(func_name, args_str)

        if func_name in _WQ_REMOTE_ONLY_OPS:
            if self.mode != "wq":
                raise ValueError(f"算子 '{func_name}' 仅在 WQ 模式下可用，不支持本地计算")
            min_args, max_args, usage = _WQ_REMOTE_ONLY_OPS[func_name]
            parts = self._split_top_level(args_str)
            if len(parts) < min_args:
                raise ValueError(f"{func_name} 至少需要 {min_args} 个参数: {usage}")
            if max_args is not None and len(parts) > max_args:
                raise ValueError(f"{func_name} 最多 {max_args} 个参数: {usage}")
            for p in parts:
                self._sub_parse(p.strip())
            def _wq_remote_stub(df, _name=func_name):
                raise RuntimeError(f"算子 '{_name}' 仅支持 WQ BRAIN 远程执行，不可本地计算")
            return _wq_remote_stub

        # Cross-sectional ops: rank() and zscore() group by trade_date
        if func_name in self._CROSS_SECTIONAL_OPS:
            inner = self._sub_parse(args_str)
            if func_name == 'rank':
                def _cs_rank(df, _inner=inner):
                    s = _inner(df)
                    if 'trade_date' in df.columns:
                        return s.groupby(df['trade_date']).rank(pct=True)
                    return s.rank(pct=True)
                return _cs_rank
            else:  # zscore
                def _cs_zscore(df, _inner=inner):
                    s = _inner(df)
                    if 'trade_date' in df.columns:
                        g = s.groupby(df['trade_date'])
                        return (s - g.transform('mean')) / (g.transform('std') + 1e-10)
                    return (s - s.mean()) / (s.std() + 1e-10)
                return _cs_zscore

        if func_name in self._UNARY_OPS:
            inner = self._sub_parse(args_str)
            op = self._UNARY_OPS[func_name]
            return lambda df, _op=op, _inner=inner: _op(_inner(df))

        if func_name in self._TS_OPS:
            parts = self._split_top_level(args_str)
            if len(parts) != 2:
                raise ValueError(
                    f"{func_name} requires exactly 2 arguments: (column, window)"
                )
            inner = self._sub_parse(parts[0].strip())
            try:
                window = self._validate_window(int(parts[1].strip()), func_name)
            except ValueError:
                raise ValueError(
                    f"{func_name} 的窗口参数必须是整数，不能是表达式。"
                    f"收到: {parts[1].strip()!r}"
                )
            op = self._TS_OPS[func_name]
            return lambda df, _op=op, _inner=inner, _w=window: ExpressionParser._apply_ts_op_per_stock(df, _inner, _op, _w)

        if func_name in self._TS_DUAL_OPS:
            parts = self._split_top_level(args_str)
            if len(parts) != 3:
                raise ValueError(
                    f"{func_name} requires exactly 3 arguments: (column1, column2, window)"
                )
            inner1 = self._sub_parse(parts[0].strip())
            inner2 = self._sub_parse(parts[1].strip())
            try:
                window = self._validate_window(int(parts[2].strip()), func_name)
            except ValueError:
                raise ValueError(
                    f"{func_name} 的窗口参数必须是整数，不能是表达式。"
                    f"收到: {parts[2].strip()!r}"
                )
            op = self._TS_DUAL_OPS[func_name]
            def _ts_dual(df, _op=op, _i1=inner1, _i2=inner2, _w=window):
                s1, s2 = _i1(df), _i2(df)
                if 'stock_code' in df.columns:
                    # Apply per-stock: build temporary frame, groupby, apply
                    tmp = pd.DataFrame({'s1': s1, 's2': s2, 'sc': df['stock_code']}, index=df.index)
                    return tmp.groupby('sc', group_keys=False).apply(
                        lambda g: _op(g['s1'], g['s2'], _w)
                    )
                return _op(s1, s2, _w)
            return _ts_dual

        if func_name in self._BINARY_OPS:
            parts = self._split_top_level(args_str)
            if len(parts) != 2:
                raise ValueError(
                    f"{func_name} requires exactly 2 arguments"
                )
            base_fn = self._sub_parse(parts[0].strip())
            exp_fn = self._sub_parse(parts[1].strip())
            op = self._BINARY_OPS[func_name]
            return lambda df, _op=op, _base=base_fn, _exp=exp_fn: _op(_base(df), _exp(df))

        if func_name == 'trade_when':
            parts = self._split_top_level(args_str)
            if len(parts) != 3:
                raise ValueError("trade_when requires 3 arguments: (condition, alpha, hold_value)")
            cond_fn = self._sub_parse(parts[0].strip())
            alpha_fn = self._sub_parse(parts[1].strip())
            # 第三个参数在 WQ 官方语义里是 hold value：**常数或任意表达式都可以**。
            # 最常见的官方标准写法就是 `trade_when(cond, returns, -returns)`
            # （条件不满足时翻转为 -returns）。上游这里只接受 float()，
            # 会让这个标准写法直接抛 ValueError —— 属于过严导致的误杀，必须放宽。
            hold_expr = parts[2].strip()
            try:
                hold_val: float | None = float(hold_expr)
                hold_fn = None
            except ValueError:
                hold_val = None
                hold_fn = self._sub_parse(hold_expr)

            def _trade_when(df, _cond=cond_fn, _alpha=alpha_fn, _hold=hold_val, _hold_fn=hold_fn):
                cond = _cond(df).astype(bool)
                alpha = _alpha(df)
                hold = _hold if _hold_fn is None else _hold_fn(df)
                result = pd.Series(np.nan, index=df.index)
                if 'stock_code' in df.columns:
                    for _, grp in df.groupby('stock_code'):
                        idx = grp.index
                        c, a, h = cond.loc[idx], alpha.loc[idx], hold.loc[idx]
                        vals = pd.Series(np.nan, index=idx)
                        prev = h.iloc[0] if _hold_fn is not None else _hold
                        for i in idx:
                            if c.loc[i]:
                                prev = a.loc[i]
                            vals.loc[i] = prev
                        result.loc[idx] = vals
                else:
                    prev = hold.iloc[0] if _hold_fn is not None else _hold
                    for i in df.index:
                        if cond.loc[i]:
                            prev = alpha.loc[i]
                        result.loc[i] = prev
                return result
            return _trade_when

        if func_name in ('group_rank', 'group_zscore'):
            parts = self._split_top_level(args_str)
            if len(parts) != 2:
                raise ValueError(f"{func_name} requires 2 arguments: (expression, group_column)")
            inner = self._sub_parse(parts[0].strip())
            group_col = parts[1].strip().strip("'\"")
            if func_name == 'group_rank':
                def _group_rank(df, _inner=inner, _gc=group_col):
                    s = _inner(df)
                    if _gc not in df.columns:
                        if 'trade_date' in df.columns:
                            return s.groupby(df['trade_date']).rank(pct=True)
                        return s.rank(pct=True)
                    if 'trade_date' in df.columns:
                        return s.groupby([df['trade_date'], df[_gc]]).rank(pct=True)
                    return s.groupby(df[_gc]).rank(pct=True)
                return _group_rank
            else:
                def _group_zscore(df, _inner=inner, _gc=group_col):
                    s = _inner(df)
                    if _gc not in df.columns:
                        if 'trade_date' in df.columns:
                            g = s.groupby(df['trade_date'])
                            return (s - g.transform('mean')) / (g.transform('std') + 1e-10)
                        return (s - s.mean()) / (s.std() + 1e-10)
                    if 'trade_date' in df.columns:
                        g = s.groupby([df['trade_date'], df[_gc]])
                    else:
                        g = s.groupby(df[_gc])
                    return (s - g.transform('mean')) / (g.transform('std') + 1e-10)
                return _group_zscore

        if func_name in self._NEUTRALIZE_OPS:
            if self.mode == "wq":
                parts = self._split_top_level(args_str)
                if len(parts) != 2:
                    raise ValueError("indneutralize requires 2 arguments: (expression, industry)")
                self._sub_parse(parts[0].strip())
                def _wq_indneut_stub(df):
                    raise RuntimeError("indneutralize 仅支持 WQ BRAIN 远程执行")
                return _wq_indneut_stub
            raise ValueError("indneutralize is not supported (requires industry classification data)")

        # ATR needs high/low/close columns, not a single series
        if func_name == 'atr':
            parts = self._split_top_level(args_str)
            if len(parts) != 1:
                raise ValueError("atr requires exactly 1 argument: (window)")
            window = self._validate_window(int(parts[0].strip()), func_name)
            def _atr(df, _w=window):
                if 'stock_code' in df.columns:
                    return df.groupby('stock_code', group_keys=False).apply(
                        lambda g: ExpressionParser._calc_atr(g, _w)
                    )
                return ExpressionParser._calc_atr(df, _w)
            return _atr

        # BOLL bands: boll_upper(col, N) / boll_lower(col, N) / boll_mid(col, N)
        if func_name in ('boll_upper', 'boll_lower', 'boll_mid'):
            parts = self._split_top_level(args_str)
            if len(parts) != 2:
                raise ValueError(f"{func_name} requires exactly 2 arguments: (column, window)")
            inner = self._sub_parse(parts[0].strip())
            window = self._validate_window(int(parts[1].strip()), func_name)
            if func_name == 'boll_upper':
                return lambda df, _i=inner, _w=window: _i(df).rolling(_w, min_periods=1).mean() + 2 * _i(df).rolling(_w, min_periods=1).std()
            elif func_name == 'boll_lower':
                return lambda df, _i=inner, _w=window: _i(df).rolling(_w, min_periods=1).mean() - 2 * _i(df).rolling(_w, min_periods=1).std()
            else:  # boll_mid
                return lambda df, _i=inner, _w=window: _i(df).rolling(_w, min_periods=1).mean()

        if func_name == 'clip':
            parts = self._split_top_level(args_str)
            if len(parts) != 3:
                raise ValueError("clip requires exactly 3 arguments: (expr, lower, upper)")
            inner = self._sub_parse(parts[0].strip())
            lower_fn = self._sub_parse(parts[1].strip())
            upper_fn = self._sub_parse(parts[2].strip())
            return lambda df, _inner=inner, _lo=lower_fn, _hi=upper_fn: _inner(df).clip(lower=_lo(df), upper=_hi(df))

        if func_name == 'where':
            parts = self._split_top_level(args_str)
            if len(parts) != 3:
                raise ValueError("where requires exactly 3 arguments: (condition, true_value, false_value)")
            cond_fn = self._sub_parse(parts[0].strip())
            true_fn = self._sub_parse(parts[1].strip())
            false_fn = self._sub_parse(parts[2].strip())
            return lambda df, _c=cond_fn, _t=true_fn, _f=false_fn: _t(df).where(_c(df).astype(bool), _f(df))

        raise ValueError(f"Unknown function: {func_name}")

    def _build_arithmetic(
        self, expression: str
    ) -> Callable[[pd.DataFrame], pd.Series]:
        """Build a callable for simple arithmetic on columns.

        Supports: col, col/col, col*col, col+col, col-col, col^col, and numeric literals.
        Also supports special variables: vwap, adv{N}, returns, cap.
        Also supports Python ternary operator: value_if_true if condition else value_if_false
        Also supports comparison operators: >, <, >=, <=, ==, !=
        """
        expression = expression.strip()

        # Check for Python ternary operator (if...else)
        # Pattern: value_if_true if condition else value_if_false
        if ' if ' in expression and ' else ' in expression:
            # Find the positions of 'if' and 'else' at the top level
            if_pos = self._find_keyword(expression, ' if ')
            else_pos = self._find_keyword(expression, ' else ')

            if if_pos is not None and else_pos is not None and if_pos < else_pos:
                true_val_expr = expression[:if_pos].strip()
                condition_expr = expression[if_pos + 4:else_pos].strip()
                false_val_expr = expression[else_pos + 6:].strip()

                true_val_fn = self._sub_parse(true_val_expr)
                condition_fn = self._sub_parse(condition_expr)
                false_val_fn = self._sub_parse(false_val_expr)

                return lambda df, _t=true_val_fn, _c=condition_fn, _f=false_val_fn: (
                    _t(df).where(_c(df) > 0, _f(df))
                )

        # Try logical operators (lowest precedence, evaluated first during parsing)
        for op_str, op_fn in [
            (' or ', lambda a, b: ((a.astype(bool)) | (b.astype(bool))).astype(float)),
            (' and ', lambda a, b: ((a.astype(bool)) & (b.astype(bool))).astype(float)),
        ]:
            idx = self._find_keyword(expression, op_str)
            if idx is not None:
                left = self._sub_parse(expression[:idx])
                right = self._sub_parse(expression[idx + len(op_str):])
                return lambda df, _l=left, _r=right, _op=op_fn: _op(_l(df), _r(df))

        # Try bitwise logical operators (& and |, same semantics as and/or for conditions)
        for op_str, op_fn in [
            ('|', lambda a, b: ((a.astype(bool)) | (b.astype(bool))).astype(float)),
            ('&', lambda a, b: ((a.astype(bool)) & (b.astype(bool))).astype(float)),
        ]:
            idx = self._find_operator(expression, op_str)
            if idx is not None:
                left = self._sub_parse(expression[:idx])
                right = self._sub_parse(expression[idx + len(op_str):])
                return lambda df, _l=left, _r=right, _op=op_fn: _op(_l(df), _r(df))

        # Try comparison operators
        for op_str, op_fn in [
            ('>=', lambda a, b: (a >= b).astype(float)),
            ('<=', lambda a, b: (a <= b).astype(float)),
            ('==', lambda a, b: (a == b).astype(float)),
            ('!=', lambda a, b: (a != b).astype(float)),
            ('>', lambda a, b: (a > b).astype(float)),
            ('<', lambda a, b: (a < b).astype(float)),
        ]:
            idx = self._find_operator(expression, op_str)
            if idx is not None:
                left = self._sub_parse(expression[:idx])
                right = self._sub_parse(expression[idx + len(op_str):])
                return lambda df, _l=left, _r=right, _op=op_fn: _op(_l(df), _r(df))

        # Try binary operators in order of precedence (lowest first)
        for op_char, op_fn in [
            ('+', lambda a, b: a + b),
            ('-', lambda a, b: a - b),
            ('*', lambda a, b: a * b),
            ('/', lambda a, b: a / b.replace(0, np.nan)),
            ('^', lambda a, b: a ** b),
        ]:
            idx = self._find_operator(expression, op_char)
            if idx is not None:
                left = self._sub_parse(expression[:idx])
                right = self._sub_parse(expression[idx + 1:])
                return lambda df, _l=left, _r=right, _op=op_fn: _op(_l(df), _r(df))

        # Unary negation: -expr  (treat as 0 - expr)
        if expression.startswith('-'):
            inner = self._sub_parse(expression[1:])
            return lambda df, _inner=inner: -_inner(df)

        # Strip outer parentheses
        if expression.startswith('(') and expression.endswith(')'):
            return self._sub_parse(expression[1:-1])

        # Numeric literal
        try:
            val = float(expression)
            return lambda df, _v=val: pd.Series(_v, index=df.index)
        except ValueError:
            pass

        # Special variables (vwap, returns, cap) — case-insensitive
        expr_lower = expression.lower()
        if expr_lower in self._SPECIAL_VARS:
            if self.mode == "wq" and expr_lower not in _WQ_SPECIAL_VARS:
                raise ValueError(f"WQ 模式下不支持变量 '{expr_lower}'")
            var_fn = self._SPECIAL_VARS[expr_lower]
            return lambda df, _fn=var_fn: _fn(df)

        # Average daily volume: adv{N} (e.g., adv20, adv60) — case-insensitive
        if expr_lower.startswith('adv'):
            digits = expr_lower[3:]
            # adv{N} 只接受 1<=N<=MAX_WINDOW。adv 后缀非数字（adv_xxx / advfoo）
            # 不再静默落到列引用分支给出含糊报错，这里显式说明。
            if not digits.isdigit():
                raise ValueError(
                    f"变量 '{expr_lower}' 不是合法的 adv{{N}}，"
                    f"N 必须是非负整数（1~{self.MAX_WINDOW}），如 adv20 / adv60"
                )
            window = self._validate_window(int(digits), 'adv')
            return lambda df, _w=window: (
                df.groupby('stock_code')['volume'].transform(lambda x: x.rolling(_w, min_periods=1).mean())
                if 'stock_code' in df.columns
                else df['volume'].rolling(_w, min_periods=1).mean()
            )

        # Column reference — only allow known columns (case-insensitive)
        col_name = expr_lower.strip()
        from .fundamental_data import ALL_FUNDAMENTAL_NAMES
        _PRICE_COLUMNS = {'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_change', 'market_cap', 'shares'}
        _ALLOWED_COLUMNS = _PRICE_COLUMNS | ALL_FUNDAMENTAL_NAMES
        _ALIAS_MAP = {
            'pe_ratio': 'pe', 'pe_ttm': 'pe', 'pb_ratio': 'pb', 'ps_ratio': 'ps',
            'eps': 'eps_ttm', 'roe_avg': 'roe', 'div_yield': 'dividend_yield',
        }
        col_name = _ALIAS_MAP.get(col_name, col_name)

        if self.mode == "wq":
            if col_name in _LOCAL_ONLY_COLUMNS:
                hint = _WQ_REPLACEMENTS.get(col_name, "")
                hint_msg = f"，替代方案：{hint}" if hint else ""
                raise ValueError(f"WQ 模式下不支持列 '{col_name}'{hint_msg}")
            is_wq_news = any(col_name.startswith(p) for p in _WQ_NEWS_PREFIXES)
            is_wq_group = any(col_name.startswith(p) for p in _WQ_GROUP_PREFIXES)
            if col_name in _WQ_COLUMNS or col_name in _WQ_EXTENDED_FIELDS or is_wq_news or is_wq_group:
                def _wq_field_stub(df, _c=col_name):
                    if _c in df.columns:
                        return df[_c]
                    raise RuntimeError(f"WQ 字段 '{_c}' 无本地数据，请通过 WQ BRAIN 远程执行")
                return _wq_field_stub
            if col_name in ALL_FUNDAMENTAL_NAMES:
                pass  # fall through to local fundamental column
            else:
                logger.warning(f"WQ 模式：未知字段 '{col_name}'，将透传给 WQ BRAIN 校验")
                def _wq_unknown_field_stub(df, _c=col_name):
                    if _c in df.columns:
                        return df[_c]
                    raise RuntimeError(f"WQ 字段 '{_c}' 无本地数据，请通过 WQ BRAIN 远程执行")
                return _wq_unknown_field_stub

        if col_name not in _ALLOWED_COLUMNS:
            raise ValueError(f"Unknown column or variable: {col_name!r}")
        return lambda df, _c=col_name: df[_c]

    @staticmethod
    def _find_keyword(expr: str, keyword: str) -> int | None:
        """Find the rightmost top-level occurrence of a keyword (e.g., ' if ', ' else ')."""
        depth = 0
        result = None
        keyword_len = len(keyword)

        for i in range(len(expr) - keyword_len + 1):
            ch = expr[i]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            elif depth == 0 and expr[i:i+keyword_len] == keyword:
                result = i

        return result

    @staticmethod
    def _find_operator(expr: str, op: str) -> int | None:
        """Find the rightmost top-level occurrence of an operator."""
        depth = 0
        result = None
        op_len = len(op)
        i = 0
        while i < len(expr):
            ch = expr[i]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            elif depth == 0 and i > 0 and expr[i:i + op_len] == op:
                if op_len == 1 and ch in '<>=!':
                    # Single-char op: skip if it's part of a two-char operator
                    next_ch = expr[i + 1] if i + 1 < len(expr) else ''
                    prev_ch = expr[i - 1] if i > 0 else ''
                    if next_ch == '=' or (ch == '=' and prev_ch in '<>!='):
                        i += 1
                        continue
                result = i
            i += 1
        return result

    @staticmethod
    def _split_top_level(s: str) -> list:
        """Split a string by commas at the top level (outside parentheses)."""
        parts = []
        depth = 0
        current = []
        for ch in s:
            if ch == '(':
                depth += 1
                current.append(ch)
            elif ch == ')':
                depth -= 1
                current.append(ch)
            elif ch == ',' and depth == 0:
                parts.append(''.join(current))
                current = []
            else:
                current.append(ch)
        if current:
            parts.append(''.join(current))
        return parts

    @staticmethod
    def _convert_ternary_operators(expression: str) -> str:
        """Convert C-style ternary operators to Python style.

        Converts: (condition) ? true_value : false_value
        To:       (true_value if condition else false_value)

        Args:
            expression: Expression that may contain C-style ternary operators

        Returns:
            Expression with Python-style ternary operators

        Examples:
            >>> ExpressionParser._convert_ternary_operators("((x > 0) ? a : b)")
            '((a if x > 0 else b))'
            >>> ExpressionParser._convert_ternary_operators("rank(ts_argmax(sign_power(((returns < 0) ? ts_std(returns, 20) : close), 2), 5))")
            'rank(ts_argmax(sign_power(((ts_std(returns, 20) if returns < 0 else close)), 2), 5))'
        """
        max_iterations = 20
        iteration = 0

        # Pattern: (condition) ? true_value : false_value
        # Use non-greedy matching to avoid crossing multiple ternary expressions
        pattern = r'\(([^()]+)\)\s*\?\s*([^:]+?)\s*:\s*([^)]+?)(?=\))'

        while '?' in expression and iteration < max_iterations:
            iteration += 1
            old_expression = expression

            def replace_ternary(match):
                condition = match.group(1).strip()
                true_val = match.group(2).strip()
                false_val = match.group(3).strip()
                return f"({true_val} if {condition} else {false_val})"

            # Replace one ternary operator at a time (from innermost)
            expression = re.sub(pattern, replace_ternary, expression, count=1)

            # If no change, stop iteration
            if expression == old_expression:
                break

        return expression


def parse_expression(expression: str, mode: str = "local") -> Callable[[pd.DataFrame], pd.Series]:
    """Convenience function to parse a factor expression.

    Args:
        expression: e.g. "rank(close/open)", "ts_mean(volume, 20)"
        mode: "local" (all operators) or "wq" (WQ BRAIN compatible only)

    Returns:
        Callable that takes a DataFrame and returns factor values as a Series.
    """
    parser = ExpressionParser(mode=mode)
    return parser.parse(expression)


_ALIAS_NORMALIZE = {
    'delta': 'ts_delta', 'delay': 'ts_shift', 'stddev': 'ts_std',
    'covariance': 'ts_cov', 'correlation': 'ts_corr',
    'ts_decay_linear': 'decay_linear', 'ts_product': 'product',
    'ts_delay': 'ts_shift', 'ts_covariance': 'ts_cov',
    'ts_arg_max': 'ts_argmax', 'ts_arg_min': 'ts_argmin',
    'indneutralize': 'indneutralize', 'IndNeutralize': 'indneutralize',
}


def normalize_expression(expression: str) -> str:
    """Canonicalize an expression for similarity comparison."""
    expr = re.sub(r'\s+', '', expression.lower())
    for alias, canonical in _ALIAS_NORMALIZE.items():
        expr = re.sub(rf'\b{re.escape(alias)}\b', canonical, expr)
    return expr


_OP_PATTERN = re.compile(r'([a-z_][a-z0-9_]*)\s*\(')
_FIELD_PATTERN = re.compile(r'\b([a-z_][a-z0-9_]*)\b')


def extract_components(expression: str) -> dict:
    """Extract operator names and field names from an expression."""
    expr_lower = expression.lower()
    operators = set(_OP_PATTERN.findall(expr_lower))
    all_words = set(_FIELD_PATTERN.findall(expr_lower))
    all_ops = _WQ_OPERATORS | _LOCAL_ONLY_OPERATORS | {'if', 'else', 'and', 'or'}
    fields = all_words - operators - all_ops - {'true', 'false'}
    try:
        fields = {w for w in fields if not float(w) and False}
    except ValueError:
        pass
    fields = {w for w in fields if not w.replace('.', '').isdigit()}
    return {"operators": operators, "fields": fields}


# ------------------------------------------------------------------
# WQ 提交前校验所需的公开查询接口（任务 2/3/4 使用）
# ------------------------------------------------------------------

def variable_category(name: str) -> str:
    """判定 FASTEXPR 里一个变量名的来源类别。

    返回 ``"builtin" | "price" | "group" | "datafield" | "unknown"``。

    三类来源完全不同，必须分开校验：
    - ``builtin``：WQ 内置变量（vwap/returns/cap/adv{N}），服务端直接认识，
      **不在** /data-fields 目录里，用目录去查会全部误判为非法。
    - ``price``：价格/成交量列（open/high/low/close/volume）。
    - ``group``：分组字段（industry/subindustry/sector/... 及其 indclass. 前缀形式）。
    - ``datafield``：真正需要查目录的一类。
    - ``unknown``：以上都不是——目录可用时即为非法字段，目录不可用时降级为警告。
    """
    key = name.strip().lower()
    if not key:
        return "unknown"
    # 本地专有别名先归一，否则 market_cap 会被误判成 data field
    key = _WQ_NAME_ALIASES.get(key, key)

    if key in _WQ_SPECIAL_VARS:
        return "builtin"
    if re.fullmatch(r"adv\d+", key):
        return "builtin"
    if key in _WQ_COLUMNS:
        return "builtin" if key in _WQ_SPECIAL_VARS else "price"
    # 分组字段：WQ 里 industry/subindustry/sector 是独立 namespace，
    # 既不是价格列也不在 data-fields 目录中，漏掉会导致误杀所有 group_* 表达式。
    if key in _WQ_GROUP_FIELDS or any(key.startswith(p) for p in _WQ_GROUP_PREFIXES):
        return "group"
    if any(key.startswith(p) for p in _WQ_NEWS_PREFIXES):
        return "datafield"
    return "datafield"


def wq_field_known(name: str) -> tuple:
    """判断 WQ data field 是否合法，返回 ``(exists, catalog_available)``。

    目录不可用时 ``exists=False`` 但 ``catalog_available=False``——调用方据此把
    "字段非法"降级成"无法校验"，避免误杀合规表达式。
    """
    from . import wq_field_catalog

    available = wq_field_catalog.catalog_status()["available"]
    if not available:
        return False, False
    return wq_field_catalog.field_exists(name), True
