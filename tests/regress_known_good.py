#!/usr/bin/env python3
"""零误杀回归：known-good 表达式一条都不能被本地校验判死。

判定标准
--------
**error 数必须为 0**。任何 ``status="error"`` 都是回归缺陷——
样本里的表达式全部在 WQ BRAIN 生产环境真实跑通过，误杀一个就是真金白银的损失
（历史上已因此误杀 Sharpe 2.29 的 ACTIVE 因子 9qWZ9G2x）。

用法（容器内，工作目录必须是 /app）::

    python tests/regress_known_good.py

也可以用 pytest 跑更细的用例矩阵::

    python -m pytest tests/test_wq_operator_catalog.py -v

退出码：0 = 通过，1 = 存在误杀。
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantgpt.wq_validator import validate_expression  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "wq_known_good_expressions.json"


def main() -> int:
    if not FIXTURE.exists():
        print(f"ERROR: 缺少样本文件 {FIXTURE}")
        return 1

    known_good = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if not known_good:
        print("ERROR: 样本文件为空，回归测试失去意义")
        return 1

    errors = []
    warnings = 0
    for item in known_good:
        result = validate_expression(item["expr"], mode="wq")
        if result.status == "error":
            errors.append((item, result))
        elif result.status == "warning":
            warnings += 1

    total = len(known_good)
    print(f"零误杀回归：{total} 条真实在 WQ BRAIN 跑通过的表达式")
    print(f"  error   = {len(errors)}   ← 必须为 0")
    print(f"  warning = {warnings}（目录不可用时的降级提示，不阻断提交）")

    if errors:
        print("\n以下表达式被误杀：\n")
        by_operator = Counter()
        for item, result in errors:
            for issue in result.errors:
                if issue.kind == "operator":
                    by_operator[issue.name] += 1
            print(f"  [{item['id']}] {item['expr'][:110]}")
            print(f"      -> {result.message[:240]}")
        if by_operator:
            print("\n误杀算子分布：")
            for name, count in by_operator.most_common():
                print(f"  {count:4d}  {name}")
        return 1

    print("\nPASS: 零误杀。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
