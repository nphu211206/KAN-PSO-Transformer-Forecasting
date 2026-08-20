"""Check that the manuscript stays within the frozen evidence boundary."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANUSCRIPT = ROOT / "paper_draft.md"

REQUIRED_TEXT = [
    "380 completed experiment records",
    "0.175458 to 0.113224",
    "0.101641",
    "0.064489 +/- 0.001843",
    "0.164913 +/- 0.014927",
    "0.127925 +/- 0.000598",
    "0.195536 +/- 0.028679",
    "four of five ETTh1 seeds and three of five ETTh2 seeds",
    "pilot_60_20_20_v1",
    "ett_canonical_12_4_4_v1",
    "## List of abbreviations",
    "### Availability of data and materials",
    "### Acknowledgements",
    "### 3.10 AI-assisted tooling",
    "### 2.4 Comparable JEAS engineering studies",
    "Table 8. Uniform-checkpoint error in recorded OT units.",
    "Table 4. Historical Kaggle HPO search space.",
    "Fig. 2. Canonical ETT target shift.",
    "K08 and K09 use different campaign and source",
    "Fig. 7. Uniform-checkpoint residual and absolute-error distributions.",
    "0.10042 +/- 0.00308",
    "Table 9. Canonical multi-horizon test MSE over five seeds.",
    "in all eight dataset-horizon cells",
    "lower MSE in 38/40 paired comparisons",
]

FORBIDDEN_PATTERNS = [
    r"\bKAN (?:is|was) state[- ]of[- ]the[- ]art\b",
    r"(?<!not a claim that )\bKAN universally outperforms\b",
    r"\bfirst (?:ever|systematic) KAN[- ]metaheuristic\b",
    r"\bp\s*[<=>]\s*0\.\d+",
    r"\bstatistically significant\b",
]


def main() -> int:
    text = MANUSCRIPT.read_text(encoding="utf-8")
    errors: list[str] = []

    for required in REQUIRED_TEXT:
        if required not in text:
            errors.append(f"missing frozen statement: {required}")

    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            errors.append(f"forbidden claim pattern: {pattern}")

    table_numbers = [
        int(number)
        for number in re.findall(r"^\*\*Table (\d+)\.", text, flags=re.MULTILINE)
    ]
    table_blocks = re.findall(
        r"^\|(?:\s*:?-+:?\s*\|)+$", text, flags=re.MULTILINE
    )
    expected_tables = list(range(1, len(table_numbers) + 1))
    if table_numbers != expected_tables:
        errors.append("table numbering is not consecutive")
    if len(table_numbers) != len(table_blocks):
        errors.append(
            f"table captions ({len(table_numbers)}) do not match table blocks "
            f"({len(table_blocks)})"
        )

    figure_numbers = [
        int(number)
        for number in re.findall(r"!\[\*\*Fig\. (\d+)\.", text)
    ]
    expected_figures = list(range(1, len(figure_numbers) + 1))
    if figure_numbers != expected_figures:
        errors.append("figure numbering is not consecutive")

    reference_section = text.split("## References", maxsplit=1)[-1]
    references = re.findall(r"^(\d+)\. ", reference_section, flags=re.MULTILINE)
    expected = [str(index) for index in range(1, len(references) + 1)]
    if references != expected:
        errors.append("reference numbering is not consecutive")

    body = text.split("## References", maxsplit=1)[0]
    for reference in references:
        if not re.search(rf"\[(?:[^\]]*,\s*)?{reference}(?:\s*,[^\]]*)?\]", body):
            errors.append(f"reference [{reference}] is not cited in the manuscript body")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print(f"Manuscript audit passed: {len(references)} references and frozen claims are consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
