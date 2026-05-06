from __future__ import annotations

import json
from typing import Any, Dict, List


def report_to_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = [
        f"# Research Report: {report.get('topic', '')}",
        "",
        "## Summary",
        str(report.get("summary", "")),
        "",
        "## Claims",
    ]

    claims = report.get("claims", [])
    if not claims:
        lines.append("- No claims produced.")
    else:
        for claim in claims:
            claim_text = str(claim.get("claim", ""))
            confidence = claim.get("confidence", 0.0)
            citations = claim.get("citations", [])
            citation_text = ", ".join(
                [f"{item.get('doc_id')}::{item.get('chunk_id')}" for item in citations]
            )
            lines.append(
                f"- {claim_text} (confidence={confidence}, citations=[{citation_text}])"
            )

    lines.extend(["", "## Open Questions"])
    for question in report.get("open_questions", []):
        lines.append(f"- {question}")

    lines.extend(
        [
            "",
            "## Validation",
            "```json",
            json.dumps(report.get("validation", {}), indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines)
