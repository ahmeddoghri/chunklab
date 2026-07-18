"""A document with planted answers, and the exact character span of each answer.

The document reads like a short internal wiki: several paragraphs, each with a
fact worth retrieving. The answers are specific sentences. We compute each
answer's character span from the document itself at import time, so the spans
are always exact no matter how the text is edited. That is what lets the
benchmark check whether a retrieved chunk truly contains the answer.
"""
from __future__ import annotations

DOCUMENT = """\
Onboarding notes for the payments service.

The payments service runs on three replicas behind a load balancer in the us-east-1 region. \
Each replica holds an in-memory idempotency cache so a retried charge is never double-billed. \
The cache entries expire after twenty-four hours.

Deployments go out through the release pipeline every Tuesday and Thursday at 10am eastern. \
A deploy is blocked automatically if the error budget for the week is already spent. \
Rolling back is a single command and takes about ninety seconds end to end.

Secrets are stored in the vault and rotated every thirty days. \
The database credentials specifically are rotated on the first of every month, out of band. \
Never paste a secret into a chat message or a ticket.

On-call runs in weekly shifts starting Monday morning. \
The primary on-call carries the pager and the secondary is the backup for escalations. \
If a page is not acknowledged within five minutes it escalates to the secondary automatically.

The service exposes a health endpoint at slash healthz and a readiness endpoint at slash readyz. \
Metrics are scraped every fifteen seconds and dashboards live in the shared observability workspace. \
Alert thresholds are defined as code in the monitoring repository.
"""


def _span(answer: str) -> tuple[int, int]:
    i = DOCUMENT.index(answer)
    return i, i + len(answer)


# (question, answer_sentence, span_start, span_end). Spans computed from the doc.
_ANSWERS = [
    ("how many replicas and what region does payments run in",
     "The payments service runs on three replicas behind a load balancer in the us-east-1 region."),
    ("how long do idempotency cache entries last",
     "The cache entries expire after twenty-four hours."),
    ("when do deployments go out",
     "Deployments go out through the release pipeline every Tuesday and Thursday at 10am eastern."),
    ("what blocks a deploy",
     "A deploy is blocked automatically if the error budget for the week is already spent."),
    ("how long does a rollback take",
     "Rolling back is a single command and takes about ninety seconds end to end."),
    ("how often are database credentials rotated",
     "The database credentials specifically are rotated on the first of every month, out of band."),
    ("who carries the pager on call",
     "The primary on-call carries the pager and the secondary is the backup for escalations."),
    ("what happens if a page is not acknowledged",
     "If a page is not acknowledged within five minutes it escalates to the secondary automatically."),
    ("how often are metrics scraped",
     "Metrics are scraped every fifteen seconds and dashboards live in the shared observability workspace."),
    ("where are alert thresholds defined",
     "Alert thresholds are defined as code in the monitoring repository."),
]

QA: list[tuple[str, str, int, int]] = [
    (q, a, *_span(a)) for q, a in _ANSWERS
]
