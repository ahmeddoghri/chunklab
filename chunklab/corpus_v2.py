"""A second, independently-written document, to check whether the
chunking-strategy ranking generalizes or was a lucky pick.

The README is upfront that the headline number comes from "one document,
ten planted answers, a lexical retriever" and explicitly invites swapping
in a different document to find your own number. This is that: a
different domain (a REST API reference instead of an internal wiki),
different sentence lengths and structure, and questions phrased
independently of the document's own wording, written before running the
benchmark against it.
"""
from __future__ import annotations

DOCUMENT = """\
API reference for the notifications service.

Authenticate every request with a bearer token in the Authorization header. \
Tokens are issued by the auth service and expire after one hour. \
There is no refresh endpoint; request a new token instead of renewing the old one.

The create endpoint accepts a POST to slash v1 slash notifications with a JSON body. \
Required fields are recipient, channel, and message. \
Requests missing a required field return a 422 with the field name in the error body.

Rate limiting allows 100 requests per minute per API key. \
Exceeding the limit returns a 429 status with a Retry-After header in seconds. \
Retries should use exponential backoff starting at one second.

Delivery status can be checked with a GET to slash v1 slash notifications slash id. \
Status values are queued, sent, delivered, and failed. \
A failed delivery includes a reason code you can look up in the errors table.

Webhooks notify your server when a delivery status changes. \
Configure the webhook URL in the dashboard under integrations. \
Every webhook payload is signed, and you should verify the signature before trusting it.

Batch sends accept up to 500 messages in a single POST to slash v1 slash batch. \
The whole batch is rejected if any single message fails validation. \
Partial success is not supported; fix the bad message and resend the entire batch.
"""


def _span(answer: str) -> tuple[int, int]:
    i = DOCUMENT.index(answer)
    return i, i + len(answer)


_ANSWERS = [
    ("how do I authenticate a request",
     "Authenticate every request with a bearer token in the Authorization header."),
    ("how long until my token expires",
     "Tokens are issued by the auth service and expire after one hour."),
    ("what happens if I forget a required field when creating a notification",
     "Requests missing a required field return a 422 with the field name in the error body."),
    ("how many requests can I make per minute",
     "Rate limiting allows 100 requests per minute per API key."),
    ("what should I do when I get rate limited",
     "Retries should use exponential backoff starting at one second."),
    ("what are the possible delivery statuses",
     "Status values are queued, sent, delivered, and failed."),
    ("how do I find out why a message failed",
     "A failed delivery includes a reason code you can look up in the errors table."),
    ("where do I set up my webhook",
     "Configure the webhook URL in the dashboard under integrations."),
    ("how many messages can I send in one batch",
     "Batch sends accept up to 500 messages in a single POST to slash v1 slash batch."),
    ("what happens if one message in a batch is invalid",
     "The whole batch is rejected if any single message fails validation."),
]

QA: list[tuple[str, str, int, int]] = [(q, a, *_span(a)) for q, a in _ANSWERS]
