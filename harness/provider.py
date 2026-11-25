"""Transport for OpenAI-compatible Chat Completions endpoints."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class ProviderError(Exception):
    """A model request failed before its response could be acted on."""

    def __init__(self, code: str, *, http_status: int | None = None):
        self.code = code
        self.http_status = http_status
        super().__init__(f"{code} (HTTP {http_status})" if http_status else code)


class ContextRejected(ProviderError):
    """Provider rejected the request because it exceeds its context window."""


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@dataclass(frozen=True)
class Completion:
    message: dict
    finish_reason: str | None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_prompt_tokens: int | None = None


def _count(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _retry_after(header: str | None) -> float:
    if not header:
        return 0
    try:
        return max(0, int(header))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(header)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return 0


def _provider_error(error: HTTPError) -> tuple[ProviderError, bool, float]:
    status = error.code
    try:
        body = json.loads(error.read(64 * 1024))
        detail = body.get("error") if isinstance(body, dict) else None
    except (ValueError, OSError, HTTPException, RecursionError):
        detail = None
    finally:
        error.close()
    code = detail.get("code") if isinstance(detail, dict) else None
    message = detail.get("message", "") if isinstance(detail, dict) else ""
    message = message.lower() if isinstance(message, str) else ""
    context_codes = ("context_length_exceeded", "context_window_exceeded", "input_too_long")
    if status in (400, 413, 422) and (
        code in context_codes or "maximum context length" in message
        or "context window is exceeded" in message
    ):
        return ContextRejected("context_length_exceeded", http_status=status), False, 0
    if status in (401, 403):
        return ProviderError("provider_access_denied", http_status=status), False, 0
    if code in ("insufficient_quota", "billing_hard_limit_reached"):
        return ProviderError("provider_quota_exhausted", http_status=status), False, 0
    retryable = status in (408, 429, 500, 502, 503, 504)
    label = "provider_rate_limited" if status == 429 else "provider_http_error"
    delay = _retry_after(error.headers.get("Retry-After") if error.headers else None)
    return ProviderError(label, http_status=status), retryable, delay


def _parse_completion(raw: bytes) -> Completion:
    try:
        data = json.loads(raw)
        choice = data["choices"][0]
        message = choice["message"]
        finish_reason = choice.get("finish_reason")
        if not isinstance(message, dict) or message.get("role", "assistant") != "assistant":
            raise ValueError
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise ValueError
        json.dumps(message, ensure_ascii=False, allow_nan=False).encode()
    except (ValueError, KeyError, IndexError, TypeError, AttributeError, RecursionError):
        raise ProviderError("invalid_provider_response") from None

    usage = data.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    details = usage.get("prompt_tokens_details")
    cached = _count(details.get("cached_tokens")) if isinstance(details, dict) else None
    prompt = _count(usage.get("prompt_tokens"))
    if cached is not None and prompt is not None and cached > prompt:
        cached = None
    return Completion(message, finish_reason, prompt, _count(usage.get("completion_tokens")), cached)


class ChatClient:
    """Retry only inference requests with transient transport or HTTP failures."""

    def __init__(
        self, base_url: str, model: str, api_key: str = "", *,
        max_output_tokens: int, timeout: float = 60, max_retries: int = 2,
        cache_key: str | None = None, output_token_field: str = "max_tokens",
        opener=None, sleep=time.sleep,
    ):
        url = urlsplit(base_url)
        if url.scheme not in ("http", "https") or not url.hostname or url.username is not None:
            raise ValueError("base_url must be an HTTP(S) endpoint without credentials")
        if url.query or url.fragment:
            raise ValueError("base_url must not contain a query or fragment")
        if output_token_field not in ("max_tokens", "max_completion_tokens"):
            raise ValueError("invalid output token field")
        if max_output_tokens < 1 or timeout <= 0 or max_retries < 0 or not model.strip():
            raise ValueError("invalid model, output limit, timeout or retry count")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_key = cache_key
        self.output_token_field = output_token_field
        self.opener = opener if opener is not None else build_opener(NoRedirects()).open
        self.sleep = sleep
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def complete(self, messages, tools, *, on_attempt=None) -> Completion:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            self.output_token_field: self.max_output_tokens,
        }
        if self.cache_key:
            payload["prompt_cache_key"] = self.cache_key
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        request = Request(self.url, data=data, headers=self.headers, method="POST")

        for attempt in range(self.max_retries + 1):
            if on_attempt:
                on_attempt(attempt > 0)
            try:
                with self.opener(request, timeout=self.timeout) as response:
                    limit = 4 * 1024 * 1024
                    raw = response.read(limit + 1)
                if len(raw) > limit:
                    raise ProviderError("provider_response_too_large")
                return _parse_completion(raw)
            except HTTPError as error:
                failure, retryable, retry_after = _provider_error(error)
            except (URLError, OSError, HTTPException):
                failure, retryable, retry_after = ProviderError("provider_unreachable"), True, 0
            if not retryable or attempt == self.max_retries or retry_after > 30:
                raise failure from None
            self.sleep(max(retry_after, 0.5 * (2 ** min(attempt, 5))))
        raise AssertionError("retry loop did not terminate")
