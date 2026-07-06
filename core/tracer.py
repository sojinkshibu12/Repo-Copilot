"""Structured tracing — JSON logging of every LLM call, tool call, and request.

Designed as a lightweight drop-in that can be swapped for OpenTelemetry later.
The `Tracer` interface mirrors OpenTelemetry's span API so migration is a
one-class replacement.
"""

import json
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)


# ── Events ───────────────────────────────────────────────────────

@dataclass
class Span:
    """A single traced operation (LLM call, tool execution, HTTP request)."""
    trace_id: str
    span_id: str
    parent_id: str | None
    name: str
    kind: str
    start_time: float  # monotonic clock
    start_datetime: str  # ISO-8601 for human readers
    end_time: float | None = None
    duration_ms: float | None = None
    attributes: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    status: str = "ok"
    error: str = ""

    def close(self, status: str = "ok", error: str = ""):
        self.end_time = time.monotonic()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.status = status
        self.error = error

    def set_attribute(self, key: str, value: Any):
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict | None = None):
        self.events.append({
            "name": name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "attributes": attributes or {},
        })

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind,
            "duration_ms": round(self.duration_ms, 1) if self.duration_ms is not None else None,
            "start_datetime": self.start_datetime,
            "status": self.status,
            "error": self.error,
            "attributes": self.attributes,
            "events": self.events,
        }


# ── Emitters ─────────────────────────────────────────────────────

class SpanEmitter:
    """Base emitter — plug in OTel exporter here later."""
    def emit(self, span: Span):
        ...


class JsonLogEmitter(SpanEmitter):
    """Emit spans as JSON log lines (one JSON object per line)."""
    def __init__(self, logger_name: str = "tracing"):
        self._log = logging.getLogger(logger_name)

    def emit(self, span: Span):
        record = span.to_dict()
        self._log.info(json.dumps(record, default=str))


class ConsoleEmitter(SpanEmitter):
    """Emit spans to stdout as JSON lines (for local dev)."""
    def emit(self, span: Span):
        print(json.dumps(span.to_dict(), default=str), flush=True)


# ── Tracer ───────────────────────────────────────────────────────

class Tracer:
    """Lightweight span tracer.

    Usage:
        tracer = Tracer(service="repo-copilot")
        with tracer.span("llm.chat", kind="llm") as span:
            span.set_attribute("model", "gpt-4o")
            result = llm.chat(...)
            span.set_attribute("tokens", 123)

    The `_local` thread-local stores the active span stack so nested spans
    automatically inherit the parent trace/span ID.
    """

    def __init__(
        self,
        service: str = "repo-copilot",
        emitter: SpanEmitter | None = None,
    ):
        self.service = service
        self.emitter = emitter or _emitter_from_env()
        self._local = threading.local()

    def _get_stack(self) -> list:
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        return self._local.stack

    def start_span(
        self,
        name: str,
        kind: str = "internal",
        attributes: dict | None = None,
    ) -> Span:
        stack = self._get_stack()
        parent = stack[-1] if stack else None
        span = Span(
            trace_id=parent.trace_id if parent else uuid.uuid4().hex[:16],
            span_id=uuid.uuid4().hex[:16],
            parent_id=parent.span_id if parent else None,
            name=name,
            kind=kind,
            start_time=time.monotonic(),
            start_datetime=datetime.now(timezone.utc).isoformat(),
            attributes=attributes or {},
        )
        stack.append(span)
        return span

    def end_span(self, span: Span, status: str = "ok", error: str = ""):
        span.close(status=status, error=error)
        self.emitter.emit(span)
        stack = self._get_stack()
        if stack and stack[-1] is span:
            stack.pop()

    @contextmanager
    def span(
        self,
        name: str,
        kind: str = "internal",
        attributes: dict | None = None,
    ) -> Iterator[Span]:
        span = self.start_span(name, kind=kind, attributes=attributes)
        try:
            yield span
        except Exception as e:
            self.end_span(span, status="error", error=str(e))
            raise
        else:
            self.end_span(span)

    def trace(self, name: str | None = None, kind: str = "internal"):
        """Decorator form — wraps a function call in a span."""
        def decorator(fn: Callable) -> Callable:
            span_name = name or fn.__qualname__
            def wrapper(*args, **kwargs):
                with self.span(span_name, kind=kind) as span:
                    return fn(*args, **kwargs)
            return wrapper
        return decorator


# ── Global singleton ─────────────────────────────────────────────

_TRACER: Tracer | None = None
_lock = threading.Lock()


def get_tracer() -> Tracer:
    global _TRACER
    if _TRACER is None:
        with _lock:
            if _TRACER is None:
                _TRACER = Tracer()
    return _TRACER


def set_tracer(tracer: Tracer):
    global _TRACER
    _TRACER = tracer


def _emitter_from_env() -> SpanEmitter:
    mode = os.environ.get("TRACE_EMITTER", "json").lower()
    if mode == "console":
        return ConsoleEmitter()
    if mode == "none":
        return SpanEmitter()  # no-op
    return JsonLogEmitter()


def reset_tracer():
    """Reset the global tracer (useful in tests)."""
    global _TRACER
    _TRACER = None
