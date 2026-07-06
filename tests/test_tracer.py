"""Tests for core/tracer.py — structured JSON tracing layer."""

import json
import time
import threading
from io import StringIO

import pytest

from core.tracer import (
    Tracer, Span, SpanEmitter, JsonLogEmitter, ConsoleEmitter,
    get_tracer, set_tracer, reset_tracer,
)


# ── Helpers ───────────────────────────────────────────────────────

class _CaptureEmitter(SpanEmitter):
    def __init__(self):
        self.spans: list[Span] = []

    def emit(self, span: Span):
        self.spans.append(span)


# ── Span ──────────────────────────────────────────────────────────

class TestSpan:
    def test_span_defaults(self):
        s = Span(
            trace_id="abc", span_id="def", parent_id=None,
            name="test", kind="internal",
            start_time=time.monotonic(), start_datetime="2025-01-01T00:00:00",
        )
        assert s.trace_id == "abc"
        assert s.span_id == "def"
        assert s.parent_id is None
        assert s.name == "test"
        assert s.kind == "internal"
        assert s.status == "ok"
        assert s.error == ""
        assert s.attributes == {}
        assert s.events == []
        assert s.end_time is None
        assert s.duration_ms is None

    def test_span_close_sets_duration(self):
        s = Span(
            trace_id="a", span_id="b", parent_id=None,
            name="t", kind="test",
            start_time=time.monotonic(), start_datetime="",
        )
        time.sleep(0.01)
        s.close()
        assert s.end_time is not None
        assert s.duration_ms is not None
        assert s.duration_ms >= 1.0

    def test_span_close_sets_status(self):
        s = Span(
            trace_id="a", span_id="b", parent_id=None,
            name="t", kind="test",
            start_time=time.monotonic(), start_datetime="",
        )
        s.close(status="error", error="something broke")
        assert s.status == "error"
        assert s.error == "something broke"

    def test_span_set_attribute(self):
        s = Span(
            trace_id="a", span_id="b", parent_id=None,
            name="t", kind="test",
            start_time=time.monotonic(), start_datetime="",
        )
        s.set_attribute("key", "value")
        assert s.attributes == {"key": "value"}

    def test_span_add_event(self):
        s = Span(
            trace_id="a", span_id="b", parent_id=None,
            name="t", kind="test",
            start_time=time.monotonic(), start_datetime="",
        )
        s.add_event("hit", {"tokens": 10})
        assert len(s.events) == 1
        assert s.events[0]["name"] == "hit"
        assert s.events[0]["attributes"]["tokens"] == 10

    def test_span_to_dict_includes_all_fields(self):
        s = Span(
            trace_id="a", span_id="b", parent_id="p",
            name="t", kind="test",
            start_time=time.monotonic(), start_datetime="2025-01-01",
        )
        s.close(status="ok")
        d = s.to_dict()
        assert d["trace_id"] == "a"
        assert d["span_id"] == "b"
        assert d["parent_id"] == "p"
        assert d["name"] == "t"
        assert d["kind"] == "test"
        assert d["duration_ms"] is not None
        assert d["status"] == "ok"


# ── Tracer ────────────────────────────────────────────────────────

class TestTracer:
    def test_start_span_generates_ids(self):
        emitter = _CaptureEmitter()
        tracer = Tracer(emitter=emitter)
        span = tracer.start_span("hello", kind="llm")
        assert span.name == "hello"
        assert span.kind == "llm"
        assert len(span.trace_id) == 16
        assert len(span.span_id) == 16
        assert span.parent_id is None
        tracer.end_span(span)
        assert len(emitter.spans) == 1

    def test_nested_spans_share_trace_id(self):
        emitter = _CaptureEmitter()
        tracer = Tracer(emitter=emitter)
        parent = tracer.start_span("parent")
        child = tracer.start_span("child")
        assert child.trace_id == parent.trace_id
        assert child.parent_id == parent.span_id
        tracer.end_span(child)
        tracer.end_span(parent)
        assert len(emitter.spans) == 2

    def test_context_manager_emits_on_exit(self):
        emitter = _CaptureEmitter()
        tracer = Tracer(emitter=emitter)
        with tracer.span("ctx", kind="tool") as span:
            span.set_attribute("k", "v")
        assert len(emitter.spans) == 1
        assert emitter.spans[0].name == "ctx"
        assert emitter.spans[0].attributes["k"] == "v"
        assert emitter.spans[0].duration_ms is not None

    def test_context_manager_sets_error_on_exception(self):
        emitter = _CaptureEmitter()
        tracer = Tracer(emitter=emitter)
        with pytest.raises(ValueError):
            with tracer.span("failing"):
                raise ValueError("boom")
        assert len(emitter.spans) == 1
        assert emitter.spans[0].status == "error"
        assert "boom" in emitter.spans[0].error

    def test_decorator_wraps_function(self):
        emitter = _CaptureEmitter()
        tracer = Tracer(emitter=emitter)

        @tracer.trace("my_func", kind="internal")
        def my_func():
            return 42

        result = my_func()
        assert result == 42
        assert len(emitter.spans) == 1
        assert emitter.spans[0].name == "my_func"

    def test_decorator_defaults_to_qualname(self):
        emitter = _CaptureEmitter()
        tracer = Tracer(emitter=emitter)

        @tracer.trace()
        def do_stuff():
            return "ok"

        do_stuff()
        assert "do_stuff" in emitter.spans[0].name

    def test_thread_local_isolation(self):
        emitter = _CaptureEmitter()
        tracer = Tracer(emitter=emitter)
        spans_in_thread = []

        def worker():
            s = tracer.start_span("thread-span")
            spans_in_thread.append(s)
            tracer.end_span(s)

        parent = tracer.start_span("main")
        t = threading.Thread(target=worker)
        t.start()
        t.join()
        tracer.end_span(parent)

        assert spans_in_thread[0].parent_id is None
        assert spans_in_thread[0].trace_id != parent.trace_id

    def test_reset_tracer_creates_new_instance(self):
        reset_tracer()
        t1 = get_tracer()
        reset_tracer()
        t2 = get_tracer()
        assert t1 is not t2

    def test_set_tracer_overrides_global(self):
        emitter = _CaptureEmitter()
        custom = Tracer(emitter=emitter)
        set_tracer(custom)
        assert get_tracer() is custom
        reset_tracer()


# ── Emitters ──────────────────────────────────────────────────────

class TestEmitters:
    def test_json_emitter_produces_valid_json(self, caplog):
        import logging
        caplog.set_level(logging.INFO)
        emitter = JsonLogEmitter(logger_name="test_tracer_json")
        s = Span(
            trace_id="a", span_id="b", parent_id=None,
            name="test", kind="internal",
            start_time=time.monotonic(), start_datetime="",
        )
        s.close()
        emitter.emit(s)

        assert len(caplog.records) == 1
        parsed = json.loads(caplog.records[0].getMessage())
        assert parsed["trace_id"] == "a"
        assert parsed["name"] == "test"
        assert parsed["duration_ms"] is not None

    def test_console_emitter_writes_to_stdout(self, capsys):
        emitter = ConsoleEmitter()
        s = Span(
            trace_id="a", span_id="b", parent_id=None,
            name="test", kind="internal",
            start_time=time.monotonic(), start_datetime="",
        )
        s.close()
        emitter.emit(s)
        captured = capsys.readouterr()
        assert captured.out
        parsed = json.loads(captured.out.strip())
        assert parsed["name"] == "test"


# ── Global tracer ─────────────────────────────────────────────────

class TestGlobalTracer:
    def setup_method(self):
        reset_tracer()

    def test_get_tracer_creates_if_unset(self):
        t = get_tracer()
        assert isinstance(t, Tracer)
        assert t.service == "repo-copilot"

    def test_get_tracer_returns_singleton(self):
        t1 = get_tracer()
        t2 = get_tracer()
        assert t1 is t2
