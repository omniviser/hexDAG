from __future__ import annotations

from contextlib import suppress
from typing import Any

from hexai.core.ports.observer_manager import Observer

from .file_sink import _event_to_dict


class OpenTelemetrySinkError(Exception):
    """Raised when OpenTelemetry sink cannot start/stop or is misconfigured."""


class OpenTelemetrySinkConfig:
    """
    Configuration for OpenTelemetry export.

    exporter:
      - "console": print spans/metrics to stdout (good for local dev)
      - "otlp": send to OTel Collector (gRPC or HTTP)
      - "jaeger": send directly to Jaeger (thrift endpoint)
      - "none": disable exporting (no-op providers), still builds attributes for future use

    otlp_protocol: "grpc" | "http"
    otlp_endpoint:
      - grpc default: "http://localhost:4317"
      - http default: "http://localhost:4318"
    """

    def __init__(
        self,
        exporter: str = "console",  # "console" | "otlp" | "jaeger" | "none"
        service_name: str = "hexDAG",
        service_version: str = "0.0.0",
        otlp_protocol: str = "grpc",  # "grpc" | "http"
        otlp_endpoint: str = "http://localhost:4317",
        otlp_headers: dict[str, str] | None = None,
        jaeger_endpoint: str = "http://localhost:14268/api/traces",
        metric_interval_sec: float = 5.0,
    ) -> None:
        self.exporter = exporter
        self.service_name = service_name
        self.service_version = service_version

        proto = (otlp_protocol or "grpc").lower()
        if proto not in {"grpc", "http"}:
            raise ValueError(f"Invalid otlp_protocol: {otlp_protocol}")
        self.otlp_protocol = proto
        if otlp_endpoint == "http://localhost:4317" and proto == "http":
            otlp_endpoint = "http://localhost:4318"
        self.otlp_endpoint = otlp_endpoint

        self.otlp_headers = otlp_headers or {}
        self.jaeger_endpoint = jaeger_endpoint
        self.metric_interval_sec = metric_interval_sec


class OpenTelemetrySinkObserver(Observer):
    """
    Observer that exports events as OpenTelemetry traces and metrics.

    - Traces: spans for pipeline/node/macro/tool with attributes and status.
    - Metrics: latency (histogram), errors (counter), LLM tokens (counter).
    """

    def __init__(self, cfg: OpenTelemetrySinkConfig) -> None:
        self.cfg = cfg

        # OTel providers/instruments (late-bound in start())
        self._tracer_provider: Any | None = None
        self._meter_provider: Any | None = None
        self._tracer: Any | None = None
        self._meter: Any | None = None

        # Instruments
        self._latency_hist: Any | None = None
        self._error_counter: Any | None = None
        self._tokens_counter: Any | None = None

    async def start(self) -> None:
        try:
            from opentelemetry import metrics, trace
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import (
                ConsoleMetricExporter,
                PeriodicExportingMetricReader,
            )
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import (
                BatchSpanProcessor,
                ConsoleSpanExporter,
            )
        except Exception as e:  # pragma: no cover - missing deps path
            raise OpenTelemetrySinkError(
                "OpenTelemetry dependencies are missing. "
                "Install extras (e.g. opentelemetry-api, opentelemetry-sdk, exporters)."
            ) from e

        resource = Resource.create({
            "service.name": self.cfg.service_name,
            "service.version": self.cfg.service_version,
        })

        # Traces
        tp = TracerProvider(resource=resource)
        exporter_mode = (self.cfg.exporter or "console").lower()

        if exporter_mode == "console":
            tp.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        elif exporter_mode == "otlp":
            if self.cfg.otlp_protocol.lower() == "grpc":
                try:
                    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                        OTLPSpanExporter as OTLPSpanExporterGrpc,
                    )
                except Exception as e:  # pragma: no cover
                    raise OpenTelemetrySinkError(
                        "Missing OTLP gRPC trace exporter. "
                        "Install opentelemetry-exporter-otlp-proto-grpc."
                    ) from e
                tp.add_span_processor(BatchSpanProcessor(OTLPSpanExporterGrpc()))
            else:
                try:
                    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                        OTLPSpanExporter as OTLPSpanExporterHttp,
                    )
                except Exception as e:  # pragma: no cover
                    raise OpenTelemetrySinkError(
                        "Missing OTLP HTTP trace exporter. "
                        "Install opentelemetry-exporter-otlp-proto-http."
                    ) from e
                tp.add_span_processor(
                    BatchSpanProcessor(
                        OTLPSpanExporterHttp(
                            endpoint=self.cfg.otlp_endpoint,
                            headers=self.cfg.otlp_headers,
                        )
                    )
                )
        elif exporter_mode == "jaeger":
            try:
                from opentelemetry.exporter.jaeger.thrift import JaegerExporter
            except Exception as e:  # pragma: no cover
                raise OpenTelemetrySinkError(
                    "Missing Jaeger exporter. Install opentelemetry-exporter-jaeger."
                ) from e
            tp.add_span_processor(
                BatchSpanProcessor(JaegerExporter(collector_endpoint=self.cfg.jaeger_endpoint))
            )
        elif exporter_mode == "none":
            pass
        else:
            raise OpenTelemetrySinkError(f"Unknown exporter mode: {self.cfg.exporter}")

        trace.set_tracer_provider(tp)
        self._tracer_provider = tp
        self._tracer = trace.get_tracer(__name__)

        # Metrics
        readers: list[Any] = []
        interval_ms = int(max(self.cfg.metric_interval_sec, 1.0) * 1000)

        if exporter_mode == "console":
            readers.append(
                PeriodicExportingMetricReader(
                    ConsoleMetricExporter(), export_interval_millis=interval_ms
                )
            )
        elif exporter_mode == "otlp":
            if self.cfg.otlp_protocol.lower() == "grpc":
                try:
                    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                        OTLPMetricExporter as OTLPMetricExporterGrpc,
                    )
                except Exception as e:  # pragma: no cover
                    raise OpenTelemetrySinkError(
                        "Missing OTLP gRPC metric exporter. "
                        "Install opentelemetry-exporter-otlp-proto-grpc."
                    ) from e
                readers.append(
                    PeriodicExportingMetricReader(
                        OTLPMetricExporterGrpc(),
                        export_interval_millis=interval_ms,
                    )
                )
            else:
                try:
                    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                        OTLPMetricExporter as OTLPMetricExporterHttp,
                    )
                except Exception as e:  # pragma: no cover
                    raise OpenTelemetrySinkError(
                        "Missing OTLP HTTP metric exporter. "
                        "Install opentelemetry-exporter-otlp-proto-http."
                    ) from e
                readers.append(
                    PeriodicExportingMetricReader(
                        OTLPMetricExporterHttp(
                            endpoint=self.cfg.otlp_endpoint,
                            headers=self.cfg.otlp_headers,
                        ),
                        export_interval_millis=interval_ms,
                    )
                )
        elif exporter_mode == "jaeger":
            # Jaeger exporter is traces-only. Metrics can still be produced to console or left off.
            pass
        elif exporter_mode == "none":
            pass

        mp = MeterProvider(resource=resource, metric_readers=readers)
        metrics.set_meter_provider(mp)
        self._meter_provider = mp
        self._meter = metrics.get_meter(__name__)

        # Instruments (names are stable and low-cardinality attributes are encouraged)
        self._latency_hist = self._meter.create_histogram(
            "hexdag.node.latency.ms", unit="ms", description="Node execution latency"
        )
        self._error_counter = self._meter.create_counter(
            "hexdag.node.errors", description="Node error count"
        )
        self._tokens_counter = self._meter.create_counter(
            "hexdag.llm.tokens", description="LLM total tokens (prompt+completion)"
        )

    async def stop(self) -> None:
        # Shutdown order: metrics then traces (flush/close)
        if self._meter_provider:
            with suppress(Exception):
                self._meter_provider.shutdown()
        if self._tracer_provider:
            with suppress(Exception):
                self._tracer_provider.shutdown()
        self._meter_provider = None
        self._tracer_provider = None
        self._meter = None
        self._tracer = None

    async def handle(self, event: Any) -> None:
        """
        Map a domain event to OTel traces/metrics.

        Strategy:
          - Create short spans per event (simple, stateless). For full duration accuracy,
            extend to correlate *:started with *:finished/*:failed using a registry.
          - Record metrics for latency/errors/tokens when present.
        """
        data = _event_to_dict(event)
        etype = str(data.get("type") or "")
        attrs = self._extract_attributes(data)
        span_name = self._span_name_for(data)

        tracer = self._tracer
        if tracer is None:
            return

        # Spans
        span = tracer.start_span(span_name, attributes=attrs)
        try:
            # Status by suffix convention
            if etype.endswith(":failed") or etype.endswith(":error"):
                self._set_span_error(span, attrs)
            elif etype.endswith(":finished") or etype.endswith(":succeeded"):
                self._set_span_ok(span)
            # else: neutral status (default)
        finally:
            span.end()

        # Metrics
        self._record_latency(attrs)
        if etype.endswith(":failed") or etype.endswith(":error"):
            self._record_error(attrs)
        self._record_llm_tokens(attrs)

    # ---------- helpers ----------

    def _span_name_for(self, data: dict[str, Any]) -> str:
        etype = str(data.get("type") or "")
        env = data.get("envelope", {}) or {}
        kind, _, _phase = etype.partition(":")
        name = (
            env.get("node")
            or env.get("macro")
            or env.get("tool")
            or env.get("pipeline")
            or "unknown"
        )

        if kind == "node":
            return f"Node/{name}"
        if kind == "macro":
            return f"Macro/{name}"
        if kind == "tool":
            model = env.get("model") or data.get("model") or ""
            adapter = env.get("adapter") or data.get("adapter") or ""
            suffix = f"{adapter}/{model}" if (adapter or model) else name
            return f"Tool/{suffix}"
        if kind == "pipeline":
            return f"Pipeline/{name}"
        return f"Event/{etype or 'unknown'}"

    def _extract_attributes(self, data: dict[str, Any]) -> dict[str, Any]:
        env = data.get("envelope", {}) or {}
        attrs: dict[str, Any] = {
            # identity
            "event.type": data.get("type"),
            "timestamp": data.get("timestamp"),
            "pipeline": env.get("pipeline"),
            "node": env.get("node"),
            "macro": env.get("macro"),
            "tool": env.get("tool"),
            "wave": env.get("wave"),
            # perf / errors
            "latency_ms": env.get("latency_ms") or data.get("latency_ms"),
            "error_message": env.get("error_message") or data.get("error_message"),
            "error_type": env.get("error_type") or data.get("error_type"),
            # LLM specifics
            "model": env.get("model") or data.get("model"),
            "adapter": env.get("adapter") or data.get("adapter"),
            "prompt_tokens": env.get("prompt_tokens") or data.get("prompt_tokens"),
            "completion_tokens": env.get("completion_tokens") or data.get("completion_tokens"),
            "total_tokens": env.get("total_tokens") or data.get("total_tokens"),
            # space for correlation ids if you have them:
            "session_id": env.get("session_id") or data.get("session_id"),
            "pipeline_id": env.get("pipeline_id") or data.get("pipeline_id"),
        }
        # Drop Nones
        return {k: v for k, v in attrs.items() if v is not None}

    def _set_span_ok(self, span: Any) -> None:
        try:
            from opentelemetry.trace import Status, StatusCode
        except Exception:
            return
        span.set_status(Status(StatusCode.OK))

    def _set_span_error(self, span: Any, attrs: dict[str, Any]) -> None:
        try:
            from opentelemetry.trace import Status, StatusCode
        except Exception:
            return
        span.set_status(Status(StatusCode.ERROR))
        msg = attrs.get("error_message")
        if msg:
            with suppress(Exception):
                span.record_exception(Exception(str(msg)))

    def _record_latency(self, attrs: dict[str, Any]) -> None:
        hist = self._latency_hist
        if hist is None:
            return
        lat = attrs.get("latency_ms")
        if isinstance(lat, (int, float)):
            hist.record(float(lat), attributes=self._metric_attrs(attrs))

    def _record_error(self, attrs: dict[str, Any]) -> None:
        c = self._error_counter
        if c is None:
            return
        c.add(1, attributes=self._metric_attrs(attrs))

    def _record_llm_tokens(self, attrs: dict[str, Any]) -> None:
        c = self._tokens_counter
        if c is None:
            return
        total = attrs.get("total_tokens")
        if isinstance(total, (int, float)):
            c.add(int(total), attributes=self._metric_attrs(attrs))

    def _metric_attrs(self, attrs: dict[str, Any]) -> dict[str, Any]:
        return {
            "pipeline": attrs.get("pipeline"),
            "node": attrs.get("node"),
            "tool": attrs.get("tool"),
        }
