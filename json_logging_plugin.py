"""
json_logging_plugin.py
----------------------

Structured, production-grade logging plugin for Google ADK agents.

This module defines:

- A cost estimation helper for supported LLM models.
- A JSON log formatter that emits single-line, ingestion-friendly log records.
- A family of helper functions that create rotating, file-backed loggers
  split by:
      * session vs. invocation
      * main / cost / error / debug channels
- The `JsonLoggingPlugin` class, a `BasePlugin` implementation that hooks into
  ADK lifecycle callbacks and emits:

  * High-signal lifecycle logs to `main.log`
      - user message previews
      - agent/model/tool lifecycle events (before/after)
      - latency metrics (agent/model/tool)
      - per-turn token usage and human-readable cost summaries
  * Structured cost and token logs to `cost.log`
      - per-turn usage
      - per-session summary (on idle)
  * Error logs to `error.log`
      - model failures and other error cases
  * Deep diagnostic payloads to `debug.log` (optional)
      - full user messages, tool args/results, etc., with optional redaction

Directory layout (per agent):

    logs/
      <agent_slug>/
        sessions/
          <session_id>/
            main.log    # high-level structured events & metrics
            cost.log    # per-turn cost + per-session summaries
            error.log   # error events
            debug.log   # detailed payloads (optional)
        invocations/
          <invocation_id>/
            main.log
            cost.log
            error.log
            debug.log

This plugin is designed to be:

- **Production ready**: rotating files, structured JSON, minimal duplication.
- **Ops friendly**: main logs stay compact; debug logs can be toggled or redacted.
- **Token & latency aware**: aggregates usage at the session level and emits
  a summary when the session has been idle for a configured interval.
"""

import json
import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
from time import perf_counter, time
from typing import Optional, Any, Dict, Tuple

from google.genai import types
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.agents import InvocationContext
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.base_tool import BaseTool
from google.adk.events import Event


# -------------------------------------------------------------------
# MODEL COST MAP
# -------------------------------------------------------------------

MODEL_PRICING = {
    # Extend with more models if needed
    "gemini-2.5-flash": {
        "input_per_1k": 0.000075,
        "output_per_1k": 0.00030,
        "currency": "USD",
    }
}


def estimate_cost(
    model_name: Optional[str],
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
) -> Optional[Dict[str, Any]]:
    """
    Estimate monetary cost for a single LLM turn based on token usage.

    Args:
        model_name: Name of the LLM model (must exist in MODEL_PRICING).
        prompt_tokens: Number of input tokens consumed by the model.
        completion_tokens: Number of output tokens produced by the model.

    Returns:
        A dictionary with cost breakdown:
            {
              "model": str,
              "input_tokens": int,
              "output_tokens": int,
              "input_cost": float,
              "output_cost": float,
              "total_cost": float,
              "currency": str,
            }
        or None if the model is not configured in MODEL_PRICING.
    """
    if not model_name or model_name not in MODEL_PRICING:
        return None

    pricing = MODEL_PRICING[model_name]
    pt = prompt_tokens or 0
    ct = completion_tokens or 0

    input_cost = (pt / 1000.0) * pricing["input_per_1k"]
    output_cost = (ct / 1000.0) * pricing["output_per_1k"]

    return {
        "model": model_name,
        "input_tokens": pt,
        "output_tokens": ct,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
        "currency": pricing["currency"],
    }


# -------------------------------------------------------------------
# PRETTY JSON FORMATTER
# -------------------------------------------------------------------

class JsonFormatter(logging.Formatter):
    """
    JSON log formatter that emits single-line, ingestion-friendly records.

    Features:
        - Adds timestamp, severity, logger name, and base message.
        - Merges extra structured fields (if provided under "extra_fields").
        - Preserves primitive and container types as true JSON structures.
        - Uses repr() for non-JSON-serializable objects (no truncation).
    """

    def format(self, record: logging.LogRecord) -> str:
        log = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        extra_fields = record.__dict__.get("extra_fields")
        if isinstance(extra_fields, dict):
            safe: Dict[str, Any] = {}
            for k, v in extra_fields.items():
                # Keep primitive + container types as true JSON
                if isinstance(v, (str, int, float, bool, dict, list)) or v is None:
                    safe[k] = v
                elif isinstance(v, tuple):
                    safe[k] = list(v)
                else:
                    # Full repr for other objects, no truncation
                    safe[k] = repr(v)
            log.update(safe)

        # Single-line JSON: better for log ingestion
        return json.dumps(log, ensure_ascii=False, separators=(",", ":"))


# -------------------------------------------------------------------
# AGENT NAME → MEANINGFUL SLUG
# -------------------------------------------------------------------

def _agent_slug(agent_name: str) -> str:
    """
    Normalize an agent name into a filesystem-safe slug.

    Example:
        "financial_data_collector_agent" → "financial_data_collector"

    Rules:
        - lowercase
        - replace '-' with '_'
        - split on '_' and drop generic suffixes:
          {"agent", "engine", "module", "service", "tool"}
        - join remaining tokens with '_'
        - keep [A-Za-z0-9_] only
        - truncate to max length of 60 characters

    Args:
        agent_name: Raw agent name from ADK.

    Returns:
        Sanitized slug string used in directory and logger names.
    """
    if not agent_name:
        return "agent"

    name = agent_name.lower().replace("-", "_").strip()
    tokens = [t for t in name.split("_") if t]

    ignore = {"agent", "engine", "module", "service", "tool"}
    tokens = [t for t in tokens if t not in ignore]

    if not tokens:
        tokens = [name]

    slug = "_".join(tokens)
    slug = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in slug)
    return slug[:60] if slug else "agent"


# -------------------------------------------------------------------
# SAFE STRING CONVERSION FOR LOGGING (NO TRUNCATION)
# -------------------------------------------------------------------

def _safe_str(value: Any) -> str:
    """
    Convert any value to a robust string representation for logging.

    Behavior:
        - Prefer repr(value) for maximum detail.
        - Fallback to str(value) if repr fails.
        - If both fail, return a sentinel "<unserializable>".

    This is used primarily in debug-level logs where full payloads
    are useful for troubleshooting.
    """
    try:
        return repr(value)
    except Exception:
        try:
            return str(value)
        except Exception:
            return "<unserializable>"


def _preview_str(value: Any, max_len: int = 200) -> str:
    """
    Return a shortened, human-friendly preview of a value.

    Purpose:
        - Keep `main.log` compact by logging only a short preview instead
          of full payloads (which go to `debug.log`).

    Args:
        value: Any object to preview.
        max_len: Maximum character length of the preview string.

    Returns:
        A string truncated to `max_len` characters, with "..." appended
        if truncation occurred.
    """
    s = _safe_str(value)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


# -------------------------------------------------------------------
# LOGGER CREATION HELPERS
# -------------------------------------------------------------------

def _get_rotating_logger(
    logger_name: str,
    base_dir: str,
    file_name: str,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create or return a logger that writes JSON logs to a rotating file.

    The logger is configured with:
        - RotatingFileHandler (5 MB max per file, 5 backups)
        - UTF-8 encoding
        - JsonFormatter
        - No propagation to root logger (avoids duplicate logs)

    Args:
        logger_name: Global logger name in the logging hierarchy.
        base_dir: Directory path under which the file will be created.
        file_name: File name (e.g., 'main.log').
        level: Logging level for this logger.

    Returns:
        A configured `logging.Logger` instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    if not logger.handlers:
        os.makedirs(base_dir, exist_ok=True)
        path = os.path.join(base_dir, file_name)

        handler = RotatingFileHandler(
            path,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=5,
            encoding="utf-8",
        )
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False

    return logger


def get_agent_session_main_logger(
    agent_name: str,
    session_id: str,
    lvl: int = logging.INFO,
) -> logging.Logger:
    """
    Get the per-session main logger for an agent.

    Writes to:
        logs/<agent_slug>/sessions/<session_id>/main.log
    """
    slug = _agent_slug(agent_name)
    base_dir = os.path.join("logs", slug, "sessions", session_id)
    return _get_rotating_logger(
        logger_name=f"{slug}_session_{session_id}_main",
        base_dir=base_dir,
        file_name="main.log",
        level=lvl,
    )


def get_agent_session_cost_logger(
    agent_name: str,
    session_id: str,
    lvl: int = logging.INFO,
) -> logging.Logger:
    """
    Get the per-session cost logger for an agent.

    Writes to:
        logs/<agent_slug>/sessions/<session_id>/cost.log
    """
    slug = _agent_slug(agent_name)
    base_dir = os.path.join("logs", slug, "sessions", session_id)
    return _get_rotating_logger(
        logger_name=f"{slug}_session_{session_id}_cost",
        base_dir=base_dir,
        file_name="cost.log",
        level=lvl,
    )


def get_agent_session_error_logger(
    agent_name: str,
    session_id: str,
    lvl: int = logging.INFO,
) -> logging.Logger:
    """
    Get the per-session error logger for an agent.

    Writes to:
        logs/<agent_slug>/sessions/<session_id>/error.log
    """
    slug = _agent_slug(agent_name)
    base_dir = os.path.join("logs", slug, "sessions", session_id)
    return _get_rotating_logger(
        logger_name=f"{slug}_session_{session_id}_error",
        base_dir=base_dir,
        file_name="error.log",
        level=lvl,
    )


def get_agent_session_debug_logger(
    agent_name: str,
    session_id: str,
    lvl: int = logging.DEBUG,
) -> logging.Logger:
    """
    Get the per-session debug logger for an agent.

    Writes to:
        logs/<agent_slug>/sessions/<session_id>/debug.log
    """
    slug = _agent_slug(agent_name)
    base_dir = os.path.join("logs", slug, "sessions", session_id)
    return _get_rotating_logger(
        logger_name=f"{slug}_session_{session_id}_debug",
        base_dir=base_dir,
        file_name="debug.log",
        level=lvl,
    )


def get_agent_invocation_main_logger(
    agent_name: str,
    invocation_id: str,
    lvl: int = logging.INFO,
) -> logging.Logger:
    """
    Get the per-invocation main logger for an agent.

    Writes to:
        logs/<agent_slug>/invocations/<invocation_id>/main.log
    """
    slug = _agent_slug(agent_name)
    base_dir = os.path.join("logs", slug, "invocations", invocation_id)
    return _get_rotating_logger(
        logger_name=f"{slug}_inv_{invocation_id}_main",
        base_dir=base_dir,
        file_name="main.log",
        level=lvl,
    )


def get_agent_invocation_cost_logger(
    agent_name: str,
    invocation_id: str,
    lvl: int = logging.INFO,
) -> logging.Logger:
    """
    Get the per-invocation cost logger for an agent.

    Writes to:
        logs/<agent_slug>/invocations/<invocation_id>/cost.log
    """
    slug = _agent_slug(agent_name)
    base_dir = os.path.join("logs", slug, "invocations", invocation_id)
    return _get_rotating_logger(
        logger_name=f"{slug}_inv_{invocation_id}_cost",
        base_dir=base_dir,
        file_name="cost.log",
        level=lvl,
    )


def get_agent_invocation_error_logger(
    agent_name: str,
    invocation_id: str,
    lvl: int = logging.INFO,
) -> logging.Logger:
    """
    Get the per-invocation error logger for an agent.

    Writes to:
        logs/<agent_slug>/invocations/<invocation_id>/error.log
    """
    slug = _agent_slug(agent_name)
    base_dir = os.path.join("logs", slug, "invocations", invocation_id)
    return _get_rotating_logger(
        logger_name=f"{slug}_inv_{invocation_id}_error",
        base_dir=base_dir,
        file_name="error.log",
        level=lvl,
    )


def get_agent_invocation_debug_logger(
    agent_name: str,
    invocation_id: str,
    lvl: int = logging.DEBUG,
) -> logging.Logger:
    """
    Get the per-invocation debug logger for an agent.

    Writes to:
        logs/<agent_slug>/invocations/<invocation_id>/debug.log
    """
    slug = _agent_slug(agent_name)
    base_dir = os.path.join("logs", slug, "invocations", invocation_id)
    return _get_rotating_logger(
        logger_name=f"{slug}_inv_{invocation_id}_debug",
        base_dir=base_dir,
        file_name="debug.log",
        level=lvl,
    )


# -------------------------------------------------------------------
# MAIN PLUGIN
# -------------------------------------------------------------------

class JsonLoggingPlugin(BasePlugin):
    """
    ADK plugin that emits structured, multi-channel JSON logs.

    Responsibilities:
        - Attach to ADK lifecycle hooks (user message, agent/model/tool events,
          generic events).
        - Track per-session aggregates (tokens, cost, latency) and emit
          a `session_summary` after idle periods.
        - Split logs into:
            * main.log   → high-level lifecycle + metrics + cost per turn
            * cost.log   → cost and token usage (per-turn + summary)
            * error.log  → model errors and failures
            * debug.log  → full payloads (optional; can be redacted)

    Configuration flags:
        - `level`: base log level for main/cost/error logs.
        - `redact_sensitive`: if True, debug logs store "<redacted>" for
          payload fields that may contain PII or sensitive data.
        - `enable_debug_logs`: if False, debug log files are not written.

    This plugin is intended to be reusable across agents in a production
    environment and safe to use under high throughput.
    """

    def __init__(
        self,
        level: int = logging.INFO,
        redact_sensitive: bool = False,
        enable_debug_logs: bool = True,
    ) -> None:
        """
        Initialize the JSON logging plugin with configurable behavior.

        Args:
            level:
                Logging level for main/cost/error logs (e.g., logging.INFO).
            redact_sensitive:
                If True, any full payloads written to debug logs are replaced
                with a "<redacted>" sentinel, preserving structure but not content.
            enable_debug_logs:
                If False, skips creating and writing to debug.log files entirely.
        """
        super().__init__(name="json_logging")
        self.level = level
        self.redact_sensitive = redact_sensitive
        self.enable_debug_logs = enable_debug_logs

        # Per-session aggregated metrics (keyed by (agent_name, session_id))
        self._session_stats: Dict[Tuple[str, str], Dict[str, Any]] = {}

        # Per-invocation context (richer activity data)
        # Key: (agent_name, session_id, invocation_id)
        self._invocation_meta: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

        # Idle threshold for session summary (seconds)
        self.idle_timeout_seconds: float = 120.0

    # -------------------- INTERNAL HELPERS --------------------

    def _session_invocation_loggers(
        self,
        ctx: InvocationContext,
    ) -> Tuple[logging.Logger, logging.Logger]:
        """
        Resolve the main session and invocation logger pair for a context.

        Args:
            ctx: The current invocation context from ADK.

        Returns:
            A tuple of (session_main_logger, invocation_main_logger).
        """
        session_logger = get_agent_session_main_logger(
            ctx.agent.name, ctx.session.id, self.level
        )
        invocation_logger = get_agent_invocation_main_logger(
            ctx.agent.name, ctx.invocation_id, self.level
        )
        return session_logger, invocation_logger

    def _session_invocation_debug_loggers(
        self,
        ctx: InvocationContext,
    ) -> Tuple[Optional[logging.Logger], Optional[logging.Logger]]:
        """
        Resolve debug loggers for a session + invocation, if enabled.

        Args:
            ctx: The current invocation context from ADK.

        Returns:
            Tuple of (session_debug_logger, invocation_debug_logger), or
            (None, None) if `enable_debug_logs` is False.
        """
        if not self.enable_debug_logs:
            return None, None

        session_debug_logger = get_agent_session_debug_logger(
            ctx.agent.name, ctx.session.id, logging.DEBUG
        )
        invocation_debug_logger = get_agent_invocation_debug_logger(
            ctx.agent.name, ctx.invocation_id, logging.DEBUG
        )
        return session_debug_logger, invocation_debug_logger

    def _resolve_model_name_from_invocation(
        self,
        invocation_context: InvocationContext,
    ) -> Optional[str]:
        """
        Resolve the underlying model name for a given invocation.

        Tries, in order:
            - `agent.model` if it's a bare string
            - `agent.model.model` if it's a Gemini(...) or similar wrapper

        Args:
            invocation_context: The ADK invocation context.

        Returns:
            The model name string if resolvable, otherwise None.
        """
        agent = invocation_context.agent
        if agent is None:
            return None

        model_attr = getattr(agent, "model", None)
        if isinstance(model_attr, str):
            return model_attr

        inner_model = getattr(model_attr, "model", None)
        if isinstance(inner_model, str):
            return inner_model

        return None

    def _get_session_key(self, agent_name: str, session_id: str) -> Tuple[str, str]:
        """
        Construct a consistent dict key for session-level stats tracking."""
        return (agent_name, session_id)

    def _get_or_create_session_stats(
        self,
        agent_name: str,
        session_id: str,
    ) -> Dict[str, Any]:
        """
        Retrieve or initialize the aggregated metrics structure for a session.

        Metrics captured include:
            - token totals (prompt, completion, total)
            - total cost (est. using MODEL_PRICING)
            - counts of model/tool/agent calls
            - latency aggregates (sum/ max)
            - last_event_ts (for idle-based summary emission)
        """
        key = self._get_session_key(agent_name, session_id)
        stats = self._session_stats.get(key)
        if stats is None:
            stats = {
                "agent_name": agent_name,
                "session_id": session_id,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "currency": None,
                "total_turns": 0,
                "total_model_calls": 0,
                "total_tool_calls": 0,
                "total_agent_calls": 0,
                "sum_model_latency_ms": 0.0,
                "max_model_latency_ms": 0.0,
                "sum_agent_latency_ms": 0.0,
                "sum_tool_latency_ms": 0.0,
                "max_tool_latency_ms": 0.0,
                "last_event_ts": None,
            }
            self._session_stats[key] = stats
        return stats

    def _reset_session_aggregates(self, stats: Dict[str, Any]) -> None:
        """
        Reset session aggregate counters while preserving identity fields.

        Called after a `session_summary` is logged due to idle timeout,
        so subsequent turns start a fresh counting window.
        """
        stats["total_prompt_tokens"] = 0
        stats["total_completion_tokens"] = 0
        stats["total_tokens"] = 0
        stats["total_cost"] = 0.0
        stats["currency"] = None
        stats["total_turns"] = 0
        stats["total_model_calls"] = 0
        stats["total_tool_calls"] = 0
        stats["total_agent_calls"] = 0
        stats["sum_model_latency_ms"] = 0.0
        stats["max_model_latency_ms"] = 0.0
        stats["sum_agent_latency_ms"] = 0.0
        stats["sum_tool_latency_ms"] = 0.0
        stats["max_tool_latency_ms"] = 0.0
        # last_event_ts is set in _touch_session_stats

    def _log_session_summary(self, stats: Dict[str, Any]) -> None:
        """
        Emit a session-level summary event to main.log and cost.log.

        Includes:
            - token totals
            - estimated total cost
            - call counts (model/agent/tool)
            - latency averages and maxima per component
        """
        agent_name = stats["agent_name"]
        session_id = stats["session_id"]

        session_logger = get_agent_session_main_logger(
            agent_name,
            session_id,
            self.level,
        )
        session_cost_logger = get_agent_session_cost_logger(
            agent_name,
            session_id,
            self.level,
        )

        total_model_calls = stats["total_model_calls"]
        total_agent_calls = stats["total_agent_calls"]
        total_tool_calls = stats["total_tool_calls"]

        avg_model_latency_ms = (
            stats["sum_model_latency_ms"] / total_model_calls
            if total_model_calls > 0
            else None
        )
        avg_agent_latency_ms = (
            stats["sum_agent_latency_ms"] / total_agent_calls
            if total_agent_calls > 0
            else None
        )
        avg_tool_latency_ms = (
            stats["sum_tool_latency_ms"] / total_tool_calls
            if total_tool_calls > 0
            else None
        )

        fields = {
            "event": "session_summary",
            "agent_name": agent_name,
            "session_id": session_id,
            "total_turns": stats["total_turns"],
            "total_prompt_tokens": stats["total_prompt_tokens"],
            "total_completion_tokens": stats["total_completion_tokens"],
            "total_tokens": stats["total_tokens"],
            "total_cost": stats["total_cost"],
            "cost_currency": stats["currency"],
            "total_model_calls": total_model_calls,
            "total_agent_calls": total_agent_calls,
            "total_tool_calls": total_tool_calls,
            "avg_model_latency_ms": avg_model_latency_ms,
            "max_model_latency_ms": stats["max_model_latency_ms"] or None,
            "avg_agent_latency_ms": avg_agent_latency_ms,
            "avg_tool_latency_ms": avg_tool_latency_ms,
            "max_tool_latency_ms": stats["max_tool_latency_ms"] or None,
        }

        session_logger.info("session_summary", extra={"extra_fields": fields})
        session_cost_logger.info("session_summary", extra={"extra_fields": fields})

    def _touch_session_stats(
        self,
        agent_name: str,
        session_id: str,
    ) -> Dict[str, Any]:
        """
        Update session activity timestamp and emit summary on idle.

        Logic:
            - If `last_event_ts` exists and the time difference from now is
              >= `idle_timeout_seconds` AND at least one turn has occurred:
                → Emit a `session_summary` event and reset aggregates.

            - Always update `last_event_ts` to current time.

        Note:
            This is a lazy strategy: summary is generated upon the first
            new event after a sufficiently long idle period (no background
            scheduler).
        """
        stats = self._get_or_create_session_stats(agent_name, session_id)
        now_ts = time()

        last_ts = stats.get("last_event_ts")
        if (
            last_ts is not None
            and (now_ts - last_ts) >= self.idle_timeout_seconds
            and stats["total_turns"] > 0
        ):
            # Idle for >= threshold → emit summary for previous segment
            self._log_session_summary(stats)
            # Reset aggregates for next segment
            self._reset_session_aggregates(stats)

        stats["last_event_ts"] = now_ts
        return stats

    def _get_invocation_meta(
        self,
        agent_name: str,
        session_id: str,
        invocation_id: str,
    ) -> Dict[str, Any]:
        """
        Retrieve or initialize per-invocation metadata used for context.

        This metadata is used to enrich event logs with:
            - last user message
            - last tool name + result preview
            - last model name

        The data can be cleaned up when `end_of_agent` is signaled.
        """
        key = (agent_name, session_id, invocation_id)
        meta = self._invocation_meta.get(key)
        if meta is None:
            meta = {}
            self._invocation_meta[key] = meta
        return meta

    def _maybe_redact(self, value: Any) -> str:
        """
        Return either a redacted sentinel or the full stringified payload.

        Used for debug logs which may contain raw user messages or tool
        inputs/outputs.

        Behavior:
            - If `self.redact_sensitive` is True → always return "<redacted>".
            - Otherwise → return `_safe_str(value)`.
        """
        if self.redact_sensitive:
            return "<redacted>"
        return _safe_str(value)

    # ---------------------------------------------------------
    # USER MESSAGE
    # ---------------------------------------------------------

    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content,
    ) -> Optional[types.Content]:
        """
        Lifecycle hook for incoming user messages.

        Behavior:
            - Updates idle-based session stats.
            - Stores the raw user message in invocation metadata.
            - Writes a compact preview to `main.log`.
            - Writes the full (or redacted) payload to `debug.log`
              when debug logging is enabled.

        Args:
            invocation_context: ADK invocation context containing session & agent info.
            user_message: The user `Content` object received by the agent.

        Returns:
            Always returns None (no message modification).
        """
        agent_name = invocation_context.agent.name
        session_id = invocation_context.session.id
        invocation_id = invocation_context.invocation_id

        # Touch stats (idle detection)
        self._touch_session_stats(agent_name, session_id)

        # Update invocation meta (for activity context)
        meta = self._get_invocation_meta(agent_name, session_id, invocation_id)
        meta["last_user_message_raw"] = _safe_str(user_message)

        session_logger = get_agent_session_main_logger(
            agent_name,
            session_id,
            self.level,
        )
        invocation_logger = get_agent_invocation_main_logger(
            agent_name,
            invocation_id,
            self.level,
        )

        # MAIN LOG: only preview
        fields_main = {
            "event": "user_message",
            "user_id": invocation_context.user_id,
            "session_id": session_id,
            "invocation_id": invocation_id,
            "app_name": invocation_context.app_name,
            "agent_name": agent_name,
            "user_message_preview": _preview_str(user_message),
        }

        session_logger.info("user_message", extra={"extra_fields": fields_main})
        invocation_logger.info("user_message", extra={"extra_fields": fields_main})

        # DEBUG LOG: full payload (maybe redacted)
        session_debug_logger, invocation_debug_logger = (
            self._session_invocation_debug_loggers(invocation_context)
        )
        if session_debug_logger and invocation_debug_logger:
            fields_debug = {
                "event": "user_message_debug",
                "user_id": invocation_context.user_id,
                "session_id": session_id,
                "invocation_id": invocation_id,
                "app_name": invocation_context.app_name,
                "agent_name": agent_name,
                "user_message_raw": self._maybe_redact(user_message),
            }
            session_debug_logger.debug(
                "user_message_debug", extra={"extra_fields": fields_debug}
            )
            invocation_debug_logger.debug(
                "user_message_debug", extra={"extra_fields": fields_debug}
            )

        return None

    # ---------------------------------------------------------
    # BEFORE AGENT
    # ---------------------------------------------------------

    async def before_agent_callback(
        self,
        *,
        agent: BaseAgent,
        callback_context: CallbackContext,
    ) -> None:
        """
        Lifecycle hook fired immediately before an agent handles a request.

        Behavior:
            - Touches session stats for idle detection.
            - Increments `total_agent_calls`.
            - Starts a high-resolution timer for agent latency.
            - Logs metadata (agent name, description) to `main.log`.
        """
        ctx = callback_context._invocation_context

        # Touch stats (idle detection)
        stats = self._touch_session_stats(ctx.agent.name, ctx.session.id)
        # Count agent calls
        stats["total_agent_calls"] += 1

        session_logger, invocation_logger = self._session_invocation_loggers(ctx)

        # Agent-level latency start
        callback_context.state["agent_start_ts"] = perf_counter()

        fields = {
            "event": "before_agent",
            "user_id": ctx.user_id,
            "session_id": ctx.session.id,
            "invocation_id": ctx.invocation_id,
            "agent_name": agent.name,
            "agent_description": getattr(agent, "description", None),
        }

        session_logger.info("before_agent", extra={"extra_fields": fields})
        invocation_logger.info("before_agent", extra={"extra_fields": fields})

    # ---------------------------------------------------------
    # AFTER AGENT
    # ---------------------------------------------------------

    async def after_agent_callback(
        self,
        *,
        agent: BaseAgent,
        callback_context: CallbackContext,
    ) -> None:
        """
        Lifecycle hook fired immediately after an agent finishes a request.

        Behavior:
            - Computes agent-level latency using stored start time.
            - Aggregates latency into per-session stats.
            - Logs latency and context to `main.log`.
        """
        ctx = callback_context._invocation_context

        # Touch stats (idle detection)
        stats = self._touch_session_stats(ctx.agent.name, ctx.session.id)

        session_logger, invocation_logger = self._session_invocation_loggers(ctx)

        start_ts = callback_context.state.get("agent_start_ts")
        agent_latency_ms = None
        if isinstance(start_ts, (int, float)):
            agent_latency_ms = (perf_counter() - start_ts) * 1000.0

        # Aggregate agent latency
        if agent_latency_ms is not None:
            stats["sum_agent_latency_ms"] += agent_latency_ms

        fields = {
            "event": "after_agent",
            "user_id": ctx.user_id,
            "session_id": ctx.session.id,
            "invocation_id": ctx.invocation_id,
            "agent_name": agent.name,
            "agent_latency_ms": agent_latency_ms,
        }

        session_logger.info("after_agent", extra={"extra_fields": fields})
        invocation_logger.info("after_agent", extra={"extra_fields": fields})

    # ---------------------------------------------------------
    # BEFORE MODEL
    # ---------------------------------------------------------

    async def before_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> None:
        """
        Lifecycle hook fired before a direct LLM/model call.

        Behavior:
            - Touches session stats for idle detection.
            - Starts a timer for model latency.
            - Stores the model name in callback state for later use.
            - Logs model configuration (temperature, top_p, etc.) to `main.log`.
        """
        ctx = callback_context._invocation_context

        # Touch stats (idle detection)
        self._touch_session_stats(ctx.agent.name, ctx.session.id)

        session_logger, invocation_logger = self._session_invocation_loggers(ctx)

        # Track model latency start
        callback_context.state["model_start_ts"] = perf_counter()

        # Keep the model in state for debugging if needed
        callback_context.state["model_name"] = llm_request.model
        config = getattr(llm_request, "config", None)

        fields = {
            "event": "before_model",
            "user_id": ctx.user_id,
            "session_id": ctx.session.id,
            "invocation_id": ctx.invocation_id,
            "agent_name": ctx.agent.name,
            "model": llm_request.model,
            "temperature": getattr(config, "temperature", None) if config else None,
            "top_p": getattr(config, "top_p", None) if config else None,
        }

        session_logger.info("before_model", extra={"extra_fields": fields})
        invocation_logger.info("before_model", extra={"extra_fields": fields})

    # ---------------------------------------------------------
    # AFTER MODEL  (NO COST HERE)
    # ---------------------------------------------------------

    async def after_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> None:
        """
        Lifecycle hook fired after a direct LLM/model call completes.

        Notes:
            - This method does NOT handle cost; cost is computed centrally
              in `on_event_callback` using turn-level usage metadata.

        Behavior:
            - Updates per-session model call count.
            - Computes model latency and aggregates its stats.
            - Reads raw usage tokens (prompt/completion) if available.
            - Stores last model name in invocation metadata.
            - Logs latency and token usage to `main.log`.
        """
        ctx = callback_context._invocation_context

        # Touch stats (idle detection) and get stats
        stats = self._touch_session_stats(ctx.agent.name, ctx.session.id)
        # Count model call
        stats["total_model_calls"] += 1

        session_logger, invocation_logger = self._session_invocation_loggers(ctx)

        # Model latency
        start_ts = callback_context.state.get("model_start_ts")
        model_latency_ms = None
        if isinstance(start_ts, (int, float)):
            model_latency_ms = (perf_counter() - start_ts) * 1000.0

        # Aggregate model latency
        if model_latency_ms is not None:
            stats["sum_model_latency_ms"] += model_latency_ms
            stats["max_model_latency_ms"] = max(
                stats["max_model_latency_ms"], model_latency_ms
            )

        usage: Optional[types.GenerateContentResponseUsageMetadata] = getattr(
            llm_response, "usage_metadata", None
        )
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None

        model_name = callback_context.state.get("model_name")

        # Update invocation meta with last model used
        meta = self._get_invocation_meta(
            ctx.agent.name, ctx.session.id, ctx.invocation_id
        )
        meta["last_model_name"] = model_name

        fields = {
            "event": "after_model",
            "user_id": ctx.user_id,
            "session_id": ctx.session.id,
            "invocation_id": ctx.invocation_id,
            "agent_name": ctx.agent.name,
            "model": model_name,
            "prompt_tokens_raw": prompt_tokens,
            "completion_tokens_raw": completion_tokens,
            "model_latency_ms": model_latency_ms,
            "error_code": getattr(llm_response, "error_code", None),
            "error_message": getattr(llm_response, "error_message", None),
            "interrupted": getattr(llm_response, "interrupted", None),
        }

        session_logger.info("after_model", extra={"extra_fields": fields})
        invocation_logger.info("after_model", extra={"extra_fields": fields})

    # ---------------------------------------------------------
    # MODEL ERROR
    # ---------------------------------------------------------

    async def on_model_error_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
        error: Exception,
    ) -> Optional[LlmResponse]:
        """
        Lifecycle hook fired when a model call raises an exception.

        Behavior:
            - Touches session stats for idle detection.
            - Logs the error (type + message + model name) to:
                * session main.log
                * invocation main.log
                * session error.log
                * invocation error.log
        """
        ctx = callback_context._invocation_context

        # Touch stats (idle detection)
        self._touch_session_stats(ctx.agent.name, ctx.session.id)

        session_logger, invocation_logger = self._session_invocation_loggers(ctx)

        fields = {
            "event": "model_error",
            "user_id": ctx.user_id,
            "session_id": ctx.session.id,
            "invocation_id": ctx.invocation_id,
            "agent_name": ctx.agent.name,
            "model": llm_request.model,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

        # main logs
        session_logger.error("model_error", extra={"extra_fields": fields})
        invocation_logger.error("model_error", extra={"extra_fields": fields})

        # error logs
        session_error_logger = get_agent_session_error_logger(
            ctx.agent.name, ctx.session.id, self.level
        )
        invocation_error_logger = get_agent_invocation_error_logger(
            ctx.agent.name, ctx.invocation_id, self.level
        )
        session_error_logger.error("model_error", extra={"extra_fields": fields})
        invocation_error_logger.error("model_error", extra={"extra_fields": fields})

        return None

    # ---------------------------------------------------------
    # TOOL CALLS
    # ---------------------------------------------------------

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_context: ToolContext,
        tool_args: Dict[str, Any],
    ) -> None:
        """
        Lifecycle hook fired immediately before a tool is invoked.

        Behavior:
            - Touches session stats for idle detection.
            - Starts a timer for tool latency.
            - Logs a compact view (tool name + arg keys) to `main.log`.
            - Logs full tool arguments to `debug.log` (redacted if configured).
        """
        ctx = tool_context._invocation_context

        # Touch stats (idle detection)
        self._touch_session_stats(ctx.agent.name, ctx.session.id)

        session_logger, invocation_logger = self._session_invocation_loggers(ctx)
        session_debug_logger, invocation_debug_logger = (
            self._session_invocation_debug_loggers(ctx)
        )

        # Tool latency start (use tool_context.state if available)
        state = getattr(tool_context, "state", None)
        if isinstance(state, dict):
            state["tool_start_ts"] = perf_counter()

        # MAIN LOG: only keys / preview
        fields_main = {
            "event": "before_tool",
            "user_id": ctx.user_id,
            "session_id": ctx.session.id,
            "invocation_id": ctx.invocation_id,
            "agent_name": ctx.agent.name,
            "tool_name": tool.name,
            "tool_arg_keys": list(tool_args.keys()),
        }

        session_logger.info("before_tool", extra={"extra_fields": fields_main})
        invocation_logger.info("before_tool", extra={"extra_fields": fields_main})

        # DEBUG LOG: full args
        if session_debug_logger and invocation_debug_logger:
            fields_debug = {
                "event": "before_tool_debug",
                "user_id": ctx.user_id,
                "session_id": ctx.session.id,
                "invocation_id": ctx.invocation_id,
                "agent_name": ctx.agent.name,
                "tool_name": tool.name,
                "tool_args_raw": self._maybe_redact(tool_args),
            }
            session_debug_logger.debug(
                "before_tool_debug", extra={"extra_fields": fields_debug}
            )
            invocation_debug_logger.debug(
                "before_tool_debug", extra={"extra_fields": fields_debug}
            )

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: Dict[str, Any],
        tool_context: ToolContext,
        result: dict,
    ) -> None:
        """
        Lifecycle hook fired immediately after a tool returns a result.

        Behavior:
            - Updates per-session tool call count.
            - Computes tool latency and aggregates per-session stats.
            - Stores last tool name + result preview in invocation metadata.
            - Logs a summary line (latency + short result preview) to `main.log`.
            - Logs full result payload to `debug.log` (redacted if configured).
        """
        ctx = tool_context._invocation_context

        # Touch stats (idle detection) and get stats
        stats = self._touch_session_stats(ctx.agent.name, ctx.session.id)
        # Count tool calls
        stats["total_tool_calls"] += 1

        session_logger, invocation_logger = self._session_invocation_loggers(ctx)
        session_debug_logger, invocation_debug_logger = (
            self._session_invocation_debug_loggers(ctx)
        )

        # Tool latency
        tool_latency_ms = None
        state = getattr(tool_context, "state", None)
        if isinstance(state, dict):
            start_ts = state.get("tool_start_ts")
            if isinstance(start_ts, (int, float)):
                tool_latency_ms = (perf_counter() - start_ts) * 1000.0

        # Aggregate tool latency
        if tool_latency_ms is not None:
            stats["sum_tool_latency_ms"] += tool_latency_ms
            stats["max_tool_latency_ms"] = max(
                stats["max_tool_latency_ms"], tool_latency_ms
            )

        # Update invocation meta with last tool info
        meta = self._get_invocation_meta(
            ctx.agent.name, ctx.session.id, ctx.invocation_id
        )
        meta["last_tool_name"] = tool.name
        meta["last_tool_result_preview"] = _preview_str(result)

        # MAIN LOG: summary
        fields_main = {
            "event": "after_tool",
            "user_id": ctx.user_id,
            "session_id": ctx.session.id,
            "invocation_id": ctx.invocation_id,
            "agent_name": ctx.agent.name,
            "tool_name": tool.name,
            "tool_latency_ms": tool_latency_ms,
            "tool_result_preview": _preview_str(result),
        }

        session_logger.info("after_tool", extra={"extra_fields": fields_main})
        invocation_logger.info("after_tool", extra={"extra_fields": fields_main})

        # DEBUG LOG: full result
        if session_debug_logger and invocation_debug_logger:
            fields_debug = {
                "event": "after_tool_debug",
                "user_id": ctx.user_id,
                "session_id": ctx.session.id,
                "invocation_id": ctx.invocation_id,
                "agent_name": ctx.agent.name,
                "tool_name": tool.name,
                "tool_latency_ms": tool_latency_ms,
                "tool_result_raw": self._maybe_redact(result),
            }
            session_debug_logger.debug(
                "after_tool_debug", extra={"extra_fields": fields_debug}
            )
            invocation_debug_logger.debug(
                "after_tool_debug", extra={"extra_fields": fields_debug}
            )

    # ---------------------------------------------------------
    # GENERIC EVENT HOOK (TURN-LEVEL TOKEN USAGE + COST + ACTIONS)
    # ---------------------------------------------------------

    async def on_event_callback(
        self,
        *,
        invocation_context: InvocationContext,
        event: Event,
    ) -> Optional[Event]:
        """
        Generic event hook used for turn-level token usage and cost logging.

        Behavior:
            - Extracts usage metadata (prompt, candidate, total tokens).
            - Updates per-session aggregates (tokens, total_turns).
            - Estimates cost using `estimate_cost` and the resolved model name.
            - Collects a rich `activity` block including:
                * event name
                * last tool + result preview
                * last user message preview
                * last model name
                * actions metadata (transfers, escalations, compaction, etc.)
            - Writes:
                * human-readable one-line cost summary to `main.log`
                * structured token/cost record to:
                    - session main.log
                    - invocation main.log
                    - session cost.log
                    - invocation cost.log

        Returns:
            Always returns None (no modification to the underlying Event).
        """
        agent_name = invocation_context.agent.name
        session_id = invocation_context.session.id
        invocation_id = invocation_context.invocation_id

        # -------------------------------
        # SESSION STATE & LOGGERS
        # -------------------------------
        stats = self._touch_session_stats(agent_name, session_id)
        session_logger, invocation_logger = self._session_invocation_loggers(
            invocation_context
        )

        # -------------------------------
        # TOKEN EXTRACTION
        # -------------------------------
        usage = getattr(event, "usage_metadata", None)

        prompt_count = getattr(usage, "prompt_token_count", 0) or 0
        candidate_count = getattr(usage, "candidates_token_count", 0) or 0
        total_count = getattr(usage, "total_token_count", 0) or 0

        if total_count == 0 and prompt_count == 0 and candidate_count == 0:
            return None  # Skip events without cost relevance

        # -------------------------------
        # PER-SESSION ACCUMULATION
        # -------------------------------
        stats["total_prompt_tokens"] += prompt_count
        stats["total_completion_tokens"] += candidate_count
        stats["total_tokens"] += total_count
        stats["total_turns"] += 1

        # -------------------------------
        # COST CALCULATION
        # -------------------------------
        model_name = self._resolve_model_name_from_invocation(invocation_context)
        cost = estimate_cost(
            model_name=model_name,
            prompt_tokens=prompt_count,
            completion_tokens=candidate_count,
        )

        if cost:
            stats["total_cost"] += cost["total_cost"]
            stats["currency"] = cost["currency"]

        # -------------------------------
        # RESOLVE EVENT / ACTIVITY NAME
        # -------------------------------
        event_name = (
            getattr(event, "event_type", None)
            or getattr(event, "event_name", None)
            or getattr(event, "type", None)
            or getattr(event, "name", None)
            or "unknown_event"
        )

        # -------------------------------
        # INVOCATION META (context you already track)
        # -------------------------------
        meta = self._invocation_meta.get(
            (agent_name, session_id, invocation_id), {}
        )

        # -------------------------------
        # ACTIONS META FROM EventActions
        # -------------------------------
        actions = getattr(event, "actions", None)
        actions_meta = None
        if actions is not None:
            # Build a compact, queryable snapshot of workflow actions
            actions_meta = {
                "skip_summarization": actions.skip_summarization,
                "transfer_to_agent": actions.transfer_to_agent,
                "escalate": actions.escalate,
                "end_of_agent": actions.end_of_agent,
                "rewind_before_invocation_id": actions.rewind_before_invocation_id,
                # lightweight sizes to avoid dumping giant blobs
                "state_delta_keys": list(actions.state_delta.keys())
                if actions.state_delta
                else [],
                "artifact_delta": actions.artifact_delta or {},
                "requested_auth_count": len(actions.requested_auth_configs)
                if actions.requested_auth_configs
                else 0,
                "requested_tool_confirmations_count": len(
                    actions.requested_tool_confirmations
                )
                if actions.requested_tool_confirmations
                else 0,
                "has_compaction": actions.compaction is not None,
                # basic compaction metadata:
                "compaction_start_ts": actions.compaction.start_timestamp
                if actions.compaction
                else None,
                "compaction_end_ts": actions.compaction.end_timestamp
                if actions.compaction
                else None,
            }

        # If end_of_agent, clean up invocation meta to avoid unbounded growth
        if actions_meta and actions_meta.get("end_of_agent"):
            self._invocation_meta.pop((agent_name, session_id, invocation_id), None)

        # -------------------------------
        # STRUCTURED activity block
        # -------------------------------
        activity_block = {
            "event_name": event_name,
            "agent_name": agent_name,
            "session_id": session_id,
            "invocation_id": invocation_id,
            "last_tool_name": meta.get("last_tool_name"),
            "last_tool_result_preview": meta.get("last_tool_result_preview"),
            "last_user_message": meta.get("last_user_message_raw"),
            "last_model_name": meta.get("last_model_name"),
            "actions": actions_meta,
        }

        # -------------------------------
        # Human readable form
        # -------------------------------
        # Add a small suffix to highlight special actions
        action_tags = []
        if actions_meta:
            if actions_meta.get("transfer_to_agent"):
                action_tags.append(f"transfer_to_agent={actions_meta['transfer_to_agent']}")
            if actions_meta.get("escalate"):
                action_tags.append("escalate=True")
            if actions_meta.get("end_of_agent"):
                action_tags.append("end_of_agent=True")
            if actions_meta.get("rewind_before_invocation_id"):
                action_tags.append(
                    f"rewind_to={actions_meta['rewind_before_invocation_id']}"
                )
            if actions_meta.get("has_compaction"):
                action_tags.append("compacted=True")

        actions_suffix = f" | actions: {', '.join(action_tags)}" if action_tags else ""
        human_activity = (
            f"Activity '{event_name}' for agent={agent_name}, "
            f"session={session_id}, invocation={invocation_id}{actions_suffix}"
        )

        # -------------------------------
        # HUMAN READABLE SUMMARY LOG
        # -------------------------------
        if cost:
            cost_str = f"{cost['total_cost']:.8f} {cost['currency']}"
        else:
            cost_str = "N/A"

        session_logger.info(
            (
                "Turn cost | %s | model=%s | tokens=%s (prompt=%s, candidates=%s) "
                "| cost=%s"
            ),
            human_activity,
            model_name or "unknown_model",
            total_count,
            prompt_count,
            candidate_count,
            cost_str,
        )

        # -------------------------------
        # STRUCTURED JSON LOG FIELDS
        # -------------------------------
        fields = {
            "event": "turn_token_usage",
            "usage_scope": "turn",

            # structured activity meta
            "activity": activity_block,
            "human_readable_activity": human_activity,

            # token info
            "session_id": session_id,
            "invocation_id": invocation_id,
            "agent_name": agent_name,
            "model": model_name,
            "prompt_tokens_turn": prompt_count,
            "candidate_tokens_turn": candidate_count,
            "total_tokens_turn": total_count,
        }

        # cost fields
        if cost:
            fields.update(
                {
                    "cost_input_tokens": cost["input_tokens"],
                    "cost_output_tokens": cost["output_tokens"],
                    "cost_input": cost["input_cost"],
                    "cost_output": cost["output_cost"],
                    "cost_total": cost["total_cost"],
                    "cost_currency": cost["currency"],
                }
            )

        # -------------------------------
        # WRITE STRUCTURED LOGS
        # -------------------------------
        session_logger.info("turn_token_usage", extra={"extra_fields": fields})
        invocation_logger.info("turn_token_usage", extra={"extra_fields": fields})

        # -------------------------------
        # COST LOG SEPARATE FILE
        # -------------------------------
        session_cost_logger = get_agent_session_cost_logger(agent_name, session_id)
        invocation_cost_logger = get_agent_invocation_cost_logger(
            agent_name, invocation_id
        )

        session_cost_logger.info("turn_cost", extra={"extra_fields": fields})
        invocation_cost_logger.info("turn_cost", extra={"extra_fields": fields})

        return None