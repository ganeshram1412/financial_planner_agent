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
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.base_agent import BaseAgent
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
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
    Estimate cost for a single logical LLM turn.

    Returns None if the model is not configured in MODEL_PRICING.
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
    """Pretty JSON log formatter with full, non-truncated content."""

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
                # Keep primitive + container types as true JSON, no repr()
                if isinstance(v, (str, int, float, bool, dict, list)) or v is None:
                    safe[k] = v
                elif isinstance(v, tuple):
                    safe[k] = list(v)
                else:
                    # Full repr for other objects, no truncation
                    safe[k] = repr(v)
            log.update(safe)

        return json.dumps(log, ensure_ascii=False, indent=2)


# -------------------------------------------------------------------
# AGENT NAME → MEANINGFUL SLUG
# -------------------------------------------------------------------

def _agent_slug(agent_name: str) -> str:
    """
    Turn 'financial_data_collector_agent' into 'financial_data_collector'.
    Rules:
      - lowercase
      - split on '_' and '-'
      - drop generic suffixes: agent, engine, module, service, tool
      - join remaining tokens with '_'
      - filesystem-safe, length-limited
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
    Convert any value to a log-safe string using repr/str.
    No truncation, just best-effort stringification.
    """
    try:
        return repr(value)
    except Exception:
        try:
            return str(value)
        except Exception:
            return "<unserializable>"


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
    Create or return a RotatingFileHandler-based logger
    writing JSON lines to base_dir/file_name.
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
    slug = _agent_slug(agent_name)
    base_dir = os.path.join("logs", slug, "sessions", session_id)
    return _get_rotating_logger(
        logger_name=f"{slug}_session_{session_id}_error",
        base_dir=base_dir,
        file_name="error.log",
        level=lvl,
    )


def get_agent_invocation_main_logger(
    agent_name: str,
    invocation_id: str,
    lvl: int = logging.INFO,
) -> logging.Logger:
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
    slug = _agent_slug(agent_name)
    base_dir = os.path.join("logs", slug, "invocations", invocation_id)
    return _get_rotating_logger(
        logger_name=f"{slug}_inv_{invocation_id}_error",
        base_dir=base_dir,
        file_name="error.log",
        level=lvl,
    )


# -------------------------------------------------------------------
# MAIN PLUGIN
# -------------------------------------------------------------------

class JsonLoggingPlugin(BasePlugin):
    """
    ADK Plugin that logs key lifecycle events as structured JSON.

    Folder layout:

      logs/
        <agent_slug>/               # e.g. financial_planner_orchestrator
          sessions/
            <session_id>/
              main.log   # lifecycle + context + usage summary + latency
              cost.log   # cost/usage per turn + per-session summary
              error.log  # errors
          invocations/
            <invocation_id>/
              main.log
              cost.log
              error.log
    """

    def __init__(self, level: int = logging.INFO) -> None:
        super().__init__(name="json_logging")
        self.level = level

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
        session_logger = get_agent_session_main_logger(
            ctx.agent.name, ctx.session.id, self.level
        )
        invocation_logger = get_agent_invocation_main_logger(
            ctx.agent.name, ctx.invocation_id, self.level
        )
        return session_logger, invocation_logger

    def _resolve_model_name_from_invocation(
        self,
        invocation_context: InvocationContext,
    ) -> Optional[str]:
        """
        Best-effort resolution of model name for cost calculation.

        Tries:
          - agent.model if it's a string
          - agent.model.model if it's a Gemini(...) or similar wrapper
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
        return (agent_name, session_id)

    def _get_or_create_session_stats(
        self,
        agent_name: str,
        session_id: str,
    ) -> Dict[str, Any]:
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
        Reset per-session aggregate counters while preserving identity.
        Called after a session_summary is logged.
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
        Emit a session-level aggregated summary log.
        Writes to:
          - session main.log
          - session cost.log
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
        Get stats object, update last_event_ts, and emit + reset session_summary
        if the session has been idle longer than idle_timeout_seconds.

        NOTE: Summary is emitted on the *next* event after the idle
        period, not exactly at the 2-minute mark (no background scheduler).
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
        Get or create the per-invocation context map.
        Used to enrich activity context (e.g. last tool, last user message).
        """
        key = (agent_name, session_id, invocation_id)
        meta = self._invocation_meta.get(key)
        if meta is None:
            meta = {}
            self._invocation_meta[key] = meta
        return meta

    # ---------------------------------------------------------
    # USER MESSAGE
    # ---------------------------------------------------------

    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content,
    ) -> Optional[types.Content]:
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

        fields = {
            "event": "user_message",
            "user_id": invocation_context.user_id,
            "session_id": session_id,
            "invocation_id": invocation_id,
            "app_name": invocation_context.app_name,
            "agent_name": agent_name,
            "user_message_raw": _safe_str(user_message),
        }

        session_logger.info("user_message", extra={"extra_fields": fields})
        invocation_logger.info("user_message", extra={"extra_fields": fields})
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
        Logs model call outcome (errors/interruption, raw usage if available).
        Cost is handled centrally in on_event_callback using turn-level usage.
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
        ctx = tool_context._invocation_context

        # Touch stats (idle detection)
        self._touch_session_stats(ctx.agent.name, ctx.session.id)

        session_logger, invocation_logger = self._session_invocation_loggers(ctx)

        # Tool latency start (use tool_context.state if available)
        state = getattr(tool_context, "state", None)
        if isinstance(state, dict):
            state["tool_start_ts"] = perf_counter()

        fields = {
            "event": "before_tool",
            "user_id": ctx.user_id,
            "session_id": ctx.session.id,
            "invocation_id": ctx.invocation_id,
            "agent_name": ctx.agent.name,
            "tool_name": tool.name,
            "tool_args_raw": _safe_str(tool_args),
        }

        session_logger.info("before_tool", extra={"extra_fields": fields})
        invocation_logger.info("before_tool", extra={"extra_fields": fields})

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: Dict[str, Any],
        tool_context: ToolContext,
        result: dict,
    ) -> None:
        ctx = tool_context._invocation_context

        # Touch stats (idle detection) and get stats
        stats = self._touch_session_stats(ctx.agent.name, ctx.session.id)
        # Count tool calls
        stats["total_tool_calls"] += 1

        session_logger, invocation_logger = self._session_invocation_loggers(ctx)

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
        meta["last_tool_result_preview"] = _safe_str(result)

        fields = {
            "event": "after_tool",
            "user_id": ctx.user_id,
            "session_id": ctx.session.id,
            "invocation_id": ctx.invocation_id,
            "agent_name": ctx.agent.name,
            "tool_name": tool.name,
            "tool_latency_ms": tool_latency_ms,
            "tool_result_raw": _safe_str(result),
        }

        session_logger.info("after_tool", extra={"extra_fields": fields})
        invocation_logger.info("after_tool", extra={"extra_fields": fields})

    # ---------------------------------------------------------
    # GENERIC EVENT HOOK (TURN-LEVEL TOKEN USAGE + COST)
    # ---------------------------------------------------------

    async def on_event_callback(
        self,
        *,
        invocation_context: InvocationContext,
        event: Event,
    ) -> Optional[Event]:

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
        # INVOCATION META (CONTEXT)
        # -------------------------------
        meta = self._invocation_meta.get(
            (agent_name, session_id, invocation_id), {}
        )

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
        }

        # -------------------------------
        # Human readable form
        # -------------------------------
        human_activity = (
            f"Activity '{event_name}' for agent={agent_name}, "
            f"session={session_id}, invocation={invocation_id}"
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