import json
import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional, Any, Dict

from google.genai import types
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.agents import InvocationContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.base_agent import BaseAgent
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.base_tool import BaseTool


# -------------------------------------------------------------------
# MODEL COST MAP
# -------------------------------------------------------------------

MODEL_PRICING = {
    "gemini-2.0-flash": {
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
    """Pretty JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        extra_fields = record.__dict__.get("extra_fields")
        if isinstance(extra_fields, dict):
            safe = {}
            for k, v in extra_fields.items():
                if isinstance(v, (str, int, float, bool)) or v is None:
                    safe[k] = v
                else:
                    safe[k] = repr(v)
            log.update(safe)

        # Pretty-print JSON
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

    IGNORE = {"agent", "engine", "module", "service", "tool"}
    tokens = [t for t in tokens if t not in IGNORE]

    if not tokens:
        tokens = [name]

    slug = "_".join(tokens)
    # keep only safe chars
    slug = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in slug)
    return slug[:60] if slug else "agent"


# -------------------------------------------------------------------
# LOGGER CREATION HELPERS
# -------------------------------------------------------------------

def _get_rotating_logger(
    logger_name: str,
    base_dir: str,
    file_name: str,
    level: int = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    if not logger.handlers:
        os.makedirs(base_dir, exist_ok=True)
        path = os.path.join(base_dir, file_name)

        handler = RotatingFileHandler(
            path,
            maxBytes=5 * 1024 * 1024,
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
    ADK Plugin that logs key lifecycle events as JSON.

    Folder layout:

      logs/
        <agent_slug>/               # e.g. financial_data_collector
          sessions/
            <session_id>/
              main.log
              cost.log
              error.log
          invocations/
            <invocation_id>/
              main.log
              cost.log
              error.log
    """

    def __init__(self, level: int = logging.INFO) -> None:
        super().__init__(name="json_logging")
        self.level = level

    def _session_invocation_loggers(self, ctx: InvocationContext):
        session_logger = get_agent_session_main_logger(
            ctx.agent.name, ctx.session.id, self.level
        )
        invocation_logger = get_agent_invocation_main_logger(
            ctx.agent.name, ctx.invocation_id, self.level
        )
        return session_logger, invocation_logger

    # ---------------------------------------------------------
    # USER MESSAGE
    # ---------------------------------------------------------
    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content,
    ) -> Optional[types.Content]:
        session_logger = get_agent_session_main_logger(
            invocation_context.agent.name,
            invocation_context.session.id,
            self.level,
        )
        invocation_logger = get_agent_invocation_main_logger(
            invocation_context.agent.name,
            invocation_context.invocation_id,
            self.level,
        )

        fields = {
            "event": "user_message",
            "user_id": invocation_context.user_id,
            "session_id": invocation_context.session.id,
            "invocation_id": invocation_context.invocation_id,
            "app_name": invocation_context.app_name,
            "agent_name": invocation_context.agent.name,
            "text_preview": str(user_message),
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
        session_logger, invocation_logger = self._session_invocation_loggers(ctx)

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
        session_logger, invocation_logger = self._session_invocation_loggers(ctx)

        fields = {
            "event": "after_agent",
            "user_id": ctx.user_id,
            "session_id": ctx.session.id,
            "invocation_id": ctx.invocation_id,
            "agent_name": agent.name,
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
        session_logger, invocation_logger = self._session_invocation_loggers(ctx)

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
    # AFTER MODEL
    # ---------------------------------------------------------
    async def after_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> None:
        ctx = callback_context._invocation_context
        session_logger, invocation_logger = self._session_invocation_loggers(ctx)

        usage: Optional[types.GenerateContentResponseUsageMetadata] = getattr(
            llm_response, "usage_metadata", None
        )
        # NOTE: adjust field names if your usage_metadata differs
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None

        model_name = callback_context.state.get("model_name")
        cost = estimate_cost(model_name, prompt_tokens, completion_tokens)

        fields = {
            "event": "after_model",
            "user_id": ctx.user_id,
            "session_id": ctx.session.id,
            "invocation_id": ctx.invocation_id,
            "agent_name": ctx.agent.name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "error_code": getattr(llm_response, "error_code", None),
            "error_message": getattr(llm_response, "error_message", None),
            "interrupted": getattr(llm_response, "interrupted", None),
        }

        if cost:
            fields.update(
                {
                    "model": cost["model"],
                    "cost_input_tokens": cost["input_tokens"],
                    "cost_output_tokens": cost["output_tokens"],
                    "cost_input": cost["input_cost"],
                    "cost_output": cost["output_cost"],
                    "cost_total": cost["total_cost"],
                    "cost_currency": cost["currency"],
                }
            )

        # main logs
        session_logger.info("after_model", extra={"extra_fields": fields})
        invocation_logger.info("after_model", extra={"extra_fields": fields})

        # cost logs
        if cost:
            session_cost_logger = get_agent_session_cost_logger(
                ctx.agent.name, ctx.session.id, self.level
            )
            invocation_cost_logger = get_agent_invocation_cost_logger(
                ctx.agent.name, ctx.invocation_id, self.level
            )
            session_cost_logger.info("model_cost", extra={"extra_fields": fields})
            invocation_cost_logger.info("model_cost", extra={"extra_fields": fields})

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
        session_logger, invocation_logger = self._session_invocation_loggers(ctx)

        fields = {
            "event": "before_tool",
            "user_id": ctx.user_id,
            "session_id": ctx.session.id,
            "invocation_id": ctx.invocation_id,
            "agent_name": ctx.agent.name,
            "tool_name": tool.name,
            "tool_arg_keys": list(tool_args.keys()),
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
        session_logger, invocation_logger = self._session_invocation_loggers(ctx)

        fields = {
            "event": "after_tool",
            "user_id": ctx.user_id,
            "session_id": ctx.session.id,
            "invocation_id": ctx.invocation_id,
            "agent_name": ctx.agent.name,
            "tool_name": tool.name,
            "tool_result_preview": str(result),
        }

        session_logger.info("after_tool", extra={"extra_fields": fields})
        invocation_logger.info("after_tool", extra={"extra_fields": fields})