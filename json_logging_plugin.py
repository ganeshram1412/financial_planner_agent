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


# --- Pricing map for cost calculation (USD per 1K tokens) ---
# Adjust these to your actual billing rates if needed.
MODEL_PRICING = {
    "gemini-2.0-flash": {
        # Example values: 0.075 USD / 1M input tokens => 0.000075 / 1K
        "input_per_1k": 0.000075,
        # Example values: 0.30 USD / 1M output tokens => 0.00030 / 1K
        "output_per_1k": 0.00030,
        "currency": "USD",
    }
}


def estimate_cost(model_name: Optional[str], prompt_tokens: Optional[int], completion_tokens: Optional[int]) -> Optional[Dict[str, Any]]:
    if not model_name:
        return None
    pricing = MODEL_PRICING.get(model_name)
    if not pricing:
        return None

    pt = prompt_tokens or 0
    ct = completion_tokens or 0

    input_cost = (pt / 1000.0) * pricing["input_per_1k"]
    output_cost = (ct / 1000.0) * pricing["output_per_1k"]
    total_cost = input_cost + output_cost

    return {
        "model": model_name,
        "input_tokens": pt,
        "output_tokens": ct,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "currency": pricing["currency"],
    }


class JsonFormatter(logging.Formatter):
    """Simple JSON line formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        extra_fields = record.__dict__.get("extra_fields")
        if isinstance(extra_fields, dict):
            safe_extra = {}
            for k, v in extra_fields.items():
                if isinstance(v, (str, int, float, bool)) or v is None:
                    safe_extra[k] = v
                else:
                    safe_extra[k] = repr(v)
            log.update(safe_extra)

        return json.dumps(log, ensure_ascii=False)


def _sanitize_for_filename(name: str, max_len: int = 80) -> str:
    """Make agent names safe for filenames."""
    safe = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        elif ch.isspace():
            safe.append("_")
        else:
            safe.append("_")
    safe_str = "".join(safe)
    return safe_str[:max_len] if len(safe_str) > max_len else safe_str


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
        log_path = os.path.join(base_dir, file_name)

        formatter = JsonFormatter()
        handler = RotatingFileHandler(
            log_path,
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger


# --- Session loggers (main, cost, error) ---

def get_session_logger(session_id: str, level: int = logging.INFO) -> logging.Logger:
    return _get_rotating_logger(
        logger_name=f"adk_session_{session_id}",
        base_dir=os.path.join("logs", "sessions"),
        file_name=f"session_{session_id}.log",
        level=level,
    )


def get_session_cost_logger(session_id: str, level: int = logging.INFO) -> logging.Logger:
    return _get_rotating_logger(
        logger_name=f"adk_session_cost_{session_id}",
        base_dir=os.path.join("logs", "sessions_cost"),
        file_name=f"session_cost_{session_id}.log",
        level=level,
    )


def get_session_error_logger(session_id: str, level: int = logging.INFO) -> logging.Logger:
    return _get_rotating_logger(
        logger_name=f"adk_session_error_{session_id}",
        base_dir=os.path.join("logs", "sessions_error"),
        file_name=f"session_error_{session_id}.log",
        level=level,
    )


# --- Agent loggers (main, cost, error) ---

def get_agent_logger(agent_name: str, level: int = logging.INFO) -> logging.Logger:
    safe_name = _sanitize_for_filename(agent_name)
    return _get_rotating_logger(
        logger_name=f"adk_agent_{safe_name}",
        base_dir=os.path.join("logs", "agents"),
        file_name=f"agent_{safe_name}.log",
        level=level,
    )


def get_agent_cost_logger(agent_name: str, level: int = logging.INFO) -> logging.Logger:
    safe_name = _sanitize_for_filename(agent_name)
    return _get_rotating_logger(
        logger_name=f"adk_agent_cost_{safe_name}",
        base_dir=os.path.join("logs", "agents_cost"),
        file_name=f"agent_cost_{safe_name}.log",
        level=level,
    )


def get_agent_error_logger(agent_name: str, level: int = logging.INFO) -> logging.Logger:
    safe_name = _sanitize_for_filename(agent_name)
    return _get_rotating_logger(
        logger_name=f"adk_agent_error_{safe_name}",
        base_dir=os.path.join("logs", "agents_error"),
        file_name=f"agent_error_{safe_name}.log",
        level=level,
    )


class JsonLoggingPlugin(BasePlugin):
    """
    ADK Plugin that logs key lifecycle events as JSON.

    It writes:
      - Session-wise main logs: logs/sessions/session_<SESSION>.log
      - Agent-wise  main logs: logs/agents/agent_<AGENT>.log
      - Session-wise cost logs: logs/sessions_cost/session_cost_<SESSION>.log
      - Agent-wise  cost logs: logs/agents_cost/agent_cost_<AGENT>.log
      - Session-wise error logs: logs/sessions_error/session_error_<SESSION>.log
      - Agent-wise  error logs: logs/agents_error/agent_error_<AGENT>.log
    """

    def __init__(self, level: int = logging.INFO) -> None:
        super().__init__(name="json_logging")
        self.level = level

    # ---------- helpers ----------
    def _session_and_agent_loggers_from_ctx(self, ctx: InvocationContext):
        session_logger = get_session_logger(ctx.session.id, self.level)
        agent_logger = get_agent_logger(ctx.agent.name, self.level)
        return session_logger, agent_logger

    # ---------- User message ----------
    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content,
    ) -> Optional[types.Content]:
        session_logger, agent_logger = self._session_and_agent_loggers_from_ctx(
            invocation_context
        )

        extra_fields = {
            "event": "user_message",
            "user_id": invocation_context.user_id, 
            "session_id": invocation_context.session.id,
            "invocation_id": invocation_context.invocation_id,
            "app_name": invocation_context.app_name, 
            "agent_name": invocation_context.agent.name,
            "text_preview": str(user_message),
        }

        session_logger.info("user_message", extra={"extra_fields": extra_fields})
        agent_logger.info("user_message", extra={"extra_fields": extra_fields})
        return None

    # ---------- Agent execution ----------
    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> None:
        ctx = callback_context._invocation_context
        session_logger, agent_logger = self._session_and_agent_loggers_from_ctx(ctx)

        extra_fields = {
            "event": "before_agent",
            "user_id": ctx.user_id,
            "session_id": ctx.session.id,
            "invocation_id": ctx.invocation_id,
            "agent_name": agent.name,
            "agent_description": getattr(agent, "description", None),
        }

        session_logger.info("before_agent", extra={"extra_fields": extra_fields})
        agent_logger.info("before_agent", extra={"extra_fields": extra_fields})

    async def after_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> None:
        ctx = callback_context._invocation_context
        session_logger, agent_logger = self._session_and_agent_loggers_from_ctx(ctx)

        extra_fields = {
            "event": "after_agent",
            "user_id": ctx.user_id,
            "session_id": ctx.session.id,
            "invocation_id": ctx.invocation_id,
            "agent_name": agent.name,
        }

        session_logger.info("after_agent", extra={"extra_fields": extra_fields})
        agent_logger.info("after_agent", extra={"extra_fields": extra_fields})

    # ---------- Model (LLM) calls ----------
    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        ctx = callback_context._invocation_context
        session_logger, agent_logger = self._session_and_agent_loggers_from_ctx(ctx)
        config = getattr(llm_request, "config", None)

        # stash model for cost calculation in after_model
        callback_context.state["model_name"] = llm_request.model

        extra_fields = {
            "event": "before_model",
            "user_id": ctx.user_id,
            "session_id": ctx.session.id,
            "invocation_id": ctx.invocation_id,
            "agent_name": ctx.agent.name,
            "model": llm_request.model,
            "temperature": getattr(config, "temperature", None) if config else None,
            "top_p": getattr(config, "top_p", None) if config else None,
            "tools_used": list(llm_request.tools_dict.keys())
            if config and getattr(config, "tools", None)
            else [],
            "max_llm_calls_limit": getattr(
                ctx.run_config, 
                "max_llm_calls", 
                -1
            ),
        }

        session_logger.info("before_model", extra={"extra_fields": extra_fields})
        agent_logger.info("before_model", extra={"extra_fields": extra_fields})

    async def after_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> None:
        ctx = callback_context._invocation_context
        session_logger, agent_logger = self._session_and_agent_loggers_from_ctx(ctx)

        usage: Optional[types.GenerateContentResponseUsageMetadata] = getattr(
            llm_response, "usage_metadata", None
        )
        error_code = getattr(llm_response, "error_code", None)
        error_msg = getattr(llm_response, "error_message", None)

        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None

        model_name = callback_context.state.get("model_name")
        cost_info = estimate_cost(model_name, prompt_tokens, completion_tokens)

        extra_fields = {
            "event": "after_model",
            "user_id": ctx.user_id,
            "session_id": ctx.session.id,
            "invocation_id": ctx.invocation_id,
            "agent_name": ctx.agent.name, 
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "error_code": error_code,
            "error_message": error_msg,
            "interrupted": getattr(llm_response, "interrupted", None),
        }

        if cost_info:
            extra_fields.update(
                {
                    "model": cost_info["model"],
                    "cost_input_tokens": cost_info["input_tokens"],
                    "cost_output_tokens": cost_info["output_tokens"],
                    "cost_input": cost_info["input_cost"],
                    "cost_output": cost_info["output_cost"],
                    "cost_total": cost_info["total_cost"],
                    "cost_currency": cost_info["currency"],
                }
            )

        # main logs
        session_logger.info("after_model", extra={"extra_fields": extra_fields})
        agent_logger.info("after_model", extra={"extra_fields": extra_fields})

        # cost-specific logs (session + agent)
        if cost_info:
            session_cost_logger = get_session_cost_logger(ctx.session.id, self.level)
            agent_cost_logger = get_agent_cost_logger(ctx.agent.name, self.level)

            session_cost_logger.info("model_cost", extra={"extra_fields": extra_fields})
            agent_cost_logger.info("model_cost", extra={"extra_fields": extra_fields})

    async def on_model_error_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
        error: Exception,
    ) -> Optional[LlmResponse]:
        ctx = callback_context._invocation_context
        session_logger, agent_logger = self._session_and_agent_loggers_from_ctx(ctx)

        extra_fields = {
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
        session_logger.error("model_error", extra={"extra_fields": extra_fields})
        agent_logger.error("model_error", extra={"extra_fields": extra_fields})

        # error-specific logs
        session_error_logger = get_session_error_logger(ctx.session.id, self.level)
        agent_error_logger = get_agent_error_logger(ctx.agent.name, self.level)
        session_error_logger.error("model_error", extra={"extra_fields": extra_fields})
        agent_error_logger.error("model_error", extra={"extra_fields": extra_fields})

        return None

    # ---------- Tool calls ----------
    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_context: ToolContext,
        tool_args: Dict[str, Any],
    ) -> None:
        ctx = tool_context._invocation_context
        session_logger, agent_logger = self._session_and_agent_loggers_from_ctx(ctx)

        extra_fields = {
            "event": "before_tool",
            "user_id": ctx.user_id,
            "session_id": ctx.session.id,
            "invocation_id": ctx.invocation_id,
            "agent_name": ctx.agent.name, 
            "tool_name": tool.name,
            "tool_arg_keys": list(tool_args.keys()),
        }

        session_logger.info("before_tool", extra={"extra_fields": extra_fields})
        agent_logger.info("before_tool", extra={"extra_fields": extra_fields})

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: Dict[str, Any],
        tool_context: ToolContext,
        result: dict
    ) -> None:
        ctx = tool_context._invocation_context
        session_logger, agent_logger = self._session_and_agent_loggers_from_ctx(ctx)

        extra_fields = {
            "event": "after_tool",
            "user_id": ctx.user_id,
            "session_id": ctx.session.id,
            "invocation_id": ctx.invocation_id,
            "agent_name": ctx.agent.name, 
            "tool_name": tool.name,
            "tool_result_preview": str(result),
        }

        session_logger.info("after_tool", extra={"extra_fields": extra_fields})
        agent_logger.info("after_tool", extra={"extra_fields": extra_fields})