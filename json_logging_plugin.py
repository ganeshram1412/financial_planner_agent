import json
import logging
from datetime import datetime
from typing import Optional, Any, Dict, List

from google.genai import types
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.agents import InvocationContext 
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.base_agent import BaseAgent
from google.adk.models.llm_request import LlmRequest 
from google.adk.models.llm_response import LlmResponse 
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.base_tool import BaseTool


class JsonFormatter(logging.Formatter):
    """Simple JSON line formatter."""

    def format(self, record: logging.LogRecord) -> str:
        # 💡 TIMESTAMP IS INCLUDED HERE (and is available in every log)
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


def configure_json_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logger (or a dedicated logger) for JSON output."""
    logger = logging.getLogger("adk_json")
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)

    logger.propagate = False
    return logger


class JsonLoggingPlugin(BasePlugin):
    """
    ADK Plugin that logs key lifecycle events as structured JSON.
    All callbacks return None so they only *observe* and don't change behavior.
    """

    def __init__(self, level: int = logging.INFO) -> None:
        super().__init__(name="json_logging")
        self.logger = configure_json_logging(level)

    # ---------- User message ----------
    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content,
    ) -> Optional[types.Content]:
        self.logger.info(
            "user_message",
            extra={
                "extra_fields": {
                    "event": "user_message",
                    "user_id": invocation_context.user_id, 
                    "session_id": invocation_context.session.id,
                    "invocation_id": invocation_context.invocation_id,
                    "app_name": invocation_context.app_name, 
                    # 💡 New: Agent name is the current agent
                    "agent_name": invocation_context.agent.name, 
                    "text_preview": _first_text_part(user_message, max_len=200),
                }
            },
        )
        return None

    # ---------- Agent execution ----------
    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> None:
        ctx = callback_context._invocation_context
        self.logger.info(
            "before_agent",
            extra={
                "extra_fields": {
                    "event": "before_agent",
                    "user_id": ctx.user_id,
                    "session_id": ctx.session.id,
                    "invocation_id": ctx.invocation_id,
                    # 💡 Agent Name is the agent starting its run
                    "agent_name": agent.name, 
                    "agent_description": getattr(agent, "description", None),
                }
            },
        )

    async def after_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> None:
        ctx = callback_context._invocation_context
        self.logger.info(
            "after_agent",
            extra={
                "extra_fields": {
                    "event": "after_agent",
                    "user_id": ctx.user_id,
                    "session_id": ctx.session.id,
                    "invocation_id": ctx.invocation_id,
                    # 💡 Agent Name is the agent that finished its run
                    "agent_name": agent.name, 
                }
            },
        )

    # ---------- Model (LLM) calls ----------
    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        ctx = callback_context._invocation_context
        config = getattr(llm_request, "config", None)
        
        self.logger.info(
            "before_model",
            extra={
                "extra_fields": {
                    "event": "before_model",
                    "user_id": ctx.user_id,
                    "session_id": ctx.session.id,
                    "invocation_id": ctx.invocation_id,
                    # 💡 Agent Name is the current agent driving the invocation
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
            },
        )

    async def after_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> None:
        ctx = callback_context._invocation_context
        usage: Optional[types.GenerateContentResponseUsageMetadata] = getattr(llm_response, "usage_metadata", None)
        error_code = getattr(llm_response, "error_code", None)
        error_msg = getattr(llm_response, "error_message", None)

        self.logger.info(
            "after_model",
            extra={
                "extra_fields": {
                    "event": "after_model",
                    "user_id": ctx.user_id,
                    "session_id": ctx.session.id,
                    "invocation_id": ctx.invocation_id,
                    # 💡 Agent Name is the current agent driving the invocation
                    "agent_name": ctx.agent.name, 
                    "completion_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
                    "prompt_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
                    "error_code": error_code,
                    "error_message": error_msg,
                    "interrupted": getattr(llm_response, "interrupted", None),
                }
            },
        )

    async def on_model_error_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
        error: Exception,
    ) -> Optional[LlmResponse]:
        ctx = callback_context._invocation_context
        self.logger.error(
            "model_error",
            extra={
                "extra_fields": {
                    "event": "model_error",
                    "user_id": ctx.user_id,
                    "session_id": ctx.session.id,
                    "invocation_id": ctx.invocation_id,
                    # 💡 Agent Name is the current agent driving the invocation
                    "agent_name": ctx.agent.name, 
                    "model": llm_request.model,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            },
        )
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
        self.logger.info(
            "before_tool",
            extra={
                "extra_fields": {
                    "event": "before_tool",
                    "user_id": ctx.user_id,
                    "session_id": ctx.session.id,
                    "invocation_id": ctx.invocation_id,
                    # 💡 Agent Name is the current agent driving the invocation
                    "agent_name": ctx.agent.name, 
                    "tool_name": tool.name,
                    "tool_arg_keys": list(tool_args.keys()),
                }
            },
        )

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: Dict[str, Any],
        tool_context: ToolContext,
        result: dict
    ) -> None:
        ctx = tool_context._invocation_context
        self.logger.info(
            "after_tool",
            extra={
                "extra_fields": {
                    "event": "after_tool",
                    "user_id": ctx.user_id,
                    "session_id": ctx.session.id,
                    "invocation_id": ctx.invocation_id,
                    # 💡 Agent Name is the current agent driving the invocation
                    "agent_name": ctx.agent.name, 
                    "tool_name": tool.name,
                    "tool_result_preview": _truncate(str(result), 500),
                }
            },
        )


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _first_text_part(content: types.Content, max_len: int = 200) -> str:
    """Extract a short preview from the user message."""
    for part in content.parts:
        if part.text:
            return _truncate(part.text, max_len)
    return ""