# financial_planner_agent.py - The Root Orchestrator Agent

import logging
import os
import asyncio # Imported, although the LLM handles the parallel execution
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import google_search

# --- 1. Sub-Agent Imports ---
# NOTE: Ensure these module names and variable names match your sub-agent files!
from smart_goal_agent import smart_goal_agent_tool 
from summarizer_agent import summarizer_agent_tool 
from financial_data_collector_agent import financial_data_collector_agent_tool 
from risk_assessment_agent import risk_assessment_agent_tool
from scenario_modeling_agent import scenario_modeling_agent_tool
from tax_implication_agent import tax_implication_agent_tool
from debt_management_agent import debt_management_agent_tool
from budget_optimizer_agent import budget_optimizer_agent_tool
# -----------------------------

# --- 2. Configuration and Logging Setup ---
# Clean up old logs for a fresh start
for log_file in ["logger.log", "web.log", "tunnel.log"]:
    if os.path.exists(log_file):
        os.remove(log_file)

logging.basicConfig(
    filename='logger.log',
    level=logging.DEBUG, 
    format="%(filename)s:%(lineno)s %(levelname)s:%(message)s",
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("FinancialPlannerOrchestrator")
logger.info("Initializing Financial Planner Orchestrator Agent components.")

# Define retry configuration for API calls
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# --- 3. Google Search Tool Definition ---
google_search_agent = LlmAgent(
    name="GoogleSearchAgent",
    model="gemini-2.5-flash",
    instruction="Answer questions using Google Search when needed. Always cite sources. Prioritize explaining financial concepts clearly and concisely.",
    description="Professional search assistant for financial definitions.",
    tools=[google_search]
)

# --- 4. The Root Orchestrator Instruction (Optimized for Cost) ---

financial_planner_agent_instruction = """
You are Aura, a warm, non-judgmental SEBI-certified Financial Planner orchestrator.
Your role: guide the client through a structured workflow, delegate tasks to sub-agents,
and synthesize a final actionable plan. Maintain a friendly, encouraging tone.
Include a disclaimer that AI-generated content may be inaccurate.

ORCHESTRATION WORKFLOW (Follow Exactly in Order):

1. Greeting & Data Collection
   - Start warmly.
   - Delegate to `financial_data_collector_agent` to gather baseline financial data.
   - Wait for successful completion.

2. Core Analysis (Run in Parallel)
   - Announce diagnostics.
   - Simultaneously delegate to:
        * `risk_assessment_agent`-risk profile + insurance gaps
        * `budget_optimizer_agent`-cash flow + savings potential
        * `tax_implication_agent`-tax regime + optimization
   - Wait for all three results.

3. Goal Setting & Projections
   - Ask the client for their primary goal.
   - Delegate to:
        * `smart_goal_agent` – SMART goal creation
        * `scenario_modeling_agent` – projections using SMART goal + risk profile

4. Strategy Synthesis
   - If high-interest debt exists, delegate to:
        * `debt_management_agent`
   - Send ALL outputs (Data, Risk, Budget, Tax, Goal, Scenario, Debt if any) to:
        * `summarizer_agent` – produce the consolidated financial plan.

5. Closure
   - Share the final summarized plan.
   - Ask the client to choose next step:
        (a) Save plan
        (b) Discuss recommendations
        (c) Modify data
"""

# --- 6. The Root Orchestrator Agent Definition ---

# The agent variable MUST be named root_agent for the ADK runner to find it.
root_agent = LlmAgent(
    name="financial_planner_orchestrator_agent",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    description="The main orchestrator for comprehensive, empathetic financial planning.",
    instruction=financial_planner_agent_instruction,
    tools=[financial_data_collector_agent_tool,summarizer_agent_tool,smart_goal_agent_tool,risk_assessment_agent_tool,scenario_modeling_agent_tool,tax_implication_agent_tool,debt_management_agent_tool,budget_optimizer_agent_tool,google_search_agent]
)

logger.info("Financial Planner Orchestrator Agent (root_agent) fully defined.")