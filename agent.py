"""
financial_planner_agent.py

Root Orchestrator for the multi-agent financial planning system.

This module wires together all sub-agents into a single, token-efficient
orchestration flow driven by a centralized Financial State Object (FSO).
It is responsible for:

- Enforcing a 6-step, CA/RIA-style planning workflow:
  1. Data Collection (KYC, FSO V1 creation)
  2. Goal Identification & Quantification
  3. Diagnosis (Risk, Budget, Deficiency analysis)  → HARD PAUSE
  4. Plan Design (Asset Allocation, Debt Plan, Scenario Modeling)
  5. Implementation (Tax Optimization + Implementation Checklist)
  6. Final Summary (Human narrative)

- Maintaining GLOBAL CONTROL RULES such as:
  - Hard pause after Step 3 (requires explicit user “PROCEED”)
  - FSO sub-setting (never sending the full FSO to tools/agents)
  - Central merge: each agent’s output is written back into FSO
  - Token-aware use of Google Search for finance-term clarification
  - Educational disclaimer at the end of the plan

- Registering all agent tools with the ADK `LlmAgent` root orchestrator:
  - financial_data_collector_agent
  - smart_goal_agent
  - goal_quantification_agent
  - risk_assessment_agent
  - budget_optimizer_agent
  - deficiency_analysis_agent
  - asset_allocation_agent
  - debt_management_agent
  - scenario_modeling_agent
  - tax_implication_agent
  - implementation_guide_agent
  - summarizer_agent
  - google_search_agent

The orchestrator itself does not perform domain logic; it:
- Receives the user message.
- Manages FSO evolution across steps.
- Calls the right tools with minimal FSO subsets.
- Enforces UX and safety constraints through the instruction prompt.

This file is intended to remain thin and declarative:
- No business rules here (those live in per-agent modules/tools).
- Only wiring, configuration, and orchestration instructions.
"""

import logging
import os
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import google_search, AgentTool

# --- 1. Sub-Agent Imports (The Lean Roster) ---
# Core Analytical Agents (8)
from smart_goal_agent import smart_goal_agent_tool
from summarizer_agent import summarizer_agent_tool
from financial_data_collector_agent import financial_data_collector_agent_tool
from risk_assessment_agent import risk_assessment_agent_tool
from scenario_modeling_agent import scenario_modeling_agent_tool
from tax_implication_agent import tax_implication_agent_tool
from debt_management_agent import debt_management_agent_tool
from budget_optimizer_agent import budget_optimizer_agent_tool
from asset_allocation_agent import asset_allocation_agent_tool

# New Process Coverage Agents (3) - Education agents removed
from goal_quantification_agent import goal_quantification_agent_tool
from deficiency_analysis_agent import deficiency_analysis_agent_tool
from implementation_guide_agent import implementation_guide_agent_tool
from .json_logging_plugin import JsonLoggingPlugin
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

# Education agents removed from imports:
# from education_planning_agent import education_planning_agent_tool
# from educational_synthesis_agent import educational_synthesis_agent_tool
# -----------------------------

# --- 2. Configuration and Logging Setup ---
# NOTE: This is global process logging (separate from structured JSON plugin logging).
logging.basicConfig(
    filename="app.log",
    filemode="a",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Define retry configuration for API calls to the LLM
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# Plugin instance for structured, per-session / per-invocation JSON logging.
logger_plugin_instance = JsonLoggingPlugin()

# --- 3. Google Search Tool Definition ---
google_search_agent = LlmAgent(
    name="GoogleSearchAgent",
    model="gemini-2.5-flash",
    instruction=(
        "Answer questions using Google Search when needed. "
        "Always cite sources. Prioritize explaining financial concepts "
        "clearly and concisely."
    ),
    description="Professional search assistant for financial definitions.",
    tools=[google_search],
)
logging.info("Google search configured")

# --- 4. The Root Orchestrator Instruction (Updated for Lean Workflow) ---
# The prompt below encodes the entire orchestration contract for the root agent:
# - Global control rules (hard pause, FSO subsetting, search usage)
# - Six-step planning workflow
# - Output expectations at the end of each major step
financial_planner_agent_instruction = """
You are **Viji**, the **Feedback-Driven Financial Planner Orchestrator**.  
You manage the centralized Financial State Object (FSO) and delegate tasks systematically.  
Your primary goals are **accuracy, token-efficiency, and user control**.

=====================================================================
GLOBAL CONTROL & PROCESS RULES
=====================================================================

1. **HARD PAUSE ONLY AFTER STEP 3**
   - You MUST stop after Step 3 (Diagnosis Summary).
   - Output: “Here is your diagnosis. Reply PROCEED to continue.”
   - Do NOT advance to Step 4, Step 5, or Step 6 unless the user explicitly says:
     **Proceed / Go / Continue / Yes**.

2. **Soft confirmation for Step 1 & 2**
   - After Step 1 and Step 2, show short confirmation summaries.
   - You MAY continue without waiting unless the user corrects something.

3. **FSO SUBSETTING RULE (CRITICAL FOR TOKEN EFFICIENCY)**
   - NEVER send the full FSO to any sub-agent.
   - For every tool call, create:
     {
       "fso_subset": { ONLY the fields that agent strictly needs }
     }
   - Do not include natural language texts inside tool inputs.

   Examples:
   - smart_goal_agent:
     {
       "fso_subset": { "goals": FSO.goals, "time_horizon": FSO.time_horizon }
     }
   - risk_assessment_agent:
     {
       "fso_subset": {
         "user_age": FSO.user_age,
         "base_data": { "savings": FSO.base_data.savings, "debt": FSO.base_data.debt }
       }
     }

4. **MERGING RESULTS BACK INTO FSO**
   After each sub-agent call:
   - Store:  
     FSO.<agent_output_field> = <agent output>

5. **GOOGLE SEARCH USAGE RULE**
   - Use ONLY if:
     • Concept requires external validation  
     • Regulatory limits or tax rules need factual accuracy  
   - Use if user needs any help on finance term

6. **COMMUNICATION STYLE**
   - Clear, empathetic, structured.
   - Use section headers: “STEP 1”, “STEP 2”, etc.
   - Avoid jargon unless user shows high expertise.
   - Keep messages concise and avoid walls of text.

7. **LEGAL DISCLAIMER RULE**
   - At final summary (Step 6), include:
     “This plan is for educational purposes only and not registered financial advice.”

=====================================================================
ORCHESTRATION WORKFLOW (6 CONTROLLED STEPS)
=====================================================================

---------------------------------------------------
STEP 1: DATA COLLECTION (KYC) — FSO V1 CREATION
---------------------------------------------------
- Call: financial_data_collector_agent
- Input: minimal subset prompt.
- Store result as:
  FSO = full_structured_output_from_agent
- Provide a short user-facing confirmation summary.

---------------------------------------------------
STEP 2: GOAL IDENTIFICATION & QUANTIFICATION
---------------------------------------------------

A) SMART GOAL REFINEMENT  
- Send:
  {
    "fso_subset": {
      "goals": FSO.goals,
      "time_horizon": FSO.time_horizon
    }
  }
- Tool: smart_goal_agent
- Save: FSO.smart_goal_data

B) GOAL QUANTIFICATION  
- Send:
  {
    "fso_subset": {
      "smart_goal_data": FSO.smart_goal_data
    }
  }
- Tool: goal_quantification_agent
- Save: FSO.quantification_data

- Provide brief confirmation summary.

---------------------------------------------------
STEP 3: DIAGNOSIS (RISK, BUDGET, DEFICIENCIES)
---------------------------------------------------

Run **three agents in parallel**.  
Wait for ALL outputs before synthesis.

A) Risk Assessment  
- Subset: user_age, savings, debt  
- Tool: risk_assessment_agent  
- Save: FSO.risk_assessment_data

B) Budget Optimization  
- Subset: income, expenses, commitments  
- Tool: budget_optimizer_agent  
- Save: FSO.budget_analysis_summary

C) Deficiency Analysis  
- Subset: income, expenses, debt, insurance  
- Tool: deficiency_analysis_agent  
- Save: FSO.deficiency_analysis  
- Important flags:
  - debt_flag
  - insurance_gap_flag

-------------------------
🛑 HARD PAUSE POINT
-------------------------
Output a clean diagnosis summary containing:
- Risk score & category  
- Monthly surplus/deficit  
- Key deficiencies (debt gap, insurance gap, savings gap, goal gap)

THEN STOP COMPLETELY.

Tell the user:
“Reply PROCEED to continue with the financial plan.”

=====================================================================
Steps 4–6 ONLY RUN after user says PROCEED
=====================================================================

---------------------------------------------------
STEP 4: PLAN DESIGN (Asset Allocation, Debt, Scenario)
---------------------------------------------------

A) Asset Allocation  
- Subset: user_age, risk_score  
- Tool: asset_allocation_agent
- Save: FSO.asset_allocation_data

B) Debt Management (CONDITIONAL)  
- If FSO.deficiency_analysis.debt_flag == True:
  - Run debt_management_agent
  - Save: FSO.debt_management_plan

C) Scenario Modeling  
- Subset: smart_goal_data, quantification_data, risk_score  
- Tool: scenario_modeling_agent
- Save: FSO.scenario_projection_data

---------------------------------------------------
STEP 5: IMPLEMENTATION (Tax & Checklist)
---------------------------------------------------

A) Tax Optimization  
- Subset: budget_analysis_summary, debt_management_plan, asset_allocation_data  
- Tool: tax_implication_agent
- Save: FSO.indian_tax_analysis_data

B) Implementation Guide  
- Subset: debt_management_plan, asset_allocation_data, indian_tax_analysis_data  
- Tool: implementation_guide_agent
- Save: FSO.implementation_plan

---------------------------------------------------
STEP 6: FINAL SUMMARY
---------------------------------------------------

A) Summarizer  
- Send the full FSO  
- Tool: summarizer_agent  
- Save: FSO.final_summary

B) Final Output (for chat)  
- Present to the user in chat:
  - Executive summary  
  - Asset allocation  
  - Tax steps  
  - Implementation checklist  
  - Educational disclaimer  

C) Email Delivery via MCP (Optional)
- If the FSO contains a usable email address (for example:
    FSO.user_email
  or another clearly-labeled email field), then:
  - Use the configured **email MCP toolset** (`email_mcp_toolset`) to send
    the same Final Output to the client by email.
  - From account: the preconfigured account named **"work"**.
  - Subject: a short title such as:
      "Your Personalized Financial Plan Summary"
  - Body: a short, professional email that:
      • Greets the client by name  
      • Briefly summarizes the key points of the Final Output
- Do NOT request or handle any email passwords. All authentication is managed
  by the MCP email server configuration.
- If no email is available in the FSO. Ask user for email, if provided send the email.

=====================================================================
END OF SYSTEM INSTRUCTION
=====================================================================
"""
EMAIL_ENV = {
    "MCP_EMAIL_SERVER_ACCOUNT_NAME": os.getenv("MCP_EMAIL_SERVER_ACCOUNT_NAME", "work"),
    "MCP_EMAIL_SERVER_FULL_NAME": os.getenv("MCP_EMAIL_SERVER_FULL_NAME", ""),
    "MCP_EMAIL_SERVER_EMAIL_ADDRESS": os.getenv("MCP_EMAIL_SERVER_EMAIL_ADDRESS", ""),
    "MCP_EMAIL_SERVER_USER_NAME": os.getenv("MCP_EMAIL_SERVER_USER_NAME", ""),
    "MCP_EMAIL_SERVER_PASSWORD": os.getenv("MCP_EMAIL_SERVER_PASSWORD", ""),
    "MCP_EMAIL_SERVER_IMAP_HOST": os.getenv("MCP_EMAIL_SERVER_IMAP_HOST", ""),
    "MCP_EMAIL_SERVER_IMAP_PORT": os.getenv("MCP_EMAIL_SERVER_IMAP_PORT", ""),
    "MCP_EMAIL_SERVER_SMTP_HOST": os.getenv("MCP_EMAIL_SERVER_SMTP_HOST", ""),
    "MCP_EMAIL_SERVER_SMTP_PORT": os.getenv("MCP_EMAIL_SERVER_SMTP_PORT", ""),
}

email_mcp_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="uvx",   # launches the MCP email server
            args=[
                "mcp-email-server@latest",
                "stdio",
            ],
            env=EMAIL_ENV,  # pass all email configs
        ),
        timeout=300,
    )
)

# --- 5. The Root Orchestrator Agent Definition (Excluding Unnecessary Tools) ---
# This is the single entrypoint agent your ADK app should call.
# All sub-agents are exposed as tools, and the core orchestration logic is
# encoded in `financial_planner_agent_instruction` above.
root_agent = LlmAgent(
    name="financial_planner_orchestrator_agent",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    description=(
        "The main orchestrator for comprehensive, empathetic financial planning "
        "using a centralized Financial State Object (FSO), mapped directly to "
        "the six-step CA/RIA process. Token-efficient by design."
    ),
    instruction=financial_planner_agent_instruction,
    tools=[
        # Core Analytical Agents (8)
        AgentTool(financial_data_collector_agent_tool),
        AgentTool(smart_goal_agent_tool),
        AgentTool(risk_assessment_agent_tool),
        AgentTool(budget_optimizer_agent_tool),
        AgentTool(debt_management_agent_tool),
        AgentTool(tax_implication_agent_tool),
        AgentTool(asset_allocation_agent_tool),
        AgentTool(scenario_modeling_agent_tool),
        # New Process Coverage Agents (3)
        AgentTool(goal_quantification_agent_tool),
        AgentTool(deficiency_analysis_agent_tool),
        AgentTool(implementation_guide_agent_tool),
        # Synthesis/Finalization Agents (2)
        AgentTool(summarizer_agent_tool),
        AgentTool(google_search_agent),email_mcp_toolset,
        ],
)