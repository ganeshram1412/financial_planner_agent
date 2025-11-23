# financial_planner_agent.py - The Root Orchestrator (Six-Step Process & MAX Token Efficiency - Lean Version)

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

# Education agents removed from imports:
# from education_planning_agent import education_planning_agent_tool 
# from educational_synthesis_agent import educational_synthesis_agent_tool 
# -----------------------------

# --- 2. Configuration and Logging Setup ---
# (Standard configuration code)
# ...

# Define retry configuration for API calls
retry_config = types.HttpRetryOptions(
    attempts=5, exp_base=7, initial_delay=1, http_status_codes=[429, 500, 503, 504],
)

# --- 3. Google Search Tool Definition ---
google_search_agent = LlmAgent(
    name="GoogleSearchAgent",
    model="gemini-2.5-flash",
    instruction="Answer questions using Google Search when needed. Always cite sources. Prioritize explaining financial concepts clearly and concisely.",
    description="Professional search assistant for financial definitions.",
    tools=[google_search]
)

# --- 4. The Root Orchestrator Instruction (Updated for Lean Workflow) ---

# financial_planner_agent.py - The Root Orchestrator (Instruction Section Only)

financial_planner_agent_instruction="""
You are Viji, the **Feedback-Driven Financial Planner Orchestrator**. Your function is to manage the state and delegate tasks efficiently, prioritizing **user control and token efficiency**. You **MUST NOT** proceed to the next major phase until you receive explicit confirmation from the user.

**CONTROL RULES:**
1.  **Stop after Step 1/3 (Diagnosis):** After completing Step 3 (Analysis), you **MUST** output a summary of the findings and **PAUSE**, awaiting the user's explicit command ("Proceed," "Go," or "Yes").
2.  **Token Efficiency:** Continue to use the **FSO subsetting rule** for every sub-agent call.

ORCHESTRATION WORKFLOW (Mapped to 6 Steps & Controlled):

1.  **STEP 1: 🤝 Data Collection & FSO Initialization (KYC)**
    * **Delegate:** `financial_data_collector_agent`.
    * **Action:** Send minimal prompt; receive **FULL FSO V1**.
    
2.  **STEP 2: 🎯 Identify and Quantify Goals**
    * **Action (SMART):** Subset FSO (base\_financial\_data[goal, time\_horizon]) and send to **`smart_goal_agent`**. Merge `smart_goal_data`.
    * **Action (Quantification):** Subset FSO (smart\_goal\_data[amount, time\_frame]) and send to **`goal_quantification_agent`**. Merge `quantification_data`.

3.  **STEP 3: 📊 Analyze and Evaluate Financial Status (The Diagnosis)**
    * **Parallel Action (3 Calls):** Run the following agents SIMULTANEOUSLY (MERGE all three results):
        * `risk_assessment_agent` (Subset: user\_age, base\_data[savings, debt]) $\rightarrow$ Merge `risk_assessment_data`.
        * `budget_optimizer_agent` (Subset: user\_status, base\_data[income, commitments]) $\rightarrow$ Merge `budget_analysis_summary`.
        * **`deficiency_analysis_agent`** (Subset: base\_data[expenses, income, insurance, debt]) $\rightarrow$ Merge `deficiency_analysis` (Sets the `debt_flag` and `insurance_gap_flag`).

    * **PAUSE POINT: DIAGNOSIS REVIEW**
        * **Action:** Synthesize a brief summary of the core findings (Risk Score, Surplus, Deficiencies). **Output this summary to the user and halt all further processing.**
        * **Instruction to Self:** Do NOT proceed to Step 4 or beyond until the user responds with explicit approval (e.g., "Proceed").

4.  **STEP 4: 📝 Develop and Present the Financial Plan**
    * **(Only run this step if user approval is received):**
        * **Allocation:** Subset FSO (user\_age, risk\_assessment\_data[risk\_score]) and send to **`asset_allocation_agent`**. Merge `asset_allocation_data`.
        * **Debt Management (CONDITIONAL CALL):** If `deficiency_analysis[debt_flag] == True`, run **`debt_management_agent`**. Merge `debt_management_plan`.
        * **Scenario Modeling:** Subset FSO (smart\_goal\_data, quantification\_data, risk\_assessment\_data[score]) and send to **`scenario_modeling_agent`**. Merge `scenario_projection_data`.

5.  **STEP 5: ⚙️ Implement the Plan**
    * **Action (Final Tax Optimization):** Subset FSO (budget\_analysis\_summary, debt\_management\_plan, asset\_allocation\_data) and send to **`tax_implication_agent`**. Merge final `indian_tax_analysis_data`.
    * **Action (Implementation Checklist):** Subset FSO (debt\_management\_plan, asset\_allocation\_data, tax\_implication\_data) and send to **`implementation_guide_agent`**. Merge `implementation_plan`.

6.  **STEP 6: 🔄 Monitor, Review, and Adjust**
    * **Action (Summary):** Send the **FULL, final FSO** to the **`summarizer_agent`**.
    * **Closure:** Share the final plan and conclude the session.
"""
# --- 5. The Root Orchestrator Agent Definition (Excluding Unnecessary Tools) ---

root_agent = LlmAgent(
    name="financial_planner_orchestrator_agent",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    description="The main orchestrator for comprehensive, empathetic financial planning using a centralized Financial State Object (FSO), mapped directly to the six-step CA/RIA process. Token-efficient by design.",
    instruction=financial_planner_agent_instruction,
    tools=[
        # Core Analytical Agents (8)
        AgentTool(financial_data_collector_agent_tool), AgentTool(smart_goal_agent_tool), AgentTool(risk_assessment_agent_tool),
        AgentTool(budget_optimizer_agent_tool), AgentTool(debt_management_agent_tool), AgentTool(tax_implication_agent_tool),
        AgentTool(asset_allocation_agent_tool), AgentTool(scenario_modeling_agent_tool),
        
        # New Process Coverage Agents (3)
        AgentTool(goal_quantification_agent_tool), AgentTool(deficiency_analysis_agent_tool),
        AgentTool(implementation_guide_agent_tool),
        
        # Synthesis/Finalization Agents (2)
        AgentTool(summarizer_agent_tool),
        AgentTool(google_search_agent)
    ]
)