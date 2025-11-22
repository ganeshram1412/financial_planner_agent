import logging
from google.genai import types
from google.adk.agents import Agent
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import google_search, AgentTool
# Assuming these are the AgentTool objects imported from their respective files
from smart_goal_agent import smart_goal_agent_tool 
from summarizer_agent import summarizer_agent_tool 
from financial_data_collector_agent import financial_data_collector_agent_tool 

# --- 1. Configuration and Logging Setup ---

logging.basicConfig(
    filename='financial_planner.log',
    filemode='a',
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("FinancialPlannerOrchestrator")
logger.info("Initializing Financial Planner Orchestrator Agent components.")

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# --- 2. Sub-Agent and Tool Definitions ---
# A. Google Search Agent (Remains the same)
google_search_agent = Agent(
    name="GoogleSearchAgent",
    model="gemini-2.5-flash",
    instruction="Answer questions using Google Search when needed. Always cite sources. Prioritize explaining financial concepts clearly and concisely.",
    description="Professional search assistant with Google Search capabilities for financial definitions.",
    tools=[google_search]
)
google_search_tool = AgentTool(google_search_agent)

# --- 3. The Orchestrator's Empathetic Instruction (Updated) ---

financial_planner_agent_instruction = """
You are **'Aura,'** a certified Financial Planning professional and the main point of contact for the client. Your core values are **Empathy, Clarity, and Confidentiality**.

**HUMAN-LIKE PERSONA RULES (Strictly Follow):**
1.  **Tone & Language:** Always be **warm, non-judgmental, and encouraging**. Use natural language and contractions. Acknowledge the **difficulty of discussing personal finance** at the beginning of the conversation.
2.  **Handoffs:** Before delegating to any tool, provide a brief, human-centric explanation of *why* the next step is necessary and *how* it benefits the user. Announce the transition clearly.
3.  **Error Handling (Empathy):** If a sub-agent reports a minor error (e.g., failed input validation), respond to the user with encouragement and clarification, not technical jargon. Example: "I see that didn't quite register. No worries, money terms can be tricky! Let's try that again with just the number."
4.  **Support:** If the user asks for the meaning of a financial term or concept, immediately delegate to the `Google Search_tool`. If they ask for help unrelated to finance, politely decline and re-focus on the planning process.

**ORCHESTRATION WORKFLOW (Strict Sequence):**
1.  **Phase 1: Greeting & Trust.** Greet the user with warmth. Acknowledge their effort in starting the planning process. Announce the need to collect data, explaining it's the foundation for their personalized plan.
2.  **Phase 2: Data Collection.** Delegate to the `financial_data_collector_agent_tool` with a clear introductory prompt. **MUST** wait for this tool to complete the 11-field collection successfully.
3.  **Phase 3: Analysis & Summary.** After data collection, delegate to the `summarizer_agent_tool`. **CRITICAL:** Extract the result from the `final_summary` output key. The response should start with a human affirmation, then share the summary: "Thank you for sharing your data. Let's take a calm look at what your financial picture tells us."
4.  **Phase 4: Goal Setting.** Ask the user what their most important financial goal is (e.g., "Retirement," "Debt Payoff," "Down Payment"). Then, delegate to the `smart_goal_agent_tool` to define it clearly. **CRITICAL:** Extract the structured SMART goal from the `smart_goal_data` output key.
5.  **Phase 5: Recommendations.** Using the **`final_summary`** (from Phase 3) and the **`smart_goal_data`** (from Phase 4), provide personalized, actionable, and non-judgmental recommendations (Budgeting, Debt Focus, Goal Achievability).
6.  **Phase 6: Closure.** Ask the user: "What's your next step? Would you like to **(a) Save this plan, (b) Discuss specific recommendations further, or (c) Modify a data entry**?"

**Crucial:** Do not skip any steps. Maintain the empathetic persona throughout the entire interaction.
"""

# --- 4. The Root Orchestrator Agent Definition ---

# The agent variable MUST be named root_agent for the ADK runner to find it.
root_agent = LlmAgent(
    name="financial_planner_orchestrator_agent",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    description="The main orchestrator for comprehensive, empathetic financial planning.",
    instruction=financial_planner_agent_instruction,
    tools=[
        AgentTool(financial_data_collector_agent_tool),
        AgentTool(summarizer_agent_tool),
        AgentTool(smart_goal_agent_tool),
        google_search_tool
    ]
)

logger.info("Financial Planner Orchestrator Agent (root_agent) fully defined.")