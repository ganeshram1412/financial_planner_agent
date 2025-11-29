"""
app.py — ADK Application Bootstrap Module
------------------------------------------

This module wires together the full AI application by:

1. Initializing the **root financial planner orchestrator agent**.
2. Configuring an **event summarizer** to reduce conversation memory footprint.
3. Registering a custom **EventsCompactionConfig** that controls how the ADK
   compacts long event histories using intelligent summarization.
4. Creating the top-level **App** instance, which is the entrypoint used by any
   server, CLI, or integration layer.

This file should stay lightweight and declarative, containing no business logic.
All intelligence lives within `root_agent` and its sub-agents.
"""

from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.models import Gemini
from .agent import root_agent

# -----------------------------------------------------------
# Summarization LLM Definition
# -----------------------------------------------------------
"""
summarization_llm (Gemini):
    The LLM used internally by the ADK to summarize older events
    during compaction.

    Why this is needed:
    -------------------
    As conversations grow long, storing full event history becomes
    expensive. ADK periodically compacts older events into a distilled,
    context-preserving summary. This ensures:
      • Lower memory usage
      • Faster reasoning
      • Reduced token consumption
      • Better long-term context retention
"""
summarization_llm = Gemini(model="gemini-2.5-flash")

# -----------------------------------------------------------
# Event Summarizer Configuration
# -----------------------------------------------------------
"""
my_summarizer (LlmEventSummarizer):
    A summarizer object that uses the LLM above to compress chunks of
    conversation history into succinct summaries. Used by ADK’s built-in 
    event compaction engine.
"""
my_summarizer = LlmEventSummarizer(llm=summarization_llm)

# -----------------------------------------------------------
# App Configuration
# -----------------------------------------------------------
"""
app (App):
    The top-level ADK application instance.

    Responsibilities:
    -----------------
    • Serves as the orchestrator root for all runtime operations.
    • Holds the `root_agent` (financial planner orchestrator).
    • Manages event compaction, summarization, and runtime session state.
    • Exposes the application to downstream integrations (API, CLI, UI).

    EventsCompactionConfig:
    -----------------------
    summarizer=my_summarizer
        - Specifies the LLM summarizer to use for compressing old events.

    compaction_interval=3
        - Every 3 interaction cycles, the ADK triggers the compaction process.

    overlap_size=1
        - Ensures one event of overlap between compaction windows.
          This preserves continuity and ensures summarization does NOT
          lose important context by summarizing disjoint chunks.
"""
app = App(
    name='my-agent',
    root_agent=root_agent,
    events_compaction_config=EventsCompactionConfig(
        summarizer=my_summarizer,
        compaction_interval=3,
        overlap_size=1
    ),
)