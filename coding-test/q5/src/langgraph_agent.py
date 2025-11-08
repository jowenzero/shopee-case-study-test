from typing import TypedDict, List, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

from src.llm_config import get_gemini_llm
from src.database import DatabaseManager
from src.storage_integration import StorageIntegration
from src.langchain_tools import create_tools


class AgentState(TypedDict):
    """State for the LangGraph agent workflow"""
    query: str
    intent: str
    tool_results: str
    response: str
    messages: List


class ReceiptQueryAgent:
    """
    LangGraph agent for querying receipt data using natural language
    """

    def __init__(
        self,
        db: DatabaseManager,
        storage: StorageIntegration,
        model_name: str = "gemini-2.5-flash"
    ):
        self.db = db
        self.storage = storage
        self.llm = get_gemini_llm(model_name=model_name, temperature=0.0)
        self.tools = create_tools(db, storage)

        self.workflow = self._build_workflow()

    def _router_node(self, state: AgentState) -> AgentState:
        """
        Classify query intent to determine which tool to use

        Args:
            state: Current agent state

        Returns:
            Updated state with intent classification
        """
        query = state["query"]

        classification_prompt = f"""Analyze this user query and classify its intent.

Query: "{query}"

Choose ONE of these intents:
1. date_query - User asks about purchases on specific date or date range
2. item_query - User asks about specific food items or products by name (e.g., "chicken", "nasi goreng")
3. expense_query - User asks about total spending or expenses
4. semantic_query - Complex queries requiring semantic search, including:
   - Price-based comparisons (cheapest, most expensive, lowest price, highest price)
   - Quality/attribute-based queries (best, worst)
   - Vague or general queries requiring understanding of context

Important: Queries about "cheapest" or "most expensive" items are ALWAYS semantic_query.

Respond with ONLY the intent name (e.g., "date_query"), nothing else."""

        response = self.llm.invoke([HumanMessage(content=classification_prompt)])
        intent = response.content.strip().lower()

        state["intent"] = intent
        state["messages"] = [HumanMessage(content=query)]

        return state

    def _extract_parameters(self, query: str, intent: str) -> dict:
        """
        Extract structured parameters from query based on intent

        Args:
            query: User query
            intent: Classified intent

        Returns:
            Dictionary with extracted parameters
        """
        from datetime import datetime

        if intent == "date_query":
            extraction_prompt = f"""Extract the date or date range from this query.

Query: "{query}"

Return ONLY the date in YYYY-MM-DD format, or a date range like "2023-01-01 to 2023-12-31", or natural language like "yesterday", "today".
If a year is mentioned (e.g., "2023"), return it as "entire year 2023".
Return only the extracted date/range, nothing else."""

        elif intent == "item_query":
            extraction_prompt = f"""Extract the food item name and time constraint from this query.

Query: "{query}"

Extract:
1. Item name - use the SINGULAR or ROOT form (e.g., "burger" not "burgers", "chicken" not "chickens")
2. Time constraint if mentioned (e.g., "last 3 months", "last 7 days", "last week")

Item name extraction rules:
- Convert plural to singular (burgers → burger, fries → fry, noodles → noodle)
- Use simple root form for better matching (e.g., "chicken", "nasi goreng", "fried rice", "burger")

Convert time constraints to number of days:
- "last X days" = X days
- "last X weeks" = X * 7 days
- "last X months" = X * 30 days (approximate)
- "last week" = 7 days
- "last month" = 30 days

Return in format: item_name|days
If no time constraint mentioned, return: item_name|none

Examples:
- "fried rice last 3 months" → fried rice|90
- "chicken from last week" → chicken|7
- "all the burgers I have eaten" → burger|none
- "nasi goreng" → nasi goreng|none

Return only in the format described, nothing else."""

        elif intent == "expense_query":
            current_year = datetime.now().year
            extraction_prompt = f"""Extract the date range for expense calculation from this query.

Query: "{query}"

IMPORTANT: Today's current year is {current_year}.

If "this year", "current year", or "year" without specific year is mentioned, use {current_year}.

Return format rules:
- Specific date range: start_date|end_date (format: YYYY-MM-DD|YYYY-MM-DD)
- Specific year mentioned (e.g., "2023"): 2023-01-01|2023-12-31
- "this year" or "current year": {current_year}-01-01|{current_year}-12-31
- Specific month (e.g., "April 2023"): 2023-04-01|2023-04-30
- No date specified: none|none

Return only in the format described, nothing else."""

        else:
            return {"query": query}

        response = self.llm.invoke([HumanMessage(content=extraction_prompt)])
        extracted = response.content.strip()

        if intent == "date_query":
            return {"date": extracted}
        elif intent == "item_query":
            if "|" in extracted:
                parts = extracted.split("|")
                item_name = parts[0].strip()
                days_str = parts[1].strip().lower()
                days = int(days_str) if days_str != "none" else None
                return {"item_name": item_name, "days": days}
            return {"item_name": extracted, "days": None}
        elif intent == "expense_query":
            if "|" in extracted:
                parts = extracted.split("|")
                start_date = parts[0] if parts[0].lower() != "none" else None
                end_date = parts[1] if parts[1].lower() != "none" else None
                return {"start_date": start_date, "end_date": end_date}
            return {"start_date": None, "end_date": None}

        return {"query": query}

    def _retriever_node(self, state: AgentState) -> AgentState:
        """
        Execute appropriate tool based on classified intent

        Args:
            state: Current agent state with intent

        Returns:
            Updated state with tool results
        """
        query = state["query"]
        intent = state["intent"]

        tool_map = {
            "date_query": self.tools[0],
            "item_query": self.tools[1],
            "expense_query": self.tools[2],
            "semantic_query": self.tools[3]
        }

        selected_tool = tool_map.get(intent, self.tools[3])

        try:
            params = self._extract_parameters(query, intent)

            if intent == "date_query":
                results = selected_tool._run(params["date"])
            elif intent == "item_query":
                results = selected_tool._run(params["item_name"], params.get("days"))
            elif intent == "expense_query":
                results = selected_tool._run(params.get("start_date"), params.get("end_date"))
            else:
                results = selected_tool._run(params["query"])

            state["tool_results"] = results
        except Exception as e:
            state["tool_results"] = f"Error executing tool: {str(e)}"

        return state

    def _generator_node(self, state: AgentState) -> AgentState:
        """
        Format tool results into natural language response

        Args:
            state: Current agent state with tool results

        Returns:
            Updated state with final response
        """
        from datetime import datetime

        query = state["query"]
        tool_results = state["tool_results"]
        today = datetime.now().date().isoformat()

        generation_prompt = f"""You are a helpful assistant that answers questions about food receipts.

IMPORTANT: Today's date is {today}.

User Query: "{query}"

Retrieved Data: {tool_results}

Generate a natural, conversational response based on the retrieved data.
- If data is found, present it clearly and concisely
- If no data found, politely inform the user
- Format prices with Rp currency symbol
- Keep response focused and relevant
- Use today's date ({today}) to correctly interpret whether dates are in the past or future

Response:"""

        response = self.llm.invoke([HumanMessage(content=generation_prompt)])
        final_response = response.content.strip()

        state["response"] = final_response
        state["messages"].append(AIMessage(content=final_response))

        return state

    def _route_after_router(self, state: AgentState) -> Literal["retriever"]:
        """Routing logic after router node - always go to retriever"""
        return "retriever"

    def _route_after_retriever(self, state: AgentState) -> Literal["generator"]:
        """Routing logic after retriever node - always go to generator"""
        return "generator"

    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow

        Returns:
            Compiled StateGraph
        """
        workflow = StateGraph(AgentState)

        workflow.add_node("router", self._router_node)
        workflow.add_node("retriever", self._retriever_node)
        workflow.add_node("generator", self._generator_node)

        workflow.set_entry_point("router")

        workflow.add_conditional_edges(
            "router",
            self._route_after_router,
            {"retriever": "retriever"}
        )

        workflow.add_conditional_edges(
            "retriever",
            self._route_after_retriever,
            {"generator": "generator"}
        )

        workflow.add_edge("generator", END)

        return workflow.compile()

    def query(self, user_query: str) -> dict:
        """
        Process a user query through the agent workflow

        Args:
            user_query: Natural language query from user

        Returns:
            Dictionary with query results and response
        """
        initial_state = {
            "query": user_query,
            "intent": "",
            "tool_results": "",
            "response": "",
            "messages": []
        }

        final_state = self.workflow.invoke(initial_state)

        return {
            "query": final_state["query"],
            "intent": final_state["intent"],
            "tool_results": final_state["tool_results"],
            "response": final_state["response"]
        }
