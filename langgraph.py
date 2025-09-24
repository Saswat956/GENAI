from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool

# -------------------
# 1) Shared State
# -------------------
class AgentState(TypedDict):
    messages: List[str]   # conversation log

# -------------------
# 2) Azure OpenAI Setup
# -------------------
llm = AzureChatOpenAI(
    deployment_name="gpt-4o-mini",   # replace with your Azure deployment
    temperature=0,
    api_version="2024-06-01"         # match your Azure API version
)

# -------------------
# 3) Define Tools
# -------------------
@tool
def web_search(query: str) -> str:
    """Simulate a web search and return some info."""
    if "solar" in query.lower():
        return "Solar panels convert sunlight into electricity using photovoltaic cells."
    return f"Stubbed search results for: {query}"

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

# -------------------
# 4) Create Agents
# -------------------
researcher = create_react_agent(llm, [web_search])   # Researcher has web search
math_agent = create_react_agent(llm, [add])          # Math agent has math tool

# -------------------
# 5) Build Graph
# -------------------
graph = StateGraph(AgentState)

graph.add_node("researcher", researcher)
graph.add_node("math_agent", math_agent)

# Flow: START -> researcher -> math_agent -> END
graph.add_edge(START, "researcher")
graph.add_edge("researcher", "math_agent")
graph.add_edge("math_agent", END)

app = graph.compile()

# -------------------
# 6) Run Multi-Agent Workflow
# -------------------
query = "Research solar panels and then calculate 12 + 8."
result = app.invoke({"messages": [query]})

print("\nFinal Collaborative Answer:\n", result["messages"][-1])