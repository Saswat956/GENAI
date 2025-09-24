from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import tool

# -------------------
# Define Tools
# -------------------
@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Stubbed results about {query}"

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

# -------------------
# Sub-Agents
# -------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Researcher agent: only has web search
researcher = initialize_agent(
    tools=[web_search],
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
)

# Math agent: only has math tool
math_agent = initialize_agent(
    tools=[add],
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
)

# -------------------
# Orchestrator Logic (Manual Example)
# -------------------
def orchestrator(query: str):
    if "calculate" in query.lower() or "add" in query.lower():
        return math_agent.run(query)
    else:
        return researcher.run(query)

# -------------------
# Test Queries
# -------------------
print("User: Tell me about solar panels")
print("Orchestrator:", orchestrator("Tell me about solar panels"))

print("\nUser: Calculate 15 + 7")
print("Orchestrator:", orchestrator("Calculate 15 + 7"))