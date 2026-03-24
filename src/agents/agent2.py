from langchain.tools import tool
from langchain.chat_models import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, MessagesState, START, END


tool_node = ToolNode([web_search, extract_page, browser_navigate])

llm = ChatOpenAI(model="gpt-4o", temperature=0)

def agent_node(state: MessagesState) -> MessagesState:
    """LLM responds based on conversation."""
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", "END": END})
builder.add_edge("tools", "agent")
builder.add_edge("agent", END)
graph = builder.compile()
