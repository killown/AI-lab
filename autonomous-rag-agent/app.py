"""Flask application integrating LangChain and LangGraph with local PDF document retrieval for portfolio querying."""

import os
from typing import Annotated, Literal
from typing_extensions import TypedDict

from flask import Flask, render_template, request, jsonify

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

app = Flask(__name__)


TEMPERATURE = 0.1
LOCAL_MODEL = "qwen2.5:7b"
HELPFUL_PROMPT = "You are a helpful assistant. "
SEARCH_PORTFOLIO_PROMPT = """You have access to a local portfolio.pdf document located in the data/ directory.
                          Use this document to answer questions about Thiago Lucio's skills, experience, and projects."""


@tool
def search_portfolio(query: str) -> str:
    """Searches the local portfolio.pdf document located in the data/ directory for information regarding Thiago Lucio.
    Use this tool whenever the user asks about Thiago Lucio, his portfolio, skills, experience, or projects.
    """
    pdf_path = os.path.join("data", "portfolio.pdf")

    if not os.path.exists(pdf_path):
        return "Portfolio file not found in data/ directory."

    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        full_text = "\n".join([page.page_content for page in pages])
        return full_text
    except Exception as e:
        return f"Error reading portfolio PDF: {str(e)}"


tools = [search_portfolio]
tools_by_name = {tool.name: tool for tool in tools}

llm = ChatOllama(model=LOCAL_MODEL, temperature=TEMPERATURE)
llm_with_tools = llm.bind_tools(tools)


class AgentState(TypedDict):
    """Represents the state of the graph containing the message history."""

    messages: Annotated[list, add_messages]


def agent_node(state: AgentState) -> dict:
    """Invokes the language model to determine the next response or tool call."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def tool_node(state: AgentState) -> dict:
    """Executes requested tools and appends their outputs to the state."""
    last_message = state["messages"][-1]
    results = []

    for tool_call in last_message.tool_calls:
        tool_obj = tools_by_name[tool_call["name"]]
        observation = tool_obj.invoke(tool_call["args"])
        results.append(
            {
                "role": "tool",
                "content": str(observation),
                "tool_call_id": tool_call["id"],
            }
        )

    return {"messages": results}


def decide_next_step(state: AgentState) -> Literal["tools", "__end__"]:
    """Determines whether to route execution to tool processing or to end the graph."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent", decide_next_step, {"tools": "tools", "__end__": END}
)
builder.add_edge("tools", "agent")

graph = builder.compile()


@app.route("/")
def home():
    """Renders the main chat interface."""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """Handles incoming user messages, executes the agent graph, and returns the response."""
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    system_prompt = f"{HELPFUL_PROMPT}{SEARCH_PORTFOLIO_PROMPT}"

    initial_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    result = graph.invoke({"messages": initial_messages})
    final_response = result["messages"][-1].content

    return jsonify({"response": final_response})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
