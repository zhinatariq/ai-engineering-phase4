from typing import TypedDict
from langgraph.graph import StateGraph, END
from groq import Groq
from dotenv import load_dotenv
import os

from main import should_search

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class AgentState(TypedDict):
    question: str
    answer: str


def search_node(state: AgentState) -> AgentState:
    # STUB: proving the graph wiring works before wiring in real RAG
    return {"answer": f"[SEARCH RESULT for: {state['question']}]"}


def direct_node(state: AgentState) -> AgentState:
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "Answer the user's question directly and clearly."
            },
            {
                "role": "user",
                "content": state["question"]
            }
        ]
    )
    return {"answer": response.choices[0].message.content}


def route(state: AgentState) -> str:
    if should_search(state["question"]):
        return "search"
    return "direct"


graph = StateGraph(AgentState)
graph.add_node("search", search_node)
graph.add_node("direct", direct_node)
graph.set_conditional_entry_point(
    route,
    {
        "search": "search",
        "direct": "direct"
    }
)
graph.add_edge("search", END)
graph.add_edge("direct", END)
app = graph.compile()


if __name__ == "__main__":
    result1 = app.invoke({"question": "What is a stack?", "answer": ""})
    print("\n- SEARCH PATH -")
    print(result1["answer"])

    result2 = app.invoke({"question": "What's 2+2?", "answer": ""})
    print("\n- DIRECT PATH -")
    print(result2["answer"])