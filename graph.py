from typing import TypedDict
import json
import os

from langgraph.graph import StateGraph, END
from groq import Groq
from dotenv import load_dotenv

from main import should_search
from rag import ask, retrieve


load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class AgentState(TypedDict):
    question: str
    answer: str


# Search tool

def search_node(state: AgentState) -> AgentState:
    return {
        "answer": ask(state["question"])
    }


# Direct answer

def direct_node(state: AgentState) -> AgentState:
    try:
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

        return {
            "answer": response.choices[0].message.content
        }

    except Exception as e:
        print(f"direct_node error: {e}")

        return {
            "answer": "Sorry, I couldn't process your request right now. Please try again."
        }


# Quiz tool

def quiz_node(state: AgentState) -> AgentState:
    documents, metadatas = retrieve(state["question"])

    context_parts = []

    for document, metadata in zip(documents, metadatas):
        context_parts.append(
            f"Source: {metadata['source']}\n"
            f"{document}"
        )

    context = "\n\n---\n\n".join(context_parts)

    system_prompt = """You are a Data Structures quiz generator.

Create exactly 3 multiple-choice questions based ONLY on the
provided course documentation.

Return ONLY valid JSON with exactly this structure:

{
  "questions": [
    {
      "question": "Question text",
      "options": [
        "Option 1",
        "Option 2",
        "Option 3",
        "Option 4"
      ],
      "correct_index": 0
    }
  ]
}

Rules:
- Generate exactly 3 questions.
- Each question must have exactly 4 options.
- correct_index must be an integer from 0 to 3.
- The correct answer must be supported by the provided documentation.
- Do not use outside knowledge.
- Do not include markdown or code fences.
- Return JSON only.

Course documentation:

""" + context

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": state["question"],
                },
            ],
        )

    except Exception as e:
        print(f"quiz_node API error: {e}")

        return {
            "answer": "Sorry, I couldn't process your request right now. Please try again."
        }

    quiz_output = response.choices[0].message.content

    try:
        json.loads(quiz_output)

        return {
            "answer": quiz_output
        }

    except json.JSONDecodeError as e:
        print(f"quiz_node JSON error: {e}")

        return {
            "answer": "Quiz generation failed: the model returned invalid JSON."
        }


# Routing

def wants_quiz(question: str) -> bool:
    quiz_words = [
        "quiz",
        "test me",
        "practice questions",
        "mcq"
    ]

    question = question.lower()

    return any(word in question for word in quiz_words)


def route(state: AgentState) -> str:
    question = state["question"]

    if wants_quiz(question):
        return "quiz"

    if should_search(question):
        return "search"

    return "direct"


# Build graph

graph = StateGraph(AgentState)

graph.add_node("search", search_node)
graph.add_node("quiz", quiz_node)
graph.add_node("direct", direct_node)

graph.set_conditional_entry_point(
    route,
    {
        "search": "search",
        "quiz": "quiz",
        "direct": "direct"
    }
)

graph.add_edge("search", END)
graph.add_edge("quiz", END)
graph.add_edge("direct", END)


app = graph.compile()


# Manual tests — repeat run to confirm no crashes across multiple inputs/rounds

if __name__ == "__main__":
    test_inputs = [
        "What is a stack?",
        "What's 2+2?",
        "Quiz me on linked lists",
    ]

    for round_num in range(3):
        print(f"\n=== ROUND {round_num + 1} ===")
        for question in test_inputs:
            result = app.invoke({"question": question, "answer": ""})
            print(f"\nQ: {question}")
            print(result["answer"])