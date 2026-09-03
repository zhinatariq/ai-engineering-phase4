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
    messages: list[dict]


# Query rewriting

def rewrite_query(messages: list[dict], question: str) -> str:
    prompt = """You rewrite user questions into standalone search queries.

Use the conversation history only when necessary to understand what the
user is referring to.

Rules:
- If the current question is already standalone and clear, return it unchanged.
- If it is a follow-up or contains references like "it", "that", "what we discussed",
  rewrite it into a self-contained search query using the relevant conversation context.
- Preserve the user's intended meaning.
- Do not add information that is not supported by the conversation.
- Return ONLY the standalone search query.
- Do not include explanations, quotes, or labels.

Conversation history:
""" + "\n".join(
        f"{message['role']}: {message['content']}"
        for message in messages
    ) + f"""

Current question:
{question}
"""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": prompt
                }
            ]
        )

        rewritten = response.choices[0].message.content.strip()

        if not rewritten:
            return question

        return rewritten

    except Exception as e:
        print(f"rewrite_query error: {e}")
        return question


# Search tool

def search_node(state: AgentState) -> AgentState:
    rewritten_query = rewrite_query(
        state["messages"],
        state["question"]
    )

    return {
        "answer": ask(rewritten_query)
    }


# Direct answer

def direct_node(state: AgentState) -> AgentState:

    messages = [
        {
            "role": "system",
            "content": "Answer the user's question directly and clearly."
        }
    ]

    messages.extend(state["messages"])

    messages.append({
        "role": "user",
        "content": state["question"]
    })

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            temperature=0,
            messages=messages
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

    rewritten_query = rewrite_query(
        state["messages"],
        state["question"]
    )

    documents, metadatas = retrieve(rewritten_query)

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

You also have prior conversation context.

Use prior conversation context only to understand what the user
is referring to. Ground all factual quiz content in the provided
course documentation.

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

    quiz_messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    quiz_messages.extend(state["messages"])

    quiz_messages.append({
        "role": "user",
        "content": state["question"],
    })

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            temperature=0,
            messages=quiz_messages,
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


# Manual tests

if __name__ == "__main__":

    print("\n=== REWRITE TEST 1: STANDALONE ===")

    result1 = rewrite_query(
        messages=[],
        question="What is a stack?"
    )

    print("Original:", "What is a stack?")
    print("Rewritten:", result1)


    print("\n=== REWRITE TEST 2: FOLLOW-UP ===")

    fake_history = [
        {
            "role": "user",
            "content": "What is a stack?"
        },
        {
            "role": "assistant",
            "content": "A stack is a LIFO data structure..."
        }
    ]

    result2 = rewrite_query(
        messages=fake_history,
        question="how is that different from a queue?"
    )

    print("Original:", "how is that different from a queue?")
    print("Rewritten:", result2)


    print("\n=== REAL MULTI-TURN TEST ===")

    messages = []

    def run_turn(question):
        global messages
        result = app.invoke({
            "question": question,
            "answer": "",
            "messages": messages
        })
        print(f"\nQ: {question}")
        print(result["answer"])
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": result["answer"]})

    run_turn("What is a stack?")
    run_turn("How is that different from a queue?")