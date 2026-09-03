# Data Structures Study Agent

An AI agent that extends a Phase 3 RAG study tool with routing, a second real tool (quiz generation), and memory across conversation turns. Instead of sending every question through the same pipeline, the agent decides how to handle each request.

Live discussion / demo: _add link here once deployed_

---

## Architecture

The system is built as a LangGraph agent that decides how to handle each user request instead of sending every question through the same pipeline.

The agent has three possible paths:

```
User Question
     │
     ▼
   route()
     │
     ├── Quiz request ──────► quiz_node
     │
     ├── Course question ───► search_node ──► RAG
     │
     └── Other question ────► direct_node
```

### 1. Quiz

If the user asks for a quiz, test, practice questions, or MCQs, `wants_quiz()` detects the request and routes it to `quiz_node`.

The quiz node retrieves relevant course material from Chroma and asks the LLM to generate exactly three multiple-choice questions grounded in that material.

### 2. Search

For non-quiz questions, the agent uses `should_search()` to decide whether the course documentation is relevant.

If the question is related to Data Structures, the request goes to `search_node`, which uses the RAG pipeline to retrieve relevant documentation and generate a grounded answer.

### 3. Direct

If the question is clearly unrelated to the course documentation, the agent routes it to `direct_node`.

This path answers the question directly with the LLM instead of performing unnecessary document retrieval.

### Why use routing?

The goal is not simply to make an LLM answer questions. The goal is to let the system choose the appropriate way to answer them.

A Data Structures question benefits from retrieval because the course documentation is the authoritative source. A general question such as `What's 2+2?` does not need a vector search. A request such as `Quiz me on linked lists` needs a different behavior entirely.

LangGraph provides the graph structure that connects these decisions to the appropriate processing path, while keeping each behavior isolated in its own node.

### 4. Memory

Memory is implemented as a cross-cutting capability rather than a separate routing path. The caller maintains a `messages` list containing previous user and assistant turns and passes that history into each new graph invocation.

There are two distinct ways this history is used:

**Conversation context for generation:**
`direct_node` and `quiz_node` include the previous message history in their LLM calls before adding the current question. This gives the model the context needed to understand references to earlier parts of the conversation.

```
System prompt
     ↓
Conversation history
     ↓
Current question
     ↓
LLM
```

**Query rewriting for retrieval:**
Conversation history also solves a separate problem in `search_node` and `quiz_node`. A follow-up such as "How is that different from a queue?" is understandable to a conversational model, but it is a poor standalone vector-search query because "that" does not identify the concept being discussed.

Before retrieval, `rewrite_query()` uses the conversation history to convert the follow-up into a self-contained query, such as "difference between a stack and a queue". The rewritten query is then sent to the RAG retriever.

This distinction is important: conversation history helps the LLM understand the conversation, while query rewriting makes follow-up questions suitable for semantic retrieval.

_Note: `quiz_node` uses history for both purposes — rewriting before retrieval and folding history into its generation call. `search_node` only rewrites for retrieval (`ask()` doesn't fold history into generation). `direct_node` only folds history into generation, since it never retrieves. This split reflects what each node actually needs, not an inconsistency._

---

## Bugs Hit and Fixed

Building the agent exposed two problems that were not obvious when each component was tested independently.

### 1. Incorrect routing for course questions

The first routing version relied too heavily on whether the model already knew the answer. When asked:

`What is a stack?`

the routing model could return `DIRECT` because it already knew what a stack was. That was the wrong behavior for this system: the course documentation is supposed to be the authoritative source for Data Structures questions, even when the model already knows the answer from general knowledge.

The routing prompt was changed to make this distinction explicit: if a question is related to a topic that may be covered by the course material, choose `SEARCH` regardless of whether the model already knows the answer.

After the change, course questions such as `What is a stack?` consistently route through the RAG path, while unrelated questions such as `What's 2+2?` remain on the direct path.

### 2. Follow-up questions were poor retrieval queries

After adding conversation history, another problem appeared. A follow-up such as:

`How is that different from a queue?`

can be understood by an LLM because the previous conversation establishes what "that" refers to. But sending the raw question directly to vector search is different: the retrieval system has no conversational understanding of what "that" means.

The solution was `rewrite_query()`. Before retrieval, it uses the conversation history to transform an ambiguous follow-up into a standalone query, for example:

`difference between a stack and a queue`

That rewritten query is then used for retrieval. Clear standalone questions are left unchanged.

This separated two responsibilities that initially looked like one problem: conversation history provides context for the model, while query rewriting provides a retrieval-friendly representation of that context.

---

## Multi-Turn Proof

A real conversation, not isolated function tests, confirms memory and query rewriting work together:

**Turn 1:** `What is a stack?`
Routed to `search`. Answered correctly from `chapter1.pdf` — LIFO principle, push/pop/top operations.

**Turn 2:** `How is that different from a queue?`
Rewritten by `rewrite_query()` to `difference between a stack and a queue` before retrieval. Routed to `search`. Answered with a grounded stack-vs-queue comparison, correctly sourced from the course documentation, even though the raw question never mentioned "stack" explicitly.

This confirms the agent can hold a real conversation, not just answer isolated one-shot questions.

---

## Tech Stack

- **Python** — core language
- **LangGraph** — agent graph: routing, nodes, state
- **Groq** (`openai/gpt-oss-120b`) — LLM calls for routing, generation, quiz creation, and query rewriting; `temperature=0` throughout for deterministic output
- **ChromaDB** — vector store for course document chunks
- **sentence-transformers** (`all-MiniLM-L6-v2`) — local, free embeddings
- **pypdf** — PDF text extraction
- **langchain-text-splitters** — document chunking
- **python-dotenv** — environment variable management

---

## Project Structure

```
ai-engineering-phase4/
├── main.py       # should_search() — LLM-based routing decision (SEARCH vs DIRECT)
├── rag.py        # Document loading, chunking, embedding, Chroma storage,
│                 # retrieve() and ask() — the RAG pipeline shared by search_node
│                 # and quiz_node
├── graph.py      # LangGraph agent: AgentState, all three nodes (search_node,
│                 # direct_node, quiz_node), rewrite_query(), routing (wants_quiz,
│                 # should_search, route), graph construction, manual tests
├── docs/         # Source course PDFs (chapter1.pdf, chapter2.pdf)
├── chroma_db/    # Generated vector store (gitignored)
├── .env          # GROQ_API_KEY (gitignored)
└── requirements.txt
```

---

## How to Run

1. Clone the repo and create a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Add a `.env` file with your Groq API key:
   ```
   GROQ_API_KEY=your_key_here
   ```

4. Run the agent's manual tests (loads PDFs, builds the Chroma store, runs rewrite tests and a real multi-turn conversation):
   ```
   python graph.py
   ```

_Note: this currently runs as a script with hardcoded test questions in `graph.py`. There is no interactive CLI or UI yet — that's the next planned addition._
