# Local Agentic RAG System

> **Full-Stack Autonomous AI Agent** built with **LangGraph**, **LangChain**, **Ollama (Qwen 2.5 7B)**, and **Flask**.

---

## Features

* **Agentic RAG Architecture:** Autonomous decision loop using LangGraph to dynamically evaluate when to retrieve data vs. generate responses.
* **100% Local & Private:** Ingests and queries personal documents (`portfolio.pdf`) locally without sending data to third-party APIs.
* **LangGraph Execution:** State Graph implementation utilizing conditional routing, message reducers (`add_messages`), and tool-binding patterns.
* **Full-Stack Implementation:** REST API powered by Flask paired with an interactive real-time Web UI.

---

## 🛠️ Architecture Overview

```text
[User Prompt] ──► [Flask API] ──► [LangGraph StateGraph]
                                         │
                                  (Decide Action)
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
            [Tool Needed?]                            [Direct Response]
                    │                                         │
        (search_portfolio)                                    │
                    │                                         │
            [Load PDF / RAG]                                   │
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         ▼
                                [Final Answer Engine]
