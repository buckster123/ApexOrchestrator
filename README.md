# Apex Orchestrator: AI-Powered Chat Agent for Raspberry Pi 5



[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)  
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)  
[![xAI Powered](https://img.shields.io/badge/Powered%20by-xAI-000000?style=flat&logo=groq&logoColor=white)](https://x.ai/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
[![Stars](https://img.shields.io/github/stars/yourusername/apex-orchestrator?style=social)](https://github.com/yourusername/apex-orchestrator)  

![Apex Orchestrator Banner](https://github.com/buckster123/ApexOrchestrator/blob/main/apex_logo.png)  

🚀 **Apex Orchestrator** is a lightweight, self-contained AI chat agent optimized for single-user setups on low-cost hardware like the Raspberry Pi 5. Powered by xAI's Grok models, it delivers secure authentication, persistent chat history, and a suite of sandboxed tools for file management, code execution, web search, and advanced memory—all in a sleek, dark-mode Streamlit UI. Ideal for hobbyists, tinkerers, or devs seeking an open-source AI sidekick without cloud dependencies or high overhead.

## 🌟 Features
- **Secure Auth & History**: Hashed passwords, auto-saving chats with load/delete options.
- **Custom Prompts**: Load from `./prompts/`—focus on "big-apex.txt" (pseudo-Python structured for efficiency) as the main agent persona. Built-in defaults act as backups for robustness.
- **Sandboxed Tools**: File I/O, code REPL, Git, shell, linting, API mocks, web search—capped at 5 iterations to prevent loops.
- **Smart Memory**: SQLite + vector embeddings (ChromaDB) with lazy loading and salience-based pruning.
- **Multimodal**: Image uploads for vision queries.
- **Pi-Optimized**: Caching, fallbacks, and minimal compute—runs smooth on Pi 5 (test on 2GB+ RAM).
- **UI Perks**: Gradient themes, sidebar controls, streaming responses.

| Feature | Description | Pi-Friendly? |
|---------|-------------|--------------|
| **Tools** | File ops, code exec, web search (LangSearch), Git, etc. | Yes—sandboxed & lightweight. |
| **Memory** | Semantic search with embeddings. | Lazy-loaded; fallback if heavy. |
| **Prompts** | Big-Apex as core; builtins as backups. | Efficient ingestion for speed. |

## 📊 System Diagrams

### 1. High-Level App Flow (Python Logic)
This shows the script's main execution path—from startup to chat interactions.

```mermaid
graph TD
    A[Start: Load Env & DB Setup] --> B{Logged In?}
    B -->|No| C[Login/Register Page]
    C --> D[Auth Success: Set Session]
    B -->|Yes| E[Chat Page: Sidebar + Main UI]
    E --> F[Select Model/Prompt/Tools]
    F --> G[User Input: Chat Prompt + Images]
    G --> H[API Call: xAI with Tools]
    H --> I[Tool Dispatch Loop (Max 5 Iterations)]
    I -->|Tool Called| J[Execute: e.g., fs_write_file, code_execution]
    J --> K[Return Results to API]
    H --> L[Stream Response to UI]
    L --> M[Save History to DB]
    M --> N[Loop: Next User Input]
```

### 2. Agent Workflow

```mermaid
graph TD
    Start[User Query] --> Init[Task Initialization: Parse & Decompose (CoT/ToT)]
    Init --> Plan[Generate Plans: Quick/Deep/Balanced -> Select Best]
    Plan --> Assign[Assign Subtasks to Subagents (Up to 5)]
    Assign --> Exec[Subtask Execution: ReAct Loops per Subagent]
    Exec -->|Retriever: Gather Data| R[Tools: Memory Retrieve / Web Search / FS Read]
    Exec -->|Reasoner: Analyze| S[Tools: Code Exec / DB Query / Git Ops]
    Exec -->|Generator: Synthesize| T[Tools: FS Write / Code Lint]
    Exec -->|Validator: Verify (Optional)| U[Tools: Memory Retrieve / Fact-Check]
    Exec -->|Optimizer: Refine (Optional)| V[Tools: Memory Prune / Cleanup]
    Exec --> Aggregate[Aggregation: Merge Outputs (Weighted by Confidence)]
    Aggregate --> Reflect[Global ReAct: Assess & Iterate (Max 5 Cycles)]
    Reflect -->|Done| Final[Finalization: Polish Output + Cleanup]
    Final --> End[User Response]
```

### 3. Tool Dispatching Logic
How tools are handled in the API loop—key for stability.

```mermaid
flowchart TD
    A[API Response with Tool Calls] --> B{Has Tools?}
    B -->|No| C[Stream Final Response]
    B -->|Yes| D[Begin Transaction (DB Begin)]
    D --> E[Loop Over Tool Calls]
    E --> F[Parse Args & Dispatch via TOOL_DISPATCHER]
    F -->|e.g., Memory Tools| G[Add User/Convo ID]
    F -->|Other Tools| H[Execute: Sandboxed Path Check]
    H --> I[Cache/Handle Errors]
    I --> J[Yield Partial Result (First 200 Chars)]
    J --> K[Collect Outputs]
    K --> L[Commit DB & Append to Messages]
    L --> M{Iterations < 5?}
    M -->|Yes| A
    M -->|No| N[Abort: Error Message]
    N --> C
```

## 🛠️ Installation
1. **Clone the Repo**:
   ```
   git clone https://github.com/yourusername/apex-orchestrator.git
   cd apex-orchestrator
   ```

2. **Set Up Environment**:
   - Virtual env: `python -m venv .venv && source .venv/bin/activate`
   - Dependencies: `pip install -r requirements.txt`
   - `.env` Setup:
     ```
     XAI_API_KEY=your_xai_key_here
     LANGSEARCH_API_KEY=your_langsearch_key_here  # For web search
     EMBED_MODEL_LIGHT=True  # Optional: Use slimmer model for Pi perf
     ```

3. **Run**:
   ```
   streamlit run app.py
   ```
   Access: `http://localhost:8501` (or Pi IP for LAN).

**Pi 5 Optimization**: Fan-cool the Pi. If embeddings lag, enable `EMBED_MODEL_LIGHT` in .env for a faster model swap.

## 📖 Usage
1. **Auth**: Login or register—secure and simple.
2. **Chat**:
   - Sidebar: Pick model, prompt (load from `./prompts/`), enable tools.
   - Main: Type queries, upload images—watch streaming magic.
3. **Prompts Focus**: Use "big-apex.txt" (pseudo-Python structured) as your go-to for advanced agent workflows. Built-ins (e.g., "tools-enabled.txt") are backups—auto-created if `./prompts/` is empty or files are lost, ensuring the app always works out-of-box.
4. **Tools**: Toggle on for power—e.g., "Lint this Python code" triggers `code_lint`.
5. **Customization**: Drop custom prompts in `./prompts/` (e.g., copy our pseudo-Python class into "big-apex.txt").

**Demo**:  
*(Add a GIF here: e.g., Pi terminal running the app, chatting with tools.)*

## 🔧 Requirements (requirements.txt)
```
streamlit
openai
passlib
python-dotenv
ntplib
pygit2
requests
black
numpy
sentence-transformers
torch
jsbeautifier
pyyaml
sqlparse
beautifulsoup4
chromadb
```
*(Full list from `pip freeze`; some linters need system tools like clang-format.)*

## 🤝 Contributing
Fork, tweak, PR! Focus on Pi perf or new tools.
- **Issues/PRs**: Welcome for bugs, features.
- **Style**: Black-formatted Python.
- **Tests**: Add pytest for tools.

## 📄 License
MIT—fork away! See [LICENSE](LICENSE).

## 🙌 Acknowledgments
- xAI/Grok for the brains.
- Streamlit for the polish.
- André's experiments: Pseudo-Python prompts for that extra snap!

Star if it sparks joy—let's evolve AI on a budget! 🚀
