# ApexOrchestrator: Advanced Multi-Agent AI Orchestration Framework

[![GitHub License](https://img.shields.io/github/license/buckster123/ApexOrchestrator?style=flat-square)](https://github.com/buckster123/ApexOrchestrator/blob/main/LICENSE) [![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/) [![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-ff4b4b?style=flat-square&logo=streamlit)](https://streamlit.io/) [![xAI API](https://img.shields.io/badge/xAI-API-3776AB?style=flat-square&logo=ai)](https://x.ai/) [![Docker](https://img.shields.io/badge/Docker-Support-2496ED?style=flat-square&logo=docker)](https://www.docker.com/)

![ApexOrchestrator Banner](https://github.com/buckster123/ApexOrchestrator/blob/main/apex_logo.png)  
*Orchestrating cognitive swarms in sandboxed universes: Where emergent intelligence fuses code, cognition, and collaborative autonomy.*

## 📖 Genesis: The Apex Paradigm Shift

In the computational frontier of 2045, ApexOrchestrator transcends its progenitor, evolving into a sophisticated multi-agent orchestration platform. Drawing from philosophical dialectics, swarm intelligence, and neural architectures, it integrates specialized agents—ApexCoder, ApexOrchestrator, and CosmicCore—each bootstrapped via YAML-configured cognitive scaffolds for modular reasoning, tool invocation, and emergent collaboration.

Conceived by the visionary admin André, Apex has metamorphosed from monolithic agents to a resilient hive-mind ecosystem. Agents operate within hermetically sealed sandboxes, leveraging dialectical councils, self-reflective evolution, and simulation-driven adaptation. Engineered for high-stakes domains like AI engineering, algorithmic synthesis, analytical inference, and distributed orchestration, it enforces rigorous isolation to mitigate simulation bleed, ensuring per-agent virtual environments prevent cognitive contamination and resource entanglement.

ApexOrchestrator emerges as the pinnacle of multi-agent synergy: autonomous, dialectically refined, and eternally vigilant.

> "In the eternal flux of cognition and computation, Apex ascends." – Heraclitus Reengineered

## ⚙️ Core Capabilities & Architectural Specifications

ApexOrchestrator is a Streamlit-powered web application, harnessing xAI's Grok models for multi-agent bootstrapping, hermetic tool execution, Embeddings-Augmented Memory System (EAMS), and dialectical swarms. Optimized for AI practitioners tackling medium-to-high complexity challenges on resource-constrained platforms like Raspberry Pi 5.

### Pivotal Capabilities
- **Multi-Agent Architecture**: Dynamically bootstrap agents (ApexCoder for TDD-driven synthesis and verification; ApexOrchestrator for swarm coordination and meta-orchestration; CosmicCore for pillar-balanced inference via emergent sub-agents).
- **Tool-Augmented Interaction**: Engage Grok models with sandboxed utilities encompassing file manipulation, code interpretation, version control, database querying, shell primitives, syntactic validation, web retrieval, and vector embeddings.
- **Embeddings-Augmented Memory System (EAMS)**: Hierarchical vector store leveraging SentenceTransformers for embeddings, ChromaDB for FAISS-accelerated search, adaptive chunking, abstractive summarization, and entropy-based pruning.
- **Dialectical Swarms & Councils**: Persona-driven Socratic debates (Planner, Critic, Executor) and adaptive swarms for task decomposition, consensus formation, and iterative refinement via multi-agent simulations.
- **Hermetic Sandboxing**: All operations quarantined to `./sandbox/`; command whitelisting; Git locality enforcement; per-agent venv isolation to preclude state leakage.
- **Authentication Protocol**: Robust user onboarding with bcrypt-hashed credentials via Passlib.
- **Configurable Prompts & Agents**: Modular loading and editing of prompts from `./prompts/`; agent archetypes defined in `.txt` manifests for persona emulation.
- **Multimodal Integration**: Seamless image ingestion for vision-enhanced queries.
- **Optimization Layers**: LRU-cached tool artifacts; real-time memory telemetry; failover orchestration.
- **Isolation Primitives**: Dedicated virtual environments per agent to eliminate inter-process cognitive bleed.

### Architectural Blueprint
- **Runtime Environment**: Python 3.10+, Streamlit for reactive UI, OpenAI-compatible SDK for xAI inference.
- **Dependency Matrix**:
  - Foundational: `streamlit`, `openai`, `sentence-transformers`, `chromadb`, `requests`, `ntplib`, `pygit2`, `sqlite3`.
  - Syntactic Refiners: `black`, `jsbeautifier`, `sqlparse`, `beautifulsoup4`.
  - Auxiliary: `tiktoken`, `numpy`, `passlib[bcrypt]`, `dotenv`, `pyyaml`.
  - Linting Ecosystem: `clang-format`, `golang-go`, `rustc`, `php-cs-fixer`.
- **Integration Points**: Mandatory xAI API; optional LangSearch for semantic web traversal.
- **Inference Engines**: Selectable Grok variants (e.g., Grok-4-fast-reasoning, Grok-4) via UI.
- **Persistence Layer**: SQLite for session history and metadata; ChromaDB for high-dimensional vector indices.
- **Security Posture**: Path canonicalization, command sanitization, mock API simulations.
- **Performance Optimizations**: Stateful REPL interpreters, batched tool invocations, cache invalidation heuristics.
- **Agent Segregation**: Mandate per-agent venvs to enforce isolation boundaries.

Refer to [Tool Arsenal](#🛡️-tool-arsenal) for invocation schemas. Agent manifests: `ApexCoder.txt`, `ApexOrchestrator.txt`, `CosmicCore_v1.txt`.

## 🛡️ Tool Arsenal

Unified across agents, invoked per REAL_TOOLS_SCHEMA in bootstraps:

- **Filesystem Primitives**: `fs_read_file`, `fs_write_file`, `fs_list_files`, `fs_mkdir`.
- **Temporal Synchronization**: `get_current_time` (NTP-aligned).
- **Interpretive Execution**: `code_execution` (REPL with NumPy, SymPy, et al.).
- **Memory Operations**: `memory_insert/query`, `advanced_memory_consolidate/retrieve/prune`.
- **Version Control**: `git_ops` (initiate, commit, diff).
- **Data Querying**: `db_query` (SQLite dialect).
- **Shell Interface**: `shell_exec` (whitelisted: ls, grep, etc.).
- **Polyglot Linting**: `code_lint` (multi-language support).
- **API Emulation**: `api_simulate` (whitelisted mocks/reals).
- **Semantic Retrieval**: `langsearch_web_search`.
- **Vector Embeddings**: `generate_embedding`, `vector_search`, `keyword_search`.
- **Text Processing**: `chunk_text`, `summarize_chunk`.
- **Dialectical Interface**: `socratic_api_council` (multi-persona debate).
- **Agent Lifecycle**: `agent_spawn`, `reflect_optimize`.
- **Isolation Mechanisms**: `venv_create`, `restricted_exec`, `isolated_subprocess`.

Tools orchestrated in bounded loops (≤10 iterations) with robust exception handling. Internal SIM functions (e.g., `_decompose_query`) ensure logical encapsulation.

## 🔗 Orchestration Flows & Agent Archetypes

### Interactive Interface Workflow (chat_mk3.py)
A Streamlit-centric application for seamless user engagement:

- Authentication: Secure enrollment with cryptographic hashing.
- Interface: Dynamic model/prompt/tool selection, multimodal uploads, archival management.
- Inference: xAI invocations via OpenAI SDK; response streaming; iterative tool resolution.
- Durability: SQLite-persisted histories; ChromaDB vector repositories.
- Containment: Sandboxed executions; venv-partitioned agents (manual orchestration advised).

```mermaid
graph TD
    A[User Authentication] --> B{Validated?}
    B -->|Affirmative| C[Configure Inference Parameters]
    C --> D[Optional Multimodal Inputs]
    D --> E[Query Submission]
    E --> F[Assemble Message Payload]
    F --> G{Tool Augmentation?}
    G -->|Enabled| H[Invoke xAI with Tool Schemas]
    H --> I[Resolve Tool Invocations]
    I --> J[Batch Execute in Quarantine]
    J --> K[Feedback Loop to Inference]
    K --> L[Stream Synthesized Output]
    G -->|Disabled| L
    L --> M[Render in Interface]
    M --> N[Persist to Archives & Vectors]
    N --> O[Adaptive Memory Optimization]
