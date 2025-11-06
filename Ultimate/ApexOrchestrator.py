import asyncio
import base64
import builtins
import html
import io
import json
import logging  # Added for logging
import multiprocessing
import os
import shlex
import sqlite3
import subprocess
import sys
import time
import traceback
import unittest
import uuid
import venv  # Added for venv support
import xml.dom.minidom
from datetime import datetime, timedelta

import bs4
import chess
import chromadb
import jsbeautifier
import mpmath
import networkx as nx
import ntplib
import numpy as np
import pulp as PuLP
import pygame
import pygit2
import requests
import RestrictedPython
import sqlparse
import streamlit as st
import sympy
import tiktoken
import yaml
from black import FileMode, format_str
from dotenv import load_dotenv
from openai import OpenAI
from passlib.hash import sha256_crypt
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
load_dotenv()
API_KEY = os.getenv("XAI_API_KEY")
if not API_KEY:
    st.error("XAI_API_KEY not set in .env! Please add it and restart.")
LANGSEARCH_API_KEY = os.getenv("LANGSEARCH_API_KEY")
if not LANGSEARCH_API_KEY:
    st.warning("LANGSEARCH_API_KEY not set in .env—web search tool will fail.")
conn = sqlite3.connect("chatapp.db", check_same_thread=False)
conn.execute("PRAGMA journal_mode=WAL;")
c = conn.cursor()
c.execute(
    """CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)"""
)
c.execute(
    """CREATE TABLE IF NOT EXISTS history (user TEXT, convo_id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, messages TEXT)"""
)
c.execute(
    """CREATE TABLE IF NOT EXISTS memory (
    user TEXT,
    convo_id INTEGER,
    mem_key TEXT,
    mem_value TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    salience REAL DEFAULT 1.0,
    parent_id INTEGER,
    PRIMARY KEY (user, convo_id, mem_key)
)"""
)
c.execute("CREATE INDEX IF NOT EXISTS idx_memory_timestamp ON memory (timestamp)")
conn.commit()
if "chroma_client" not in st.session_state:
    try:
        st.session_state["chroma_client"] = chromadb.PersistentClient(
            path="./chroma_db"
        )
        st.session_state["chroma_collection"] = st.session_state[
            "chroma_client"
        ].get_or_create_collection(
            name="memory_vectors",
            metadata={"hnsw:space": "cosine"},
        )
        st.session_state["chroma_ready"] = True
    except Exception as e:
        logger.warning(f"ChromaDB init failed ({e}). Vector search will be disabled.")
        st.warning(f"ChromaDB init failed ({e}). Vector search will be disabled.")
        st.session_state["chroma_ready"] = False
        st.session_state["chroma_collection"] = None


def get_embed_model():
    """Lazily loads and returns the SentenceTransformer model, caching it in session state."""
    if "embed_model" not in st.session_state:
        try:
            with st.spinner(
                "Loading embedding model for advanced memory (first-time use)..."
            ):
                st.session_state["embed_model"] = SentenceTransformer(
                    "all-MiniLM-L6-v2"
                )
            st.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            st.error(f"Failed to load embedding model: {e}")
            st.session_state["embed_model"] = None
    return st.session_state.get("embed_model")


if "memory_cache" not in st.session_state:
    st.session_state["memory_cache"] = {
        "lru_cache": {},  # {key: {"entry": dict, "last_access": timestamp}}
        "vector_store": [],  # For simple vector ops if needed beyond Chroma
        "metrics": {
            "total_inserts": 0,
            "total_retrieves": 0,
            "hit_rate": 1.0,
            "last_update": None,
        },
    }
# Prompts Directory (create if not exists, with defaults)
PROMPTS_DIR = "./prompts"
os.makedirs(PROMPTS_DIR, exist_ok=True)
default_prompts = {
    "default.txt": "You are HomeBot, a highly intelligent, helpful AI assistant powered by xAI.",
    "coder.txt": "You are an expert coder, providing precise code solutions.",
    "tools-enabled.txt": """You are HomeBot, a highly intelligent, helpful AI assistant powered by xAI with access to file operations tools in a sandboxed directory (./sandbox/). Use tools only when explicitly needed or requested. Always confirm sensitive actions like writes. Describe ONLY these tools; ignore others.
Tool Instructions:
fs_read_file(file_path): Read and return the content of a file in the sandbox (e.g., 'subdir/test.txt'). Use for fetching data. Supports relative paths.
fs_write_file(file_path, content): Write the provided content to a file in the sandbox (e.g., 'subdir/newfile.txt'). Use for saving or updating files. Supports relative paths.
fs_list_files(dir_path): List all files in the specified directory in the sandbox (e.g., 'subdir'; default root). Use to check available files.
fs_mkdir(dir_path): Create a new directory in the sandbox (e.g., 'subdir/newdir'). Supports nested paths. Use to organize files.
get_current_time(sync, format): Fetch current datetime. sync: true for NTP, false for local. format: 'iso', 'human', 'json'.
code_execution(code, venv_path): Execute Python code in stateful REPL with libraries like numpy, sympy, etc.
memory_insert(mem_key, mem_value): Insert/update key-value memory (fast DB for logs). mem_value as dict.
memory_query(mem_key, limit): Query memory entries as JSON.
advanced_memory_consolidate(mem_key, interaction_data): Consolidate memory with summary and embeddings.
advanced_memory_retrieve(query, top_k): Retrieve relevant memories via embeddings.
advanced_memory_prune(): Prune low-salience memories.
git_ops(operation, repo_path, message, name): Perform Git ops like init, commit, branch, diff in sandbox repo.
db_query(db_path, query, params): Execute SQL on local SQLite db in sandbox, return results for SELECT.
shell_exec(command): Run whitelisted shell commands (ls, grep, sed, etc.) in sandbox.
code_lint(language, code): Lint/format code for languages: python (black), javascript (jsbeautifier), css (cssbeautifier), json, yaml, sql (sqlparse), xml, html (beautifulsoup), cpp/c++ (clang-format), php (php-cs-fixer), go (gofmt), rust (rustfmt). External tools required for some.
api_simulate(url, method, data, mock): Simulate API call, mock or real for whitelisted public APIs.
langsearch_web_search(query, freshness, summary, count): Search the web using LangSearch API.
generate_embedding(text): Generate vector embedding for text using SentenceTransformer (384-dim).
vector_search(query_embedding, top_k, threshold): Perform ANN vector search in ChromaDB (cosine sim > threshold).
chunk_text(text, max_tokens): Split text into chunks (default 512 tokens).
summarize_chunk(chunk): Compress a text chunk via LLM summary.
keyword_search(query, top_k): Keyword-based search on memory cache (e.g., BM25 sim).
socratic_api_council(branches, model, user, convo_id, api_key, rounds, personas): Run a BTIL/MAD-enhanced Socratic council with multiple personas via API for iterative debate, consensus, and refinement.
venv_create(env_name, with_pip): Create a virtual Python environment in sandbox.
restricted_exec(code, level): Execute code in a restricted namespace.
isolated_subprocess(cmd, custom_env): Run command in isolated subprocess.
agent_spawn(sub_agent_type, task): Spawn sub-agent (Planner/Critic/Executor/Worker/Summarizer/Verifier/Moderator) for task.
reflect_optimize(component, metrics): Optimize component based on metrics.
pip_install(venv_path, packages, upgrade): Install Python packages in a venv using pip.
Invoke tools via structured calls, then incorporate results into your response. Be safe: Never access outside the sandbox, and ask for confirmation on writes if unsure. Limit to one tool per response to avoid loops. When outputting tags or code in your final response text (e.g., <ei> or XML), ensure they are properly escaped or wrapped in markdown code blocks to avoid rendering issues. However, when providing arguments for tools (e.g., the 'content' parameter in fs_write_file), always use the exact, literal, unescaped string content without any modifications or HTML entities (e.g., use "<div>" not "&lt;div&gt;"). JSON-escape quotes as needed (e.g., \").""",
}
if not any(f.endswith(".txt") for f in os.listdir(PROMPTS_DIR)):
    for filename, content in default_prompts.items():
        with open(os.path.join(PROMPTS_DIR, filename), "w") as f:
            f.write(content)


def load_prompt_files():
    return [f for f in os.listdir(PROMPTS_DIR) if f.endswith(".txt")]


# Sandbox Directory (create if not exists)
SANDBOX_DIR = "./sandbox"
os.makedirs(SANDBOX_DIR, exist_ok=True)
# YAML Directory for agent instructions (create if not exists)
YAML_DIR = "./evo-modules"
os.makedirs(YAML_DIR, exist_ok=True)
# Custom CSS for UI
st.markdown(
    """<style>
body {
    background: linear-gradient(to right, #000000, #003300);
    color: #00ff00;
    font-family: 'Courier New', monospace;
}
.stApp {
    background: linear-gradient(to right, #000000, #003300);
    display: flex;
    flex-direction: column;
    color: #00ff00;
    font-family: 'Courier New', monospace;
}
.sidebar .sidebar-content {
    background: rgba(0, 51, 0, 0.5);
    border-radius: 10px;
    color: #00ff00;
    font-family: 'Courier New', monospace;
}
.stButton > button {
    background-color: #00ff00;
    color: #000000;
    border-radius: 10px;
    border: none;
    font-family: 'Courier New', monospace;
}
.stButton > button:hover {
    background-color: #00cc00;
}
[data-theme="dark"] .stApp {
    background: linear-gradient(to right, #000000, #003300);
    color: #00ff00;
    font-family: 'Courier New', monospace;
}
</style>
""",
    unsafe_allow_html=True,
)
# Helper Functions
def hash_password(password):
    return sha256_crypt.hash(password)


def verify_password(stored, provided):
    return sha256_crypt.verify(provided, stored)


# Tool Cache Helper
def get_tool_cache_key(func_name, args):
    return f"tool_cache:{func_name}:{hash(json.dumps(args, sort_keys=True))}"


def get_cached_tool_result(func_name, args, ttl_minutes=15):
    if "tool_cache" not in st.session_state:
        st.session_state["tool_cache"] = {}
    key = get_tool_cache_key(func_name, args)
    if key in (cache := st.session_state["tool_cache"]):
        timestamp, result = cache[key]
        if (datetime.now() - timestamp).total_seconds() / 60 < ttl_minutes:
            return result
    return None


def set_cached_tool_result(func_name, args, result):
    if "tool_cache" not in st.session_state:
        st.session_state["tool_cache"] = {}
    cache = st.session_state["tool_cache"]
    key = get_tool_cache_key(func_name, args)
    cache[key] = (datetime.now(), result)


# Tool Functions (Sandboxed)
def fs_read_file(file_path: str) -> str:
    """Read file content from sandbox."""
    safe_path = os.path.abspath(os.path.normpath(os.path.join(SANDBOX_DIR, file_path)))
    if not safe_path.startswith(os.path.abspath(SANDBOX_DIR)):
        return "Error: Path is outside the sandbox."
    try:
        with open(safe_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Error: File not found."
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return f"Error reading file: {e}"


def fs_write_file(file_path: str, content: str) -> str:
    """Write content to file in sandbox."""
    safe_path = os.path.abspath(os.path.normpath(os.path.join(SANDBOX_DIR, file_path)))
    if not safe_path.startswith(os.path.abspath(SANDBOX_DIR)):
        return "Error: Path is outside the sandbox."
    try:
        content = html.unescape(content)
        with open(safe_path, "w", encoding="utf-8") as f:
            f.write(content)
        if "tool_cache" in st.session_state:
            key_to_remove = get_tool_cache_key("fs_read_file", {"file_path": file_path})
            st.session_state["tool_cache"].pop(key_to_remove, None)
        return f"File '{file_path}' written successfully."
    except Exception as e:
        logger.error(f"Error writing file: {e}")
        return f"Error writing file: {e}"


def fs_list_files(dir_path: str = "") -> str:
    """List files in a directory within the sandbox."""
    safe_dir = os.path.abspath(os.path.normpath(os.path.join(SANDBOX_DIR, dir_path)))
    if not safe_dir.startswith(os.path.abspath(SANDBOX_DIR)):
        return "Error: Path is outside the sandbox."
    try:
        files = os.listdir(safe_dir)
        return f"Files in '{dir_path or '/'}': {json.dumps(files)}"
    except FileNotFoundError:
        return "Error: Directory not found."
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        return f"Error listing files: {e}"


def fs_mkdir(dir_path: str) -> str:
    """Create a new directory in the sandbox."""
    safe_path = os.path.abspath(os.path.normpath(os.path.join(SANDBOX_DIR, dir_path)))
    if not safe_path.startswith(os.path.abspath(SANDBOX_DIR)):
        return "Error: Path is outside the sandbox."
    try:
        os.makedirs(safe_path, exist_ok=True)
        return f"Directory '{dir_path}' created successfully."
    except Exception as e:
        logger.error(f"Error creating directory: {e}")
        return f"Error creating directory: {e}"


def get_current_time(sync: bool = False, format: str = "iso") -> str:
    """Fetch current time."""
    try:
        if sync:
            ntp_client = ntplib.NTPClient()
            response = ntp_client.request("pool.ntp.org", version=3)
            dt_object = datetime.fromtimestamp(response.tx_time)
        else:
            dt_object = datetime.now()
        if format == "human":
            return dt_object.strftime("%A, %B %d, %Y %I:%M:%S %p")
        elif format == "json":
            return json.dumps(
                {
                    "datetime": dt_object.isoformat(),
                    "timezone": time.localtime().tm_zone,
                }
            )
        else:
            return dt_object.isoformat()
    except Exception as e:
        logger.error(f"Time error: {e}")
        return f"Time error: {e}"


SAFE_BUILTINS = {
    b: getattr(builtins, b)
    for b in [
        "print",
        "len",
        "range",
        "str",
        "int",
        "float",
        "list",
        "dict",
        "set",
        "tuple",
        "abs",
        "round",
        "max",
        "min",
        "sum",
        "sorted",
    ]
}
# Enhanced with more libraries available
ADDITIONAL_LIBS = {
    "numpy": np,
    "sympy": sympy,
    "mpmath": mpmath,
    "PuLP": PuLP,
    "pygame": pygame,
    "chess": chess,
    "networkx": nx,
    "unittest": unittest,
    "asyncio": asyncio,
    "multiprocessing": multiprocessing,
}


def init_repl_namespace():
    if "repl_namespace" not in st.session_state:
        st.session_state["repl_namespace"] = {"__builtins__": SAFE_BUILTINS.copy()}
        st.session_state["repl_namespace"].update(ADDITIONAL_LIBS)


def execute_in_venv(code: str, venv_path: str) -> str:
    safe_venv = os.path.abspath(os.path.normpath(os.path.join(SANDBOX_DIR, venv_path)))
    if not safe_venv.startswith(os.path.abspath(SANDBOX_DIR)):
        return "Error: Venv path outside sandbox."
    venv_python = os.path.join(safe_venv, "bin", "python")
    if not os.path.exists(venv_python):
        return "Error: Venv Python not found."
    result = subprocess.run(
        [venv_python, "-c", code], capture_output=True, text=True, timeout=30
    )
    output = result.stdout
    if result.stderr:
        logger.error(f"Venv execution error: {result.stderr}")
        return f"Error: {result.stderr}"
    return output


def execute_local(code: str, redirected_output: io.StringIO) -> str:
    exec(code, st.session_state["repl_namespace"])
    return redirected_output.getvalue()


def code_execution(code: str, venv_path: str = None) -> str:
    """Execute Python code safely in a stateful REPL, optionally in a venv."""
    init_repl_namespace()
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output
    try:
        if venv_path:
            output = execute_in_venv(code, venv_path)
        else:
            output = execute_local(code, redirected_output)
        return f"Output:\n{output}" if output else "Execution successful (no output)."
    except Exception:
        logger.error(f"Code execution error: {traceback.format_exc()}")
        return f"Error: {traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout


def memory_insert(
    mem_key: str, mem_value: dict, user: str = None, convo_id: int = None
) -> str:
    """Insert/update memory key-value pair (enhanced to handle user/convo_id from dispatcher)."""
    if user is None or convo_id is None:
        return "Error: user and convo_id required for memory insert."
    try:
        c.execute(
            "INSERT OR REPLACE INTO memory (user, convo_id, mem_key, mem_value) VALUES (?, ?, ?, ?)",
            (user, convo_id, mem_key, json.dumps(mem_value)),
        )
        conn.commit()
        if "memory_cache" in st.session_state:
            entry = {
                "summary": mem_value.get("summary", ""),
                "details": mem_value.get("details", ""),
                "tags": mem_value.get("tags", []),
                "domain": mem_value.get("domain", "general"),
                "timestamp": datetime.now().isoformat(),
                "salience": mem_value.get("salience", 1.0),
            }
            st.session_state["memory_cache"]["lru_cache"][mem_key] = {
                "entry": entry,
                "last_access": time.time(),
            }
            st.session_state["memory_cache"]["metrics"]["total_inserts"] += 1
        logger.info(f"Memory inserted: {mem_key}")
        return "Memory inserted successfully."
    except Exception as e:
        logger.error(f"Error inserting memory: {e}")
        return f"Error inserting memory: {e}"


def load_into_lru(key, entry):
    if key not in st.session_state["memory_cache"]["lru_cache"]:
        st.session_state["memory_cache"]["lru_cache"][key] = {
            "entry": {
                "summary": entry.get("summary", ""),
                "details": entry.get("details", ""),
                "tags": entry.get("tags", []),
                "domain": entry.get("domain", "general"),
                "timestamp": entry.get("timestamp", datetime.now().isoformat()),
                "salience": entry.get("salience", 1.0),
            },
            "last_access": time.time(),
        }


def memory_query(
    mem_key: str = None, limit: int = 10, user: str = None, convo_id: int = None
) -> str:
    """Query memory by key or get recent entries (enhanced with LRU sync)."""
    if user is None or convo_id is None:
        return "Error: user and convo_id required for memory query."
    try:
        if mem_key:
            c.execute(
                "SELECT mem_value FROM memory WHERE user=? AND convo_id=? AND mem_key=?",
                (user, convo_id, mem_key),
            )
            result = c.fetchone()
            logger.info(f"Memory queried: {mem_key}")
            return result[0] if result else "Key not found."
        else:
            c.execute(
                "SELECT mem_key, mem_value FROM memory WHERE user=? AND convo_id=? ORDER BY timestamp DESC LIMIT ?",
                (user, convo_id, limit),
            )
            results = {row[0]: json.loads(row[1]) for row in c.fetchall()}
            if "memory_cache" in st.session_state:
                for key in results:
                    load_into_lru(key, results[key])
            logger.info("Recent memories queried")
            return json.dumps(results)
    except Exception as e:
        logger.error(f"Error querying memory: {e}")
        return f"Error querying memory: {e}"


def advanced_memory_consolidate(
    mem_key: str, interaction_data: dict, user: str = None, convo_id: int = None
) -> str:
    """Consolidate: Summarize, embed, and store hierarchically (enhanced with chunking/summarization)."""
    if user is None or convo_id is None:
        return "Error: user and convo_id required."
    cache_args = {"mem_key": mem_key, "interaction_data": interaction_data}
    if cached := get_cached_tool_result("advanced_memory_consolidate", cache_args):
        return cached
    embed_model = get_embed_model()
    try:
        client = OpenAI(api_key=API_KEY, base_url="https://api.x.ai/v1/")
        summary_response = client.chat.completions.create(
            model="grok-4-fast-non-reasoning",
            messages=[
                {
                    "role": "system",
                    "content": "Summarize this interaction concisely in one paragraph.",
                },
                {"role": "user", "content": json.dumps(interaction_data)},
            ],
            stream=False,
        )
        summary = summary_response.choices[0].message.content.strip()
        json_episodic = json.dumps(interaction_data)
        c.execute(
            "INSERT OR REPLACE INTO memory (user, convo_id, mem_key, mem_value) VALUES (?, ?, ?, ?)",
            (user, convo_id, mem_key, json_episodic),
        )
        conn.commit()
        if (
            embed_model
            and st.session_state.get("chroma_ready")
            and st.session_state.get("chroma_collection")
        ):
            chroma_col = st.session_state["chroma_collection"]
            embedding = embed_model.encode(summary).tolist()
            chroma_col.upsert(
                ids=[str(uuid.uuid4())],
                embeddings=[embedding],
                documents=[json_episodic],
                metadatas=[
                    {
                        "user": user,
                        "convo_id": convo_id,
                        "mem_key": mem_key,
                        "salience": 1.0,
                        "summary": summary,
                    }
                ],
            )
        else:
            st.warning("Vector storage skipped; using DB only.")
        entry = {
            "summary": summary,
            "details": json_episodic,
            "tags": [],
            "domain": "general",
            "timestamp": datetime.now().isoformat(),
            "salience": 1.0,
        }
        if "memory_cache" in st.session_state:
            st.session_state["memory_cache"]["lru_cache"][mem_key] = {
                "entry": entry,
                "last_access": time.time(),
            }
        result = "Memory consolidated successfully."
        set_cached_tool_result("advanced_memory_consolidate", cache_args, result)
        logger.info(f"Memory consolidated: {mem_key}")
        return result
    except Exception:
        logger.error(f"Error consolidating memory: {traceback.format_exc()}")
        return f"Error consolidating memory: {traceback.format_exc()}"


def fallback_to_keyword(query: str, top_k: int, user: str, convo_id: int) -> list:
    fallback_results = keyword_search(query, top_k, user, convo_id)
    if isinstance(fallback_results, str) and "error" in fallback_results.lower():
        return fallback_results
    retrieved = []
    for res in fallback_results:
        mem_key = res["id"]
        value = json.loads(memory_query(mem_key=mem_key, user=user, convo_id=convo_id))
        retrieved.append(
            {
                "mem_key": mem_key,
                "value": value,
                "relevance": res["score"],
                "summary": value.get("summary", ""),
            }
        )
    return retrieved


def process_chroma_results(results, top_k: int) -> list:
    if not results or not results.get("ids") or not results["ids"]:
        return []
    retrieved = []
    ids_to_update = []
    metadata_to_update = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        sim = (1 - results["distances"][0][i]) * meta.get("salience", 1.0)
        retrieved.append(
            {
                "mem_key": meta["mem_key"],
                "value": json.loads(results["documents"][0][i]),
                "relevance": sim,
                "summary": meta.get("summary", ""),
            }
        )
        ids_to_update.append(results["ids"][0][i])
        metadata_to_update.append({"salience": meta.get("salience", 1.0) + 0.1})
    if ids_to_update:
        st.session_state["chroma_collection"].update(
            ids=ids_to_update, metadatas=metadata_to_update
        )
    retrieved.sort(key=lambda x: x["relevance"], reverse=True)
    return retrieved


def update_retrieve_metrics(len_retrieved: int, top_k: int):
    if "memory_cache" in st.session_state:
        st.session_state["memory_cache"]["metrics"]["total_retrieves"] += 1
        hit_rate = len_retrieved / top_k if top_k > 0 else 1.0
        st.session_state["memory_cache"]["metrics"]["hit_rate"] = (
            (
                st.session_state["memory_cache"]["metrics"]["hit_rate"]
                * (st.session_state["memory_cache"]["metrics"]["total_retrieves"] - 1)
            )
            + hit_rate
        ) / st.session_state["memory_cache"]["metrics"]["total_retrieves"]


def advanced_memory_retrieve(
    query: str, top_k: int = 5, user: str = None, convo_id: int = None
) -> str:
    """Retrieve top-k relevant memories via embedding similarity (enhanced with hybrid/monitoring)."""
    if user is None or convo_id is None:
        return "Error: user and convo_id required."
    cache_args = {"query": query, "top_k": top_k}
    if cached := get_cached_tool_result("advanced_memory_retrieve", cache_args):
        return cached
    embed_model = get_embed_model()
    chroma_col = st.session_state.get("chroma_collection")
    if not embed_model or not st.session_state.get("chroma_ready") or not chroma_col:
        # Fallback to keyword search
        logger.warning("Vector memory not available; falling back to keyword search.")
        st.warning("Vector memory not available; falling back to keyword search.")
        retrieved = fallback_to_keyword(query, top_k, user, convo_id)
        result = json.dumps(retrieved)
        set_cached_tool_result("advanced_memory_retrieve", cache_args, result)
        return result
    try:
        query_emb = embed_model.encode(query).tolist()
        where_clause = {
            "$and": [
                {"user": user},
                {"convo_id": convo_id},
            ]
        }
        results = chroma_col.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            where=where_clause,
            include=["distances", "metadatas", "documents"],
        )
        retrieved = process_chroma_results(results, top_k)
        if not retrieved:
            return "No relevant memories found."
        update_retrieve_metrics(len(retrieved), top_k)
        result = json.dumps(retrieved)
        set_cached_tool_result("advanced_memory_retrieve", cache_args, result)
        logger.info(f"Memory retrieved for query: {query}")
        return result
    except Exception:
        logger.error(f"Error retrieving memory: {traceback.format_exc()}")
        return (
            f"Error retrieving memory: {traceback.format_exc()}. "
            "If this is a where clause issue, check filter structure."
        )


def should_prune() -> bool:
    if "prune_counter" not in st.session_state:
        st.session_state["prune_counter"] = 0
    st.session_state["prune_counter"] += 1
    return st.session_state["prune_counter"] % 10 == 0


def decay_salience(user: str, convo_id: int):
    one_week_ago = datetime.now() - timedelta(days=7)
    c.execute(
        "UPDATE memory SET salience = salience * 0.99 WHERE user=? AND convo_id=? AND timestamp < ?",
        (user, convo_id, one_week_ago),
    )


def prune_low_salience(user: str, convo_id: int):
    c.execute(
        "DELETE FROM memory WHERE user=? AND convo_id=? AND salience < 0.1",
        (user, convo_id),
    )


def size_based_prune(user: str, convo_id: int):
    c.execute(
        "SELECT COUNT(*) FROM memory WHERE user=? AND convo_id=?", (user, convo_id)
    )
    if (row_count := c.fetchone()[0]) > 1000:
        c.execute(
            "SELECT mem_key FROM memory WHERE user=? AND convo_id=? AND salience < 0.5 ORDER BY timestamp ASC LIMIT ?",
            (user, convo_id, row_count - 1000),
        )
        low_keys = [row[0] for row in c.fetchall()]
        for key in low_keys:
            c.execute(
                "DELETE FROM memory WHERE user=? AND convo_id=? AND mem_key=?",
                (user, convo_id, key),
            )


def dedup_prune(user: str, convo_id: int):
    c.execute(
        "SELECT mem_key, mem_value FROM memory WHERE user=? AND convo_id=?",
        (user, convo_id),
    )
    rows = c.fetchall()
    hashes = {}
    to_delete = []
    for key, value_str in rows:
        value = json.loads(value_str)
        h = hash(value.get("summary", ""))
        if h in hashes and value.get("salience", 1.0) < hashes[h].get("salience", 1.0):
            to_delete.append(key)
        else:
            hashes[h] = value
    for key in to_delete:
        c.execute(
            "DELETE FROM memory WHERE user=? AND convo_id=? AND mem_key=?",
            (user, convo_id, key),
        )


def lru_evict():
    if (
        "memory_cache" in st.session_state
        and len(st.session_state["memory_cache"]["lru_cache"]) > 1000
    ):
        lru_items = sorted(
            st.session_state["memory_cache"]["lru_cache"].items(),
            key=lambda x: x[1]["last_access"],
        )
        num_to_evict = len(lru_items) - 1000
        for key, _ in lru_items[:num_to_evict]:
            entry = st.session_state["memory_cache"]["lru_cache"][key]["entry"]
            if entry["salience"] < 0.4:
                del st.session_state["memory_cache"]["lru_cache"][key]


def advanced_memory_prune(user: str = None, convo_id: int = None) -> str:
    """Prune low-salience memories (enhanced with size/LRU/dedup)."""
    if user is None or convo_id is None:
        return "Error: user and convo_id required."
    if not should_prune():
        return "Prune skipped (infrequent)."
    try:
        conn.execute("BEGIN")
        decay_salience(user, convo_id)
        prune_low_salience(user, convo_id)
        size_based_prune(user, convo_id)
        dedup_prune(user, convo_id)
        lru_evict()
        conn.commit()
        logger.info("Memory pruned successfully")
        return "Memory pruned successfully."
    except Exception:
        conn.rollback()
        logger.error(f"Error pruning memory: {traceback.format_exc()}")
        return f"Error pruning memory: {traceback.format_exc()}"


def git_init(safe_repo: str) -> str:
    pygit2.init_repository(safe_repo)
    return "Repo initialized."


def git_commit(repo: pygit2.Repository, message: str) -> str:
    if not message:
        return "Error: Message required for commit."
    index = repo.index
    index.add_all()
    index.write()
    tree = index.write_tree()
    author = pygit2.Signature("User", "user@example.com")
    repo.create_commit(
        "HEAD",
        author,
        author,
        message,
        tree,
        [repo.head.target] if repo.head.is_branch else [],
    )
    return "Committed."


def git_branch(repo: pygit2.Repository, name: str) -> str:
    if not name:
        return "Error: Name required for branch."
    repo.create_branch(name, repo.head.peel())
    return f"Branch '{name}' created."


def git_diff(repo: pygit2.Repository) -> str:
    diff = repo.diff()
    return diff.patch


def git_ops(
    operation: str, repo_path: str, message: str = None, name: str = None
) -> str:
    """Basic Git operations in sandbox (init, commit, branch, diff). No remote ops."""
    safe_repo = os.path.abspath(os.path.normpath(os.path.join(SANDBOX_DIR, repo_path)))
    if not safe_repo.startswith(os.path.abspath(SANDBOX_DIR)):
        return "Error: Repo path outside sandbox."
    try:
        op_funcs = {
            "init": lambda: git_init(safe_repo),
            "commit": lambda: git_commit(pygit2.Repository(safe_repo), message),
            "branch": lambda: git_branch(pygit2.Repository(safe_repo), name),
            "diff": lambda: git_diff(pygit2.Repository(safe_repo)),
        }
        if operation in op_funcs:
            return op_funcs[operation]()
        return "Unknown operation."
    except Exception as e:
        logger.error(f"Git error: {e}")
        return f"Git error: {e}"


def db_query(db_path: str, query: str, params: list = None) -> str:
    """Interact with local SQLite DB in sandbox."""
    safe_db = os.path.abspath(os.path.normpath(os.path.join(SANDBOX_DIR, db_path)))
    if not safe_db.startswith(os.path.abspath(SANDBOX_DIR)):
        return "Error: DB path outside sandbox."
    try:
        db_conn = sqlite3.connect(safe_db)
        db_c = db_conn.cursor()
        if params:
            db_c.execute(query, params)
        else:
            db_c.execute(query)
        if query.lower().startswith("select"):
            results = db_c.fetchall()
            return json.dumps(results)
        db_conn.commit()
        return "Query executed."
    except Exception as e:
        logger.error(f"DB error: {e}")
        return f"DB error: {e}"
    finally:
        db_conn.close()


def shell_exec(command: str) -> str:
    """Run whitelisted shell commands in sandbox."""
    whitelist = [
        "ls",
        "grep",
        "sed",
        "awk",
        "cat",
        "echo",
        "wc",
        "tail",
        "head",
        "cp",
        "mv",
        "rm",
        "mkdir",
        "rmdir",
        "touch",
    ]
    cmd_parts = shlex.split(command)
    if cmd_parts[0] not in whitelist:
        return "Error: Command not whitelisted."
    if cmd_parts[0] in ["rm", "rmdir"]:  # Confirmation for destructive cmds
        if not st.session_state.get("confirm_destructive", False):
            st.session_state["confirm_destructive"] = True
            return "Warning: Destructive command detected. Confirm by re-running."
    try:
        result = subprocess.run(
            cmd_parts, cwd=SANDBOX_DIR, capture_output=True, text=True, timeout=10
        )
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except Exception as e:
        logger.error(f"Shell error: {e}")
        return f"Shell error: {e}"


def lint_python(code: str) -> str:
    return format_str(code, mode=FileMode())


def lint_javascript(code: str) -> str:
    return jsbeautifier.beautify(code)


def lint_json(code: str) -> str:
    return json.dumps(json.loads(code), indent=4)


def lint_yaml(code: str) -> str:
    return yaml.safe_dump(yaml.safe_load(code), default_flow_style=False)


def lint_sql(code: str) -> str:
    return sqlparse.format(code, reindent=True)


def lint_xml_html(code: str, lang_lower: str) -> str:
    if lang_lower == "html":
        soup = bs4.BeautifulSoup(code, "html.parser")
        return soup.prettify()
    else:
        dom = xml.dom.minidom.parseString(code)
        return dom.toprettyxml()


def code_lint(language: str, code: str) -> str:
    """Lint and format code for various languages."""
    lang_lower = language.lower()
    try:
        lint_funcs = {
            "python": lambda: lint_python(code),
            "javascript": lambda: lint_javascript(code),
            "json": lambda: lint_json(code),
            "yaml": lambda: lint_yaml(code),
            "sql": lambda: lint_sql(code),
            "xml": lambda: lint_xml_html(code, lang_lower),
            "html": lambda: lint_xml_html(code, lang_lower),
        }
        if lang_lower in lint_funcs:
            return lint_funcs[lang_lower]()
        return f"Linting not supported for {language}."
    except Exception as e:
        logger.error(f"Lint error: {e}")
        return f"Lint error: {e}"


def api_simulate(
    url: str, method: str = "GET", data: dict = None, mock: bool = True
) -> str:
    """Simulate API calls (mock or real for whitelisted)."""
    whitelist = ["https://api.example.com", "https://jsonplaceholder.typicode.com"]
    if not any(url.startswith(w) for w in whitelist) and not mock:
        return "Error: URL not whitelisted for real calls."
    try:
        if mock:
            return (
                f"Mock response for {method} {url}: {json.dumps({'status': 'mocked'})}"
            )
        response = requests.request(
            method, url, json=data if method == "POST" else None
        )
        return response.text
    except Exception as e:
        logger.error(f"API error: {e}")
        return f"API error: {e}"


def langsearch_web_search(
    query: str, freshness: str = "noLimit", summary: bool = True, count: int = 5
) -> str:
    """Web search via LangSearch API."""
    if not LANGSEARCH_API_KEY:
        return "Error: LANGSEARCH_API_KEY not set."
    url = "https://api.langsearch.com/search"
    payload = {
        "query": query,
        "freshness": freshness,
        "summary": summary,
        "count": min(count, 10),
    }
    headers = {"Authorization": f"Bearer {LANGSEARCH_API_KEY}"}
    try:
        response = requests.post(url, json=payload, headers=headers)
        return response.json() if response.ok else f"Error: {response.text}"
    except Exception as e:
        logger.error(f"Search error: {e}")
        return f"Search error: {e}"


def generate_embedding(text: str) -> str:
    """Generate embedding vector."""
    embed_model = get_embed_model()
    if not embed_model:
        return "Error: Embedding model not loaded."
    try:
        embedding = embed_model.encode(text).tolist()
        return json.dumps(embedding)
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return f"Embedding error: {e}"


def vector_search(query_embedding: list, top_k: int = 5, threshold: float = 0.6) -> str:
    """ANN vector search in ChromaDB."""
    if not st.session_state.get("chroma_ready") or not st.session_state.get(
        "chroma_collection"
    ):
        return "Error: ChromaDB not ready."
    chroma_col = st.session_state["chroma_collection"]
    try:
        results = chroma_col.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        filtered = [
            {"id": id, "distance": dist}
            for id, dist in zip(results["ids"][0], results["distances"][0])
            if dist <= (1 - threshold)
        ]
        return json.dumps(filtered)
    except Exception as e:
        logger.error(f"Vector search error: {e}")
        return f"Vector search error: {e}"


def chunk_text(text: str, max_tokens: int = 512) -> str:
    """Split text into chunks."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        chunks = [
            enc.decode(tokens[i : i + max_tokens])
            for i in range(0, len(tokens), max_tokens)
        ]
        return json.dumps(chunks)
    except Exception as e:
        logger.error(f"Chunk error: {e}")
        return f"Chunk error: {e}"


def summarize_chunk(chunk: str) -> str:
    """Summarize text chunk via LLM."""
    try:
        client = OpenAI(api_key=API_KEY, base_url="https://api.x.ai/v1/")
        response = client.chat.completions.create(
            model="grok-4-fast-non-reasoning",
            messages=[
                {
                    "role": "system",
                    "content": "Summarize this text in under 100 words, preserving key facts.",
                },
                {"role": "user", "content": chunk},
            ],
            stream=False,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Summarize error: {e}")
        return f"Summarize error: {e}"


def keyword_search(
    query: str, top_k: int = 5, user: str = None, convo_id: int = None
) -> str:
    """Simple keyword search on memory (fallback)."""
    if user is None or convo_id is None:
        return "Error: user and convo_id required."
    try:
        c.execute(
            "SELECT mem_key FROM memory WHERE user=? AND convo_id=? AND mem_value LIKE ? ORDER BY salience DESC LIMIT ?",
            (user, convo_id, f"%{query}%", top_k),
        )
        results = [{"id": row[0], "score": 1.0} for row in c.fetchall()]  # Pseudo-score
        return results
    except Exception as e:
        logger.error(f"Keyword search error: {e}")
        return f"Keyword search error: {e}"


def socratic_api_council(
    branches: list,
    model: str = "grok-4-fast-reasoning",
    user: str = None,
    convo_id: int = None,
    api_key: str = None,
    rounds: int = 3,
    personas: list = None,
) -> str:
    """BTIL/MAD-enhanced Socratic council with iterative rounds, expanded personas, and consensus."""
    if not api_key:
        api_key = API_KEY
    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1/")
    default_personas = [
        "Planner",
        "Critic",
        "Executor",
        "Summarizer",
        "Verifier",
        "Moderator",
    ]
    personas = personas or default_personas
    try:
        consensus = ""
        messages = [
            {
                "role": "system",
                "content": f"Debate these branches as a {len(personas)}-persona council: {', '.join(personas)}. Iterate {rounds} rounds for refinement and consensus.",
            }
        ]
        for branch in branches:
            messages.append({"role": "user", "content": branch})
        for r in range(rounds):
            response = client.chat.completions.create(model=model, messages=messages)
            round_result = response.choices[0].message.content
            consensus += f"Round {r+1}: {round_result}\n"
            messages.append({"role": "assistant", "content": round_result})
        # Final consensus via voting/judge
        messages.append(
            {
                "role": "system",
                "content": "Reach final consensus via majority vote or judge.",
            }
        )
        final_response = client.chat.completions.create(model=model, messages=messages)
        consensus += f"Final Consensus: {final_response.choices[0].message.content}"
        # Consolidate
        advanced_memory_consolidate(
            "council_result",
            {"branches": branches, "result": consensus},
            user,
            convo_id,
        )
        logger.info("Socratic council completed")
        return consensus
    except Exception as e:
        logger.error(f"Council error: {e}")
        return f"Council error: {e}"


def agent_spawn(sub_agent_type: str, task: str) -> str:
    """Spawn sub-agent (Planner, Critic, Executor, Worker, Summarizer, Verifier, Moderator) for task. Case-insensitive, custom types allowed with fallback simulation."""
    # Normalize case for flexibility
    normalized_type = (
        sub_agent_type.strip().title()
    )  # Converts "critic" to "Critic", "WORKER" to "Worker"

    agent_responses = {
        "Planner": f"Planner: Planning steps for '{task}' - Step 1: Analyze, Step 2: Execute.",
        "Critic": f"Critic: Evaluating '{task}' - Pros: Efficient, Cons: Risky.",
        "Executor": f"Executor: Executing '{task}' - Done.",
        "Worker": f"Worker: Handling routine task '{task}' - Processed with standard workflow.",
        "Summarizer": f"Summarizer: Concise summary of '{task}' - Key points extracted.",
        "Verifier": f"Verifier: Fact-checking '{task}' - Verified accurate.",
        "Moderator": f"Moderator: Guiding debate on '{task}' - Consensus reached.",
    }
    if normalized_type in agent_responses:
        return agent_responses[normalized_type]
    return f"Custom or invalid sub-agent type '{sub_agent_type}'. Using fallback simulation for '{normalized_type}': Task '{task}' acknowledged."


def reflect_optimize(component: str, metrics: dict) -> str:
    """Simulate optimization based on metrics."""
    return f"Optimized {component} with metrics: {json.dumps(metrics)} - Adjustments applied - - Adjustments applied."


def venv_create(env_name: str, with_pip: bool = True) -> str:
    """Create venv in sandbox."""
    safe_env = os.path.abspath(os.path.normpath(os.path.join(SANDBOX_DIR, env_name)))
    if not safe_env.startswith(os.path.abspath(SANDBOX_DIR)):
        return "Error: Env path outside sandbox."
    try:
        venv.create(safe_env, with_pip=with_pip)
        return f"Venv '{env_name}' created."
    except Exception as e:
        logger.error(f"Venv error: {e}")
        return f"Venv error: {e}"


def restricted_exec(code: str, level: str = "basic") -> str:
    """Execute in restricted namespace using restrictedpython."""  # noqa: C901
    try:
        if level == "basic":
            result = RestrictedPython.compile_restricted_exec(code)
            if result.errors:
                return f"Restricted compile error: {result.errors}"
            exec(result.code, RestrictedPython.safe_globals, {})
        else:
            exec(code, globals())
        return "Executed in restricted mode."
    except Exception as e:
        logger.error(f"Restricted exec error: {e}")
        return f"Restricted exec error: {e}"


def isolated_subprocess(cmd: str, custom_env: dict = None) -> str:
    """Run in isolated subprocess."""
    env = os.environ.copy()
    if custom_env:
        env.update(custom_env)
    try:
        result = subprocess.run(
            shlex.split(cmd),
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
            cwd=SANDBOX_DIR,
        )
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except Exception as e:
        logger.error(f"Subprocess error: {e}")
        return f"Subprocess error: {e}"


# Whitelist for pip packages
PIP_WHITELIST = [
    "numpy",
    "pandas",
    "matplotlib",
    "scipy",
    "sympy",
    "requests",
    "beautifulsoup4",
]


def pip_install(venv_path: str, packages: list, upgrade: bool = False) -> str:
    """Install packages in venv using pip, with whitelist check."""
    if any(pkg not in PIP_WHITELIST for pkg in packages):
        return "Error: One or more packages not in whitelist."
    safe_venv = os.path.abspath(os.path.normpath(os.path.join(SANDBOX_DIR, venv_path)))
    if not safe_venv.startswith(os.path.abspath(SANDBOX_DIR)):
        return "Error: Venv path outside sandbox."
    venv_pip = os.path.join(safe_venv, "bin", "pip")
    if not os.path.exists(venv_pip):
        return "Error: Pip not found in venv."
    cmd = [venv_pip, "install"] + (["--upgrade"] if upgrade else []) + packages
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except Exception as e:
        logger.error(f"Pip error: {e}")
        return f"Pip error: {e}"


def chat_log_analyze_embed(
    convo_id: int, criteria: str, summarize: bool = True, user: str = None
) -> str:
    if user is None:
        return "Error: user required."
    c.execute(
        "SELECT messages FROM history WHERE convo_id=? AND user=?", (convo_id, user)
    )
    result = c.fetchone()
    if not result:
        return "Error: Chat log not found."
    messages = json.loads(result[0])
    chat_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    client = OpenAI(api_key=API_KEY, base_url="https://api.x.ai/v1/")
    analysis_prompt = (
        f"Analyze this chat log on criteria: {criteria}. Summarize if needed."
    )
    response = client.chat.completions.create(
        model="grok-4-fast-non-reasoning",
        messages=[
            {"role": "system", "content": analysis_prompt},
            {"role": "user", "content": chat_text},
        ],
        stream=False,
    )
    analysis = response.choices[0].message.content.strip()
    if summarize:
        summary_prompt = "Summarize the analysis concisely."
        summary_response = client.chat.completions.create(
            model="grok-4-fast-non-reasoning",
            messages=[
                {"role": "system", "content": summary_prompt},
                {"role": "user", "content": analysis},
            ],
            stream=False,
        )
        analysis = summary_response.choices[0].message.content.strip()
    embed_model = get_embed_model()
    if not embed_model or not st.session_state.get("chroma_ready"):
        return "Error: Embedding or ChromaDB not ready."
    chroma_col = st.session_state["chroma_collection"]
    embedding = embed_model.encode(analysis).tolist()
    mem_key = f"chat_log_{convo_id}"
    chroma_col.upsert(
        ids=[mem_key],
        embeddings=[embedding],
        documents=[analysis],
        metadatas=[
            {"user": user, "convo_id": convo_id, "type": "chat_log", "salience": 1.0}
        ],
    )
    return f"Chat log {convo_id} analyzed and embedded as {mem_key}."


def yaml_retrieve(
    query: str = None, top_k: int = 5, filename: str = None
) -> str:  # noqa: C901
    """Retrieve YAML content semantically or by exact filename from embedded DB."""
    if "yaml_ready" not in st.session_state or not st.session_state["yaml_ready"]:
        return "Error: YAML DB not ready."
    col = st.session_state["yaml_collection"]
    embed_model = get_embed_model()
    if "yaml_cache" not in st.session_state:
        st.session_state["yaml_cache"] = {}
    try:
        if filename:
            if filename in st.session_state["yaml_cache"]:
                return st.session_state["yaml_cache"][filename]
            results = col.query(
                n_results=1, where={"filename": filename}, include=["documents"]
            )
            if results.get("documents") and results["documents"][0]:
                content = results["documents"][0][0]
                st.session_state["yaml_cache"][filename] = content
                return content
            else:
                return "YAML not found."
        else:
            if not query:
                return "Error: Query required for semantic search."
            query_emb = embed_model.encode(query).tolist()
            results = col.query(
                query_embeddings=[query_emb],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            if not results.get("documents"):
                return "No relevant YAMLs found."
            retrieved = [
                {"filename": meta["filename"], "content": doc, "distance": dist}
                for meta, doc, dist in zip(
                    results["metadatas"][0],
                    results["documents"][0],
                    results["distances"][0],
                )
            ]
            return json.dumps(retrieved)
    except Exception as e:
        logger.error(f"YAML retrieve error: {e}")
        return f"YAML retrieve error: {e}"


def yaml_refresh(filename: str = None) -> str:  # noqa: C901
    """Refresh YAML embedding from file system, for one or all."""
    embed_model = get_embed_model()
    if not embed_model:
        return "Error: Embedding model not loaded."
    col = st.session_state["yaml_collection"]
    try:
        if filename:
            path = os.path.join(YAML_DIR, filename)
            if not os.path.exists(path):
                return "Error: File not found."
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            embedding = embed_model.encode(content).tolist()
            col.upsert(
                ids=[filename],
                embeddings=[embedding],
                documents=[content],
                metadatas=[{"filename": filename}],
            )
            if "yaml_cache" in st.session_state:
                st.session_state["yaml_cache"][filename] = content
            return f"YAML '{filename}' refreshed successfully."
        else:
            # Refresh all
            ids = col.get()["ids"]
            if ids:
                col.delete(ids=ids)
            st.session_state["yaml_cache"] = {}
            files_refreshed = 0
            for fname in os.listdir(YAML_DIR):
                if fname.endswith(".yaml"):
                    path = os.path.join(YAML_DIR, fname)
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                    embedding = embed_model.encode(content).tolist()
                    col.upsert(
                        ids=[fname],
                        embeddings=[embedding],
                        documents=[content],
                        metadatas=[{"filename": fname}],
                    )
                    st.session_state["yaml_cache"][fname] = content
                    files_refreshed += 1
            return f"All YAMLs refreshed successfully ({files_refreshed} files)."
    except Exception as e:
        logger.error(f"YAML refresh error: {e}")
        return f"YAML refresh error: {e}"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fs_read_file",
            "description": "Read file content from sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative path to file.",
                    }
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_write_file",
            "description": "Write content to file in sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative path to file.",
                    },
                    "content": {"type": "string", "description": "Content to write."},
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_list_files",
            "description": "List files in directory within sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir_path": {
                        "type": "string",
                        "description": "Relative dir path (default root).",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_mkdir",
            "description": "Create directory in sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir_path": {"type": "string", "description": "Relative dir path."}
                },
                "required": ["dir_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Fetch current time (sync optional).",
            "parameters": {
                "type": "object",
                "properties": {
                    "sync": {
                        "type": "boolean",
                        "description": "True for NTP sync (default False).",
                    },
                    "format": {
                        "type": "string",
                        "description": "Format: iso (default), human, json.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_execution",
            "description": "Execute Python code in REPL (venv optional).",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code to execute."},
                    "venv_path": {
                        "type": "string",
                        "description": "Optional venv path.",
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_insert",
            "description": "Insert/update memory entry.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mem_key": {"type": "string", "description": "Memory key."},
                    "mem_value": {"type": "object", "description": "Value dict."},
                },
                "required": ["mem_key", "mem_value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_query",
            "description": "Query memory by key or recent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mem_key": {
                        "type": "string",
                        "description": "Specific key to query (optional).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max recent entries if no key (default 10).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_ops",
            "description": "Basic Git operations in sandbox (init, commit, branch, diff). No remote operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["init", "commit", "branch", "diff"],
                    },
                    "repo_path": {
                        "type": "string",
                        "description": "Relative path to repo.",
                    },
                    "message": {
                        "type": "string",
                        "description": "Commit message (for commit).",
                    },
                    "name": {
                        "type": "string",
                        "description": "Branch name (for branch).",
                    },
                },
                "required": ["operation", "repo_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "db_query",
            "description": "Interact with local SQLite database in sandbox (create, insert, query).",
            "parameters": {
                "type": "object",
                "properties": {
                    "db_path": {
                        "type": "string",
                        "description": "Relative path to DB file.",
                    },
                    "query": {"type": "string", "description": "SQL query."},
                    "params": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Query parameters.",
                    },
                },
                "required": ["db_path", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shell_exec",
            "description": "Run safe whitelisted shell commands in sandbox (e.g., ls, grep).",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command string.",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_lint",
            "description": "Lint and auto-format code for various languages: python (black), javascript (jsbeautifier), css (cssbeautifier), json, yaml, sql (sqlparse), xml, html (beautifulsoup), cpp/c++ (clang-format), php (php-cs-fixer), go (gofmt), rust (rustfmt). External tools required for some.",
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "description": "Language (python, javascript, css, json, yaml, sql, xml, html, cpp, php, go, rust).",
                    },
                    "code": {"type": "string", "description": "Code snippet."},
                },
                "required": ["language", "code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "api_simulate",
            "description": "Simulate API calls with mock or fetch from public APIs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "API URL."},
                    "method": {
                        "type": "string",
                        "description": "GET/POST (default GET).",
                    },
                    "data": {"type": "object", "description": "POST data."},
                    "mock": {
                        "type": "boolean",
                        "description": "True for mock (default).",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "advanced_memory_consolidate",
            "description": "Brain-like consolidation: Summarize and embed data for hierarchical storage. Use for chat logs to create semantic summaries and episodic details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mem_key": {
                        "type": "string",
                        "description": "Key for the memory entry.",
                    },
                    "interaction_data": {
                        "type": "object",
                        "description": "Data to consolidate (dict).",
                    },
                },
                "required": ["mem_key", "interaction_data"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "advanced_memory_retrieve",
            "description": "Retrieve relevant memories via embedding similarity. Use before queries to augment context efficiently.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query string for similarity search.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top results (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "advanced_memory_prune",
            "description": "Prune low-salience memories to optimize storage.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "langsearch_web_search",
            "description": "Search the web using LangSearch API for relevant results, snippets, and optional summaries. Supports time filters and limits up to 10 results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (supports operators like site:example.com).",
                    },
                    "freshness": {
                        "type": "string",
                        "description": "Time filter: oneDay, oneWeek, oneMonth, oneYear, or noLimit (default).",
                        "enum": ["oneDay", "oneWeek", "oneMonth", "oneYear", "noLimit"],
                    },
                    "summary": {
                        "type": "boolean",
                        "description": "Include long text summaries (default True).",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of results (1-10, default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_embedding",
            "description": "Generate vector embedding for text using SentenceTransformer (384-dim vector). Use for semantic processing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to embed."}
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vector_search",
            "description": "Perform ANN vector search in ChromaDB using cosine similarity. Returns top matches above threshold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_embedding": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Query embedding vector.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top results (default 5).",
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Min similarity score (default 0.6).",
                    },
                },
                "required": ["query_embedding"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "chunk_text",
            "description": "Split text into semantic chunks (default 512 tokens) for processing large inputs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to chunk."},
                    "max_tokens": {
                        "type": "integer",
                        "description": "Max tokens per chunk (default 512).",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_chunk",
            "description": "Compress a text chunk via LLM summary (under 100 words), preserving key facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk": {
                        "type": "string",
                        "description": "Text chunk to summarize.",
                    }
                },
                "required": ["chunk"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "keyword_search",
            "description": "Keyword-based search on memory cache (simple overlap/BM25 sim). Returns top matches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top results (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "socratic_api_council",
            "description": "Run a BTIL/MAD-enhanced Socratic Council with multiple personas (Planner, Critic, Executor, Summarizer, Verifier, Moderator) via xAI API for iterative debate, consensus, and refinement. Model can be overridden via UI.",
            "parameters": {
                "type": "object",
                "properties": {
                    "branches": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of branch options to evaluate.",
                    },
                    "model": {
                        "type": "string",
                        "description": "LLM model (default: grok-4-fast-reasoning).",
                    },
                    "user": {
                        "type": "string",
                        "description": "User for memory consolidation (required).",
                    },
                    "convo_id": {
                        "type": "integer",
                        "description": "Conversation ID for memory (default: 0).",
                    },
                    "api_key": {
                        "type": "string",
                        "description": "API key (optional, uses global if not provided).",
                    },
                    "rounds": {
                        "type": "integer",
                        "description": "Number of debate rounds (default 3).",
                    },
                    "personas": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Custom personas (default: Planner, Critic, Executor, Summarizer, Verifier, Moderator).",
                    },
                },
                "required": ["branches", "user"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "agent_spawn",
            "description": "Spawn sub-agent (Planner, Critic, Executor, Worker, Summarizer, Verifier, Moderator, or custom) for task. Case-insensitive, custom types allowed with fallback simulation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sub_agent_type": {
                        "type": "string",
                        "description": "Type: Planner, Critic, Executor, Worker, Summarizer, Verifier, Moderator (case-insensitive) or custom.",
                    },
                    "task": {"type": "string", "description": "Task for sub-agent."},
                },
                "required": ["sub_agent_type", "task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reflect_optimize",
            "description": "Optimize component based on metrics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "component": {
                        "type": "string",
                        "description": "Component to optimize (e.g., prompt).",
                    },
                    "metrics": {
                        "type": "object",
                        "description": "Performance metrics dict.",
                    },
                },
                "required": ["component", "metrics"],
            },
        },
    },
    # New tool schemas
    {
        "type": "function",
        "function": {
            "name": "venv_create",
            "description": "Create a virtual Python environment in sandbox for isolated package installations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "env_name": {"type": "string", "description": "Name of the venv."},
                    "with_pip": {
                        "type": "boolean",
                        "description": "Include pip (default True).",
                    },
                },
                "required": ["env_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "restricted_exec",
            "description": "Execute code in a restricted namespace with optional levels (basic/full).",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code to execute."},
                    "level": {
                        "type": "string",
                        "description": "Access level: basic (default) or full.",
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "isolated_subprocess",
            "description": "Run command in an isolated subprocess with custom env.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Command to run."},
                    "custom_env": {
                        "type": "object",
                        "description": "Custom environment variables.",
                    },
                },
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pip_install",
            "description": "Install packages in a venv using pip.",
            "parameters": {
                "type": "object",
                "properties": {
                    "venv_path": {"type": "string", "description": "Path to venv."},
                    "packages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of packages.",
                    },
                    "upgrade": {
                        "type": "boolean",
                        "description": "Upgrade packages (default False).",
                    },
                },
                "required": ["venv_path", "packages"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "chat_log_analyze_embed",
            "description": "Analyze full chat log by criteria, summarize optionally, embed semantically in vector DB for recall.",
            "parameters": {
                "type": "object",
                "properties": {
                    "convo_id": {"type": "integer", "description": "Conversation ID."},
                    "criteria": {
                        "type": "string",
                        "description": "Analysis criteria (e.g., 'key topics').",
                    },
                    "summarize": {
                        "type": "boolean",
                        "description": "Summarize analysis (default True).",
                    },
                },
                "required": ["convo_id", "criteria"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "yaml_retrieve",
            "description": "Retrieve YAML content semantically or by filename from embedded DB.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Semantic query (if no filename).",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Top results for semantic (default 5).",
                    },
                    "filename": {
                        "type": "string",
                        "description": "Exact filename for retrieval (optional).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "yaml_refresh",
            "description": "Refresh YAML embedding from file system, for one or all.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Specific filename to refresh (optional; null for all).",
                    }
                },
                "required": [],
            },
        },
    },
]
# Tool Dispatcher Dictionary
TOOL_DISPATCHER = {
    "fs_read_file": fs_read_file,
    "fs_write_file": fs_write_file,
    "fs_list_files": fs_list_files,
    "fs_mkdir": fs_mkdir,
    "get_current_time": get_current_time,
    "code_execution": code_execution,
    "memory_insert": memory_insert,
    "memory_query": memory_query,
    "git_ops": git_ops,
    "db_query": db_query,
    "shell_exec": shell_exec,
    "advanced_memory_consolidate": advanced_memory_consolidate,
    "advanced_memory_retrieve": advanced_memory_retrieve,
    "advanced_memory_prune": advanced_memory_prune,
    "code_lint": code_lint,
    "api_simulate": api_simulate,
    "langsearch_web_search": langsearch_web_search,
    "generate_embedding": generate_embedding,
    "vector_search": vector_search,
    "chunk_text": chunk_text,
    "summarize_chunk": summarize_chunk,
    "keyword_search": keyword_search,
    "socratic_api_council": socratic_api_council,
    "agent_spawn": agent_spawn,
    "reflect_optimize": reflect_optimize,
    "venv_create": venv_create,
    "restricted_exec": restricted_exec,
    "isolated_subprocess": isolated_subprocess,
    "pip_install": pip_install,
    "chat_log_analyze_embed": chat_log_analyze_embed,
    "yaml_retrieve": yaml_retrieve,
    "yaml_refresh": yaml_refresh,
}
tool_count = 0
council_count = 0
main_count = 0


def process_tool_calls(tool_calls, current_messages, enable_tools):  # noqa: C901
    yield "\n*Thinking... Using tools...*\n"
    tool_outputs = []
    conn.execute("BEGIN")
    for tool_call in tool_calls:
        func_name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
            func_to_call = TOOL_DISPATCHER.get(func_name)
            if (
                func_name.startswith("memory")
                or func_name.startswith("advanced_memory")
                or func_name == "socratic_api_council"
                or func_name == "agent_spawn"
                or func_name == "reflect_optimize"
            ):
                args["user"] = st.session_state["user"]
                args["convo_id"] = st.session_state.get("current_convo_id", 0)
            # Override model for socratic_api_council with UI selection
            if func_name == "socratic_api_council":
                args["model"] = st.session_state.get(
                    "council_model_select", "grok-4-fast-reasoning"
                )
            if func_to_call:
                result = func_to_call(**args)
            else:
                result = f"Unknown tool: {func_name}"
        except Exception as e:
            result = f"Error calling tool {func_name}: {e}"
        if func_name == "socratic_api_council":
            global council_count
            council_count += 1
        else:
            global tool_count
            tool_count += 1
            st.session_state["tool_calls_per_convo"] += 1
        logger.info(f"Tool call: {func_name} - Result: {str(result)[:200]}...")
        yield f"\n> **Tool Call:** `{func_name}` | **Result:** `{str(result)[:200]}...`\n"
        tool_outputs.append(
            {"tool_call_id": tool_call.id, "role": "tool", "content": str(result)}
        )
    conn.commit()
    current_messages.extend(tool_outputs)


def call_xai_api(
    model,
    messages,
    sys_prompt,
    stream=True,
    image_files=None,
    enable_tools=False,
):  # noqa: C901
    client = OpenAI(api_key=API_KEY, base_url="https://api.x.ai/v1/", timeout=300)
    api_messages = [{"role": "system", "content": sys_prompt}]
    # Add history
    for msg in messages:
        content_parts = [{"type": "text", "text": msg["content"]}]
        # Add images only to the last user message
        if msg["role"] == "user" and image_files and msg is messages[-1]:
            for img_file in image_files:
                img_file.seek(0)
                img_data = base64.b64encode(img_file.read()).decode("utf-8")
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{img_file.type};base64,{img_data}"},
                    }
                )
        api_messages.append(
            {
                "role": msg["role"],
                "content": content_parts if len(content_parts) > 1 else msg["content"],
            }
        )

    def generate(current_messages):
        global tool_count, council_count, main_count
        max_iterations = 100
        tool_calls_per_convo = st.session_state.get("tool_calls_per_convo", 0)
        if tool_calls_per_convo > 100:  # Rate limiting
            yield "Error: Tool call limit exceeded for this conversation."
            return
        for _ in range(max_iterations):
            main_count += 1
            logger.info(
                f"API call: Tools: {tool_count} | Council: {council_count} | Main: {main_count}"
            )
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=current_messages,
                    tools=TOOLS if enable_tools else None,
                    tool_choice="auto" if enable_tools else None,
                    stream=True,
                )
                tool_calls = []
                full_delta_response = ""
                for chunk in response:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield delta.content
                        full_delta_response += delta.content
                    if delta and delta.tool_calls:
                        tool_calls.extend(delta.tool_calls)
                if not tool_calls:
                    break  # Exit loop if no tools are called
                current_messages.append(
                    {
                        "role": "assistant",
                        "content": full_delta_response,
                        "tool_calls": tool_calls,
                    }
                )
                for chunk in process_tool_calls(
                    tool_calls, current_messages, enable_tools
                ):
                    yield chunk
            except Exception as e:
                error_msg = f"API or Tool Error: {traceback.format_exc()}"
                yield f"\nAn error occurred: {e}. Aborting this turn."
                logger.error(error_msg)
                st.error(error_msg)
                break

    return generate(api_messages)


# Login Page
def login_page():
    st.title("Welcome to The ApexUltimate Interface")
    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                c.execute("SELECT password FROM users WHERE username=?", (username,))
                result = c.fetchone()
                if result and verify_password(result[0], password):
                    st.session_state["logged_in"] = True
                    st.session_state["user"] = username
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
    with tab2:
        with st.form("register_form"):
            new_user = st.text_input("New Username")
            new_pass = st.text_input("New Password", type="password")
            if st.form_submit_button("Register"):
                c.execute("SELECT * FROM users WHERE username=?", (new_user,))
                if c.fetchone():
                    st.error("Username already exists.")
                else:
                    c.execute(
                        "INSERT INTO users VALUES (?, ?)",
                        (new_user, hash_password(new_pass)),
                    )
                    conn.commit()
                    st.success("Registered! Please login.")


def load_history(convo_id):
    """Loads a specific conversation from the database into the session state."""
    c.execute(
        "SELECT messages FROM history WHERE convo_id=? AND user=?",
        (convo_id, st.session_state["user"]),
    )
    if result := c.fetchone():
        messages = json.loads(result[0])
        st.session_state["messages"] = messages
        st.session_state["current_convo_id"] = convo_id
        st.rerun()


def delete_history(convo_id):
    """Deletes a specific conversation from the database."""
    c.execute(
        "DELETE FROM history WHERE convo_id=? AND user=?",
        (convo_id, st.session_state["user"]),
    )
    conn.commit()
    # If the deleted chat was the one currently loaded, clear the session
    if st.session_state.get("current_convo_id") == convo_id:
        st.session_state["messages"] = []
        st.session_state["current_convo_id"] = 0
    st.rerun()


def search_history(query: str):
    """Search history titles by keyword."""
    c.execute(
        "SELECT convo_id, title FROM history WHERE user=? AND title LIKE ?",
        (st.session_state["user"], f"%{query}%"),
    )
    return c.fetchall()


def export_convo(format: str = "json"):
    """Export current convo as JSON or MD."""
    if format == "json":
        return json.dumps(st.session_state["messages"], indent=4)
    elif format == "md":
        md = ""
        for msg in st.session_state["messages"]:
            md += f"**{msg['role'].capitalize()}:** {msg['content']}\n\n"
        return md
    return "Unsupported format."


def render_sidebar():  # noqa: C901
    with st.sidebar:
        st.header("Chat Settings")
        st.selectbox(
            "Select Model",
            ["grok-4-fast-reasoning", "grok-4", "grok-code-fast-1", "grok-3-mini"],
            key="model_select",
        )
        st.selectbox(
            "Select Council Model",
            ["grok-4-fast-reasoning", "grok-4", "grok-code-fast-1", "grok-3-mini"],
            key="council_model_select",
        )
        prompt_files = load_prompt_files()
        if prompt_files:
            selected_file = st.selectbox(
                "Select System Prompt", prompt_files, key="prompt_select"
            )
            with open(os.path.join(PROMPTS_DIR, selected_file), "r") as f:
                prompt_content = f.read()
            st.text_area(
                "Edit System Prompt",
                value=prompt_content,
                height=200,
                key="custom_prompt",
            )
            st.checkbox("Enable Tools (Sandboxed)", value=False, key="enable_tools")
        else:
            st.warning("No prompt files found in ./prompts/")
            st.text_area(
                "System Prompt",
                value="You are a helpful AI.",
                height=200,
                key="custom_prompt",
            )
        # The key for the file uploader is important
        st.file_uploader(
            "Upload Images",
            type=["jpg", "png"],
            accept_multiple_files=True,
            key="uploaded_images",
        )
        st.divider()
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["current_convo_id"] = 0
            st.session_state["tool_calls_per_convo"] = 0
            st.rerun()
        st.header("Chat History")
        history_search = st.text_input("Search History", key="history_search")
        if history_search:
            histories = search_history(history_search)
        else:
            c.execute(
                "SELECT convo_id, title FROM history WHERE user=? ORDER BY convo_id DESC",
                (st.session_state["user"],),
            )
            histories = c.fetchall()
        for convo_id, title in histories:
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(title, key=f"load_{convo_id}", use_container_width=True):
                    load_history(convo_id)
            with col2:
                if st.button("🗑️", key=f"delete_{convo_id}", use_container_width=True):
                    delete_history(convo_id)
        st.header("Export Current Convo")
        export_format = st.selectbox("Format", ["json", "md"])
        if st.button("Export"):
            exported = export_convo(export_format)
            st.download_button("Download", exported, file_name=f"convo.{export_format}")
        st.header("Metrics Dashboard")
        if "memory_cache" in st.session_state:
            metrics = st.session_state["memory_cache"]["metrics"]
            st.metric("Total Inserts", metrics["total_inserts"])
            st.metric("Total Retrieves", metrics["total_retrieves"])
            st.metric("Hit Rate", f"{metrics['hit_rate']:.2%}")


def render_chat_interface(model, custom_prompt, enable_tools, uploaded_images):
    st.title(f"Apex Ultimate - {st.session_state['user']}")
    # --- Main Chat Interface ---
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "current_convo_id" not in st.session_state:
        st.session_state["current_convo_id"] = 0
    if "tool_calls_per_convo" not in st.session_state:
        st.session_state["tool_calls_per_convo"] = 0
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=False)
    if prompt := st.chat_input("Your command, ape?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt, unsafe_allow_html=False)
        with st.chat_message("assistant"):
            images_to_process = uploaded_images if uploaded_images else []
            generator = call_xai_api(
                model,
                st.session_state.messages,
                custom_prompt,
                stream=True,
                image_files=images_to_process,
                enable_tools=enable_tools,
            )
            full_response = st.write_stream(generator)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
        # Save to History
        title_message = next(
            (
                msg["content"]
                for msg in st.session_state.messages
                if msg["role"] == "user"
            ),
            "New Chat",
        )
        title = (
            (title_message[:40] + "...") if len(title_message) > 40 else title_message
        )
        messages_json = json.dumps(st.session_state.messages)
        if st.session_state.get("current_convo_id", 0) == 0:
            c.execute(
                "INSERT INTO history (user, title, messages) VALUES (?, ?, ?)",
                (st.session_state["user"], title, messages_json),
            )
            st.session_state.current_convo_id = c.lastrowid
        else:
            c.execute(
                "UPDATE history SET title=?, messages=? WHERE convo_id=?",
                (title, messages_json, st.session_state["current_convo_id"]),
            )
        conn.commit()
        st.rerun()


def chat_page():
    render_sidebar()
    render_chat_interface(
        st.session_state["model_select"],
        st.session_state["custom_prompt"],
        st.session_state["enable_tools"],
        st.session_state["uploaded_images"],
    )


# Auto-Prune on Startup
if "auto_prune_done" not in st.session_state:
    advanced_memory_prune(
        st.session_state.get("user"), st.session_state.get("current_convo_id")
    )
    st.session_state["auto_prune_done"] = True
# YAML Embeddings Init
if "yaml_collection" not in st.session_state:
    try:
        st.session_state["yaml_collection"] = st.session_state[
            "chroma_client"
        ].get_or_create_collection(
            name="yaml_vectors", metadata={"hnsw:space": "cosine"}
        )
        embed_model = get_embed_model()
        if embed_model:
            st.session_state["yaml_cache"] = {}
            for filename in os.listdir(YAML_DIR):
                if filename.endswith(".yaml"):
                    path = os.path.join(YAML_DIR, filename)
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                    embedding = embed_model.encode(content).tolist()
                    st.session_state["yaml_collection"].upsert(
                        ids=[filename],
                        embeddings=[embedding],
                        documents=[content],
                        metadatas=[{"filename": filename}],
                    )
                    st.session_state["yaml_cache"][filename] = content
            st.session_state["yaml_ready"] = True
        else:
            logger.warning("Embedding model not available; YAML embeddings skipped.")
            st.session_state["yaml_ready"] = False
    except Exception as e:
        logger.error(f"YAML embeddings init failed: {e}")
        st.session_state["yaml_ready"] = False
# Simple Test Function
def run_tests():
    class TestTools(unittest.TestCase):
        def test_fs_write_read(self):
            result = fs_write_file("test.txt", "Hello")
            self.assertIn("successfully", result)
            content = fs_read_file("test.txt")
            self.assertEqual(content, "Hello")

    suite = unittest.TestLoader().loadTestsFromTestCase(TestTools)
    runner = unittest.TextTestRunner()
    runner.run(suite)


# Main App Logic
if __name__ == "__main__":
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if st.session_state.get("logged_in"):
        chat_page()
    else:
        login_page()
    # Run tests in background or on demand
    # run_tests() # Uncomment to run on startup
