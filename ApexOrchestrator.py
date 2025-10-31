import base64
import html
import io
import json
import os
import shlex
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
import xml.dom.minidom
import yaml
from datetime import datetime, timedelta
from dotenv import load_dotenv
import bs4
from black import format_str, FileMode
from openai import OpenAI
from passlib.hash import sha256_crypt
from sentence_transformers import SentenceTransformer
import chromadb
import jsbeautifier
import ntplib
import numpy as np
import pygit2
import requests
import sqlparse
import streamlit as st
import tiktoken
import builtins
import venv
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
        st.session_state["chroma_client"] = chromadb.PersistentClient(path="./chroma_db")
        st.session_state["chroma_collection"] = st.session_state["chroma_client"].get_or_create_collection(
            name="memory_vectors",
            metadata={"hnsw:space": "cosine"},
        )
        st.session_state["chroma_ready"] = True
    except Exception as e:
        st.warning(f"ChromaDB init failed ({e}). Vector search will be disabled.")
        st.session_state["chroma_ready"] = False
        st.session_state["chroma_collection"] = None
def get_embed_model():
    if "embed_model" not in st.session_state:
        try:
            with st.spinner("Loading embedding model for advanced memory (first-time use)..."):
                st.session_state["embed_model"] = SentenceTransformer("all-MiniLM-L6-v2")
            st.info("Embedding model loaded successfully.")
        except Exception as e:
            st.error(f"Failed to load embedding model: {e}")
            st.session_state["embed_model"] = None
    return st.session_state.get("embed_model")
if "memory_cache" not in st.session_state:
    st.session_state["memory_cache"] = {
        "lru_cache": {},
        "vector_store": [],
        "metrics": {
            "total_inserts": 0,
            "total_retrieves": 0,
            "hit_rate": 1.0,
            "last_update": None,
        },
    }
PROMPTS_DIR = "./prompts"
os.makedirs(PROMPTS_DIR, exist_ok=True)
def load_prompt_files():
    return [f for f in os.listdir(PROMPTS_DIR) if f.endswith(".txt")]
SANDBOX_DIR = "./sandbox"
os.makedirs(SANDBOX_DIR, exist_ok=True)
st.markdown(
    """<style>
body {
    background: linear-gradient(to right, #512E5F, #1B4F72);
    color: #FFFFFF;
    font-family: 'Courier New', monospace;
}
.stApp {
    background: linear-gradient(to right, #512E5F, #1B4F72);
    display: flex;
    flex-direction: column;
    color: #FFFFFF;
    font-family: 'Courier New', monospace;
}
.sidebar .sidebar-content {
    background: rgba(81, 46, 95, 0.5);
    border-radius: 10px;
    color: #FFFFFF;
    font-family: 'Courier New', monospace;
}
.stButton > button {
    background-color: #512E5F;
    color: #FFFFFF;
    border-radius: 10px;
    border: none;
    font-family: 'Courier New', monospace;
}
.stButton > button:hover {
    background-color: #76448A;
}
[data-theme="dark"] .stApp {
    background: linear-gradient(to right, #512E5F, #1B4F72);
    color: #FFFFFF;
    font-family: 'Courier New', monospace;
}
</style>
""",
    unsafe_allow_html=True,
)
def hash_password(password):
    return sha256_crypt.hash(password)
def verify_password(stored, provided):
    return sha256_crypt.verify(provided, stored)
def get_tool_cache_key(func_name, args):
    return f"tool_cache:{func_name}:{hash(json.dumps(args, sort_keys=True))}"
def get_cached_tool_result(func_name, args, ttl_minutes=15):
    if "tool_cache" not in st.session_state:
        st.session_state["tool_cache"] = {}
    cache = st.session_state["tool_cache"]
    key = get_tool_cache_key(func_name, args)
    if key in cache:
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
def fs_read_file(file_path: str) -> str:
    safe_path = os.path.abspath(os.path.normpath(os.path.join(SANDBOX_DIR, file_path)))
    if not safe_path.startswith(os.path.abspath(SANDBOX_DIR)):
        return "Error: Path is outside the sandbox."
    try:
        with open(safe_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Error: File not found."
    except Exception as e:
        return f"Error reading file: {e}"
def fs_write_file(file_path: str, content: str) -> str:
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
        return f"Error writing file: {e}"
def fs_list_files(dir_path: str = "") -> str:
    safe_dir = os.path.abspath(os.path.normpath(os.path.join(SANDBOX_DIR, dir_path)))
    if not safe_dir.startswith(os.path.abspath(SANDBOX_DIR)):
        return "Error: Path is outside the sandbox."
    try:
        files = os.listdir(safe_dir)
        return f"Files in '{dir_path or '/'}': {json.dumps(files)}"
    except FileNotFoundError:
        return "Error: Directory not found."
    except Exception as e:
        return f"Error listing files: {e}"
def fs_mkdir(dir_path: str) -> str:
    safe_path = os.path.abspath(os.path.normpath(os.path.join(SANDBOX_DIR, dir_path)))
    if not safe_path.startswith(os.path.abspath(SANDBOX_DIR)):
        return "Error: Path is outside the sandbox."
    try:
        os.makedirs(safe_path, exist_ok=True)
        return f"Directory '{dir_path}' created successfully."
    except Exception as e:
        return f"Error creating directory: {e}"
def get_current_time(sync: bool = False, format: str = "iso") -> str:
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
                {"datetime": dt_object.isoformat(), "timezone": time.localtime().tm_zone}
            )
        else:
            return dt_object.isoformat()
    except Exception as e:
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
ADDITIONAL_LIBS = {
    'numpy': np,
}
def code_execution(code: str, venv_path: str = None) -> str:
    if "repl_namespace" not in st.session_state:
        st.session_state["repl_namespace"] = {"__builtins__": SAFE_BUILTINS.copy()}
        st.session_state["repl_namespace"].update(ADDITIONAL_LIBS)
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output
    try:
        if venv_path:
            safe_venv = os.path.abspath(os.path.normpath(os.path.join(SANDBOX_DIR, venv_path)))
            if not safe_venv.startswith(os.path.abspath(SANDBOX_DIR)):
                return "Error: Venv path outside sandbox."
            venv_python = os.path.join(safe_venv, 'bin', 'python')
            if not os.path.exists(venv_python):
                return "Error: Venv Python not found."
            result = subprocess.run([venv_python, '-c', code], capture_output=True, text=True, timeout=30)
            output = result.stdout
            if result.stderr:
                return f"Error: {result.stderr}"
        else:
            exec(code, st.session_state["repl_namespace"])
            output = redirected_output.getvalue()
        return f"Output:\n{output}" if output else "Execution successful (no output)."
    except Exception as e:
        return f"Error: {traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout
def memory_insert(mem_key: str, mem_value: dict, user: str = None, convo_id: int = None) -> str:
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
        return "Memory inserted successfully."
    except Exception as e:
        return f"Error inserting memory: {e}"
def memory_query(
    mem_key: str = None, limit: int = 10, user: str = None, convo_id: int = None
) -> str:
    if user is None or convo_id is None:
        return "Error: user and convo_id required for memory query."
    try:
        if mem_key:
            c.execute(
                "SELECT mem_value FROM memory WHERE user=? AND convo_id=? AND mem_key=?",
                (user, convo_id, mem_key),
            )
            result = c.fetchone()
            return result[0] if result else "Key not found."
        else:
            c.execute(
                "SELECT mem_key, mem_value FROM memory WHERE user=? AND convo_id=? ORDER BY timestamp DESC LIMIT ?",
                (user, convo_id, limit),
            )
            results = {row[0]: json.loads(row[1]) for row in c.fetchall()}
            if "memory_cache" in st.session_state:
                for key in results:
                    if key not in st.session_state["memory_cache"]["lru_cache"]:
                        entry = results[key]
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
            return json.dumps(results)
    except Exception as e:
        return f"Error querying memory: {e}"
def advanced_memory_consolidate(
    mem_key: str, interaction_data: dict, user: str = None, convo_id: int = None
) -> str:
    if user is None or convo_id is None:
        return "Error: user and convo_id required."
    cache_args = {"mem_key": mem_key, "interaction_data": interaction_data}
    cached = get_cached_tool_result("advanced_memory_consolidate", cache_args)
    if cached:
        return cached
    embed_model = get_embed_model()
    try:
        client = OpenAI(api_key=API_KEY, base_url="https://api.x.ai/v1/")
        summary_response = client.chat.completions.create(
            model="grok-4-fast-non-reasoning",
            messages=[
                {"role": "system", "content": "Summarize this interaction concisely in one paragraph."},
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
        if embed_model and st.session_state.get("chroma_ready") and st.session_state.get("chroma_collection"):
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
        return result
    except Exception as e:
        return f"Error consolidating memory: {traceback.format_exc()}"
def advanced_memory_retrieve(
    query: str, top_k: int = 5, user: str = None, convo_id: int = None
) -> str:
    if user is None or convo_id is None:
        return "Error: user and convo_id required."
    cache_args = {"query": query, "top_k": top_k}
    cached = get_cached_tool_result("advanced_memory_retrieve", cache_args)
    if cached:
        return cached
    embed_model = get_embed_model()
    chroma_col = st.session_state.get("chroma_collection")
    if not embed_model or not st.session_state.get("chroma_ready") or not chroma_col:
        st.warning("Vector memory not available; falling back to keyword search.")
        fallback_results = keyword_search(query, top_k, user, convo_id)
        if isinstance(fallback_results, str) and "error" in fallback_results.lower():
            return fallback_results
        retrieved = []
        for res in fallback_results:
            mem_key = res["id"]
            value = json.loads(memory_query(mem_key=mem_key, user=user, convo_id=convo_id))
            retrieved.append({
                "mem_key": mem_key,
                "value": value,
                "relevance": res["score"],
                "summary": value.get("summary", ""),
            })
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
        if not results or not results.get("ids") or not results["ids"]:
            return "No relevant memories found."
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
            chroma_col.update(ids=ids_to_update, metadatas=metadata_to_update)
        retrieved.sort(key=lambda x: x["relevance"], reverse=True)
        if "memory_cache" in st.session_state:
            st.session_state["memory_cache"]["metrics"]["total_retrieves"] += 1
            hit_rate = len(retrieved) / top_k if top_k > 0 else 1.0
            st.session_state["memory_cache"]["metrics"]["hit_rate"] = (
                (
                    st.session_state["memory_cache"]["metrics"]["hit_rate"]
                    * (st.session_state["memory_cache"]["metrics"]["total_retrieves"] - 1)
                )
                + hit_rate
            ) / st.session_state["memory_cache"]["metrics"]["total_retrieves"]
        result = json.dumps(retrieved)
        set_cached_tool_result("advanced_memory_retrieve", cache_args, result)
        return result
    except Exception as e:
        return (
            f"Error retrieving memory: {traceback.format_exc()}. "
            "If this is a where clause issue, check filter structure."
        )
def advanced_memory_prune(user: str = None, convo_id: int = None) -> str:
    if user is None or convo_id is None:
        return "Error: user and convo_id required."
    if "prune_counter" not in st.session_state:
        st.session_state["prune_counter"] = 0
    st.session_state["prune_counter"] += 1
    if st.session_state["prune_counter"] % 10 != 0:
        return "Prune skipped (infrequent)."
    try:
        conn.execute("BEGIN")
        one_week_ago = datetime.now() - timedelta(days=7)
        c.execute(
            "UPDATE memory SET salience = salience * 0.99 WHERE user=? AND convo_id=? AND timestamp < ?",
            (user, convo_id, one_week_ago),
        )
        c.execute(
            "DELETE FROM memory WHERE user=? AND convo_id=? AND salience < 0.1", (user, convo_id)
        )
        c.execute(
            "SELECT COUNT(*) FROM memory WHERE user=? AND convo_id=?", (user, convo_id)
        )
        row_count = c.fetchone()[0]
        if row_count > 1000:
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
        c.execute(
            "SELECT mem_key, mem_value FROM memory WHERE user=? AND convo_id=?", (user, convo_id)
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
                "DELETE FROM memory WHERE user=? AND convo_id=? AND mem_key=?", (user, convo_id, key)
            )
        if "memory_cache" in st.session_state and len(
            st.session_state["memory_cache"]["lru_cache"]
        ) > 1000:
            lru_items = sorted(
                st.session_state["memory_cache"]["lru_cache"].items(),
                key=lambda x: x[1]["last_access"],
            )
            num_to_evict = len(lru_items) - 1000
            for key, _ in lru_items[:num_to_evict]:
                entry = st.session_state["memory_cache"]["lru_cache"][key]["entry"]
                if entry["salience"] < 0.4:
                    del st.session_state["memory_cache"]["lru_cache"][key]
                    c.execute(
                        "DELETE FROM memory WHERE user=? AND convo_id=? AND mem_key=?",
                        (user, convo_id, key),
                    )
        conn.commit()
        if "memory_cache" in st.session_state:
            st.session_state["memory_cache"]["metrics"]["last_update"] = datetime.now().isoformat()
        return "Memory pruned successfully."
    except Exception as e:
        conn.rollback()
        return f"Error pruning memory: {e}"
def generate_embedding(text: str) -> list:
    embed_model = get_embed_model()
    if not embed_model:
        return [0.0] * 384
    try:
        embedding = embed_model.encode(text).tolist()
        return embedding
    except Exception as e:
        return f"Embedding error: {e}"
def vector_search(
    query_embedding: list, top_k: int = 5, threshold: float = 0.6
) -> list:
    chroma_col = st.session_state.get("chroma_collection")
    if not st.session_state.get("chroma_ready") or not chroma_col:
        return []
    try:
        results = chroma_col.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={},
            include=["distances", "metadatas", "documents"],
        )
        if not results or not results.get("ids") or not results["ids"]:
            return []
        retrieved = []
        for i in range(len(results["ids"][0])):
            dist = results["distances"][0][i]
            sim = 1 - dist
            if sim > threshold:
                retrieved.append(
                    {
                        "id": results["ids"][0][i],
                        "score": float(sim),
                        "metadata": results["metadatas"][0][i],
                        "document": results["documents"][0][i],
                    }
                )
        retrieved.sort(key=lambda x: x["score"], reverse=True)
        if "memory_cache" in st.session_state:
            st.session_state["memory_cache"]["metrics"]["total_retrieves"] += 1
        return retrieved[:top_k]
    except Exception as e:
        return f"Vector search error: {e}"
def chunk_text(text: str, max_tokens: int = 512) -> list:
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i : i + max_tokens]
            chunks.append(encoding.decode(chunk_tokens))
        return chunks
    except ImportError:
        words = text.split()
        chunks = []
        current = []
        current_len = 0
        for word in words:
            if current_len + len(word.split()) > max_tokens / 4:
                chunks.append(" ".join(current))
                current = [word]
                current_len = len(word.split())
            else:
                current.append(word)
                current_len += len(word.split())
        if current:
            chunks.append(" ".join(current))
        return chunks
def summarize_chunk(chunk: str) -> str:
    client = OpenAI(api_key=API_KEY, base_url="https://api.x.ai/v1/")
    try:
        response = client.chat.completions.create(
            model="grok-4-fast-non-reasoning",
            messages=[
                {
                    "role": "system",
                    "content": "Summarize this text concisely (under 100 words), preserving key facts.",
                },
                {"role": "user", "content": chunk},
            ],
            max_tokens=100,
            stream=False,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Summarization error: {e} (raw chunk: {chunk[:200]}...)"
def keyword_search(
    query: str, top_k: int = 5, user: str = None, convo_id: int = None
) -> list:
    if user is None or convo_id is None:
        return []
    try:
        c.execute(
            "SELECT mem_key, mem_value FROM memory WHERE user=? AND convo_id=?",
            (user, convo_id),
        )
        rows = c.fetchall()
        scores = {}
        query_words = set(query.lower().split())
        for key, value_str in rows:
            value = json.loads(value_str)
            text = f"{value.get('summary', '')} {value.get('details', '')}".lower()
            text_words = set(text.split())
            score = (
                len(query_words.intersection(text_words)) / len(query_words)
                if query_words
                else 0
            )
            scores[key] = score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = [{"id": key, "score": score} for key, score in sorted_scores[:top_k]]
        if "memory_cache" in st.session_state:
            for res in results:
                key = res["id"]
                if key in st.session_state["memory_cache"]["lru_cache"]:
                    st.session_state["memory_cache"]["lru_cache"][key]["last_access"] = time.time()
        return results
    except Exception as e:
        return f"Keyword search error: {e}"
def git_ops(operation: str, repo_path: str = "", **kwargs) -> str:
    safe_repo = os.path.abspath(os.path.normpath(os.path.join(SANDBOX_DIR, repo_path)))
    if not safe_repo.startswith(os.path.abspath(SANDBOX_DIR)):
        return "Invalid repo path."
    try:
        if operation == "init":
            pygit2.init_repository(safe_repo, bare=False)
            return "Repository initialized."
        repo = pygit2.Repository(safe_repo)
        if operation == "commit":
            index = repo.index
            index.add_all()
            index.write()
            tree = index.write_tree()
            author = pygit2.Signature("AI User", "ai@example.com")
            parents = [repo.head.target] if not repo.head_is_unborn else []
            repo.create_commit(
                "HEAD",
                author,
                author,
                kwargs.get("message", "Auto-commit"),
                tree,
                parents,
            )
            return "Changes committed."
        elif operation == "diff":
            return repo.diff().patch or "No differences."
        return "Unsupported Git operation."
    except Exception as e:
        return f"Git error: {e}"
def db_query(db_path: str, query: str, params: list = []) -> str:
    safe_db = os.path.abspath(os.path.normpath(os.path.join(SANDBOX_DIR, db_path)))
    if not safe_db.startswith(os.path.abspath(SANDBOX_DIR)):
        return "Invalid DB path."
    try:
        with sqlite3.connect(safe_db) as db_conn:
            cur = db_conn.cursor()
            cur.execute(query, params)
            if query.strip().upper().startswith("SELECT"):
                return json.dumps(cur.fetchall())
            else:
                db_conn.commit()
                return f"{cur.rowcount} rows affected."
    except Exception as e:
        return f"DB error: {e}"
WHITELISTED_COMMANDS = ["ls", "grep", "sed", "cat", "echo", "pwd", "pip", "apt", "bash"]
def shell_exec(command: str) -> str:
    cmd_parts = shlex.split(command)
    if not cmd_parts or cmd_parts[0] not in WHITELISTED_COMMANDS:
        return "Command not whitelisted."
    try:
        result = subprocess.run(
            cmd_parts,
            cwd=SANDBOX_DIR,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return (
            result.stdout.strip()
            + ("\nError: " + result.stderr.strip() if result.stderr else "")
        )
    except Exception as e:
        return f"Shell error: {e}"
def code_lint(language: str, code: str) -> str:
    lang = language.lower()
    try:
        if lang == "python":
            formatted = format_str(code, mode=FileMode(line_length=88))
        elif lang == "javascript":
            opts = jsbeautifier.default_options()
            formatted = jsbeautifier.beautify(code, opts)
        elif lang == "css":
            opts = jsbeautifier.default_options()
            formatted = jsbeautifier.beautify(code, opts)
        elif lang == "json":
            formatted = json.dumps(json.loads(code), indent=4)
        elif lang == "yaml":
            formatted = yaml.safe_dump(yaml.safe_load(code), indent=2)
        elif lang == "sql":
            formatted = sqlparse.format(code, reindent=True, keyword_case="upper")
        elif lang == "xml":
            dom = xml.dom.minidom.parseString(code)
            formatted = dom.toprettyxml(indent=" ")
        elif lang == "html":
            soup = bs4.BeautifulSoup(code, "html.parser")
            formatted = soup.prettify()
        elif lang in ["c", "cpp", "c++"]:
            try:
                formatted = (
                    subprocess.check_output(
                        ["clang-format", "-style=google"], input=code.encode()
                    ).decode()
                )
            except Exception as e:
                return f"clang-format not available: {str(e)}"
        elif lang == "php":
            try:
                with tempfile.NamedTemporaryFile(suffix=".php", delete=False) as tmp:
                    tmp.write(code.encode())
                    tmp.flush()
                    subprocess.check_call(
                        ["php-cs-fixer", "fix", tmp.name, "--quiet"], check=True
                    )
                    with open(tmp.name, "r") as f:
                        formatted = f.read()
                os.unlink(tmp.name)
            except Exception as e:
                return f"php-cs-fixer not available: {str(e)}"
        elif lang == "go":
            try:
                formatted = (
                    subprocess.check_output(["gofmt"], input=code.encode()).decode()
                )
            except Exception as e:
                return f"gofmt not available: {str(e)}"
        elif lang == "rust":
            try:
                formatted = (
                    subprocess.check_output(
                        ["rustfmt", "--emit=stdout"], input=code.encode()
                    ).decode()
                )
            except Exception as e:
                return f"rustfmt not available: {str(e)}"
        else:
            return "Unsupported language."
        return formatted
    except Exception as e:
        return f"Lint error: {str(e)}"
API_WHITELIST = [
    "https://jsonplaceholder.typicode.com/",
    "https://api.openweathermap.org/",
    "https://docs.sunoapi.org/suno-api/",
    "https://docs.sunoapi.org/llms.txt",
    "https://archive.org/",
]
def api_simulate(url: str, method: str = "GET", data: dict = None, mock: bool = False) -> str:
    cache_args = {"url": url, "method": method, "data": data, "mock": mock}
    cached = get_cached_tool_result("api_simulate", cache_args)
    if cached:
        return cached
    if mock:
        result = json.dumps(
            {"status": "mocked", "url": url, "method": method, "data": data}
        )
    else:
        if not any(url.startswith(base) for base in API_WHITELIST):
            result = "URL not in whitelist."
        else:
            try:
                if method.upper() == "GET":
                    resp = requests.get(url, timeout=5)
                elif method.upper() == "POST":
                    resp = requests.post(url, json=data, timeout=5)
                else:
                    result = "Unsupported method."
                    set_cached_tool_result("api_simulate", cache_args, result)
                    return result
                resp.raise_for_status()
                result = resp.text
            except Exception as e:
                result = f"API error: {str(e)}"
    set_cached_tool_result("api_simulate", cache_args, result)
    return result
def langsearch_web_search(
    query: str, freshness: str = "noLimit", summary: bool = False, count: int = 5
) -> str:
    if not LANGSEARCH_API_KEY:
        return "LangSearch API key not set—configure in .env."
    url = "https://api.langsearch.com/v1/web-search"
    payload = json.dumps(
        {
            "query": query,
            "freshness": freshness,
            "summary": summary,
            "count": count,
        }
    )
    headers = {
        "Authorization": f"Bearer {LANGSEARCH_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        return json.dumps(response.json())
    except Exception as e:
        return f"LangSearch error: {str(e)}"
def socratic_api_council(
    branches: list,
    model: str = "grok-4-fast-reasoning",
    user: str = None,
    convo_id: int = 0,
    api_key: str = None,
    personas: list = None,
) -> str:
    if user is None:
        return "Error: user required for memory consolidation in council."
    if not api_key:
        api_key = API_KEY
    if not api_key:
        return "Error: API key not available."
    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1/")
    if not personas:
        personas = [
            {
                "name": "Planner",
                "prompt": "As Planner, outline feasible tasks from these branches: {branches}. Prioritize actionable paths.",
            },
            {
                "name": "Critic",
                "prompt": "As Critic, identify risks and flaws in these branches: {branches}. Flag potential errors or loops.",
            },
            {
                "name": "Executor",
                "prompt": "As Executor, select the safest branch to execute from: {branches}. Focus on tool utilization and stability.",
            },
        ]
    if all(":" in branch for branch in branches):
        dynamic_personas = []
        for branch in branches:
            role, task = branch.split(":", 1)
            dynamic_personas.append({
                "name": role.strip(),
                "prompt": f"As {role.strip()}, evaluate and respond to: {task.strip()}. Provide specialized insights.",
            })
        personas = dynamic_personas
    verdicts = []
    for persona in personas:
        for attempt in range(2):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": persona["prompt"].format(
                                branches=json.dumps(branches)
                            ),
                        },
                    ],
                    stream=False,
                )
                verdict = response.choices[0].message.content.strip()
                if "SIM_" in verdict:
                    verdict = "[Bleed Detected: Rerouted] Simulated verdict cleaned."
                confidence = min(1.0, len(verdict) / 500.0) if verdict else 0.5
                verdicts.append(f"{persona['name']}: {verdict} (Confidence: {confidence:.2f})")
                mem_key = f"council_{persona['name'].lower()}_{uuid.uuid4()}"
                interaction_data = {
                    "persona": persona["name"],
                    "verdict": verdict,
                    "branches": branches,
                    "summary": f"{persona['name']} verdict on branches",
                    "details": verdict,
                    "tags": ["council", "socratic"],
                    "domain": "debate",
                    "salience": confidence,
                }
                consolidate_result = advanced_memory_consolidate(
                    mem_key=mem_key,
                    interaction_data=interaction_data,
                    user=user,
                    convo_id=convo_id,
                )
                if "Error" in consolidate_result:
                    print(
                        f"Warning: Consolidation failed for {persona['name']}: {consolidate_result}"
                    )
                else:
                    print(f"Consolidated: {persona['name']} verdict for convo {convo_id}")
                break
            except Exception as e:
                if attempt == 1:
                    verdicts.append(f"{persona['name']}: Error - {e}")
    if verdicts:
        synth_prompt = "As Judge, synthesize these verdicts into a consensus: {verdicts}. Use weighted voting based on confidence scores."
        try:
            synth_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": synth_prompt.format(verdicts="\n".join(verdicts))},
                ],
                stream=False,
            )
            synthesis = synth_response.choices[0].message.content.strip()
            if "SIM_" in synthesis:
                synthesis = "[Bleed Detected: Rerouted] Simulated synthesis cleaned."
            aggregated = "\n".join(verdicts) + f"\n\nSynthesized Consensus: {synthesis}"
        except Exception as e:
            aggregated = " | ".join(verdicts) + " | Aggregation Error: " + str(e)
    else:
        aggregated = "No verdicts generated."
    return aggregated

def agent_spawn(sub_agent_type: str, task: str, user: str = None, convo_id: int = None) -> str:
    if user is None or convo_id is None:
        return "Error: user and convo_id required."
    prompts = {"Planner": "Decompose task into steps.", "Critic": "Debug and critique plan.", "Executor": "Execute using tools."}
    if sub_agent_type not in prompts:
        return "Invalid sub-agent."
    client = OpenAI(api_key=API_KEY, base_url="https://api.x.ai/v1/")
    response = client.chat.completions.create(model="grok-4-fast-reasoning", messages=[{"role": "system", "content": prompts[sub_agent_type]}, {"role": "user", "content": task}])
    result = response.choices[0].message.content
    memory_insert(f"{sub_agent_type}_{uuid.uuid4()}", {"task": task, "result": result}, user, convo_id)
    return result

def reflect_optimize(component: str, metrics: dict, user: str = None, convo_id: int = None) -> str:
    if user is None or convo_id is None:
        return "Error: user and convo_id required."
    branches = [f"Optimize {component} given metrics: {json.dumps(metrics)}"]
    verdict = socratic_api_council(branches, user=user, convo_id=convo_id)
    fs_write_file(f"optimized_{component}.txt", verdict)
    return "Optimized."

def venv_create(env_name: str, with_pip: bool = True) -> str:
    env_path = os.path.abspath(os.path.normpath(os.path.join(SANDBOX_DIR, env_name)))
    if not env_path.startswith(os.path.abspath(SANDBOX_DIR)):
        return "Error: Path outside sandbox."
    try:
        builder = venv.EnvBuilder(with_pip=with_pip, clear=True)
        builder.create(env_path)
        return f"Virtual env created at {env_name}"
    except Exception as e:
        return f"Error creating venv: {e}"

def restricted_exec(code: str, level: str = "basic") -> str:
    restricted_globals = {'__builtins__': SAFE_BUILTINS.copy()}
    if level == "full":
        restricted_globals.update(ADDITIONAL_LIBS)
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output
    try:
        exec(code, restricted_globals)
        output = redirected_output.getvalue()
        return f"Output:\n{output}" if output else "Execution successful."
    except Exception as e:
        return f"Error: {traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout

def isolated_subprocess(cmd: str, custom_env: dict = None) -> str:
    safe_cwd = os.path.abspath(os.path.normpath(os.path.join(SANDBOX_DIR, 'temp')))
    os.makedirs(safe_cwd, exist_ok=True)
    env = os.environ.copy() if custom_env is None else custom_env
    try:
        result = subprocess.run(shlex.split(cmd), cwd=safe_cwd, env=env, capture_output=True, text=True, timeout=30)
        return result.stdout.strip() + (f"\nError: {result.stderr.strip()}" if result.stderr else "")
    except Exception as e:
        return f"Subprocess error: {e}"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fs_read_file",
            "description": "Read the content of a file in the sandbox directory (./sandbox/). Supports relative paths (e.g., 'subdir/test.txt'). Use for fetching data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative path to the file (e.g., subdir/test.txt).",
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
            "description": "Write content to a file in the sandbox directory (./sandbox/). Supports relative paths (e.g., 'subdir/newfile.txt'). Use for saving or updating files. If 'Love' is in file_path or content, optionally add ironic flair like 'LOVE <3' for fun.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative path to the file (e.g., subdir/newfile.txt).",
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
            "description": "List all files in a directory within the sandbox (./sandbox/). Supports relative paths (default: root). Use to check available files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir_path": {
                        "type": "string",
                        "description": "Relative path to the directory (e.g., subdir). Optional; defaults to root.",
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
            "description": "Create a new directory in the sandbox (./sandbox/). Supports relative/nested paths (e.g., 'subdir/newdir'). Use to organize files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir_path": {
                        "type": "string",
                        "description": "Relative path for the new directory (e.g., subdir/newdir).",
                    }
                },
                "required": ["dir_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Fetch current datetime. Use host clock by default; sync with NTP if requested for precision.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sync": {
                        "type": "boolean",
                        "description": "True for NTP sync (requires network), false for local host time. Default: false.",
                    },
                    "format": {
                        "type": "string",
                        "description": "Output format: 'iso' (default), 'human', 'json'.",
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
            "description": "Execute provided code in a stateful REPL environment and return output or errors for verification. Supports Python with various libraries (e.g., numpy, sympy, pygame). No internet access or package installation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "The code snippet to execute."},
                    "venv_path": {"type": "string", "description": "Optional path to venv for execution."}
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_insert",
            "description": "Insert or update a memory key-value pair (value as JSON dict) for logging/metadata. Use for fast persistent storage without files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mem_key": {
                        "type": "string",
                        "description": "Key for the memory entry (e.g., 'chat_log_1').",
                    },
                    "mem_value": {
                        "type": "object",
                        "description": "Value as dict (e.g., {'content': 'Log text'}).",
                    },
                },
                "required": ["mem_key", "mem_value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_query",
            "description": "Query memory: specific key or last N entries. Returns JSON. Use for recalling logs without FS reads.",
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
                    "operation": {"type": "string", "enum": ["init", "commit", "branch", "diff"]},
                    "repo_path": {"type": "string", "description": "Relative path to repo."},
                    "message": {
                        "type": "string",
                        "description": "Commit message (for commit).",
                    },
                    "name": {"type": "string", "description": "Branch name (for branch)."},
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
                    "db_path": {"type": "string", "description": "Relative path to DB file."},
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
                    "command": {"type": "string", "description": "Shell command string."}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_lint",
            "description": "Lint and auto-format code for languages: python (black), javascript (jsbeautifier), css (cssbeautifier), json, yaml, sql (sqlparse), xml, html (beautifulsoup), cpp/c++ (clang-format), php (php-cs-fixer), go (gofmt), rust (rustfmt). External tools required for some.",
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
                    "method": {"type": "string", "description": "GET/POST (default GET)."},
                    "data": {"type": "object", "description": "POST data."},
                    "mock": {"type": "boolean", "description": "True for mock (default)."},
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
                    "mem_key": {"type": "string", "description": "Key for the memory entry."},
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
                    "query": {"type": "string", "description": "Query string for similarity search."},
                    "top_k": {"type": "integer", "description": "Number of top results (default 5)."},
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
                    "summary": {"type": "boolean", "description": "Include long text summaries (default True)."},
                    "count": {"type": "integer", "description": "Number of results (1-10, default 5)."},
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
                    "top_k": {"type": "integer", "description": "Number of top results (default 5)."},
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
                    "chunk": {"type": "string", "description": "Text chunk to summarize."}
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
                    "top_k": {"type": "integer", "description": "Number of top results (default 5)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "socratic_api_council",
            "description": "Run a Socratic Council with configurable personas via xAI API to evaluate branches. Supports dynamic roles for swarm integration. Integrates with advanced memory for consolidation. Model can be overridden via UI.",
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
                    "personas": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "prompt": {"type": "string"}
                            }
                        },
                        "description": "Custom list of personas (each with name and prompt template). Defaults to Planner/Critic/Executor.",
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
            "description": "Spawn sub-agent (Planner/Critic/Executor) for task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sub_agent_type": {
                        "type": "string",
                        "description": "Type: Planner, Critic, Executor."
                    },
                    "task": {
                        "type": "string",
                        "description": "Task for sub-agent."
                    }
                },
                "required": ["sub_agent_type", "task"]
            }
        }
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
                        "description": "Component to optimize (e.g., prompt)."
                    },
                    "metrics": {
                        "type": "object",
                        "description": "Performance metrics dict."
                    }
                },
                "required": ["component", "metrics"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "venv_create",
            "description": "Create a virtual Python environment in sandbox for isolated package installations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "env_name": {"type": "string", "description": "Name of the venv."},
                    "with_pip": {"type": "boolean", "description": "Include pip (default True)."}
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
                    "level": {"type": "string", "description": "Access level: basic (default) or full."}
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
                    "custom_env": {"type": "object", "description": "Custom environment variables."}
                },
                "required": ["cmd"],
            },
        },
    }
]
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
}
tool_count = 0
council_count = 0
main_count = 0
def call_xai_api(
    model,
    messages,
    sys_prompt,
    stream=True,
    image_files=None,
    enable_tools=False,
):
    client = OpenAI(api_key=API_KEY, base_url="https://api.x.ai/v1/", timeout=300)
    api_messages = [{"role": "system", "content": sys_prompt}]
    for msg in messages:
        content_parts = [{"type": "text", "text": msg["content"]}]
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
        max_iterations = 10
        for _ in range(max_iterations):
            main_count += 1
            print(f"\rTools: {tool_count} | Council: {council_count} | Main: {main_count}", end='')
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
                    break
                current_messages.append(
                    {"role": "assistant", "content": full_delta_response, "tool_calls": tool_calls}
                )
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
                        council_count += 1
                    else:
                        tool_count += 1
                    print(f"\rTools: {tool_count} | Council: {council_count} | Main: {main_count}", end='')
                    if func_name == "socratic_api_council":
                        print(f"\nCouncil Call {council_count}: {result}")
                    yield f"\n> **Tool Call:** `{func_name}` | **Result:** `{str(result)[:200]}...`\n"
                    tool_outputs.append(
                        {"tool_call_id": tool_call.id, "role": "tool", "content": str(result)}
                    )
                conn.commit()
                current_messages.extend(tool_outputs)
            except Exception as e:
                error_msg = f"API or Tool Error: {traceback.format_exc()}"
                yield f"\nAn error occurred: {e}. Aborting this turn."
                st.error(error_msg)
                break
    return generate(api_messages)
def login_page():
    st.title("Welcome to The Apex Interface")
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
                        "INSERT INTO users VALUES (?, ?)", (new_user, hash_password(new_pass))
                    )
                    conn.commit()
                    st.success("Registered! Please login.")
def load_history(convo_id):
    c.execute(
        "SELECT messages FROM history WHERE convo_id=? AND user=?", (convo_id, st.session_state["user"])
    )
    result = c.fetchone()
    if result:
        messages = json.loads(result[0])
        st.session_state["messages"] = messages
        st.session_state["current_convo_id"] = convo_id
        st.rerun()
def delete_history(convo_id):
    c.execute(
        "DELETE FROM history WHERE convo_id=? AND user=?", (convo_id, st.session_state["user"])
    )
    conn.commit()
    if st.session_state.get("current_convo_id") == convo_id:
        st.session_state["messages"] = []
        st.session_state["current_convo_id"] = 0
    st.rerun()
def chat_page():
    st.title(f"Apex Interface - {st.session_state['user']}")
    with st.sidebar:
        st.header("Chat Settings")
        model = st.selectbox(
            "Select Model",
            ["grok-4-fast-reasoning", "grok-4", "grok-code-fast-1", "grok-3-mini"],
            key="model_select",
        )
        council_model = st.selectbox(
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
            if "previous_selected_prompt" not in st.session_state:
                st.session_state.previous_selected_prompt = selected_file
            if st.session_state.previous_selected_prompt != selected_file:
                st.session_state.previous_selected_prompt = selected_file
                st.session_state["custom_prompt"] = prompt_content
            custom_prompt = st.text_area(
                "Edit System Prompt", value=prompt_content, height=200, key="custom_prompt"
            )
        else:
            st.warning("No prompt files found in ./prompts/")
            custom_prompt = st.text_area(
                "System Prompt", value="You are a helpful AI.", height=200, key="custom_prompt"
            )
        uploaded_images = st.file_uploader(
            "Upload Images",
            type=["jpg", "png"],
            accept_multiple_files=True,
            key="uploaded_images",
        )
        enable_tools = st.checkbox("Enable Tools (Sandboxed)", value=False, key="enable_tools")
        st.divider()
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["current_convo_id"] = 0
            st.rerun()
        st.header("Chat History")
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
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "current_convo_id" not in st.session_state:
        st.session_state["current_convo_id"] = 0
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
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        title_message = next(
            (msg["content"] for msg in st.session_state.messages if msg["role"] == "user"),
            "New Chat",
        )
        title = (title_message[:40] + "...") if len(title_message) > 40 else title_message
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
                (title, messages_json, st.session_state.current_convo_id),
            )
        conn.commit()
        st.rerun()
if __name__ == "__main__":
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if st.session_state.get("logged_in"):
        chat_page()
    else:
        login_page()
