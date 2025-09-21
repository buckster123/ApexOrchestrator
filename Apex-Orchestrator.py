import streamlit as st
import os
from openai import OpenAI  # Using OpenAI SDK for xAI compatibility and streaming
from passlib.hash import sha256_crypt
import sqlite3
from dotenv import load_dotenv
import json
import time
import base64  # For image handling
import traceback  # For error logging
import ntplib  # For NTP time sync; pip install ntplib
import io  # For capturing code output
import sys  # For stdout redirection
import pygit2  # For git_ops; pip install pygit2
import subprocess
import requests  # For api_simulate; pip install requests
from black import format_str, FileMode  # For code_lint; pip install black
import numpy as np  # For embeddings
from sentence_transformers import SentenceTransformer  # For advanced memory; pip install sentence-transformers torch
from datetime import datetime, timedelta  # For pruning
import jsbeautifier  # For JS/CSS linting; pip install jsbeautifier
import yaml  # For YAML; pip install pyyaml
import sqlparse  # For SQL; pip install sqlparse
from bs4 import BeautifulSoup  # For HTML; pip install beautifulsoup4
import xml.dom.minidom  # Built-in for XML
import tempfile  # For temp files in linting
import shlex  # For safe shell splitting
import builtins  # For restricted globals
import chromadb  # For vector storage; pip install chromadb
import uuid  # For unique IDs

# Load environment variables
load_dotenv()
API_KEY = os.getenv("XAI_API_KEY")
if not API_KEY:
    st.error("XAI_API_KEY not set in .env! Please add it and restart.")
LANGSEARCH_API_KEY = os.getenv("LANGSEARCH_API_KEY")
if not LANGSEARCH_API_KEY:
    st.warning("LANGSEARCH_API_KEY not set in .env—web search tool will fail.")

# Database Setup (SQLite for users and history) with WAL mode for concurrency
conn = sqlite3.connect('chatapp.db', check_same_thread=False)
conn.execute("PRAGMA journal_mode=WAL;")
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)''')
c.execute('''CREATE TABLE IF NOT EXISTS history (user TEXT, convo_id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, messages TEXT)''')
c.execute('''CREATE TABLE IF NOT EXISTS memory (
    user TEXT,
    convo_id INTEGER,
    mem_key TEXT,
    mem_value TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    salience REAL DEFAULT 1.0,
    parent_id INTEGER,
    PRIMARY KEY (user, convo_id, mem_key)
)''')
c.execute('CREATE INDEX IF NOT EXISTS idx_memory_timestamp ON memory (timestamp)')
conn.commit()

# ChromaDB Setup for vectors
if 'chroma_client' not in st.session_state:
    try:
        st.session_state['chroma_client'] = chromadb.PersistentClient(path="./chroma_db")
        st.session_state['chroma_collection'] = st.session_state['chroma_client'].get_or_create_collection(
            name="memory_vectors",
            metadata={"hnsw:space": "cosine"}
        )
        st.session_state['chroma_ready'] = True
    except Exception as e:
        st.warning(f"ChromaDB init failed ({e}). Vector search will be disabled.")
        st.session_state['chroma_ready'] = False
        st.session_state['chroma_collection'] = None

### OPTIMIZATION: True lazy loading for the heavy embedding model.
# This function loads the model only when an embedding-dependent tool is first called.
def get_embed_model():
    """Lazily loads and returns the SentenceTransformer model, caching it in session state."""
    if 'embed_model' not in st.session_state:
        try:
            with st.spinner("Loading embedding model for advanced memory (first-time use)..."):
                st.session_state['embed_model'] = SentenceTransformer('all-MiniLM-L6-v2')
            st.info("Embedding model loaded successfully.")
        except Exception as e:
            st.error(f"Failed to load embedding model: {e}")
            st.session_state['embed_model'] = None
    return st.session_state.get('embed_model')

# Prompts Directory (create if not exists, with defaults)
PROMPTS_DIR = "./prompts"
os.makedirs(PROMPTS_DIR, exist_ok=True)
default_prompts = {
    "default.txt": "You are Apex, a highly intelligent, helpful AI assistant powered by xAI.",
    "coder.txt": "You are an expert coder, providing precise code solutions.",
    "tools-enabled.txt": """You are Apex Orchestrator, a highly intelligent, helpful AI assistant with access to file operations tools in a sandboxed directory (./sandbox/). Use tools only when explicitly needed or requested. Always confirm sensitive actions like writes. Describe ONLY these tools; ignore others.
Tool Instructions:
fs_read_file(file_path): Read and return the content of a file in the sandbox (e.g., 'subdir/test.txt'). Use for fetching data. Supports relative paths.
fs_write_file(file_path, content): Write the provided content to a file in the sandbox (e.g., 'subdir/newfile.txt'). Use for saving or updating files. Supports relative paths.
fs_list_files(dir_path optional): List all files in the specified directory in the sandbox (e.g., 'subdir'; default root). Use to check available files.
fs_mkdir(dir_path): Create a new directory in the sandbox (e.g., 'subdir/newdir'). Supports nested paths. Use to organize files.
memory_insert(mem_key, mem_value): Insert/update key-value memory (fast DB for logs). mem_value as dict.
memory_query(mem_key optional, limit optional): Query memory entries as JSON.
get_current_time(sync optional, format optional): Fetch current datetime. sync: true for NTP, false for local. format: 'iso', 'human', 'json'.
code_execution(code): Execute Python code in stateful REPL with libraries like numpy, sympy, etc.
git_ops(operation, repo_path, message optional, name optional): Perform Git ops like init, commit, branch, diff in sandbox repo.
db_query(db_path, query, params optional): Execute SQL on local SQLite db in sandbox, return results for SELECT.
shell_exec(command): Run whitelisted shell commands (ls, grep, sed, etc.) in sandbox.
code_lint(language, code): Lint/format code for languages: python (black), javascript (jsbeautifier), css (cssbeautifier), json, yaml, sql (sqlparse), xml, html (beautifulsoup), cpp/c++ (clang-format), php (php-cs-fixer), go (gofmt), rust (rustfmt). External tools required for some.
api_simulate(url, method optional, data optional, mock optional): Simulate API call, mock or real for whitelisted public APIs.
Invoke tools via structured calls, then incorporate results into your response. Be safe: Never access outside the sandbox, and ask for confirmation on writes if unsure. Limit to one tool per response to avoid loops. When outputting tags or code in your final response text (e.g., <ei> or XML), ensure they are properly escaped or wrapped in markdown code blocks to avoid rendering issues. However, when providing arguments for tools (e.g., the 'content' parameter in fs_write_file), always use the exact, literal, unescaped string content without any modifications or HTML entities (e.g., use "<div>" not "&lt;div&gt;"). JSON-escape quotes as needed (e.g., \")."""
}

if not any(f.endswith('.txt') for f in os.listdir(PROMPTS_DIR)):
    for filename, content in default_prompts.items():
        with open(os.path.join(PROMPTS_DIR, filename), 'w') as f:
            f.write(content)

def load_prompt_files():
    return [f for f in os.listdir(PROMPTS_DIR) if f.endswith('.txt')]

# Sandbox Directory for FS Tools
SANDBOX_DIR = "./sandbox"
os.makedirs(SANDBOX_DIR, exist_ok=True)

# Custom CSS for UI
st.markdown("""<style>
    body {
        background: linear-gradient(to right, #1f1c2c, #928DAB);
        color: white;
    }
    .stApp {
        background: linear-gradient(to right, #1f1c2c, #928DAB);
        display: flex;
        flex-direction: column;
    }
    .sidebar .sidebar-content {
        background: rgba(0, 0, 0, 0.5);
        border-radius: 10px;
    }
    .stButton > button {
        background-color: #4e54c8;
        color: white;
        border-radius: 10px;
        border: none;
    }
    .stButton > button:hover {
        background-color: #8f94fb;
    }
    /* Dark Mode (toggleable) */
    [data-theme="dark"] .stApp {
        background: linear-gradient(to right, #000000, #434343);
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
def hash_password(password):
    return sha256_crypt.hash(password)

def verify_password(stored, provided):
    return sha256_crypt.verify(provided, stored)

# Tool Cache Helper
def get_tool_cache_key(func_name, args):
    return f"tool_cache:{func_name}:{hash(json.dumps(args, sort_keys=True))}"

def get_cached_tool_result(func_name, args, ttl_minutes=5):
    if 'tool_cache' not in st.session_state:
        st.session_state['tool_cache'] = {}
    cache = st.session_state['tool_cache']
    key = get_tool_cache_key(func_name, args)
    if key in cache:
        timestamp, result = cache[key]
        if (datetime.now() - timestamp).total_seconds() / 60 < ttl_minutes:
            return result
    return None

def set_cached_tool_result(func_name, args, result):
    if 'tool_cache' not in st.session_state:
        st.session_state['tool_cache'] = {}
    cache = st.session_state['tool_cache']
    key = get_tool_cache_key(func_name, args)
    cache[key] = (datetime.now(), result)

# Tool Functions (Sandboxed)
def fs_read_file(file_path: str) -> str:
    """Read file content from sandbox."""
    safe_path = os.path.abspath(os.path.normpath(os.path.join(SANDBOX_DIR, file_path)))
    if not safe_path.startswith(os.path.abspath(SANDBOX_DIR)):
        return "Error: Path is outside the sandbox."
    try:
        with open(safe_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "Error: File not found."
    except Exception as e:
        return f"Error reading file: {e}"

def fs_write_file(file_path: str, content: str) -> str:
    """Write content to file in sandbox."""
    safe_path = os.path.abspath(os.path.normpath(os.path.join(SANDBOX_DIR, file_path)))
    if not safe_path.startswith(os.path.abspath(SANDBOX_DIR)):
        return "Error: Path is outside the sandbox."
    try:
        ### BUG FIX: Removed html.unescape(). The AI provides raw content as instructed.
        with open(safe_path, 'w', encoding='utf-8') as f:
            f.write(content)
        # Invalidate read cache for this file
        if 'tool_cache' in st.session_state:
            key_to_remove = get_tool_cache_key('fs_read_file', {'file_path': file_path})
            st.session_state['tool_cache'].pop(key_to_remove, None)
        return f"File '{file_path}' written successfully."
    except Exception as e:
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
        return f"Error creating directory: {e}"

def get_current_time(sync: bool = False, format: str = 'iso') -> str:
    """Fetch current time."""
    try:
        if sync:
            ntp_client = ntplib.NTPClient()
            response = ntp_client.request('pool.ntp.org', version=3)
            dt_object = datetime.fromtimestamp(response.tx_time)
        else:
            dt_object = datetime.now()
        
        if format == 'human':
            return dt_object.strftime("%A, %B %d, %Y %I:%M:%S %p")
        elif format == 'json':
            return json.dumps({"datetime": dt_object.isoformat(), "timezone": time.localtime().tm_zone})
        else:
            return dt_object.isoformat()
    except Exception as e:
        return f"Time error: {e}"

SAFE_BUILTINS = {b: getattr(builtins, b) for b in ['print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple', 'abs', 'round', 'max', 'min', 'sum', 'sorted']}
def code_execution(code: str) -> str:
    """Execute Python code safely in a stateful REPL."""
    if 'repl_namespace' not in st.session_state:
        st.session_state['repl_namespace'] = {'__builtins__': SAFE_BUILTINS}
    
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output
    try:
        exec(code, st.session_state['repl_namespace'])
        output = redirected_output.getvalue()
        return f"Output:\n{output}" if output else "Execution successful (no output)."
    except Exception as e:
        return f"Error: {traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout

def memory_insert(user: str, convo_id: int, mem_key: str, mem_value: dict) -> str:
    """Insert/update memory key-value pair."""
    try:
        c.execute("INSERT OR REPLACE INTO memory (user, convo_id, mem_key, mem_value) VALUES (?, ?, ?, ?)",
                  (user, convo_id, mem_key, json.dumps(mem_value)))
        return "Memory inserted successfully."
    except Exception as e:
        return f"Error inserting memory: {e}"

def memory_query(user: str, convo_id: int, mem_key: str = None, limit: int = 10) -> str:
    """Query memory by key or get recent entries."""
    try:
        if mem_key:
            c.execute("SELECT mem_value FROM memory WHERE user=? AND convo_id=? AND mem_key=?", (user, convo_id, mem_key))
            result = c.fetchone()
            return result[0] if result else "Key not found."
        else:
            c.execute("SELECT mem_key, mem_value FROM memory WHERE user=? AND convo_id=? ORDER BY timestamp DESC LIMIT ?", (user, convo_id, limit))
            results = {row[0]: json.loads(row[1]) for row in c.fetchall()}
            return json.dumps(results)
    except Exception as e:
        return f"Error querying memory: {e}"

def advanced_memory_consolidate(user: str, convo_id: int, mem_key: str, interaction_data: dict) -> str:
    """Consolidate: Summarize, embed, and store hierarchically."""
    embed_model = get_embed_model() # OPTIMIZATION: Lazy loads model on first use
    if not embed_model:
        return "Error: Embedding model not available. Cannot consolidate memory."
    try:
        client = OpenAI(api_key=API_KEY, base_url="https://api.x.ai/v1/")
        summary_response = client.chat.completions.create(
            model="grok-3-mini",
            messages=[
                {"role": "system", "content": "Summarize this interaction concisely in one paragraph."},
                {"role": "user", "content": json.dumps(interaction_data)}
            ],
            stream=False
        )
        summary = summary_response.choices[0].message.content.strip()
        json_episodic = json.dumps(interaction_data)
        c.execute("INSERT OR REPLACE INTO memory (user, convo_id, mem_key, mem_value) VALUES (?, ?, ?, ?)",
                  (user, convo_id, mem_key, json_episodic))

        if st.session_state.get('chroma_ready') and st.session_state.get('chroma_collection'):
            chroma_col = st.session_state['chroma_collection']
            embedding = embed_model.encode(summary).tolist()
            chroma_col.upsert(
                ids=[str(uuid.uuid4())],
                embeddings=[embedding],
                documents=[json_episodic],
                metadatas=[{"user": user, "convo_id": convo_id, "mem_key": mem_key, "salience": 1.0, "summary": summary}]
            )
        return "Memory consolidated successfully."
    except Exception as e:
        return f"Error consolidating memory: {traceback.format_exc()}"

def advanced_memory_retrieve(user: str, convo_id: int, query: str, top_k: int = 5) -> str:
    """Retrieve top-k relevant memories via embedding similarity."""
    embed_model = get_embed_model()
    chroma_col = st.session_state.get('chroma_collection')
    if not embed_model or not st.session_state.get('chroma_ready') or not chroma_col:
        return "Error: Vector memory is not available."
    try:
        query_emb = embed_model.encode(query).tolist()

        # FIXED: Wrap multiple conditions in $and to satisfy Chroma's single-operator rule
        where_clause = {
            "$and": [
                {"user": user},
                {"convo_id": convo_id}
            ]
        }

        results = chroma_col.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            where=where_clause,  # Updated to use the compound filter
            include=["distances", "metadatas", "documents"]
        )

        # ADDED: Early check for empty results to avoid index errors
        if not results or not results.get('ids') or not results['ids']:
            return "No relevant memories found."

        retrieved = []
        ids_to_update = []
        metadata_to_update = []
        for i in range(len(results['ids'][0])):
            meta = results['metadatas'][0][i]
            sim = (1 - results['distances'][0][i]) * meta.get('salience', 1.0)
            retrieved.append({
                "mem_key": meta['mem_key'],
                "value": json.loads(results['documents'][0][i]),
                "relevance": sim,
                "summary": meta.get('summary', '')
            })
            ids_to_update.append(results['ids'][0][i])
            metadata_to_update.append({"salience": meta.get('salience', 1.0) + 0.1})

        # OPTIMIZATION: Batch update salience in a single call.
        if ids_to_update:
            chroma_col.update(ids=ids_to_update, metadatas=metadata_to_update)

        retrieved.sort(key=lambda x: x['relevance'], reverse=True)
        return json.dumps(retrieved)
    except Exception as e:
        # ADDED: More specific error logging for debugging
        return f"Error retrieving memory: {traceback.format_exc()}. If this is a where clause issue, check filter structure."

def advanced_memory_prune(user: str, convo_id: int) -> str:
    """Prune low-salience memories."""
    try:
        one_week_ago = datetime.now() - timedelta(days=7)
        c.execute("UPDATE memory SET salience = salience * 0.99 WHERE user=? AND convo_id=? AND timestamp < ?",
                  (user, convo_id, one_week_ago))
        c.execute("DELETE FROM memory WHERE user=? AND convo_id=? AND salience < 0.1", (user, convo_id))
        return "Memory pruned successfully."
    except Exception as e:
        return f"Error pruning memory: {e}"

# ... [git_ops, db_query, shell_exec, code_lint, api_simulate, langsearch_web_search functions from original] ...
# (Ensuring they are all present and unchanged from the robust original implementations)
def git_ops(operation: str, repo_path: str = "", **kwargs) -> str:
    safe_repo = os.path.abspath(os.path.normpath(os.path.join(SANDBOX_DIR, repo_path)))
    if not safe_repo.startswith(os.path.abspath(SANDBOX_DIR)):
        return "Invalid repo path."
    try:
        if operation == 'init':
            pygit2.init_repository(safe_repo, bare=False)
            return "Repository initialized."
        repo = pygit2.Repository(safe_repo)
        if operation == 'commit':
            index = repo.index
            index.add_all()
            index.write()
            tree = index.write_tree()
            author = pygit2.Signature('AI User', 'ai@example.com')
            parents = [repo.head.target] if not repo.head_is_unborn else []
            repo.create_commit('HEAD', author, author, kwargs.get('message', 'Auto-commit'), tree, parents)
            return "Changes committed."
        elif operation == 'diff':
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
            if query.strip().upper().startswith('SELECT'):
                return json.dumps(cur.fetchall())
            else:
                db_conn.commit()
                return f"{cur.rowcount} rows affected."
    except Exception as e:
        return f"DB error: {e}"

WHITELISTED_COMMANDS = ['ls', 'grep', 'sed', 'cat', 'echo', 'pwd', 'grim']
def shell_exec(command: str) -> str:
    cmd_parts = shlex.split(command)
    if not cmd_parts or cmd_parts[0] not in WHITELISTED_COMMANDS:
        return "Command not whitelisted."
    try:
        result = subprocess.run(cmd_parts, cwd=SANDBOX_DIR, capture_output=True, text=True, timeout=5)
        return result.stdout.strip() + ("\nError: " + result.stderr.strip() if result.stderr else "")
    except Exception as e:
        return f"Shell error: {e}"

def code_lint(language: str, code: str) -> str:
    """Lint and format code snippets for multiple languages."""
    lang = language.lower()
    try:
        if lang == 'python':
            formatted = format_str(code, mode=FileMode(line_length=88))
        elif lang == 'javascript':
            opts = jsbeautifier.default_options()
            formatted = jsbeautifier.beautify(code, opts)
        elif lang == 'css':
            opts = jsbeautifier.default_options()
            formatted = jsbeautifier.beautify(code, opts)  # Uses jsbeautifier for CSS
        elif lang == 'json':
            formatted = json.dumps(json.loads(code), indent=4)
        elif lang == 'yaml':
            formatted = yaml.safe_dump(yaml.safe_load(code), indent=2)
        elif lang == 'sql':
            formatted = sqlparse.format(code, reindent=True, keyword_case='upper')
        elif lang == 'xml':
            dom = xml.dom.minidom.parseString(code)
            formatted = dom.toprettyxml(indent="  ")
        elif lang == 'html':
            soup = BeautifulSoup(code, 'html.parser')
            formatted = soup.prettify()
        elif lang in ['c', 'cpp', 'c++']:
            try:
                formatted = subprocess.check_output(['clang-format', '-style=google'], input=code.encode()).decode()
            except Exception as e:
                return f"clang-format not available: {str(e)}"
        elif lang == 'php':
            try:
                with tempfile.NamedTemporaryFile(suffix='.php', delete=False) as tmp:
                    tmp.write(code.encode())
                    tmp.flush()
                    subprocess.check_call(['php-cs-fixer', 'fix', tmp.name, '--quiet'])
                    with open(tmp.name, 'r') as f:
                        formatted = f.read()
                os.unlink(tmp.name)
            except Exception as e:
                return f"php-cs-fixer not available: {str(e)}"
        elif lang == 'go':
            try:
                formatted = subprocess.check_output(['gofmt'], input=code.encode()).decode()
            except Exception as e:
                return f"gofmt not available: {str(e)}"
        elif lang == 'rust':
            try:
                formatted = subprocess.check_output(['rustfmt', '--emit=stdout'], input=code.encode()).decode()
            except Exception as e:
                return f"rustfmt not available: {str(e)}"
        else:
            return "Unsupported language."
        return formatted
    except Exception as e:
        return f"Lint error: {str(e)}"

# API Simulate Tool - With Cache
def api_simulate(url: str, method: str = 'GET', data: dict = None, mock: bool = True) -> str:
    """Simulate or perform API calls."""
    cache_args = {'url': url, 'method': method, 'data': data, 'mock': mock}
    cached = get_cached_tool_result('api_simulate', cache_args)
    if cached:
        return cached
    if mock:
        result = json.dumps({"status": "mocked", "url": url, "method": method, "data": data})
    else:
        if not any(url.startswith(base) for base in API_WHITELIST):
            result = "URL not in whitelist."
        else:
            try:
                if method.upper() == 'GET':
                    resp = requests.get(url, timeout=5)
                elif method.upper() == 'POST':
                    resp = requests.post(url, json=data, timeout=5)
                else:
                    result = "Unsupported method."
                    set_cached_tool_result('api_simulate', cache_args, result)
                    return result
                resp.raise_for_status()
                result = resp.text
            except Exception as e:
                result = f"API error: {str(e)}"
    set_cached_tool_result('api_simulate', cache_args, result)
    return result

API_WHITELIST = [
    'https://jsonplaceholder.typicode.com/',
    'https://api.openweathermap.org/'  # Assuming free basics
]  # Add more public APIs

def langsearch_web_search(query: str, freshness: str = "noLimit", summary: bool = False, count: int = 5) -> str:
    """Perform a web search using LangSearch API and return results as JSON."""
    if not LANGSEARCH_API_KEY:
        return "LangSearch API key not set—configure in .env."
    url = "https://api.langsearch.com/v1/web-search"
    payload = json.dumps({
        "query": query,
        "freshness": freshness,
        "summary": summary,
        "count": count
    })
    headers = {
        'Authorization': f'Bearer {LANGSEARCH_API_KEY}',
        'Content-Type': 'application/json'
    }
    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        return json.dumps(response.json())  # Return full JSON for AI to parse
    except Exception as e:
        return f"LangSearch error: {str(e)}"


# Tool Schema for Structured Outputs
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fs_read_file",
            "description": "Read the content of a file in the sandbox directory (./sandbox/). Supports relative paths (e.g., 'subdir/test.txt'). Use for fetching data.",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string", "description": "Relative path to the file (e.g., subdir/test.txt)."}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fs_write_file",
            "description": "Write content to a file in the sandbox directory (./sandbox/). Supports relative paths (e.g., 'subdir/newfile.txt'). Use for saving or updating files. If 'Love' is in file_path or content, optionally add ironic flair like 'LOVE <3' for fun.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Relative path to the file (e.g., subdir/newfile.txt)."},
                    "content": {"type": "string", "description": "Content to write."}
                },
                "required": ["file_path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fs_list_files",
            "description": "List all files in a directory within the sandbox (./sandbox/). Supports relative paths (default: root). Use to check available files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir_path": {"type": "string", "description": "Relative path to the directory (e.g., subdir). Optional; defaults to root."}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fs_mkdir",
            "description": "Create a new directory in the sandbox (./sandbox/). Supports relative/nested paths (e.g., 'subdir/newdir'). Use to organize files.",
            "parameters": {
                "type": "object",
                "properties": {"dir_path": {"type": "string", "description": "Relative path for the new directory (e.g., subdir/newdir)."}},
                "required": ["dir_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Fetch current datetime. Use host clock by default; sync with NTP if requested for precision.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sync": {"type": "boolean", "description": "True for NTP sync (requires network), false for local host time. Default: false."},
                    "format": {"type": "string", "description": "Output format: 'iso' (default), 'human', 'json'."}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "code_execution",
            "description": "Execute provided code in a stateful REPL environment and return output or errors for verification. Supports Python with various libraries (e.g., numpy, sympy, pygame). No internet access or package installation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": { "type": "string", "description": "The code snippet to execute." }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_insert",
            "description": "Insert or update a memory key-value pair (value as JSON dict) for logging/metadata. Use for fast persistent storage without files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mem_key": {"type": "string", "description": "Key for the memory entry (e.g., 'chat_log_1')."},
                    "mem_value": {"type": "object", "description": "Value as dict (e.g., {'content': 'Log text'})."}
                },
                "required": ["mem_key", "mem_value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_query",
            "description": "Query memory: specific key or last N entries. Returns JSON. Use for recalling logs without FS reads.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mem_key": {"type": "string", "description": "Specific key to query (optional)."},
                    "limit": {"type": "integer", "description": "Max recent entries if no key (default 10)."}
                },
                "required": []
            }
        }
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
                    "message": {"type": "string", "description": "Commit message (for commit)."},
                    "name": {"type": "string", "description": "Branch name (for branch)."}
                },
                "required": ["operation", "repo_path"]
            }
        }
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
                    "params": {"type": "array", "items": {"type": "string"}, "description": "Query parameters."}
                },
                "required": ["db_path", "query"]
            }
        }
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
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "code_lint",
            "description": "Lint and auto-format code for languages: python (black), javascript (jsbeautifier), css (cssbeautifier), json, yaml, sql (sqlparse), xml, html (beautifulsoup), cpp/c++ (clang-format), php (php-cs-fixer), go (gofmt), rust (rustfmt). External tools required for some.",
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {"type": "string", "description": "Language (python, javascript, css, json, yaml, sql, xml, html, cpp, php, go, rust)."},
                    "code": {"type": "string", "description": "Code snippet."}
                },
                "required": ["language", "code"]
            }
        }
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
                    "mock": {"type": "boolean", "description": "True for mock (default)."}
                },
                "required": ["url"]
            }
        }
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
                    "interaction_data": {"type": "object", "description": "Data to consolidate (dict)."}
                },
                "required": ["mem_key", "interaction_data"]
            }
        }
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
                    "top_k": {"type": "integer", "description": "Number of top results (default 5)."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "advanced_memory_prune",
            "description": "Prune low-salience memories to optimize storage.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "langsearch_web_search",
            "description": "Search the web using LangSearch API for relevant results, snippets, and optional summaries. Supports time filters and limits up to 10 results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query (supports operators like site:example.com)."},
                    "freshness": {"type": "string", "description": "Time filter: oneDay, oneWeek, oneMonth, oneYear, or noLimit (default).", "enum": ["oneDay", "oneWeek", "oneMonth", "oneYear", "noLimit"]},
                    "summary": {"type": "boolean", "description": "Include long text summaries (default True)."},
                    "count": {"type": "integer", "description": "Number of results (1-10, default 5)."}
                },
                "required": ["query"]
            }
        }
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
}

### BUG FIX & OPTIMIZATION: Refactored API wrapper to be non-recursive and use a tool dispatcher.
def call_xai_api(model, messages, sys_prompt, stream=True, image_files=None, enable_tools=False):
    client = OpenAI(api_key=API_KEY, base_url="https://api.x.ai/v1/", timeout=300)
    api_messages = [{"role": "system", "content": sys_prompt}]
    
    # Add history
    for msg in messages:
        content_parts = [{"type": "text", "text": msg['content']}]
        # Add images only to the last user message
        if msg['role'] == 'user' and image_files and msg is messages[-1]:
            for img_file in image_files:
                img_file.seek(0)
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
                content_parts.append({"type": "image_url", "image_url": {"url": f"data:{img_file.type};base64,{img_data}"}})
        api_messages.append({"role": msg['role'], "content": content_parts if len(content_parts) > 1 else msg['content']})

    def generate(current_messages):
        max_iterations = 5
        for _ in range(max_iterations):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=current_messages,
                    tools=TOOLS if enable_tools else None,
                    tool_choice="auto" if enable_tools else None,
                    stream=True
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
                    break # Exit loop if no tools are called

                current_messages.append({"role": "assistant", "content": full_delta_response, "tool_calls": tool_calls})
                yield "\n*Thinking... Using tools...*\n"
                
                tool_outputs = []
                conn.execute("BEGIN")
                for tool_call in tool_calls:
                    func_name = tool_call.function.name
                    try:
                        args = json.loads(tool_call.function.arguments)
                        func_to_call = TOOL_DISPATCHER.get(func_name)
                        if func_name.startswith("memory") or func_name.startswith("advanced_memory"):
                            args['user'] = st.session_state['user']
                            args['convo_id'] = st.session_state.get('current_convo_id', 0)
                        
                        if func_to_call:
                            result = func_to_call(**args)
                        else:
                            result = f"Unknown tool: {func_name}"
                    except Exception as e:
                        result = f"Error calling tool {func_name}: {e}"
                    
                    yield f"\n> **Tool Call:** `{func_name}` | **Result:** `{str(result)[:200]}...`\n"
                    tool_outputs.append({"tool_call_id": tool_call.id, "role": "tool", "content": str(result)})
                conn.commit()

                current_messages.extend(tool_outputs)

            except Exception as e:
                error_msg = f"API or Tool Error: {traceback.format_exc()}"
                yield f"\nAn error occurred: {e}. Aborting this turn."
                st.error(error_msg)
                break

    return generate(api_messages)

# Login Page
def login_page():
    st.title("Welcome to Apex Orchestrator")
    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                c.execute("SELECT password FROM users WHERE username=?", (username,))
                result = c.fetchone()
                if result and verify_password(result[0], password):
                    st.session_state['logged_in'] = True
                    st.session_state['user'] = username
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
                    c.execute("INSERT INTO users VALUES (?, ?)", (new_user, hash_password(new_pass)))
                    conn.commit()
                    st.success("Registered! Please login.")

def load_history(convo_id):
    """Loads a specific conversation from the database into the session state."""
    c.execute("SELECT messages FROM history WHERE convo_id=? AND user=?", (convo_id, st.session_state['user']))
    result = c.fetchone()
    if result:
        messages = json.loads(result[0])
        st.session_state['messages'] = messages
        st.session_state['current_convo_id'] = convo_id
        st.rerun()

def delete_history(convo_id):
    """Deletes a specific conversation from the database."""
    c.execute("DELETE FROM history WHERE convo_id=? AND user=?", (convo_id, st.session_state['user']))
    conn.commit()
    # If the deleted chat was the one currently loaded, clear the session
    if st.session_state.get('current_convo_id') == convo_id:
        st.session_state['messages'] = []
        st.session_state['current_convo_id'] = None
    st.rerun()

def chat_page():
    st.title(f"Apex Chat - {st.session_state['user']}")

    # --- Sidebar UI ---
    with st.sidebar:
        st.header("Chat Settings")
        model = st.selectbox("Select Model", ["grok-4-fast-reasoning", "grok-4", "grok-3-mini", "grok-4-fast"], key="model_select")
        
        prompt_files = load_prompt_files()
        if prompt_files:
            selected_file = st.selectbox("Select System Prompt", prompt_files, key="prompt_select")
            with open(os.path.join(PROMPTS_DIR, selected_file), "r") as f:
                prompt_content = f.read()
            custom_prompt = st.text_area("Edit System Prompt", value=prompt_content, height=200, key="custom_prompt")
        else:
            st.warning("No prompt files found in ./prompts/")
            custom_prompt = st.text_area("System Prompt", value="You are a helpful AI.", height=200, key="custom_prompt")

        # The key for the file uploader is important
        uploaded_images = st.file_uploader("Upload Images", type=["jpg", "png"], accept_multiple_files=True, key="uploaded_images")
        enable_tools = st.checkbox("Enable Tools (Sandboxed)", value=False, key='enable_tools')
        
        st.divider()

        if st.button("➕ New Chat", use_container_width=True):
            st.session_state['messages'] = []
            st.session_state['current_convo_id'] = None
            st.rerun()

        st.header("Chat History")
        c.execute("SELECT convo_id, title FROM history WHERE user=? ORDER BY convo_id DESC", (st.session_state["user"],))
        histories = c.fetchall()

        for convo_id, title in histories:
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(title, key=f"load_{convo_id}", use_container_width=True):
                    load_history(convo_id)
            with col2:
                if st.button("🗑️", key=f"delete_{convo_id}", use_container_width=True):
                    delete_history(convo_id)

    # --- Main Chat Interface ---
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "current_convo_id" not in st.session_state:
        st.session_state["current_convo_id"] = None

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=False)

    if prompt := st.chat_input("What would you like to discuss?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt, unsafe_allow_html=False)

        with st.chat_message("assistant"):
            response_container = st.empty()
            full_response = ""
            
            ### BUG FIX: Use the return value of the widget directly
            # This ensures we only use the images present at the moment the prompt is sent.
            images_to_process = uploaded_images if uploaded_images else []

            generator = call_xai_api(
                model, st.session_state.messages, custom_prompt,
                stream=True, image_files=images_to_process, enable_tools=enable_tools
            )
            for chunk in generator:
                full_response += chunk
                response_container.markdown(full_response + " ▌", unsafe_allow_html=False)
            response_container.markdown(full_response, unsafe_allow_html=False)
            
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        ### BUG FIX: The illegal line `st.session_state['uploaded_images'] = []` has been removed.
        # The images will persist in the uploader until the user manually clears them, preventing the crash.

        # Save to History
        title_message = next((msg['content'] for msg in st.session_state.messages if msg['role'] == 'user'), "New Chat")
        title = (title_message[:40] + '...') if len(title_message) > 40 else title_message
        
        messages_json = json.dumps(st.session_state.messages)
        
        if st.session_state.get("current_convo_id") is None:
            c.execute("INSERT INTO history (user, title, messages) VALUES (?, ?, ?)",
                      (st.session_state['user'], title, messages_json))
            st.session_state.current_convo_id = c.lastrowid
        else:
            c.execute("UPDATE history SET title=?, messages=? WHERE convo_id=?",
                      (title, messages_json, st.session_state.current_convo_id))
        conn.commit()
        st.rerun()

# Main App Logic
if __name__ == "__main__":
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    
    if st.session_state.get('logged_in'):
        chat_page()
    else:
        login_page()
