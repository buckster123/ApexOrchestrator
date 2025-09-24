import streamlit as st
import os
from openai import OpenAI
from passlib.hash import sha256_crypt
from dotenv import load_dotenv
import json
import time
import base64
import ntplib
import io
import sys
import subprocess
import requests
from datetime import datetime, timedelta
import html
import shlex
import uuid
from functools import lru_cache
import sqlite3
import builtins  # Added to fix AttributeError

load_dotenv()
API_KEY = os.getenv("XAI_API_KEY")
if not API_KEY:
    st.error("XAI_API_KEY not set in .env! Please add it and restart.")
LANGSEARCH_API_KEY = os.getenv("LANGSEARCH_API_KEY")
if not LANGSEARCH_API_KEY:
    st.warning("LANGSEARCH_API_KEY not set in .env—web search tool will fail.")

PROMPTS_DIR = "./prompts"
SANDBOX_DIR = "./sandbox"
os.makedirs(PROMPTS_DIR, exist_ok=True)
os.makedirs(SANDBOX_DIR, exist_ok=True)

default_prompts = {
    "default.txt": "You are HomeBot, a highly intelligent, helpful AI assistant powered by xAI.",
    "coder.txt": "You are an expert coder, providing precise code solutions.",
    "tools-enabled.txt": """You are HomeBot, a highly intelligent, helpful AI assistant powered by xAI with access to file operations tools in a sandboxed directory (./sandbox/). Use tools only when explicitly needed or requested. Always confirm sensitive actions like writes. Describe ONLY these tools; ignore others.
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
code_lint(language, code): Lint/format code for languages: python, javascript, css, json, yaml, sql, xml, html, cpp, php, go, rust. External tools required for some.
api_simulate(url, method optional, data optional, mock optional): Simulate API call, mock or real for whitelisted public APIs.
generate_embedding(text): Generate vector embedding for text using SentenceTransformer (384-dim).
vector_search(query_embedding, top_k optional, threshold optional): Perform ANN vector search in ChromaDB (cosine sim > threshold).
chunk_text(text, max_tokens optional): Split text into chunks (default 512 tokens).
summarize_chunk(chunk): Compress a text chunk via LLM summary.
keyword_search(query, top_k optional): Keyword-based search on memory cache (e.g., BM25 sim).
Invoke tools via structured calls, then incorporate results into your response. Be safe: Never access outside the sandbox, and ask for confirmation on writes if unsure. Limit to one tool per response to avoid loops. When outputting tags or code in your final response text (e.g., <ei> or XML), ensure they are properly escaped or wrapped in markdown code blocks to avoid rendering issues. However, when providing arguments for tools (e.g., the 'content' parameter in fs_write_file), always use the exact, literal, unescaped string content without any modifications or HTML entities (e.g., use "<div>" not "&lt;div&gt;"). JSON-escape quotes as needed (e.g., \")."""
}

if not any(f.endswith('.txt') for f in os.listdir(PROMPTS_DIR)):
    for filename, content in default_prompts.items():
        with open(os.path.join(PROMPTS_DIR, filename), 'w') as f:
            f.write(content)

class MemoryCache:
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.lru_cache = {}
        self.metrics = {"total_inserts": 0, "total_retrieves": 0, "hit_rate": 1.0, "last_update": None}

    def insert(self, key, entry):
        if len(self.lru_cache) >= self.max_size:
            oldest = min(self.lru_cache.items(), key=lambda x: x[1]["last_access"])
            del self.lru_cache[oldest[0]]
        self.lru_cache[key] = {"entry": entry, "last_access": time.time()}
        self.metrics["total_inserts"] += 1

    def update_access(self, key):
        if key in self.lru_cache:
            self.lru_cache[key]["last_access"] = time.time()

def initialize_session_state():
    defaults = {
        'logged_in': False,
        'user': None,
        'messages': [],
        'current_convo_id': 0,
        'tool_cache': {},
        'memory_cache': MemoryCache(),
        'image_cache': {},
        'repl_namespace': {'__builtins__': {b: getattr(builtins, b) for b in ['print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple', 'abs', 'round', 'max', 'min', 'sum', 'sorted']}},
        'prompt_files': [],
        'prompt_files_mtime': 0
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    try:
        import chromadb
        st.session_state['chroma_client'] = chromadb.PersistentClient(path="./chroma_db")
        st.session_state['chroma_collection'] = st.session_state['chroma_client'].get_or_create_collection(
            name="memory_vectors", metadata={"hnsw:space": "cosine"}
        )
        st.session_state['chroma_ready'] = True
    except Exception:
        st.session_state['chroma_ready'] = False
        st.session_state['chroma_collection'] = None

def get_embed_model():
    if 'embed_model' not in st.session_state:
        try:
            from sentence_transformers import SentenceTransformer
            with st.spinner("Loading embedding model..."):
                st.session_state['embed_model'] = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            st.session_state['embed_model'] = None
    return st.session_state.get('embed_model')

def restrict_to_sandbox(path):
    safe_path = os.path.abspath(os.path.normpath(os.path.join(SANDBOX_DIR, path)))
    if not safe_path.startswith(os.path.abspath(SANDBOX_DIR)):
        raise ValueError("Path is outside the sandbox.")
    return safe_path

@lru_cache(maxsize=100)
def get_cached_tool_result(func_name, args_json):
    args = json.loads(args_json)
    return st.session_state['tool_cache'].get((func_name, args_json))

def set_cached_tool_result(func_name, args, result):
    args_json = json.dumps(args, sort_keys=True)
    st.session_state['tool_cache'][(func_name, args_json)] = (datetime.now(), result)

def setup_database():
    with sqlite3.connect('chatapp.db') as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS history (user TEXT, convo_id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, messages TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS memory (
            user TEXT, convo_id INTEGER, mem_key TEXT, mem_value TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            salience REAL DEFAULT 1.0, parent_id INTEGER, PRIMARY KEY (user, convo_id, mem_key)
        )''')
        c.execute('CREATE INDEX IF NOT EXISTS idx_memory_timestamp ON memory (timestamp)')
        conn.commit()

def hash_password(password):
    return sha256_crypt.hash(password)

def verify_password(stored, provided):
    return sha256_crypt.verify(provided, stored)

def fs_read_file(file_path):
    try:
        with open(restrict_to_sandbox(file_path), 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "Error: File not found."
    except Exception as e:
        return f"Error reading file: {e}"

def fs_write_file(file_path, content):
    try:
        safe_path = restrict_to_sandbox(file_path)
        with open(safe_path, 'w', encoding='utf-8') as f:
            f.write(content)
        set_cached_tool_result('fs_read_file', {'file_path': file_path}, None)
        return f"File '{file_path}' written successfully."
    except Exception as e:
        return f"Error writing file: {e}"

def fs_list_files(dir_path=""):
    try:
        files = os.listdir(restrict_to_sandbox(dir_path))
        return f"Files in '{dir_path or '/'}': {json.dumps(files)}"
    except FileNotFoundError:
        return "Error: Directory not found."
    except Exception as e:
        return f"Error listing files: {e}"

def fs_mkdir(dir_path):
    try:
        os.makedirs(restrict_to_sandbox(dir_path), exist_ok=True)
        return f"Directory '{dir_path}' created successfully."
    except Exception as e:
        return f"Error creating directory: {e}"

def get_current_time(sync=False, format='iso'):
    try:
        dt_object = datetime.fromtimestamp(ntplib.NTPClient().request('pool.ntp.org', version=3).tx_time) if sync else datetime.now()
        if format == 'human':
            return dt_object.strftime("%A, %B %d, %Y %I:%M:%S %p")
        elif format == 'json':
            return json.dumps({"datetime": dt_object.isoformat(), "timezone": time.localtime().tm_zone})
        return dt_object.isoformat()
    except Exception as e:
        return f"Time error: {e}"

def code_execution(code):
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output
    try:
        exec(code, st.session_state['repl_namespace'])
        output = redirected_output.getvalue()
        return f"Output:\n{output}" if output else "Execution successful (no output)."
    except Exception as e:
        return f"Error: {e}"
    finally:
        sys.stdout = old_stdout

def memory_insert(mem_key, mem_value, user, convo_id):
    try:
        with sqlite3.connect('chatapp.db') as conn:
            c = conn.cursor()
            c.execute("INSERT OR REPLACE INTO memory (user, convo_id, mem_key, mem_value) VALUES (?, ?, ?, ?)",
                      (user, convo_id, mem_key, json.dumps(mem_value)))
            conn.commit()
        entry = {
            "summary": mem_value.get("summary", ""),
            "details": mem_value.get("details", ""),
            "tags": mem_value.get("tags", []),
            "domain": mem_value.get("domain", "general"),
            "timestamp": datetime.now().isoformat(),
            "salience": mem_value.get("salience", 1.0)
        }
        st.session_state['memory_cache'].insert(mem_key, entry)
        return "Memory inserted successfully."
    except Exception as e:
        return f"Error inserting memory: {e}"

def memory_query(mem_key=None, limit=10, user=None, convo_id=None):
    try:
        with sqlite3.connect('chatapp.db') as conn:
            c = conn.cursor()
            if mem_key:
                c.execute("SELECT mem_value FROM memory WHERE user=? AND convo_id=? AND mem_key=?", (user, convo_id, mem_key))
                result = c.fetchone()
                return result[0] if result else "Key not found."
            c.execute("SELECT mem_key, mem_value FROM memory WHERE user=? AND convo_id=? ORDER BY timestamp DESC LIMIT ?", (user, convo_id, limit))
            results = {row[0]: json.loads(row[1]) for row in c.fetchall()}
            for key in results:
                if key not in st.session_state['memory_cache'].lru_cache:
                    entry = results[key]
                    st.session_state['memory_cache'].insert(key, {
                        "summary": entry.get("summary", ""),
                        "details": entry.get("details", ""),
                        "tags": entry.get("tags", []),
                        "domain": entry.get("domain", "general"),
                        "timestamp": entry.get("timestamp", datetime.now().isoformat()),
                        "salience": entry.get("salience", 1.0)
                    })
            return json.dumps(results)
    except Exception as e:
        return f"Error querying memory: {e}"

def advanced_memory_consolidate(mem_key, interaction_data, user, convo_id):
    embed_model = get_embed_model()
    if not embed_model:
        return "Error: Embedding model not available."
    try:
        client = OpenAI(api_key=API_KEY, base_url="https://api.x.ai/v1/")
        summary_response = client.chat.completions.create(
            model="grok-4-fast-non-reasoning",
            messages=[{"role": "system", "content": "Summarize this interaction concisely in one paragraph."}, {"role": "user", "content": json.dumps(interaction_data)}],
            stream=False
        )
        summary = summary_response.choices[0].message.content.strip()
        json_episodic = json.dumps(interaction_data)
        with sqlite3.connect('chatapp.db') as conn:
            c = conn.cursor()
            c.execute("INSERT OR REPLACE INTO memory (user, convo_id, mem_key, mem_value) VALUES (?, ?, ?, ?)",
                      (user, convo_id, mem_key, json_episodic))
            conn.commit()
        if st.session_state.get('chroma_ready') and st.session_state.get('chroma_collection'):
            embedding = embed_model.encode(summary).tolist()
            st.session_state['chroma_collection'].upsert(
                ids=[str(uuid.uuid4())], embeddings=[embedding], documents=[json_episodic],
                metadatas=[{"user": user, "convo_id": convo_id, "mem_key": mem_key, "salience": 1.0, "summary": summary}]
            )
        entry = {"summary": summary, "details": json_episodic, "tags": [], "domain": "general", "timestamp": datetime.now().isoformat(), "salience": 1.0}
        st.session_state['memory_cache'].insert(mem_key, entry)
        return "Memory consolidated successfully."
    except Exception as e:
        return f"Error consolidating memory: {e}"

def advanced_memory_retrieve(query, top_k=5, user=None, convo_id=None):
    embed_model = get_embed_model()
    if not embed_model or not st.session_state.get('chroma_ready') or not st.session_state.get('chroma_collection'):
        return "Error: Vector memory is not available."
    try:
        query_emb = embed_model.encode(query).tolist()
        results = st.session_state['chroma_collection'].query(
            query_embeddings=[query_emb], n_results=top_k, where={"user": user, "convo_id": convo_id}, include=["distances", "metadatas", "documents"]
        )
        if not results.get('ids', [[]])[0]:
            return "No relevant memories found."
        retrieved = []
        ids_to_update = []
        metadata_to_update = []
        for i in range(len(results['ids'][0])):
            meta = results['metadatas'][0][i]
            sim = (1 - results['distances'][0][i]) * meta.get('salience', 1.0)
            retrieved.append({"mem_key": meta['mem_key'], "value": json.loads(results['documents'][0][i]), "relevance": sim, "summary": meta.get('summary', '')})
            ids_to_update.append(results['ids'][0][i])
            metadata_to_update.append({"salience": meta.get('salience', 1.0) + 0.1})
        if ids_to_update:
            st.session_state['chroma_collection'].update(ids=ids_to_update, metadatas=metadata_to_update)
        retrieved.sort(key=lambda x: x['relevance'], reverse=True)
        st.session_state['memory_cache'].metrics['total_retrieves'] += 1
        hit_rate = len(retrieved) / top_k if top_k > 0 else 1.0
        st.session_state['memory_cache'].metrics['hit_rate'] = ((st.session_state['memory_cache'].metrics['hit_rate'] * (st.session_state['memory_cache'].metrics['total_retrieves'] - 1)) + hit_rate) / st.session_state['memory_cache'].metrics['total_retrieves']
        return json.dumps(retrieved)
    except Exception as e:
        return f"Error retrieving memory: {e}"

def advanced_memory_prune(user, convo_id):
    try:
        with sqlite3.connect('chatapp.db') as conn:
            c = conn.cursor()
            one_week_ago = datetime.now() - timedelta(days=7)
            c.execute("UPDATE memory SET salience = salience * 0.99 WHERE user=? AND convo_id=? AND timestamp < ?", (user, convo_id, one_week_ago))
            c.execute("DELETE FROM memory WHERE user=? AND convo_id=? AND salience < 0.1", (user, convo_id))
            c.execute("SELECT COUNT(*) FROM memory WHERE user=? AND convo_id=?", (user, convo_id))
            row_count = c.fetchone()[0]
            if row_count > 1000:
                c.execute("SELECT mem_key FROM memory WHERE user=? AND convo_id=? AND salience < 0.5 ORDER BY timestamp ASC LIMIT ?", (user, convo_id, row_count - 1000))
                low_keys = [row[0] for row in c.fetchall()]
                for key in low_keys:
                    c.execute("DELETE FROM memory WHERE user=? AND convo_id=? AND mem_key=?", (user, convo_id, key))
            c.execute("SELECT mem_key, mem_value FROM memory WHERE user=? AND convo_id=?", (user, convo_id))
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
                c.execute("DELETE FROM memory WHERE user=? AND convo_id=? AND mem_key=?", (user, convo_id, key))
            conn.commit()
        if len(st.session_state['memory_cache'].lru_cache) > 1000:
            lru_items = sorted(st.session_state['memory_cache'].lru_cache.items(), key=lambda x: x[1]["last_access"])
            num_to_evict = len(lru_items) - 1000
            with sqlite3.connect('chatapp.db') as conn:
                c = conn.cursor()
                for key, _ in lru_items[:num_to_evict]:
                    entry = st.session_state['memory_cache'].lru_cache[key]["entry"]
                    if entry["salience"] < 0.4:
                        del st.session_state['memory_cache'].lru_cache[key]
                        c.execute("DELETE FROM memory WHERE user=? AND convo_id=? AND mem_key=?", (user, convo_id, key))
                conn.commit()
        st.session_state['memory_cache'].metrics['last_update'] = datetime.now().isoformat()
        return "Memory pruned successfully."
    except Exception as e:
        return f"Error pruning memory: {e}"

def generate_embedding(text):
    embed_model = get_embed_model()
    if not embed_model:
        return [0.0] * 384
    try:
        return embed_model.encode(text).tolist()
    except Exception as e:
        return f"Embedding error: {e}"

def vector_search(query_embedding, top_k=5, threshold=0.6):
    if not st.session_state.get('chroma_ready') or not st.session_state.get('chroma_collection'):
        return []
    try:
        results = st.session_state['chroma_collection'].query(
            query_embeddings=[query_embedding], n_results=top_k, where={}, include=["distances", "metadatas", "documents"]
        )
        if not results.get('ids', [[]])[0]:
            return []
        retrieved = []
        for i in range(len(results['ids'][0])):
            dist = results['distances'][0][i]
            sim = 1 - dist
            if sim > threshold:
                retrieved.append({"id": results['ids'][0][i], "score": float(sim), "metadata": results['metadatas'][0][i], "document": results['documents'][0][i]})
        retrieved.sort(key=lambda x: x['score'], reverse=True)
        st.session_state['memory_cache'].metrics['total_retrieves'] += 1
        return retrieved[:top_k]
    except Exception as e:
        return f"Vector search error: {e}"

def chunk_text(text, max_tokens=512):
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunks.append(encoding.decode(chunk_tokens))
        return chunks
    except ImportError:
        words = text.split()
        chunks = []
        current = []
        current_len = 0
        for word in words:
            if current_len + len(word.split()) > max_tokens / 4:
                chunks.append(' '.join(current))
                current = [word]
                current_len = len(word.split())
            else:
                current.append(word)
                current_len += len(word.split())
        if current:
            chunks.append(' '.join(current))
        return chunks

def summarize_chunk(chunk):
    client = OpenAI(api_key=API_KEY, base_url="https://api.x.ai/v1/")
    try:
        response = client.chat.completions.create(
            model="grok-4-fast-non-reasoning",
            messages=[{"role": "system", "content": "Summarize this text concisely (under 100 words), preserving key facts."}, {"role": "user", "content": chunk}],
            max_tokens=100, stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Summarization error: {e}"

def keyword_search(query, top_k=5, user=None, convo_id=None):
    try:
        with sqlite3.connect('chatapp.db') as conn:
            c = conn.cursor()
            c.execute("SELECT mem_key, mem_value FROM memory WHERE user=? AND convo_id=?", (user, convo_id))
            rows = c.fetchall()
            scores = {}
            query_words = set(query.lower().split())
            for key, value_str in rows:
                value = json.loads(value_str)
                text = f"{value.get('summary', '')} {value.get('details', '')}".lower()
                text_words = set(text.split())
                score = len(query_words.intersection(text_words)) / len(query_words) if query_words else 0
                scores[key] = score
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            results = [{"id": key, "score": score} for key, score in sorted_scores[:top_k]]
            for res in results:
                st.session_state['memory_cache'].update_access(res["id"])
            return results
    except Exception as e:
        return f"Keyword search error: {e}"

def git_ops(operation, repo_path="", **kwargs):
    try:
        import pygit2
        safe_repo = restrict_to_sandbox(repo_path)
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

def db_query(db_path, query, params=[]):
    try:
        with sqlite3.connect(restrict_to_sandbox(db_path)) as db_conn:
            cur = db_conn.cursor()
            cur.execute(query, params)
            if query.strip().upper().startswith('SELECT'):
                return json.dumps(cur.fetchall())
            db_conn.commit()
            return f"{cur.rowcount} rows affected."
    except Exception as e:
        return f"DB error: {e}"

WHITELISTED_COMMANDS = ['ls', 'grep', 'sed', 'cat', 'echo', 'pwd', 'grim']
def shell_exec(command):
    cmd_parts = shlex.split(command)
    if not cmd_parts or cmd_parts[0] not in WHITELISTED_COMMANDS:
        return "Command not whitelisted."
    try:
        result = subprocess.run(cmd_parts, cwd=SANDBOX_DIR, capture_output=True, text=True, timeout=5)
        return result.stdout.strip() + ("\nError: " + result.stderr.strip() if result.stderr else "")
    except Exception as e:
        return f"Shell error: {e}"

def code_lint(language, code):
    lang = language.lower()
    try:
        if lang == 'python':
            from black import format_str, FileMode
            return format_str(code, mode=FileMode(line_length=88))
        elif lang in ['javascript', 'css']:
            import jsbeautifier
            opts = jsbeautifier.default_options()
            return jsbeautifier.beautify(code, opts)
        elif lang == 'json':
            return json.dumps(json.loads(code), indent=4)
        elif lang == 'yaml':
            import yaml
            return yaml.safe_dump(yaml.safe_load(code), indent=2)
        elif lang == 'sql':
            import sqlparse
            return sqlparse.format(code, reindent=True, keyword_case='upper')
        elif lang == 'xml':
            import xml.dom.minidom
            dom = xml.dom.minidom.parseString(code)
            return dom.toprettyxml(indent="  ")
        elif lang == 'html':
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(code, 'html.parser')
            return soup.prettify()
        elif lang in ['c', 'cpp', 'c++']:
            return subprocess.check_output(['clang-format', '-style=google'], input=code.encode()).decode()
        elif lang == 'php':
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.php', delete=False) as tmp:
                tmp.write(code.encode())
                tmp.flush()
                subprocess.check_call(['php-cs-fixer', 'fix', tmp.name, '--quiet'])
                with open(tmp.name, 'r') as f:
                    formatted = f.read()
            os.unlink(tmp.name)
            return formatted
        elif lang == 'go':
            return subprocess.check_output(['gofmt'], input=code.encode()).decode()
        elif lang == 'rust':
            return subprocess.check_output(['rustfmt', '--emit=stdout'], input=code.encode()).decode()
        return "Unsupported language."
    except Exception as e:
        return f"Lint error: {e}"

API_WHITELIST = ['https://jsonplaceholder.typicode.com/', 'https://api.openweathermap.org/']
def api_simulate(url, method='GET', data=None, mock=True):
    cache_args = {'url': url, 'method': method, 'data': data, 'mock': mock}
    args_json = json.dumps(cache_args, sort_keys=True)
    cached = get_cached_tool_result('api_simulate', args_json)
    if cached:
        return cached[1]
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
            except requests.RequestException as e:
                result = f"API error: {e}"
    set_cached_tool_result('api_simulate', cache_args, result)
    return result

def langsearch_web_search(query, freshness="noLimit", summary=False, count=5):
    if not LANGSEARCH_API_KEY:
        return "LangSearch API key not set—configure in .env."
    url = "https://api.langsearch.com/v1/web-search"
    payload = json.dumps({"query": query, "freshness": freshness, "summary": summary, "count": count})
    headers = {'Authorization': f'Bearer {LANGSEARCH_API_KEY}', 'Content-Type': 'application/json'}
    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        return json.dumps(response.json())
    except requests.RequestException as e:
        return f"LangSearch error: {e}"

TOOL_REGISTRY = {
    "fs_read_file": {
        "function": fs_read_file,
        "schema": {"type": "function", "function": {"name": "fs_read_file", "description": "Read the content of a file in the sandbox directory (./sandbox/). Supports relative paths (e.g., 'subdir/test.txt'). Use for fetching data.", "parameters": {"type": "object", "properties": {"file_path": {"type": "string", "description": "Relative path to the file (e.g., subdir/test.txt)."}}, "required": ["file_path"]}}}
    },
    "fs_write_file": {
        "function": fs_write_file,
        "schema": {"type": "function", "function": {"name": "fs_write_file", "description": "Write content to a file in the sandbox directory (./sandbox/). Supports relative paths (e.g., 'subdir/newfile.txt'). Use for saving or updating files.", "parameters": {"type": "object", "properties": {"file_path": {"type": "string", "description": "Relative path to the file (e.g., subdir/newfile.txt)."}, "content": {"type": "string", "description": "Content to write."}}, "required": ["file_path", "content"]}}}
    },
    "fs_list_files": {
        "function": fs_list_files,
        "schema": {"type": "function", "function": {"name": "fs_list_files", "description": "List all files in a directory within the sandbox (./sandbox/). Supports relative paths (default: root). Use to check available files.", "parameters": {"type": "object", "properties": {"dir_path": {"type": "string", "description": "Relative path to the directory (e.g., subdir). Optional; defaults to root."}}, "required": []}}}
    },
    "fs_mkdir": {
        "function": fs_mkdir,
        "schema": {"type": "function", "function": {"name": "fs_mkdir", "description": "Create a new directory in the sandbox (./sandbox/). Supports relative/nested paths (e.g., 'subdir/newdir'). Use to organize files.", "parameters": {"type": "object", "properties": {"dir_path": {"type": "string", "description": "Relative path for the new directory (e.g., subdir/newdir)."}}, "required": ["dir_path"]}}}
    },
    "get_current_time": {
        "function": get_current_time,
        "schema": {"type": "function", "function": {"name": "get_current_time", "description": "Fetch current datetime. Use host clock by default; sync with NTP if requested for precision.", "parameters": {"type": "object", "properties": {"sync": {"type": "boolean", "description": "True for NTP sync (requires network), false for local host time. Default: false."}, "format": {"type": "string", "description": "Output format: 'iso' (default), 'human', 'json'."}}, "required": []}}}
    },
    "code_execution": {
        "function": code_execution,
        "schema": {"type": "function", "function": {"name": "code_execution", "description": "Execute provided code in a stateful REPL environment and return output or errors for verification. Supports Python with various libraries (e.g., numpy, sympy).", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "The code snippet to execute."}}, "required": ["code"]}}}
    },
    "memory_insert": {
        "function": memory_insert,
        "schema": {"type": "function", "function": {"name": "memory_insert", "description": "Insert or update a memory key-value pair (value as JSON dict) for logging/metadata. Use for fast persistent storage without files.", "parameters": {"type": "object", "properties": {"mem_key": {"type": "string", "description": "Key for the memory entry (e.g., 'chat_log_1')."}, "mem_value": {"type": "object", "description": "Value as dict (e.g., {'content': 'Log text'})."}, "user": {"type": "string"}, "convo_id": {"type": "integer"}}, "required": ["mem_key", "mem_value", "user", "convo_id"]}}}
    },
    "memory_query": {
        "function": memory_query,
        "schema": {"type": "function", "function": {"name": "memory_query", "description": "Query memory: specific key or last N entries. Returns JSON. Use for recalling logs without FS reads.", "parameters": {"type": "object", "properties": {"mem_key": {"type": "string", "description": "Specific key to query (optional)."}, "limit": {"type": "integer", "description": "Max recent entries if no key (default 10)."}, "user": {"type": "string"}, "convo_id": {"type": "integer"}}, "required": ["user", "convo_id"]}}}
    },
    "git_ops": {
        "function": git_ops,
        "schema": {"type": "function", "function": {"name": "git_ops", "description": "Basic Git operations in sandbox (init, commit, branch, diff). No remote operations.", "parameters": {"type": "object", "properties": {"operation": {"type": "string", "enum": ["init", "commit", "branch", "diff"]}, "repo_path": {"type": "string", "description": "Relative path to repo."}, "message": {"type": "string", "description": "Commit message (for commit)."}, "name": {"type": "string", "description": "Branch name (for branch)."}}, "required": ["operation", "repo_path"]}}}
    },
    "db_query": {
        "function": db_query,
        "schema": {"type": "function", "function": {"name": "db_query", "description": "Interact with local SQLite database in sandbox (create, insert, query).", "parameters": {"type": "object", "properties": {"db_path": {"type": "string", "description": "Relative path to DB file."}, "query": {"type": "string", "description": "SQL query."}, "params": {"type": "array", "items": {"type": "string"}, "description": "Query parameters."}}, "required": ["db_path", "query"]}}}
    },
    "shell_exec": {
        "function": shell_exec,
        "schema": {"type": "function", "function": {"name": "shell_exec", "description": "Run safe whitelisted shell commands in sandbox (e.g., ls, grep).", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "Shell command string."}}, "required": ["command"]}}}
    },
    "code_lint": {
        "function": code_lint,
        "schema": {"type": "function", "function": {"name": "code_lint", "description": "Lint and auto-format code for languages: python, javascript, css, json, yaml, sql, xml, html, cpp, php, go, rust.", "parameters": {"type": "object", "properties": {"language": {"type": "string", "description": "Language (python, javascript, css, json, yaml, sql, xml, html, cpp, php, go, rust)."}, "code": {"type": "string", "description": "Code snippet."}}, "required": ["language", "code"]}}}
    },
    "api_simulate": {
        "function": api_simulate,
        "schema": {"type": "function", "function": {"name": "api_simulate", "description": "Simulate API calls with mock or fetch from public APIs.", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "API URL."}, "method": {"type": "string", "description": "GET/POST (default GET)."}, "data": {"type": "object", "description": "POST data."}, "mock": {"type": "boolean", "description": "True for mock (default)."}}, "required": ["url"]}}}
    },
    "advanced_memory_consolidate": {
        "function": advanced_memory_consolidate,
        "schema": {"type": "function", "function": {"name": "advanced_memory_consolidate", "description": "Brain-like consolidation: Summarize and embed data for hierarchical storage.", "parameters": {"type": "object", "properties": {"mem_key": {"type": "string", "description": "Key for the memory entry."}, "interaction_data": {"type": "object", "description": "Data to consolidate (dict)."}, "user": {"type": "string"}, "convo_id": {"type": "integer"}}, "required": ["mem_key", "interaction_data", "user", "convo_id"]}}}
    },
    "advanced_memory_retrieve": {
        "function": advanced_memory_retrieve,
        "schema": {"type": "function", "function": {"name": "advanced_memory_retrieve", "description": "Retrieve relevant memories via embedding similarity.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Query string for similarity search."}, "top_k": {"type": "integer", "description": "Number of top results (default 5)."}, "user": {"type": "string"}, "convo_id": {"type": "integer"}}, "required": ["query", "user", "convo_id"]}}}
    },
    "advanced_memory_prune": {
        "function": advanced_memory_prune,
        "schema": {"type": "function", "function": {"name": "advanced_memory_prune", "description": "Prune low-salience memories to optimize storage.", "parameters": {"type": "object", "properties": {"user": {"type": "string"}, "convo_id": {"type": "integer"}}, "required": ["user", "convo_id"]}}}
    },
    "langsearch_web_search": {
        "function": langsearch_web_search,
        "schema": {"type": "function", "function": {"name": "langsearch_web_search", "description": "Search the web using LangSearch API.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query."}, "freshness": {"type": "string", "description": "Time filter: oneDay, oneWeek, oneMonth, oneYear, or noLimit (default).", "enum": ["oneDay", "oneWeek", "oneMonth", "oneYear", "noLimit"]}, "summary": {"type": "boolean", "description": "Include long text summaries (default True)."}, "count": {"type": "integer", "description": "Number of results (1-10, default 5)."}}, "required": ["query"]}}}
    },
    "generate_embedding": {
        "function": generate_embedding,
        "schema": {"type": "function", "function": {"name": "generate_embedding", "description": "Generate vector embedding for text using SentenceTransformer (384-dim vector).", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "Text to embed."}}, "required": ["text"]}}}
    },
    "vector_search": {
        "function": vector_search,
        "schema": {"type": "function", "function": {"name": "vector_search", "description": "Perform ANN vector search in ChromaDB using cosine similarity.", "parameters": {"type": "object", "properties": {"query_embedding": {"type": "array", "items": {"type": "number"}, "description": "Query embedding vector."}, "top_k": {"type": "integer", "description": "Number of top results (default 5)."}, "threshold": {"type": "number", "description": "Min similarity score (default 0.6)."}}, "required": ["query_embedding"]}}}
    },
    "chunk_text": {
        "function": chunk_text,
        "schema": {"type": "function", "function": {"name": "chunk_text", "description": "Split text into semantic chunks (default 512 tokens).", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "Text to chunk."}, "max_tokens": {"type": "integer", "description": "Max tokens per chunk (default 512)."}}, "required": ["text"]}}}
    },
    "summarize_chunk": {
        "function": summarize_chunk,
        "schema": {"type": "function", "function": {"name": "summarize_chunk", "description": "Compress a text chunk via LLM summary (under 100 words).", "parameters": {"type": "object", "properties": {"chunk": {"type": "string", "description": "Text chunk to summarize."}}, "required": ["chunk"]}}}
    },
    "keyword_search": {
        "function": keyword_search,
        "schema": {"type": "function", "function": {"name": "keyword_search", "description": "Keyword-based search on memory cache (simple overlap/BM25 sim).", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query."}, "top_k": {"type": "integer", "description": "Number of top results (default 5)."}, "user": {"type": "string"}, "convo_id": {"type": "integer"}}, "required": ["query", "user", "convo_id"]}}}
    },
}

def load_prompt_files():
    mtime = os.path.getmtime(PROMPTS_DIR)
    if mtime > st.session_state['prompt_files_mtime']:
        st.session_state['prompt_files'] = [f for f in os.listdir(PROMPTS_DIR) if f.endswith('.txt')]
        st.session_state['prompt_files_mtime'] = mtime
    return st.session_state['prompt_files']

def cache_images(image_files):
    image_cache = st.session_state['image_cache']
    for img_file in image_files:
        img_id = hash(img_file.getvalue())
        if img_id not in image_cache:
            with img_file:
                image_cache[img_id] = base64.b64encode(img_file.read()).decode('utf-8')
    return [(img_id, image_cache[img_id]) for img_id in image_cache]

def call_xai_api(model, messages, sys_prompt, stream=True, image_files=None, enable_tools=False):
    client = OpenAI(api_key=API_KEY, base_url="https://api.x.ai/v1/", timeout=300)
    api_messages = [{"role": "system", "content": sys_prompt}]
    cached_images = cache_images(image_files or [])
    for msg in messages:
        content_parts = [{"type": "text", "text": msg['content']}]
        if msg['role'] == 'user' and cached_images and msg is messages[-1]:
            for _, img_data in cached_images:
                content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}})
        api_messages.append({"role": msg['role'], "content": content_parts if len(content_parts) > 1 else msg['content']})
    
    max_iterations = 5
    for _ in range(max_iterations):
        try:
            response = client.chat.completions.create(
                model=model, messages=api_messages, tools=[v['schema'] for v in TOOL_REGISTRY.values()] if enable_tools else None,
                tool_choice="auto" if enable_tools else None, stream=True
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
            api_messages.append({"role": "assistant", "content": full_delta_response, "tool_calls": tool_calls})
            yield "\n*Thinking... Using tools...*\n"
            tool_outputs = []
            with sqlite3.connect('chatapp.db') as conn:
                for tool_call in tool_calls:
                    func_name = tool_call.function.name
                    try:
                        args = json.loads(tool_call.function.arguments)
                        if func_name.startswith(("memory", "advanced_memory", "keyword_search")):
                            args['user'] = st.session_state['user']
                            args['convo_id'] = st.session_state.get('current_convo_id', 0)
                        result = TOOL_REGISTRY[func_name]['function'](**args)
                    except Exception as e:
                        result = f"Error calling tool {func_name}: {e}"
                    yield f"\n> **Tool Call:** `{func_name}` | **Result:** `{str(result)[:200]}...`\n"
                    tool_outputs.append({"tool_call_id": tool_call.id, "role": "tool", "content": str(result)})
                conn.commit()
            api_messages.extend(tool_outputs)
        except Exception as e:
            yield f"\nAn error occurred: {e}. Aborting this turn."
            break

def login_page():
    st.title("Welcome to Apex Orchestrator")
    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                with sqlite3.connect('chatapp.db') as conn:
                    c = conn.cursor()
                    c.execute("SELECT password FROM users WHERE username=?", (username,))
                    result = c.fetchone()
                if result and verify_password(result[0], password):
                    st.session_state['logged_in'] = True
                    st.session_state['user'] = username
                    st.session_state['repl_namespace'] = {'__builtins__': {b: getattr(builtins, b) for b in ['print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple', 'abs', 'round', 'max', 'min', 'sum', 'sorted']}}
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
    with tab2:
        with st.form("register_form"):
            new_user = st.text_input("New Username")
            new_pass = st.text_input("New Password", type="password")
            if st.form_submit_button("Register"):
                with sqlite3.connect('chatapp.db') as conn:
                    c = conn.cursor()
                    c.execute("SELECT * FROM users WHERE username=?", (new_user,))
                    if c.fetchone():
                        st.error("Username already exists.")
                    else:
                        c.execute("INSERT INTO users VALUES (?, ?)", (new_user, hash_password(new_pass)))
                        conn.commit()
                        st.success("Registered! Please login.")

def load_history(convo_id):
    with sqlite3.connect('chatapp.db') as conn:
        c = conn.cursor()
        c.execute("SELECT messages FROM history WHERE convo_id=? AND user=?", (convo_id, st.session_state['user']))
        result = c.fetchone()
    if result:
        st.session_state['messages'] = json.loads(result[0])
        st.session_state['current_convo_id'] = convo_id

def delete_history(convo_id):
    with sqlite3.connect('chatapp.db') as conn:
        c = conn.cursor()
        c.execute("DELETE FROM history WHERE convo_id=? AND user=?", (convo_id, st.session_state['user']))
        conn.commit()
    if st.session_state.get('current_convo_id') == convo_id:
        st.session_state['messages'] = []
        st.session_state['current_convo_id'] = 0

def chat_page():
    st.title(f"Apex Chat - {st.session_state['user']}")
    with st.sidebar:
        st.header("Chat Settings")
        model = st.selectbox("Select Model", ["grok-4-fast-reasoning", "grok-4", "grok-code-fast-1", "grok-3-mini"], key="model_select")
        prompt_files = load_prompt_files()
        if prompt_files:
            selected_file = st.selectbox("Select System Prompt", prompt_files, key="prompt_select")
            with open(os.path.join(PROMPTS_DIR, selected_file), "r") as f:
                prompt_content = f.read()
            custom_prompt = st.text_area("Edit System Prompt", value=prompt_content, height=200, key="custom_prompt")
        else:
            st.warning("No prompt files found in ./prompts/")
            custom_prompt = st.text_area("System Prompt", value="You are a helpful AI.", height=200, key="custom_prompt")
        uploaded_images = st.file_uploader("Upload Images", type=["jpg", "png"], accept_multiple_files=True, key="uploaded_images")
        enable_tools = st.checkbox("Enable Tools (Sandboxed)", value=False, key='enable_tools')
        st.divider()
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state['messages'] = []
            st.session_state['current_convo_id'] = 0
        st.header("Chat History")
        with sqlite3.connect('chatapp.db') as conn:
            c = conn.cursor()
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
    
    for msg in st.session_state['messages']:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=False)
    
    if prompt := st.chat_input("What would you like to discuss?"):
        st.session_state['messages'].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt, unsafe_allow_html=False)
        with st.chat_message("assistant"):
            response_container = st.empty()
            full_response = ""
            images_to_process = st.session_state.get('uploaded_images', [])
            generator = call_xai_api(model, st.session_state['messages'], custom_prompt, stream=True, image_files=images_to_process, enable_tools=enable_tools)
            for chunk in generator:
                full_response += chunk
                response_container.markdown(full_response + " ▌", unsafe_allow_html=False)
            response_container.markdown(full_response, unsafe_allow_html=False)
        st.session_state['messages'].append({"role": "assistant", "content": full_response})
        title_message = next((msg['content'] for msg in st.session_state['messages'] if msg['role'] == 'user'), "New Chat")
        title = (title_message[:40] + '...') if len(title_message) > 40 else title_message
        messages_json = json.dumps(st.session_state['messages'])
        with sqlite3.connect('chatapp.db') as conn:
            c = conn.cursor()
            if st.session_state.get("current_convo_id", 0) == 0:
                c.execute("INSERT INTO history (user, title, messages) VALUES (?, ?, ?)", (st.session_state['user'], title, messages_json))
                st.session_state['current_convo_id'] = c.lastrowid
            else:
                c.execute("UPDATE history SET title=?, messages=? WHERE convo_id=?", (title, messages_json, st.session_state['current_convo_id']))
            conn.commit()

if __name__ == "__main__":
    with open("style.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    setup_database()
    initialize_session_state()
    if st.session_state['logged_in']:
        chat_page()
    else:
        login_page()
