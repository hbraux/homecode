#!/usr/bin/env python3
# a simple local coding agent using llama-server and gemma4
# Copyright 2026 Harold Braux. MIT License

import argparse
import glob
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import readline
import requests
from rich.console import Console
from rich.markdown import Markdown

console = Console()

VERSION = "0.3"
BASE_URL = "http://localhost:8080"
MODEL_ID = os.environ.get("HOMECODE_MODEL_ID", "yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF:Q4_K_M")
MODEL_PARAMS = os.environ.get("HOMECODE_MODEL_PARAMS", "--ctx-size 16384 --n-gpu-layers 99 --no-mmap  -fa on --jinja --top-p 0.95 --top-k 64 --temp 0.1")
LLAMA_DIR = os.path.expanduser("~/.llama")
LLAMA_BIN = os.path.join(LLAMA_DIR, "llama-server")
LLAMA_LOG = os.path.join(LLAMA_DIR, "llama.log")
if platform.system() == "Darwin":
    LLAMA_ASSET_PATTERN = r"macos.*arm64"
else:
    LLAMA_ASSET_PATTERN = r"ubuntu.*vulkan.*x64"
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
HISTORY_FILE = os.path.expanduser("~/.homecode_history")

CYAN        = "\033[36m"
YELLOW      = "\033[33m"
BOLD        = "\033[1m"
BOLD_YELLOW = "\033[1;33m"
BOLD_BLUE   = "\033[1;34m"
DIM_PURPLE  = "\033[22;38;5;177m"
RESET       = "\033[0m"
AGENT_FILE = "AGENT.md"
SYSTEM_PROMPT = (
    "You are an expert coding assistant. "
    "You help with any programming language, framework, or tool. "
    "You can read, write, and edit files, search code, and run shell commands to assist with software engineering tasks. "
    "Only use tools when strictly necessary to answer the question. "
    "For simple questions that can be answered directly, respond with text only — do not call any tools. "
    "Only write or edit files when the user explicitly asks you to create or modify a file. "
    "Be concise."
)

# ── client-defined tools ──────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file. Only use when you need to inspect a specific file to answer the user's question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file, relative to the working directory"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Only use when the user explicitly asks to create or modify a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file, relative to the working directory"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files matching a glob pattern within the working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.py' or '*.md'"}
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search for a text pattern across files in the working directory using grep.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "file_glob": {"type": "string", "description": "Glob to restrict which files to search, e.g. '*.py'", "default": "*"},
                    "max_lines": {"type": "integer", "description": "Maximum number of result lines to return (default 50)"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "exec_shell_command",
            "description": "Run a shell command in the working directory. Use only when no other tool can answer the question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"}
                },
                "required": ["command"],
            },
        },
    },
]

TAVILY_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web and return the top results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"],
        },
    },
}

# Tools that require user confirmation before running
CONFIRM_TOOLS = {"exec_shell_command", "write_file"}


def _safe_path(path):
    """Resolve path relative to cwd, refuse traversal outside it."""
    cwd = os.getcwd()
    full = os.path.realpath(os.path.join(cwd, path))
    if not full.startswith(cwd):
        raise ValueError(f"Path outside working directory: {path}")
    return full


def tool_read_file(path):
    full = _safe_path(path)
    with open(full) as f:
        return f.read()


def tool_write_file(path, content):
    full = _safe_path(path)
    os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
    with open(full, "w") as f:
        f.write(content)
    return f"Written {path}"


def tool_list_files(pattern):
    cwd = os.getcwd()
    matches = glob.glob(pattern, root_dir=cwd, recursive=True)
    return "\n".join(sorted(matches)) or "(no matches)"


def tool_search_code(pattern, file_glob="*", max_lines=50):
    result = subprocess.run(
        ["grep", "-rn", "--include", file_glob, pattern, "."],
        capture_output=True, text=True,
    )
    output = result.stdout
    if not output:
        return "(no matches)"
    lines = output.splitlines()
    truncated = len(lines) > max_lines
    lines = lines[:max_lines]
    out = "\n".join(lines)
    if truncated:
        out += f"\n... (truncated to {max_lines} lines)"
    return out


def tool_exec_shell_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    out = result.stdout + result.stderr
    return out.strip() or "(no output)"


def web_search(query):
    resp = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": TAVILY_API_KEY, "query": query, "max_results": 5},
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    return "\n\n".join(f"{r['title']}\n{r['url']}\n{r.get('content', '')}" for r in results)


def execute_tool(name, args):
    if name == "read_file":
        return tool_read_file(args["path"])
    if name == "write_file":
        return tool_write_file(args["path"], args["content"])
    if name == "list_files":
        return tool_list_files(args["pattern"])
    if name == "search_code":
        return tool_search_code(args["pattern"], args.get("file_glob", "*"), args.get("max_lines", 50))
    if name == "exec_shell_command":
        return tool_exec_shell_command(args["command"])
    if name == "web_search":
        return web_search(args["query"])
    return f"Unknown tool: {name}"


# ── llama-server management ───────────────────────────────────────────────────

def install_llama():
    print("Fetching latest llama.cpp release ...")
    release = requests.get("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest").json()
    latest = release["tag_name"]
    version_file = os.path.join(LLAMA_DIR, ".version")
    if os.path.isfile(version_file):
        with open(version_file) as f:
            if f.read().strip() == latest:
                print(f"Already up to date ({latest})")
                return
    assets = [a for a in release["assets"] if re.search(LLAMA_ASSET_PATTERN + r"\.tar\.gz$", a["name"])]
    if not assets:
        print("No matching asset found for pattern: " + LLAMA_ASSET_PATTERN, file=sys.stderr)
        sys.exit(1)
    url = assets[0]["browser_download_url"]
    print(f"Installing {latest} to {LLAMA_DIR}/ ...")
    os.makedirs(LLAMA_DIR, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = os.path.join(tmp, "llama.tar.gz")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(archive, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
        with tarfile.open(archive) as tar:  # type: ignore[attr-defined]
            tar.extractall(tmp)
        for root, _, files in os.walk(tmp):
            depth = root[len(tmp):].count(os.sep)
            if depth > 3:
                continue
            for name in files:
                src = os.path.join(root, name)
                if ".so" in name or os.access(src, os.X_OK):
                    shutil.copy2(src, os.path.join(LLAMA_DIR, name))
    with open(version_file, "w") as f:
        f.write(latest)
    print(f"Installed {latest}")


def update_script():
    url = "https://raw.githubusercontent.com/hbraux/homecode/main/homecode.py"
    print("Checking for updates ...")
    resp = requests.get(url)
    resp.raise_for_status()
    new_source = resp.text
    match = re.search(r'^VERSION\s*=\s*"([^"]+)"', new_source, re.MULTILINE)
    if not match:
        print("Could not determine remote version.", file=sys.stderr)
        sys.exit(1)
    remote_version = match.group(1)
    if remote_version == VERSION:
        print(f"Already up to date (v{VERSION})")
        return
    script_path = os.path.realpath(__file__)
    tmp = script_path + ".tmp"
    with open(tmp, "w") as f:
        f.write(new_source)
    os.chmod(tmp, os.stat(script_path).st_mode)
    os.replace(tmp, script_path)
    print(f"Updated v{VERSION} → v{remote_version}")


def ensure_server():
    def is_ready():
        try:
            return requests.get(f"{BASE_URL}/health", timeout=1).json().get("status") == "ok"
        except requests.exceptions.ConnectionError:
            return False
    if is_ready():
        return None
    if not os.path.isfile(LLAMA_BIN):
        print(f"{BOLD_YELLOW}llama-server not found — run: ./homecode.py --install{RESET}", file=sys.stderr)
        sys.exit(1)
    version_file = os.path.join(LLAMA_DIR, ".version")
    version = open(version_file).read().strip() if os.path.isfile(version_file) else "unknown"
    print(f"{BOLD_YELLOW}Starting llama-server {version} ...{RESET}", file=sys.stderr)
    os.makedirs(LLAMA_DIR, exist_ok=True)
    with open(LLAMA_LOG, "w") as log:
        proc = subprocess.Popen(
            [LLAMA_BIN, "-hf", MODEL_ID] + shlex.split(MODEL_PARAMS),
            stdout=log, stderr=log,
        )
    print(f"PID {proc.pid} — logs: {LLAMA_LOG}", file=sys.stderr)
    for _ in range(120):
        time.sleep(1)
        if is_ready():
            return proc
    print(f"{BOLD_YELLOW}Server failed to start — check {LLAMA_LOG}{RESET}", file=sys.stderr)
    sys.exit(1)


# ── chat loop ─────────────────────────────────────────────────────────────────

def chat(messages, tools, show_timings=False):
    while True:
        resp = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={"model": "local", "messages": messages, "tools": tools},
        )
        resp.raise_for_status()
        body = resp.json()
        choice = body["choices"][0]
        msg = choice["message"]
        messages.append(msg)
        if choice["finish_reason"] == "tool_calls":
            aborted = False
            for tc in msg["tool_calls"]:
                name = tc["function"]["name"]
                args = json.loads(tc["function"]["arguments"])
                detail = args.get("path") or args.get("command") or args.get("pattern") or args.get("query") or ""
                print(f"{YELLOW}  {name} {detail}{RESET}", file=sys.stderr)
                if name in CONFIRM_TOOLS:
                    print(f"{BOLD}  run? [Y/n] {RESET}", end="", flush=True, file=sys.stderr)
                    if input().strip().lower() == "n":
                        print(f"{BOLD_YELLOW}  Aborted{RESET}", file=sys.stderr)
                        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": "User aborted this tool call."})
                        aborted = True
                        break
                try:
                    result = execute_tool(name, args)
                except Exception as e:
                    result = f"Error: {e}"
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
            if aborted:
                return
        else:
            reasoning = msg.get("reasoning_content") or ""
            if reasoning:
                print(f"{DIM_PURPLE}{reasoning}{RESET}\n")
            content = msg.get("content") or ""
            console.print(Markdown(content))
            if show_timings:
                tps = body.get("timings", {}).get("predicted_per_second")
                if tps:
                    print(f"{YELLOW}  {tps:.1f} tok/s{RESET}")
            print()
            return


def main():
    parser = argparse.ArgumentParser(description=f"homecode v{VERSION} — local AI coding agent")
    parser.add_argument("--install", action="store_true", help="install or update llama.cpp to ~/.llama")
    parser.add_argument("--update", action="store_true", help="update homecode.py from GitHub")
    parser.add_argument("--timings", action="store_true", help="show token rate after each response")
    parser.add_argument("--keep", action="store_true", help="do not stop llama-server on exit")
    args = parser.parse_args()
    if args.install:
        install_llama()
        return
    if args.update:
        update_script()
        return
    readline.parse_and_bind("tab: complete")
    readline.parse_and_bind("set editing-mode emacs")
    try:
        readline.read_history_file(HISTORY_FILE)
    except (FileNotFoundError, PermissionError):
        pass
    server_proc = ensure_server()
    tools = list(TOOLS)
    if TAVILY_API_KEY:
        tools.append(TAVILY_SEARCH_TOOL)
    tool_names = ", ".join(t["function"]["name"] for t in tools)
    system_prompt = SYSTEM_PROMPT + "\n"
    agent_md = os.path.join(os.getcwd(), AGENT_FILE)
    if os.path.isfile(agent_md):
        with open(agent_md) as f:
            system_prompt += f.read().strip()
        print(f"File {AGENT_FILE} added to prompt")
    messages = [{"role": "system", "content": system_prompt}]
    model_name = requests.get(f"{BASE_URL}/v1/models").json()["data"][0]["id"]
    print(f"{BOLD_BLUE}homecode v{VERSION} — using {model_name}{RESET}")
    print(f"Available tools: {tool_names}")
    print("ctrl+d to exit\n")
    try:
        while True:
            try:
                user_input = input(f"\001{CYAN}\002>\001{RESET}\002 ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_input:
                continue
            print()
            messages.append({"role": "user", "content": user_input})
            chat(messages, tools, show_timings=args.timings)
    finally:
        readline.write_history_file(HISTORY_FILE)
        if server_proc is not None and not args.keep:
            print(f"{BOLD_YELLOW}Stopping llama-server ...{RESET}", file=sys.stderr)
            server_proc.terminate()
            server_proc.wait()


if __name__ == "__main__":
    main()
