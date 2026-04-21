#!/usr/bin/env python3
# a simple local coding agent using llama-server and gemma4
# Copyright 2026 Harold Braux. MIT License

import argparse
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

BASE_URL = "http://localhost:8080"
MODEL_ID = os.environ.get("HOMECODE_MODEL_ID", "bartowski/google_gemma-4-E4B-it-GGUF:Q5_K_M")
MODEL_PARAMS = os.environ.get("HOMECODE_MODEL_PARAMS", "--no-mmproj --ctx-size 16384 --flash-attn on --temp 0.1 --n-gpu-layers all")
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
RESET       = "\033[0m"
SYSTEM_PROMPT = "You are an expert coding assistant. You help with any programming language, framework, or tool. You can read, write, and edit files, search code, and run shell commands to assist with software engineering tasks."


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
    with open(LLAMA_LOG, "a") as log:
        proc = subprocess.Popen(
            [LLAMA_BIN, "-v", "-hf", MODEL_ID] + shlex.split(MODEL_PARAMS) + ["--tools", "all"],
            stdout=log, stderr=log,
        )
    print(f"PID {proc.pid} — logs: {LLAMA_LOG}", file=sys.stderr)
    for _ in range(120):
        time.sleep(1)
        if is_ready():
            return proc
    print(f"{BOLD_YELLOW}Server failed to start — check {LLAMA_LOG}{RESET}", file=sys.stderr)
    sys.exit(1)


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


def web_search(query):
    resp = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": TAVILY_API_KEY, "query": query, "max_results": 5},
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    return "\n\n".join(f"{r['title']}\n{r['url']}\n{r.get('content', '')}" for r in results)


def fetch_tools():
    return [t["definition"] for t in requests.get(f"{BASE_URL}/tools").json()]


def execute_tool(name, args):
    if name == "web_search":
        return web_search(args["query"])
    resp = requests.post(f"{BASE_URL}/tools", json={"tool": name, "params": args})
    resp.raise_for_status()
    data = resp.json()
    return data.get("plain_text_response") or str(data)


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
            for tc in msg["tool_calls"]:
                name = tc["function"]["name"]
                args = json.loads(tc["function"]["arguments"])
                print(f"{YELLOW}  [tool] {name} {args}{RESET}", file=sys.stderr)
                if name == "exec_shell_command":
                    print(f"{BOLD}  run? [Y/n] {RESET}", end="", flush=True, file=sys.stderr)
                    if input().strip().lower() == "n":
                        print(f"{BOLD_YELLOW}  Aborted{RESET}", file=sys.stderr)
                        return
                result = execute_tool(name, args)
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
        else:
            content = msg.get("content") or ""
            console.print(Markdown(content))
            if show_timings:
                tps = body.get("timings", {}).get("predicted_per_second")
                if tps:
                    print(f"{YELLOW}  {tps:.1f} tok/s{RESET}")
            print()
            return


def main():
    parser = argparse.ArgumentParser(description="CodePy — local AI coding agent")
    parser.add_argument("--install", action="store_true", help="install or update llama.cpp to ~/.llama")
    parser.add_argument("--timings", action="store_true", help="show token rate after each response")
    parser.add_argument("--keep-server", action="store_true", help="do not stop llama-server on exit")
    args = parser.parse_args()
    if args.install:
        install_llama()
        return
    readline.parse_and_bind("tab: complete")
    readline.parse_and_bind("set editing-mode emacs")
    try:
        readline.read_history_file(HISTORY_FILE)
    except (FileNotFoundError, PermissionError):
        pass
    server_proc = ensure_server()
    tools = fetch_tools()
    if TAVILY_API_KEY:
        tools.append(TAVILY_SEARCH_TOOL)
    tool_names = ", ".join(t["function"]["name"] for t in tools)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    model_name = requests.get(f"{BASE_URL}/v1/models").json()["data"][0]["id"]
    print(f"{BOLD_BLUE}CodePy v0.1 — using {model_name}{RESET}")
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
        if server_proc is not None and not args.keep_server:
            print(f"{BOLD_YELLOW}Stopping llama-server ...{RESET}", file=sys.stderr)
            server_proc.terminate()
            server_proc.wait()


if __name__ == "__main__":
    main()
