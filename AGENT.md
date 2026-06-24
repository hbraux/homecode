Language: Python 3, idiomatic
Code style: no comments, no docstrings, minimal inline explanation
Single-file project: homecode.py — do not create additional modules
Entry point: main() in homecode.py
Key constants at top of file: BASE_URL, MODEL_ID, MODEL_PARAMS, LLAMA_DIR, LLAMA_BIN, CONFIRM_TOOLS, SYSTEM_PROMPT
Agentic loop: chat() — posts to /v1/chat/completions, handles tool_calls finish_reason, loops until stop
Tool dispatch: execute_tool() — web_search handled locally via Tavily, all other tools delegated to llama-server /tools endpoint
Server lifecycle: ensure_server() auto-starts llama-server subprocess; terminated on exit unless --keep
Install target: ~/.llama/ (binary + model weights)
History file: ~/.homecode_history
