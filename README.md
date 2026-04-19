# homecode

A simple local AI coding agent in Python. Sends prompts to a local `llama-server` (OpenAI-compatible API) and executes tool calls client-side in an agentic loop.

## Requirements

- Python 3.x with `requests` and `rich` (`pip install requests rich`)
- A GPU with Vulkan support (Linux) or Apple Silicon (macOS)

## Install

```bash
python3 homecode.py --install   # download latest llama.cpp to ~/.llama
```

## Usage

```bash
python3 homecode.py             # start the REPL (auto-starts llama-server)
python3 homecode.py --timings   # show token/s after each response
python3 homecode.py --keep-server  # leave llama-server running on exit
```

Press `ctrl+d` to exit. Input history is saved to `~/.homecode_history`.

## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `HOMECODE_MODEL_ID` | `bartowski/google_gemma-4-E4B-it-GGUF:Q5_K_M` | HuggingFace model repo |
| `HOMECODE_MODEL_PARAMS` | see source | Extra flags passed to `llama-server` |
| `TAVILY_API_KEY` | *(unset)* | Enable web search via Tavily |

## How it works

`homecode.py` → POST `/v1/chat/completions` → llama-server → `finish_reason: tool_calls` → execute tool client-side → append result → loop until `finish_reason: stop` → print response.

Tool calls are delegated to llama-server's `/tools` endpoint. Shell commands require confirmation before running. If `TAVILY_API_KEY` is set, a `web_search` tool is added automatically.

llama-server logs go to `~/.llama/llama.log`.
