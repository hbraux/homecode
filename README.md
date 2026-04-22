# homecode

A simple local AI coding agent in Python. Sends prompts to a local `llama-server` (OpenAI-compatible API) and 
executes tool calls client-side in an agentic loop.

## Requirements

- Python 3.x with `requests` and `rich` (usually already in the OS python env, otherwise run `pip install requests rich`)
- A GPU with Vulkan support (Linux) or Apple Silicon (macOS)

## Installation

```bash
curl -O https://raw.githubusercontent.com/hbraux/homecode/main/homecode.py
chmod +x homecode.py
ln -fs "$(pwd)/homecode.py" ~/.local/bin/homecode
```

Make sure `~/.local/bin` is in your `PATH`

## Usage

```bash
homecode             # start the REPL (auto-starts llama-server)
homecode --install   # download latest llama.cpp version to ~/.llama
homecode --timings   # show token/s after each response
homecode --keep      # leave llama-server running on exit
homecode --update    # update the tool to the latest version
```

Press `ctrl+d` to exit. Input history is saved to `~/.homecode_history`.

## Configuration

| Environment variable    | Default                                       | Description                          |
|-------------------------|-----------------------------------------------|--------------------------------------|
| `HOMECODE_MODEL_ID`     | `bartowski/google_gemma-4-E4B-it-GGUF:Q5_K_M` | HuggingFace model repo               |
| `HOMECODE_MODEL_PARAMS` | see source                                    | Extra flags passed to `llama-server` |
| `TAVILY_API_KEY`        | *(unset)*                                     | Enable web search via Tavily         |

## Project context (AGENT.md)

If an `AGENT.md` file exists in the current directory when `homecode` starts, it is automatically loaded and 
appended to the system prompt. Use it to provide project-specific context to the model (language, build tool, framework, conventions, etc.).\
Example:

```
Project context:
* Language: Kotlin (Idiomatic)
* Build Tool: Maven (pom.xml)
* Game framework: LibGDX 1.12.1 (backend LWJGL3)
```

A confirmation message is printed at startup when `AGENT.md` is found.

## How it works

`homecode.py` → POST `/v1/chat/completions` → llama-server → `finish_reason: tool_calls` → execute tool client-side → append result → loop until `finish_reason: stop` → print response.

Tool calls are delegated to llama-server's `/tools` endpoint (except WebSearch). 
Shell commands require confirmation before running. If `TAVILY_API_KEY` is set, a `web_search` tool is added automatically.

llama-server logs go to `~/.llama/llama.log`.
