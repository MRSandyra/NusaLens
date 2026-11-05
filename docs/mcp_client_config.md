# NusaLens MCP server: client configuration

Add this block to a desktop MCP client's config (for Claude Desktop, `claude_desktop_config.json`; for other clients, the equivalent stdio-server entry) to try the server directly:

```json
{
  "mcpServers": {
    "nusalens": {
      "command": "D:\\My Life\\Projects\\nusalens\\.venv\\Scripts\\python.exe",
      "args": ["-m", "nusalens.mcp_server.main", "--transport", "stdio"],
      "cwd": "D:\\My Life\\Projects\\nusalens"
    }
  }
}
```

Adjust `command`/`cwd` to the actual checkout path. The server needs the stack already running: PostgreSQL on `localhost:55433` (`docker compose up -d` or equivalent) and Ollama with `qwen3:4b` and `qwen2.5:3b-instruct` pulled, since `answer_question` and `search_documents` load the embedding/reranker/LLM clients on first use.

After the client restarts, NusaLens's 6 tools, 3 resources, and 2 prompts should appear in its tool list (see `reports/10_mcp.md` for what each one does).

## Running it manually without a desktop client

```bash
python -m nusalens.mcp_server.main --transport stdio
```

Starts the server over stdio and blocks, waiting for a client's handshake on stdin/stdout. `scripts/mcp_smoke_client.py` is a minimal real client (not a unit test double) that launches this exact command as a subprocess, performs the handshake, and exercises every tool; its full transcript is saved at `reports/raw/10_mcp_transcript.json`.
