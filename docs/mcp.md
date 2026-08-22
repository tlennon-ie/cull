# Cull MCP Server

`cull` ships a **Model Context Protocol** server (`cull-mcp`) so any MCP client — Claude Desktop, Cursor, Codex, or a custom agent — can drive an end-to-end curation run over stdio: list and create jobs, tune scoring, start the pipeline, sample the gallery, and export the result.

The server is a thin wrapper over the *same* public APIs the dashboard and CLI use (`job_config`, `paths`, `queue_manager`, `index_store`, `hf_export`, `export_profiles`). It never reaches into private state.

---

## Install

```bash
pip install -e ".[mcp]"
```

This pulls the [`mcp`](https://pypi.org/project/mcp/) Python SDK and registers a console script called `cull-mcp` on your `PATH`.

Verify the install:

```bash
cull-mcp --help    # (nothing to output — this just proves the binary resolves)
python -c "import mcp_server; print(len(mcp_server.list_tool_names()), 'tools')"
```

If the `mcp` extra is missing, `cull-mcp` still runs — it just prints a friendly install hint on stderr and exits non-zero.

---

## Client setup

### Claude Desktop

Edit `~/.config/Claude/claude_desktop_config.json` (macOS / Linux) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows) and add a `cull` entry:

```json
{
  "mcpServers": {
    "cull": {
      "command": "cull-mcp",
      "env": {
        "PIPELINE_BASE_DIR": "/home/you/cull-data",
        "FLASK_PORT": "5000"
      }
    }
  }
}
```

Restart Claude Desktop. The tools appear in the tool tray as `cull_list_jobs`, `cull_create_job`, …

### Cursor

Cursor uses the same shape. In Cursor Settings → MCP Servers, add:

```json
{
  "cull": {
    "command": "cull-mcp",
    "env": {"PIPELINE_BASE_DIR": "/home/you/cull-data"}
  }
}
```

### Codex / other stdio MCP clients

Any client that speaks the MCP stdio transport can wire up the same command. The typical shape:

```
command: cull-mcp
args:    []
env:     PIPELINE_BASE_DIR=/absolute/path/to/cull/data
```

`cull-mcp` reads no CLI args — configuration flows through the environment (same variables cull's dashboard and CLI use).

---

## Environment

The server inherits every env var the rest of cull reads (see `.env.example`). The ones that matter most for an agent-driven install:

| Var | Purpose |
|-----|---------|
| `PIPELINE_BASE_DIR` | Root of cull's data dir. Contains `queue/`, `sorted/`, `jobs/`, `logs/`, `cull_categories.json`. |
| `FLASK_PORT` | Port the local dashboard listens on. `cull_start_pipeline` / `cull_stop_pipeline` / `cull_pipeline_status` talk to it. Defaults to 5000. |
| `HF_TOKEN` | HuggingFace token for `cull_export_hf`. Never round-tripped to the client — the server logs a masked confirmation and the token stays in memory. |
| `LOG_DIR` | Where the supervisor and workers write their log files. |

Anything set in `.env` at cull's repo root is picked up automatically.

---

## Tool reference

Every tool returns MCP `TextContent` blocks; `cull_sample_gallery` also returns an `ImageContent` block for the top result so the agent can literally *see* the top keeper. Errors surface as `isError=true` responses with a fixed message; the detail is in the server-side log (never leaked).

### Job management

| Tool | Input | Returns |
|------|-------|---------|
| `cull_list_jobs` | – | `{jobs: [...], active: [slug, ...]}` |
| `cull_get_job` | `slug` | Full effective config (preset ⊕ overrides), secrets masked |
| `cull_create_job` | `slug`, `subject?`, `preset?`, `base_on?` | New job record |
| `cull_delete_job` | `slug` | `{deleted: true}` (refuses to delete the active job) |
| `cull_activate_job` | `slug`, `exclusive?` | Projects env + categories; `exclusive=true` resets active set to `[slug]` |
| `cull_deactivate_job` | `slug` | Removes `slug` from active set (idempotent) |
| `cull_set_job_priority` | `slug`, `priority` (1-10) | Clamped weight actually stored |

### Presets

| Tool | Input | Returns |
|------|-------|---------|
| `cull_list_presets` | – | `[{name, source: builtin|custom, description}]` |
| `cull_get_preset` | `name` | Full preset envelope, secrets masked |
| `cull_clone_preset` | `source_name`, `new_name` | `{cloned: true}` |

### Pipeline control

The MCP server never spawns the supervisor — the dashboard owns that subprocess handle. These tools proxy to the local dashboard's REST API:

| Tool | Input | Returns |
|------|-------|---------|
| `cull_start_pipeline` | – | Instructions + current state, or a "dashboard unreachable" error |
| `cull_stop_pipeline` | – | Same |
| `cull_pipeline_status` | – | `{running, active_slugs, queue_totals, worker_health}` — falls back to filesystem when the dashboard is down |

### Scoring + scraper config

| Tool | Input | Returns |
|------|-------|---------|
| `cull_set_scoring` | `slug`, `min_ovr?`, `min_rel?`, `require_prompt?` | New effective scoring block |
| `cull_get_scoring` | `slug` | `{scoring, require_prompt}` |
| `cull_add_scraper_url` | `slug`, `source` (`gallery_dl` / `yt_dlp`), `url` | Updated URL list (deduped) |
| `cull_toggle_scraper` | `slug`, `name`, `enabled` | New per-job enabled map |

### Data inspection (read-only)

| Tool | Input | Returns |
|------|-------|---------|
| `cull_stats` | `slug?` | `{by_source, by_category, ovr_histogram}` |
| `cull_sample_gallery` | `slug`, `category?` (default `Keep`), `n?` (1-50, default 10) | List of `{path, category, ovr, rel, prompt, source}` + `ImageContent` thumbnail of the top result |
| `cull_get_vision_meta` | `image_path` | The `.vision.json` audit record (path-injection barred — only paths inside queue/sorted are allowed) |

### Export

| Tool | Input | Returns |
|------|-------|---------|
| `cull_export_kohya` | `slug`, `out_dir` | `{sample_count, categories, out_dir}` |
| `cull_export_hf` | `slug`, `repo` (namespace/name), `private?` (default true) | `{uploaded, repo, url}` — requires `HF_TOKEN` in the environment |

---

## Example prompts

Once wired into Claude Desktop, natural-language prompts do the work:

> "Create a job for cyberpunk portraits from the `photoreal_portrait` preset, activate it, and start the pipeline. Tell me when there are 500 keepers."

> "Show me the top 10 sorted Keep images from the `wildlife_macro` job. If any look mis-classified, print their `.vision.json` records."

> "Raise the OVR floor on the `product_photography` job to 80, then re-run the pipeline for an hour."

> "Export the `default` job's Keep bucket as a Kohya training set to `~/lora/cyberpunk`."

> "Push the `anime_style` job to a private HuggingFace dataset repo at `me/anime-style-v1`."

---

## Sample transcripts

**1. Create + run**

```
User:    Create a job for aerial drone photography, use the `aerial` preset,
         and activate it.

Claude:  I'll create the job now.
         [tool] cull_create_job({slug: "aerial_drone", subject:
                                  "aerial drone photography", preset: "aerial"})
         [tool] cull_activate_job({slug: "aerial_drone", exclusive: true})

         Created job `aerial_drone` from the aerial preset. Active jobs now:
         [aerial_drone]. Start the pipeline from the dashboard when ready.
```

**2. Tune + inspect**

```
User:    The current gallery has too much noise. Raise the OVR floor to 75 and
         show me the top 5 keepers.

Claude:  [tool] cull_set_scoring({slug: "aerial_drone", min_ovr: 75})
         [tool] cull_sample_gallery({slug: "aerial_drone", n: 5})

         OVR floor is now 75 (from 0). Here are the top 5 by OVR — the first
         image is attached inline. Prompts and scores are in the accompanying
         JSON block.
```

**3. Export**

```
User:    Export the current job as a Kohya dataset to /tmp/lora/aerial.

Claude:  [tool] cull_export_kohya({slug: "aerial_drone",
                                    out_dir: "/tmp/lora/aerial"})

         Exported 812 samples to /tmp/lora/aerial (categories: Keep). Ready
         for training.
```

---

## Security notes

* Every tool that dereferences a caller-supplied path (currently only `cull_get_vision_meta`) runs it through the same containment check the dashboard uses (`safe_inside()` mirrored as `_safe_media_path`). Paths outside `queue/` and `sorted/` are refused.
* `_mask_secrets` redacts every known secret env-var and every fleet `api_key` before shipping a preset or job envelope over MCP. Real credentials never leave the process.
* All logging routes to stderr via `pipeline_logging` — stdout is the MCP transport, so a stray `print()` would corrupt the JSON-RPC frame. If you extend the server, keep to `logger.info(...)` / `logger.debug(...)`, never `print(...)`.
* The server never auto-runs privileged actions: starting/stopping the supervisor still requires the local dashboard (which enforces its own auth in the API-auth PR), and the HF push requires an explicit token.

---

## Troubleshooting

* **`cull-mcp` prints "requires the 'mcp' extra"** — run `pip install -e ".[mcp]"`.
* **Claude Desktop shows no tools** — check the daemon logs (`~/Library/Logs/Claude/mcp-server-cull.log` on macOS). A common cause is a mistyped `PIPELINE_BASE_DIR` — the server can start with a nonexistent path, but tools that hit the index fail.
* **`cull_start_pipeline` returns "dashboard unreachable"** — start the dashboard first (`python pipeline_code/dashboard_enhanced.py` or `cull run`), or set `FLASK_PORT` if you moved it off 5000.
* **`cull_export_hf` errors with "no token"** — set `HF_TOKEN` in the MCP client's env block (the dashboard's Settings tab is another way, but MCP-only installs should use the env).
