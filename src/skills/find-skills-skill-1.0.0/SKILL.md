---
name: find-skills
description: "Search ClawHub and install skills into this repo’s src/skills/<slug>/ via `npx clawhub --dir src/skills install <slug>`. Bare `npx clawhub install` and `--dir` with a drive path are blocked by run_command. Use repo root as cwd or `--workdir`; optional file-by-file fallback with `clawhub inspect --file`."
homepage: https://clawhub.com
metadata: { "openclaw": { "emoji": "🔍", "requires": { "bins": [] } } }
---

# Find Skills Skill

Search **ClawHub** and install packs under **`src/skills/<slug>/`** (each with a root **`SKILL.md`**). Do not hardcode absolute paths in docs; use your clone’s repo root as cwd or pass **`--workdir "<path-to-repo-root>"`**.

## Why `--dir src/skills` is required

ClawHub’s global **`--dir`** defaults to **`skills`**, so installs land in **`<workdir>/skills/<slug>`**. This project’s **`SkillsManager`** only loads **`src/skills/<slug>/SKILL.md`**. Put **`--dir src/skills`** before subcommands **`install`**, **`update`**, **`list`**, **`uninstall`**.

```bash
# From repo root (recommended)
npx clawhub --dir src/skills install <slug>

# When cwd is not repo root
npx clawhub --workdir "<path-to-repo-root>" --dir src/skills install <slug>
```

**`run_command` behavior:** for any command containing **`clawhub`**, it runs with **cwd = this Agent repo root** and sets **`CLAWHUB_WORKDIR`** to that root unless **`--workdir`** is already in the command (avoids installs under `WorkDatabase/` or a global OpenClaw workspace). It **blocks** bare **`npx clawhub install`** and **`--dir`** values that look like Windows drive paths (use **`src/skills`**, not the repo root as `--dir`).

## When to Use

✅ **USE this skill when:**

- "Find skills for [task]"
- "Search for OpenClaw skills"
- "What skills are available?"
- "Discover new skills"
- "Find skills by category"

## When NOT to Use

❌ **DON'T use this skill when:**

- Listing or enabling skills already in this project → use project tooling / config
- Creating new skills from scratch → use skill-creator

## Finding skills (ClawHub)

```bash
npx clawhub search "keyword"
npx clawhub explore
```

Optional: browse **https://clawhub.com** for slugs; CLI search is still primary.

## Installing into `src/skills/`

**Preferred:** `npx clawhub --dir src/skills install <slug>` (add `--force` to overwrite). You may set **`CLAWHUB_WORKDIR`** to repo root; see ClawHub **`--help`** for precedence with **`--workdir`**.

### Fallback: `inspect` (file by file)

1. Search (above).
2. `npx clawhub inspect <slug> --files`
3. `mkdir -p src/skills/<slug>`
4. `npx clawhub inspect <slug> --file SKILL.md > src/skills/<slug>/SKILL.md`
5. Repeat **`--file`** for other paths (`references/`, `scripts/`, …).
6. Ensure **`src/skills/<slug>/SKILL.md`** exists.
7. Reload: **`SkillsManager`** hot-reload or restart the app.

## Search examples

```bash
npx clawhub search "web search"
npx clawhub search "tavily"
npx clawhub search --sort installs
npx clawhub search --sort stars
```

## Tips

1. Always pair **`--dir src/skills`** with **`install` / `update` / `list` / `uninstall`** for this repo.
2. Prefer cwd = repo root, or pass **`--workdir`**; never bake machine-specific paths into shared docs.
3. Preview packs: `npx clawhub inspect <slug>`.
4. Read each pack’s **`SKILL.md`** for **`metadata`** and required bins.
5. Updates: `npx clawhub --dir src/skills update <slug>` or re-run **`inspect --file`**.

## Common categories (examples)

- Core: `weather`, `skill-creator`, `healthcheck`
- Integrations: `github`, `feishu`, `notion`
- Search: `tavily-search`, `web-search-plus`
- Agents: `proactive-agent`, `coding-agent`

## Troubleshooting

- **Rate limits:** wait, use the website, or search GitHub.
- **Install path:** confirm **`src/skills/<slug>/SKILL.md`** exists; check **`HTTPS_PROXY`** if needed.

## Related

- `npx clawhub --help`
- Local conventions: root **`SKILL.md`** per skill folder
