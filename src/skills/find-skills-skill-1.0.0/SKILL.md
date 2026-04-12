---
name: find-skills
description: "Search OpenClaw skills via ClawHub; install by downloading and extracting into the project src/skills/ folder (per-skill subdirectory with SKILL.md). Use when: user wants to find skills, search by keyword, or install a skill into this repo layout."
homepage: https://clawhub.com
metadata: { "openclaw": { "emoji": "🔍", "requires": { "bins": [] } } }
---

# Find Skills Skill

Search skills on **ClawHub**; install them by placing an extracted package under the repo **`skills/`** directory, following the per-skill folder layout below.

## When to Use

✅ **USE this skill when:**

- "Find skills for [task]"
- "Search for OpenClaw skills"
- "What skills are available?"
- "Discover new skills"
- "Find skills by category"

## When NOT to Use

❌ **DON'T use this skill when:**

- Listing or enabling skills already present in this project → use project skills tooling / config
- Creating new skills from scratch → use skill-creator skill

## Finding skills (ClawHub)

Use the ClawHub CLI against the registry (default `https://clawhub.ai` unless overridden):

```bash
# Search by keyword
npx clawhub search "keyword"

# Explore / sort (see clawhub --help for current flags)
npx clawhub explore
```

Optional: open **https://clawhub.com** in a browser to browse or copy slugs; search remains CLI-first.

## Installing into this project (`skills/`)

This repo expects **one skill per subdirectory** under **`src/skills/`**, each folder containing a root **`SKILL.md`** (YAML front matter + body). Same layout as existing packs (e.g. `src/skills/find-skills-skill-1.0.0/`).

> **⚠️ Do NOT use `npx clawhub install`** — it extracts into its own layout with `.clawhub/lock.json` metadata, which is incompatible with this project's skills directory. Instead, use `clawhub inspect` to download files directly.

### Step-by-step flow

**1. Search** for the skill on ClawHub (see [Finding skills](#finding-skills-clawhub) above).

**2. List files** in the skill pack to see what it contains:

```bash
npx clawhub inspect <slug> --files
```

**3. Create the skill folder** under `src/skills/`:

```bash
mkdir -p src/skills/<slug>
```

**4. Download `SKILL.md`** via `inspect --file` and redirect to the target path:

```bash
npx clawhub inspect <slug> --file SKILL.md > src/skills/<slug>/SKILL.md
```

**5. Download additional files** (if the skill bundles `references/`, `scripts/`, `rules/`, etc.):

```bash
# Example: download a references file
mkdir -p src/skills/<slug>/references
npx clawhub inspect <slug> --file references/OVERVIEW.md > src/skills/<slug>/references/OVERVIEW.md
```

Repeat for each file listed in step 2.

**6. Verify** the folder structure looks correct:

```
src/skills/<slug>/
├── SKILL.md          # required
├── references/       # optional
│   └── ...
├── scripts/          # optional
│   └── ...
└── ...
```

**7. Reload** — The `SkillsManager` will auto-detect new skills via hot-reload. If hot-reload is not active, restart the app to trigger a rescan.

## Search Strategies

### By Functionality
```bash
# Web search skills
npx clawhub search "web search"

# Weather skills
npx clawhub search "weather"

# Document skills
npx clawhub search "document"
```

### By Provider
```bash
# Tavily skills
npx clawhub search "tavily"

# GitHub skills
npx clawhub search "github"

# Calendar skills
npx clawhub search "calendar"
```

### By Popularity
```bash
# Most installed skills
npx clawhub search --sort installs

# Most starred skills
npx clawhub search --sort stars
```

## Installation tips

1. **Workdir** — Run all commands with cwd = repo root so paths resolve to `src/skills/` correctly.
2. **Preview first** — Use `npx clawhub inspect <slug>` (no flags) to view skill metadata before downloading.
3. **Requirements** — Check `metadata` / bins in `SKILL.md` before relying on a skill.
4. **Read `SKILL.md`** — Usage and constraints are in the pack.
5. **Updates** — Re-run `inspect --file` for each file to overwrite with the newer version, or delete the folder and repeat the download steps.

## Common Skill Categories

### Core Skills
- `weather` - Weather forecasts
- `skill-creator` - Create new skills
- `healthcheck` - Security audits

### Integration Skills
- `github` - GitHub operations
- `feishu` - Feishu integration
- `notion` - Notion API

### Search Skills
- `tavily-search` - Web search via Tavily
- `web-search-plus` - Enhanced web search

### Agent Skills
- `proactive-agent` - Proactive automation
- `coding-agent` - Code generation

## Troubleshooting

### Rate Limits
If you hit rate limits with clawhub:
1. Wait 1 hour before retrying
2. Use alternative sources (websites)
3. Search manually on GitHub

### Installation issues
1. Confirm extraction path: `src/skills/<slug>/SKILL.md` must exist
2. Verify network / proxy (`HTTPS_PROXY` if needed) for CLI download
3. Check skill requirements and OpenClaw compatibility in `SKILL.md`

## Best Practices

1. **Search before creating** - Don't reinvent the wheel
2. **Read documentation** - Understand skill capabilities
3. **Start simple** - Install one skill at a time
4. **Test thoroughly** - Verify skill works as expected
5. **Provide feedback** - Help improve skills

## Related

- ClawHub CLI: `npx clawhub --help`
- Skill authoring: skill-creator / local `SKILL.md` conventions