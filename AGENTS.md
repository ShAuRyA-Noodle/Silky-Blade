# Finance Project — Agent Configuration

## Active Plugins

| Plugin | Version | Purpose |
|---|---|---|
| superpowers@claude-plugins-official | 5.1.0 | Development methodology (TDD, planning, subagents, debugging) |
| claude-mem@thedotmack | 13.2.0 | Persistent cross-session memory, MCP search tools |
| caveman@caveman | ef6050c | Token-efficient compressed communication mode |
| taste-skill (Leonxlnx) | latest | Premium frontend design skills (12 skills) |

## Superpowers Skill Map

Invoke these before acting. Non-negotiable.

| Trigger | Skill |
|---|---|
| Starting any feature | `superpowers:brainstorming` |
| Writing code | `superpowers:test-driven-development` |
| Multi-step task | `superpowers:writing-plans` |
| Parallel independent tasks | `superpowers:dispatching-parallel-agents` |
| Executing approved plan | `superpowers:subagent-driven-development` |
| Claiming work done | `superpowers:verification-before-completion` |
| After major step | `superpowers:requesting-code-review` |
| Receiving review feedback | `superpowers:receiving-code-review` |
| Any bug/failure | `superpowers:systematic-debugging` |
| Feature work needing isolation | `superpowers:using-git-worktrees` |

## Claude-Mem MCP Tools

Three-layer search workflow:

```
mcp-search:search           # Layer 1 — compact index, fast (50-100 tokens)
mcp-search:timeline         # Layer 2 — chronological context around results
mcp-search:get_observations # Layer 3 — full detail by observation ID
```

Worker: `127.0.0.1:37701` | DB: `~/.claude-mem/claude-mem.db`
Privacy: `<private>content</private>` stripped before storage.

## Design Skills (Taste-Skill)

12 skills at `.agents/skills/`, symlinked to Claude Code. Invoke before ANY frontend work.

| Use case | Skill |
|---|---|
| Default premium UI | `design-taste-frontend` |
| GSAP + editorial | `gpt-taste` |
| Luxury agency | `high-end-visual-design` |
| Clean/editorial | `minimalist-ui` |
| Raw mechanical | `industrial-brutalist-ui` |
| Generate design images → code | `image-to-code` |
| Web hero/landing image | `imagegen-frontend-web` |
| Mobile screen concepts | `imagegen-frontend-mobile` |
| Upgrade existing pages | `redesign-existing-projects` |
| Brand system | `brandkit` |

## Domain Knowledge

- Python: FastAPI, LightGBM, Polars, SQLAlchemy, Pydantic v2, Typer, pytest
- Frontend: Next.js 16, TypeScript, Tailwind CSS
- ML: walk-forward backtest, purged K-fold, isotonic calibration, SHAP
- Finance: momentum signals, value signals (1/PE), sentiment NLP, Alpaca broker API
- Infra: GitHub Actions, MLflow, SQLite/PostgreSQL

## Project Root

`/Users/shauryapunj/Desktop/ShAuRyA_Side_Projects/Finance_Project`

Sub-projects:
- `apps/api/` — Python backend (FastAPI + quant engine)
- `apps/web/` — Next.js frontend
- `AUDIT.md` — Current gap analysis (read before new features)
- `CLAUDE.md` — Full project config for Claude Code
