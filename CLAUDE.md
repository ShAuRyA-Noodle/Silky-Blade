# Finance Project — Claude Code Configuration

## Project Overview

World-class quant trading platform. Real data only — zero synthetic/mock/simulated data in any path.

**Stack:** Python (FastAPI, LightGBM, Polars, Alpaca), Next.js 16, PostgreSQL, MLflow, GitHub Actions.

**Current state (2026-05-15):** Research prototype with solid quant infrastructure. Paper trading wired to Alpaca ($100k account) but cron only reads status — does not submit orders yet. Priority: wire cron to actually trade + surface ML recommendations in web UI.

---

## Mandatory Workflow (Superpowers)

Superpowers is active. Follow it every time:

1. **Before any feature:** Invoke `superpowers:brainstorming` — no coding before spec alignment.
2. **Before writing code:** Invoke `superpowers:test-driven-development` — write tests first, always.
3. **Before multi-step work:** Invoke `superpowers:writing-plans` — get plan approved first.
4. **Executing plans:** Use `superpowers:subagent-driven-development` for parallel independent tasks.
5. **Before claiming done:** Invoke `superpowers:verification-before-completion` — run actual commands, show real output.
6. **After major steps:** Invoke `superpowers:requesting-code-review`.
7. **When debugging:** Invoke `superpowers:systematic-debugging` — no guessing.
8. **When receiving feedback:** Invoke `superpowers:receiving-code-review` — verify before agreeing.

---

## Memory System (Claude-Mem)

Claude-mem tracks all work across sessions.

- **Worker port:** 37701 (localhost only)
- **Database:** `~/.claude-mem/claude-mem.db`
- **Viewer UI:** http://localhost:37701 (if worker running)
- **MCP tools available:** `mcp-search` (search, timeline, get_observations)
- **Privacy:** Wrap sensitive content in `<private>content</private>` — stripped before storage. Never include API keys, credentials, or personal data in tool output.

Search past work:
```
/mem-search <query>         # search observations
/make-plan <feature>        # plan with doc discovery
/do                         # execute plan via subagents
```

---

## Design Skills (Taste-Skill)

Installed at `.agents/skills/` (symlinked to Claude Code).

**Available design skills:**
- `design-taste-frontend` — default, premium UI with metric-based rules
- `gpt-taste` — stricter variant, GSAP animations, editorial typography
- `high-end-visual-design` — luxury agency aesthetic
- `minimalist-ui` — clean, warm monochrome, editorial
- `industrial-brutalist-ui` — raw mechanical, Swiss typography
- `imagegen-frontend-web` — generate design reference images before coding
- `imagegen-frontend-mobile` — mobile-specific image direction
- `image-to-code` — image-first workflow: generate image → analyze → implement
- `redesign-existing-projects` — upgrade existing pages without breaking functionality
- `stitch-design-taste` — Google Stitch design system skill
- `brandkit` — brand identity and logo systems

**Rule:** For ANY frontend work, invoke the relevant design skill first. Never write UI without it.

---

## Architecture Decisions (Locked)

| Decision | Choice | Reason |
|---|---|---|
| Universe | SP500 (500 names) | Liquidity, data coverage |
| Data source | Alpaca (OHLCV), Finnhub (fundamentals), Groq (sentiment) | Real-only |
| ML model | LightGBM + isotonic calibration | Speed, interpretability, SHAP |
| Backtest engine | Polars walk-forward, purged K-fold | No look-ahead |
| Broker | Alpaca paper → Alpaca live | API quality |
| Frontend | Next.js 16, Vercel | Static prerender |

---

## Critical Rules — Never Violate

### Data integrity
- Zero synthetic/mock data in any non-test path.
- `ValueSignal` (P/E from Finnhub) and `SentimentSignal` (news CSV) are **live-only** — today's snapshot. Never use in historical backtest. Add runtime guard if wiring to backtest runner.
- Backtest uses `MLPredictionsSignal` (replays OOF predictions) or `MLBundleSignal` (trained model + features). Not sentiment, not value.

### ML / signals
- Sentiment is a **parallel signal**, not an ML feature. `trainer.py` trains on 26 technical features only. CompositeSignal is a late-stage blend, not feature injection. If adding sentiment to ML: add `sentiment_*` column to `FEATURE_COLUMNS` and retrain.
- Report macro_AUC and calibration ECE only on held-out block (last 6-12 months), not in-sample OOF.
- Backtest Sharpe always includes survivorship bias disclosure. Never headline the 1.703 number alone.

### Live trading safety
- Triple gate: `ALPACA_PAPER=true` + `TRADING_ENABLED=true` + `--confirm` flag. All three required.
- Never flip `ALPACA_PAPER=false` without explicit user confirmation.
- Orders go through `execution/live_session.py` → `execution/risk_gate.py`. The `risk/manager.py` is dormant (DB-backed, not wired). Don't confuse them.
- Add order persistence to `Trade`/`Position` DB models when wiring cron to trade.

### Code quality
- `mypy --strict` must pass on all new files.
- `ruff` must pass (no ignores without comment).
- Test coverage: every new signal/feature needs unit tests.
- No fake data in tests that touch real data paths — use real fixtures or mark explicitly.

---

## Testing

```bash
# Run unit tests
cd apps/api && python -m pytest tests/unit/ -v

# Run integration tests  
cd apps/api && python -m pytest tests/integration/ -v -m integration

# Type check
cd apps/api && mypy --strict src/quant/

# Lint
cd apps/api && ruff check src/
```

---

## Environment Variables (Required)

```
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_PAPER=true          # NEVER false without deliberate intent
TRADING_ENABLED=false      # flip true only for intentional paper trade session
GROQ_API_KEY=...
FINNHUB_API_KEY=...
MARKETAUX_API_KEY=...
NEWS_API_KEY=...
```

---

## Key File Paths

| Path | What |
|---|---|
| `apps/api/src/quant/ml/trainer.py` | LightGBM training, walk-forward, calibration |
| `apps/api/src/quant/ml/predict.py` | ModelBundle, load_bundle, BUY/HOLD/SELL recommend |
| `apps/api/src/quant/backtest/engine.py` | Walk-forward backtest engine |
| `apps/api/src/quant/backtest/signals.py` | MomentumSignal, ValueSignal, SentimentSignal, CompositeSignal |
| `apps/api/src/quant/backtest/runner.py` | SignalSpec, build_signal registry |
| `apps/api/src/quant/features/sentiment.py` | News fetch → Groq score → CSV |
| `apps/api/src/quant/features/technical.py` | 26 OHLCV-based ML features |
| `apps/api/src/quant/execution/live_session.py` | Paper trade orchestrator |
| `apps/api/src/quant/execution/risk_gate.py` | Live risk controls |
| `apps/api/src/quant/cli.py` | All CLI commands (quant backtest/paper/ml/features) |
| `apps/web/app/paper/page.tsx` | Paper trading dashboard (static snapshot) |
| `apps/web/app/results/page.tsx` | Backtest results page |
| `.github/workflows/daily-paper.yml` | Cron: currently READ-ONLY (no trading) |
| `AUDIT.md` | Brutal gap analysis — read before any major feature |

---

## Open Gaps (From Audit 2026-05-15)

Priority order:
1. Wire cron to compute + submit paper orders daily (`ml_bundle` signal, SP500 universe)
2. Add order persistence to DB (Trade/Position models)
3. Surface BUY/HOLD/SELL recommendations on web (`/recommendations` page)
4. Add runtime guard blocking `SentimentSignal`/`ValueSignal` in historical backtest
5. Connect `risk/manager.py` to live path (or delete it)
6. Add monitoring/alerting (Sentry or Slack hook on order rejection)
7. Rate limit FastAPI endpoints
8. Post-tuning holdout evaluation for ML (last 6 months OOF block)

---

## Caveman Mode

Active. Terse responses, no fluff. Switch: `/caveman lite|full|ultra`. Off: "stop caveman".
Commits/PRs/security warnings: written normally regardless of mode.
