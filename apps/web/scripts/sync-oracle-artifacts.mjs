#!/usr/bin/env node
/**
 * Copy the committed Oracle backtest artifact bundles into
 * `apps/web/.oracle-artifacts/` so the Docker build context (which is
 * `apps/web/` only) can find them at static-prerender time.
 *
 * Idempotent — safe to run on every `npm run build`. Skips silently if
 * a source bundle is missing (e.g. a fresh clone before any backtest
 * has been run). The web build will fall back to its in-tree path
 * `../api/examples/backtest/artifacts/<run>/` for local dev.
 */

import { cpSync, existsSync, mkdirSync } from "node:fs"
import { dirname, join } from "node:path"
import { fileURLToPath } from "node:url"

const HERE = dirname(fileURLToPath(import.meta.url))
const WEB_ROOT = join(HERE, "..")
const API_ARTIFACTS = join(
  WEB_ROOT,
  "..",
  "api",
  "examples",
  "backtest",
  "artifacts",
)
const OUT_ROOT = join(WEB_ROOT, ".oracle-artifacts")

const RUNS = [
  "sp500_momentum_126",
  "sp500_momentum_126_pit",
  "sp500_momentum_126_2026",
  "sp500_ml_predictions_v1",
  "sp500_momentum_sweep",
]

mkdirSync(OUT_ROOT, { recursive: true })

let copied = 0
for (const run of RUNS) {
  const src = join(API_ARTIFACTS, run)
  if (!existsSync(src)) {
    console.warn(`[sync-oracle-artifacts] missing source bundle: ${src} (skipping)`)
    continue
  }
  const dst = join(OUT_ROOT, run)
  cpSync(src, dst, { recursive: true })
  copied += 1
  console.log(`[sync-oracle-artifacts] ${run} -> ${dst}`)
}

if (copied === 0) {
  console.warn(
    "[sync-oracle-artifacts] no bundles synced — /results will use the in-tree fallback path",
  )
}
