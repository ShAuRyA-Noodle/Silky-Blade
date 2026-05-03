/**
 * Server-only: load the real backtest artifact bundle from disk at build time.
 *
 * House rule (TRUST.md §2.6): no manifest → no publish. If the bundle is
 * missing or shaped wrong, this module THROWS, which makes `next build` fail
 * loudly. There is no fallback to placeholder data.
 *
 * Source of truth on disk:
 *   apps/api/examples/backtest/artifacts/sp500_momentum_126/{report,manifest}.json
 *   apps/api/examples/backtest/artifacts/sp500_momentum_126/equity_curve.csv
 *
 * Regenerate with:
 *   cd apps/api && .venv/bin/python examples/backtest/prepare_sp500_5yr.py \
 *     && .venv/bin/python -m quant.cli backtest run examples/backtest/sp500_momentum.yaml
 */

import { existsSync, readFileSync } from "node:fs"
import { join } from "node:path"

import type {
  BacktestManifest,
  BacktestReport,
  BacktestSweepReport,
  EquityPoint,
  OracleArtifacts,
  PaperStatusSnapshot,
  PointInTimeComparison,
} from "./types"

// Headline run = the freshly-backfilled 2019-2026 momentum on Alpaca data.
// PIT comparison + LightGBM run still point at the 2014-2018 Kaggle bundles
// because those are the apples-to-apples comparisons that exist on disk.
const RUN_NAME = "sp500_momentum_126_2026"
const LEGACY_RAW_NAME = "sp500_momentum_126"
const PIT_RUN_NAME = "sp500_momentum_126_pit"
const ML_RUN_NAME = "sp500_ml_predictions_v1"
const SWEEP_NAME = "sp500_momentum_sweep"

const REGEN_HINT =
  "Regenerate with: cd apps/api && .venv/bin/python examples/backtest/prepare_sp500_5yr.py " +
  "&& .venv/bin/python -m quant.cli backtest run examples/backtest/sp500_momentum.yaml"

/**
 * Two artifact roots, tried in order:
 *   1. `<cwd>/.oracle-artifacts/` — populated by the npm `prebuild` hook
 *      and by the CI "Sync artifacts" step. Always inside the Docker
 *      build context, so the `web` image can render `/results`.
 *   2. `<cwd>/../api/examples/backtest/artifacts/` — the canonical source
 *      in a checked-out monorepo. Used in local dev when prebuild hasn't
 *      run yet, and on CI before the sync step.
 */
const ARTIFACT_ROOTS: readonly string[] = [
  join(process.cwd(), ".oracle-artifacts"),
  join(process.cwd(), "..", "api", "examples", "backtest", "artifacts"),
]

function resolveArtifact(runName: string, file: string): string {
  for (const root of ARTIFACT_ROOTS) {
    const candidate = join(root, runName, file)
    if (existsSync(candidate)) return candidate
  }
  // Fall through to the canonical path so the caller's error message
  // names a path the user can act on.
  return join(ARTIFACT_ROOTS[ARTIFACT_ROOTS.length - 1] ?? "", runName, file)
}

function artifactPath(file: string): string {
  return resolveArtifact(RUN_NAME, file)
}

function sweepArtifactPath(file: string): string {
  return resolveArtifact(SWEEP_NAME, file)
}

function readJson<T>(file: string): T {
  const path = artifactPath(file)
  let raw: string
  try {
    raw = readFileSync(path, "utf8")
  } catch (err) {
    throw new Error(
      `[oracle] missing artifact ${file} at ${path}. ${REGEN_HINT}\n${(err as Error).message}`,
    )
  }
  try {
    return JSON.parse(raw) as T
  } catch (err) {
    throw new Error(
      `[oracle] artifact ${file} at ${path} is not valid JSON. ${REGEN_HINT}\n${(err as Error).message}`,
    )
  }
}

function parseEquityCurve(csv: string): readonly EquityPoint[] {
  const lines = csv.trim().split(/\r?\n/)
  if (lines.length < 2) {
    throw new Error(
      `[oracle] equity_curve.csv has no data rows. ${REGEN_HINT}`,
    )
  }
  const header = lines[0]?.split(",") ?? []
  if (header[0] !== "date" || header[1] !== "equity") {
    throw new Error(
      `[oracle] equity_curve.csv header mismatch: expected "date,equity", got "${lines[0] ?? ""}". ${REGEN_HINT}`,
    )
  }
  const points: EquityPoint[] = []
  for (let i = 1; i < lines.length; i += 1) {
    const line = lines[i]
    if (!line) continue
    const [date, equityStr] = line.split(",")
    if (!date || !equityStr) {
      throw new Error(
        `[oracle] equity_curve.csv row ${i} malformed: "${line}". ${REGEN_HINT}`,
      )
    }
    const equity = Number(equityStr)
    if (!Number.isFinite(equity)) {
      throw new Error(
        `[oracle] equity_curve.csv row ${i} has non-numeric equity: "${equityStr}". ${REGEN_HINT}`,
      )
    }
    points.push({ date, equity })
  }
  return points
}

function validateReport(report: BacktestReport): void {
  const m = report.metrics
  if (
    !Number.isFinite(m.sharpe) ||
    !Number.isFinite(m.annualized_return) ||
    !Number.isFinite(m.max_drawdown) ||
    !Number.isFinite(m.deflated_sharpe_p)
  ) {
    throw new Error(
      `[oracle] report.json has non-finite metrics. ${REGEN_HINT}`,
    )
  }
}

function validateManifest(manifest: BacktestManifest): void {
  if (!manifest.code_sha || manifest.code_sha.length < 7) {
    throw new Error(`[oracle] manifest.json missing code_sha. ${REGEN_HINT}`)
  }
  if (!manifest.data_fingerprint || manifest.data_fingerprint.length < 32) {
    throw new Error(
      `[oracle] manifest.json missing data_fingerprint. ${REGEN_HINT}`,
    )
  }
  if (!manifest.config_hash || manifest.config_hash.length < 32) {
    throw new Error(
      `[oracle] manifest.json missing config_hash. ${REGEN_HINT}`,
    )
  }
}

function tryLoadMetricsBundle(
  runName: string,
): Pick<
  BacktestReport["metrics"],
  "sharpe" | "annualized_return" | "max_drawdown" | "deflated_sharpe_p"
> | null {
  const path = resolveArtifact(runName, "report.json")
  let raw: string
  try {
    raw = readFileSync(path, "utf8")
  } catch {
    return null
  }
  let parsed: BacktestReport
  try {
    parsed = JSON.parse(raw) as BacktestReport
  } catch {
    return null
  }
  const m = parsed.metrics
  if (
    !Number.isFinite(m.sharpe) ||
    !Number.isFinite(m.annualized_return) ||
    !Number.isFinite(m.max_drawdown) ||
    !Number.isFinite(m.deflated_sharpe_p)
  ) {
    return null
  }
  return {
    sharpe: m.sharpe,
    annualized_return: m.annualized_return,
    max_drawdown: m.max_drawdown,
    deflated_sharpe_p: m.deflated_sharpe_p,
  }
}

function loadOptionalPitComparison(
  rawReport: BacktestReport,
): PointInTimeComparison | null {
  // For the survivorship-bias panel we want apples-to-apples: the Kaggle
  // 2014-2018 raw run vs. the same window with PIT enforcement. The
  // headline Sharpe (now 2019-2026) is shown elsewhere on the page.
  const legacyRaw = tryLoadMetricsBundle(LEGACY_RAW_NAME)
  const pit = tryLoadMetricsBundle(PIT_RUN_NAME)
  if (legacyRaw === null || pit === null) {
    // No legacy bundle — fall back to comparing the headline 2019-2026 run
    // against PIT (still informative even if windows differ).
    if (pit === null) return null
    const r = rawReport.metrics
    const ml = tryLoadMetricsBundle(ML_RUN_NAME)
    const out: PointInTimeComparison = {
      raw: {
        sharpe: r.sharpe,
        annualized_return: r.annualized_return,
        max_drawdown: r.max_drawdown,
        deflated_sharpe_p: r.deflated_sharpe_p,
      },
      pit,
    }
    return ml ? { ...out, ml } : out
  }
  const ml = tryLoadMetricsBundle(ML_RUN_NAME)
  const out: PointInTimeComparison = { raw: legacyRaw, pit }
  return ml ? { ...out, ml } : out
}

function loadOptionalSweep(): BacktestSweepReport | null {
  // Sweep is optional — its absence does not break the build. PBO is a
  // diagnostic, not a contract. If the user hasn't run the sweep yet the
  // page renders without the PBO panel rather than fabricating one.
  const path = sweepArtifactPath("sweep_report.json")
  let raw: string
  try {
    raw = readFileSync(path, "utf8")
  } catch {
    return null
  }
  let parsed: BacktestSweepReport
  try {
    parsed = JSON.parse(raw) as BacktestSweepReport
  } catch (err) {
    throw new Error(
      `[oracle] sweep_report.json at ${path} is not valid JSON. ${(err as Error).message}`,
    )
  }
  if (!Number.isFinite(parsed.pbo) || parsed.pbo < 0 || parsed.pbo > 1) {
    throw new Error(
      `[oracle] sweep_report.json at ${path} has invalid pbo: ${String(parsed.pbo)}`,
    )
  }
  return parsed
}

/**
 * Load and validate the full artifact bundle. Called once at build time
 * from the Oracle page; the parsed result is then statically embedded in
 * the bundled HTML/JS (no runtime fetch, no client-side parsing).
 */
export function loadOracleArtifacts(): OracleArtifacts {
  const report = readJson<BacktestReport>("report.json")
  validateReport(report)

  const manifest = readJson<BacktestManifest>("manifest.json")
  validateManifest(manifest)

  let equityRaw: string
  const csvPath = artifactPath("equity_curve.csv")
  try {
    equityRaw = readFileSync(csvPath, "utf8")
  } catch (err) {
    throw new Error(
      `[oracle] missing artifact equity_curve.csv at ${csvPath}. ${REGEN_HINT}\n${(err as Error).message}`,
    )
  }
  const equityCurve = parseEquityCurve(equityRaw)
  const sweep = loadOptionalSweep()
  const pitComparison = loadOptionalPitComparison(report)
  const paperStatus = loadOptionalPaperStatus()

  return { report, manifest, equityCurve, sweep, pitComparison, paperStatus }
}

function loadOptionalPaperStatus(): PaperStatusSnapshot | null {
  // Optional. Written by `quant paper status --json-out`. Sits inside the
  // synced .oracle-artifacts directory so the Docker build picks it up.
  for (const root of ARTIFACT_ROOTS) {
    const candidate = join(root, "paper-status.json")
    if (!existsSync(candidate)) continue
    try {
      const parsed = JSON.parse(readFileSync(candidate, "utf8")) as PaperStatusSnapshot
      if (
        typeof parsed.account?.equity === "string" &&
        Array.isArray(parsed.positions)
      ) {
        return parsed
      }
    } catch {
      // fall through
    }
  }
  return null
}
