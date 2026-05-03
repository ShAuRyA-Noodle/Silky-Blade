/**
 * Server-only: load the optional paper-trading snapshot.
 *
 * The snapshot is written by `quant paper status --json-out` and synced
 * into apps/web/.oracle-artifacts/ by the prebuild script. Absence is a
 * valid state — the page renders a "not connected" panel rather than
 * fabricating any account number.
 */

import { existsSync, readFileSync } from "node:fs"
import { join } from "node:path"

import type { PaperStatusSnapshot } from "./types"

export interface PaperEquityPoint {
  readonly timestamp: string
  readonly equity: number
  readonly n_positions: number
}

const ARTIFACT_ROOTS: readonly string[] = [
  join(process.cwd(), ".oracle-artifacts"),
  join(process.cwd(), "..", "api", "examples", "backtest", "artifacts"),
]

export function loadOraclePaperSnapshot(): PaperStatusSnapshot | null {
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
      // fall through; next root
    }
  }
  return null
}

/**
 * Optional time-series CSV written by `quant paper status --history-csv`.
 * Returns the parsed equity track (timestamp, equity, n_positions) so the
 * /paper page can draw a line chart of paper-account growth over time.
 *
 * Schema: timestamp,equity,cash,buying_power,status,n_positions
 */
export function loadOraclePaperHistory(): readonly PaperEquityPoint[] {
  for (const root of ARTIFACT_ROOTS) {
    const candidate = join(root, "paper-history.csv")
    if (!existsSync(candidate)) continue
    try {
      const raw = readFileSync(candidate, "utf8").trim()
      if (!raw) return []
      const lines = raw.split(/\r?\n/)
      const header = lines[0]?.split(",") ?? []
      const tsIdx = header.indexOf("timestamp")
      const eqIdx = header.indexOf("equity")
      const npIdx = header.indexOf("n_positions")
      if (tsIdx < 0 || eqIdx < 0) return []
      const out: PaperEquityPoint[] = []
      for (let i = 1; i < lines.length; i += 1) {
        const cells = lines[i]?.split(",") ?? []
        const ts = cells[tsIdx]
        const eqStr = cells[eqIdx]
        if (!ts || !eqStr) continue
        const equity = Number(eqStr)
        if (!Number.isFinite(equity)) continue
        const np = npIdx >= 0 ? Number(cells[npIdx] ?? "0") : 0
        out.push({
          timestamp: ts,
          equity,
          n_positions: Number.isFinite(np) ? np : 0,
        })
      }
      return out
    } catch {
      // fall through
    }
  }
  return []
}
