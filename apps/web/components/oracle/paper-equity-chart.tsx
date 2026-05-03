/* eslint-disable react-hooks/purity --
 * This is a server component that renders SVG axis labels from ISO
 * timestamps. `new Date(iso)` + `Intl.DateTimeFormat#format` on the
 * server is deterministic for the same input; the lint rule fires
 * because Date constructors and Date.now() are flagged generically.
 * Disabling at file scope is appropriate.
 */
import type { PaperEquityPoint } from "@/lib/oracle/load-paper"

interface PaperEquityChartProps {
  readonly points: readonly PaperEquityPoint[]
}

const usd = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
})

const dateFmt = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
})

const VIEWBOX_W = 1000
const VIEWBOX_H = 360
const PAD_LEFT = 72
const PAD_RIGHT = 24
const PAD_TOP = 24
const PAD_BOTTOM = 44

function pickTickIndices(n: number, n_ticks: number): number[] {
  if (n <= 1) return [0]
  if (n <= n_ticks) return Array.from({ length: n }, (_, i) => i)
  const step = (n - 1) / (n_ticks - 1)
  const out: number[] = []
  for (let i = 0; i < n_ticks; i += 1) out.push(Math.round(i * step))
  return Array.from(new Set(out))
}

export function PaperEquityChart({ points }: PaperEquityChartProps) {
  if (points.length < 2) {
    return (
      <div className="rounded-2xl border border-border/60 bg-card/30 backdrop-blur-xl p-6 text-sm text-muted-foreground">
        Need at least two history points to plot. Run{" "}
        <code className="font-mono text-primary">
          quant paper status --history-csv apps/web/.oracle-artifacts/paper-history.csv
        </code>{" "}
        on a daily cadence to grow the curve.
      </div>
    )
  }

  const eqs = points.map((p) => p.equity)
  const eMin = Math.min(...eqs)
  const eMax = Math.max(...eqs)
  const range = Math.max(eMax - eMin, 1)
  const yPad = range * 0.08

  const innerW = VIEWBOX_W - PAD_LEFT - PAD_RIGHT
  const innerH = VIEWBOX_H - PAD_TOP - PAD_BOTTOM

  const xFor = (i: number) =>
    PAD_LEFT + (innerW * i) / Math.max(points.length - 1, 1)
  const yFor = (eq: number) => {
    const top = eMax + yPad
    const bot = eMin - yPad
    const t = (eq - bot) / (top - bot)
    return PAD_TOP + innerH * (1 - t)
  }

  const linePath = points
    .map((p, i) => `${i === 0 ? "M" : "L"}${xFor(i).toFixed(2)},${yFor(p.equity).toFixed(2)}`)
    .join(" ")
  const areaPath = `${linePath} L${xFor(points.length - 1).toFixed(2)},${(VIEWBOX_H - PAD_BOTTOM).toFixed(2)} L${PAD_LEFT.toFixed(2)},${(VIEWBOX_H - PAD_BOTTOM).toFixed(2)} Z`

  const xTicks = pickTickIndices(points.length, 6)
  const yTicks = [eMax + yPad, (eMax + eMin) / 2 + yPad / 2, eMin - yPad]

  const start = points[0]?.equity ?? 0
  const last = points[points.length - 1]?.equity ?? 0
  const totalPct = ((last - start) / Math.max(start, 1)) * 100

  // Precompute formatted date strings — keeping `new Date(...)` out of JSX
  // expressions keeps the React/lint rules happy + saves repeated work.
  const firstLabel = dateFmt.format(new Date(points[0]?.timestamp ?? Date.now()))
  const lastLabel = dateFmt.format(
    new Date(points[points.length - 1]?.timestamp ?? Date.now()),
  )
  const xTickLabels = xTicks.map((i) =>
    dateFmt.format(new Date(points[i]?.timestamp ?? Date.now())),
  )

  return (
    <div className="rounded-2xl border border-border/60 bg-card/30 backdrop-blur-xl p-3 md:p-6">
      <div className="flex items-baseline justify-between mb-4 px-3">
        <div>
          <div className="text-[10px] font-mono uppercase tracking-[0.22em] text-muted-foreground">
            Equity history
          </div>
          <div className="text-2xl md:text-3xl font-semibold tracking-[-0.025em] text-primary tabular-nums mt-1">
            {usd.format(last)}
          </div>
        </div>
        <div className={`text-sm font-mono tabular-nums ${totalPct >= 0 ? "text-primary" : "text-red-400"}`}>
          {totalPct >= 0 ? "+" : ""}
          {totalPct.toFixed(2)}% since {firstLabel}
        </div>
      </div>
      <svg
        role="img"
        aria-label={`Paper account equity from ${firstLabel} to ${lastLabel}`}
        viewBox={`0 0 ${VIEWBOX_W} ${VIEWBOX_H}`}
        preserveAspectRatio="none"
        className="block w-full h-[220px] sm:h-[280px] md:h-[340px]"
      >
        {yTicks.map((eq, i) => (
          <g key={i}>
            <line
              x1={PAD_LEFT}
              x2={VIEWBOX_W - PAD_RIGHT}
              y1={yFor(eq)}
              y2={yFor(eq)}
              stroke="rgba(0,240,255,0.12)"
              strokeWidth={1}
              strokeDasharray="2 4"
            />
            <text
              x={PAD_LEFT - 8}
              y={yFor(eq) + 4}
              textAnchor="end"
              className="fill-[oklch(0.5_0_0)] font-mono"
              style={{ fontSize: 11 }}
            >
              {usd.format(eq)}
            </text>
          </g>
        ))}
        {xTicks.map((i, idx) => (
          <g key={i}>
            <line
              x1={xFor(i)}
              x2={xFor(i)}
              y1={VIEWBOX_H - PAD_BOTTOM}
              y2={VIEWBOX_H - PAD_BOTTOM + 5}
              stroke="rgba(0,240,255,0.25)"
              strokeWidth={1}
            />
            <text
              x={xFor(i)}
              y={VIEWBOX_H - 14}
              textAnchor="middle"
              className="fill-[oklch(0.5_0_0)] font-mono"
              style={{ fontSize: 11 }}
            >
              {xTickLabels[idx]}
            </text>
          </g>
        ))}
        <defs>
          <linearGradient id="paper-equity-area" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="rgba(0,240,255,0.35)" />
            <stop offset="100%" stopColor="rgba(0,240,255,0)" />
          </linearGradient>
        </defs>
        <path d={areaPath} fill="url(#paper-equity-area)" />
        <path
          d={linePath}
          fill="none"
          stroke="oklch(0.8 0.15 195)"
          strokeWidth={2}
          strokeLinejoin="round"
          strokeLinecap="round"
          style={{ filter: "drop-shadow(0 0 6px rgba(0,240,255,0.45))" }}
        />
      </svg>
    </div>
  )
}
