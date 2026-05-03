import type { Metadata } from "next"
import Link from "next/link"
import { ArrowUpRight } from "lucide-react"

import { PaperEquityChart } from "@/components/oracle/paper-equity-chart"
import {
  loadOraclePaperHistory,
  loadOraclePaperSnapshot,
} from "@/lib/oracle/load-paper"

// Build-time read of `.oracle-artifacts/paper-status.json`. The file is
// produced by `quant paper status --json-out`. If absent the page renders
// a "not connected" state with regen instructions — never invented data.
export const dynamic = "force-static"

export const metadata: Metadata = {
  title: "Paper account — ORACLE",
  description: "Live Alpaca paper trading account state and open positions.",
}

const usd = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 2,
})

const num = new Intl.NumberFormat("en-US", { maximumFractionDigits: 4 })

function fmtPct(s: string): string {
  const n = Number(s)
  if (!Number.isFinite(n)) return s
  return `${(n * 100).toFixed(2)}%`
}

export default function PaperPage() {
  const snapshot = loadOraclePaperSnapshot()
  const history = loadOraclePaperHistory()

  return (
    <main id="oracle-paper" className="relative">
      <section className="relative px-6 py-20 md:py-28">
        <div className="container mx-auto max-w-6xl">
          <div className="text-[11px] font-mono tracking-[0.3em] uppercase text-primary mb-3">
            Paper account · Alpaca
          </div>
          <h1 className="text-4xl md:text-6xl font-semibold tracking-[-0.025em] mb-6">
            Live paper trading state.
          </h1>
          <p className="text-base md:text-lg text-muted-foreground max-w-2xl leading-relaxed">
            Snapshot of the operator&rsquo;s Alpaca paper account at last
            refresh. Read-only view, sourced from
            {" "}<code className="font-mono text-primary">paper-status.json</code> on
            disk. No real money. No order routing. No promises about future
            returns. To regenerate run{" "}
            <code className="font-mono text-primary">
              quant paper status --json-out apps/web/.oracle-artifacts/paper-status.json
            </code>
            .
          </p>
        </div>
      </section>

      {snapshot ? (
        <>
          <section className="relative px-6 py-12 md:py-16 border-t border-border/40">
            <div className="container mx-auto max-w-6xl">
              <dl className="grid grid-cols-2 md:grid-cols-4 gap-3 md:gap-4">
                {[
                  ["Equity", usd.format(Number(snapshot.account.equity)), "Account net liquidity"],
                  ["Cash", usd.format(Number(snapshot.account.cash)), "Settled cash"],
                  ["Buying power", usd.format(Number(snapshot.account.buying_power)), "Margin · 2x intraday"],
                  ["Status", snapshot.account.status, snapshot.account.paper ? "Paper account" : "LIVE — refuse"],
                ].map(([label, value, note]) => (
                  <div
                    key={label}
                    className="rounded-2xl border border-border/60 bg-card/40 backdrop-blur-xl p-5 md:p-6"
                  >
                    <dt className="text-[10px] md:text-[11px] font-mono uppercase tracking-[0.22em] text-muted-foreground">
                      {label}
                    </dt>
                    <dd className="mt-3 text-2xl md:text-3xl font-semibold tracking-[-0.025em] text-primary tabular-nums">
                      {value}
                    </dd>
                    <p className="mt-3 text-[11px] md:text-xs font-mono text-muted-foreground/80 leading-relaxed">
                      {note}
                    </p>
                  </div>
                ))}
              </dl>
            </div>
          </section>

          {history.length >= 2 && (
            <section className="relative px-6 py-12 md:py-16 border-t border-border/40">
              <div className="container mx-auto max-w-6xl">
                <div className="text-[11px] font-mono tracking-[0.3em] uppercase text-primary mb-6">
                  Equity over time · {history.length} snapshots
                </div>
                <PaperEquityChart points={history} />
                <p className="mt-4 text-[11px] md:text-xs font-mono text-muted-foreground/80 leading-relaxed">
                  Daily snapshot written by the GitHub Actions cron job
                  (apps/api: <code className="text-primary">quant paper status --history-csv apps/web/.oracle-artifacts/paper-history.csv</code>).
                  Past performance does not predict future returns.
                </p>
              </div>
            </section>
          )}

          <section className="relative px-6 py-12 md:py-16 border-t border-border/40">
            <div className="container mx-auto max-w-6xl">
              <div className="text-[11px] font-mono tracking-[0.3em] uppercase text-primary mb-3">
                Open positions ({snapshot.positions.length})
              </div>
              <h2 className="text-2xl md:text-3xl font-semibold tracking-[-0.02em] mb-6">
                {snapshot.positions.length === 0
                  ? "Account is flat."
                  : "Mark-to-market of every long position."}
              </h2>

              {snapshot.positions.length === 0 ? (
                <p className="text-sm text-muted-foreground max-w-xl">
                  No open positions. Run{" "}
                  <code className="font-mono text-primary">quant paper now --signal-kind ml_bundle</code>{" "}
                  to compute today&rsquo;s plan, then re-run with{" "}
                  <code className="font-mono text-primary">--submit --confirm</code>{" "}
                  + <code className="font-mono text-primary">TRADING_ENABLED=true</code>{" "}
                  to actually place paper orders.
                </p>
              ) : (
                <div className="overflow-x-auto rounded-2xl border border-border/60 bg-card/30">
                  <table className="w-full text-sm font-mono">
                    <thead>
                      <tr className="text-left text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
                        <th className="px-4 py-3">Symbol</th>
                        <th className="px-4 py-3 text-right">Qty</th>
                        <th className="px-4 py-3 text-right">Avg entry</th>
                        <th className="px-4 py-3 text-right">Last</th>
                        <th className="px-4 py-3 text-right">Market value</th>
                        <th className="px-4 py-3 text-right">Unrealized PnL</th>
                        <th className="px-4 py-3 text-right">%</th>
                      </tr>
                    </thead>
                    <tbody>
                      {snapshot.positions.map((p) => {
                        const raw = p.raw as Record<string, string>
                        const upl = Number(raw.unrealized_pl ?? "0")
                        return (
                          <tr key={p.symbol} className="border-t border-border/40">
                            <td className="px-4 py-3 text-foreground">{p.symbol}</td>
                            <td className="px-4 py-3 text-right tabular-nums">{num.format(Number(p.quantity))}</td>
                            <td className="px-4 py-3 text-right tabular-nums">{usd.format(Number(raw.avg_entry_price ?? "0"))}</td>
                            <td className="px-4 py-3 text-right tabular-nums">{usd.format(Number(p.last_price))}</td>
                            <td className="px-4 py-3 text-right tabular-nums">{usd.format(Number(raw.market_value ?? "0"))}</td>
                            <td className={`px-4 py-3 text-right tabular-nums ${upl >= 0 ? "text-primary" : "text-red-400"}`}>
                              {usd.format(upl)}
                            </td>
                            <td className={`px-4 py-3 text-right tabular-nums ${upl >= 0 ? "text-primary" : "text-red-400"}`}>
                              {fmtPct(String(raw.unrealized_plpc ?? "0"))}
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </section>
        </>
      ) : (
        <section className="relative px-6 py-12 md:py-16 border-t border-border/40">
          <div className="container mx-auto max-w-3xl">
            <div className="rounded-2xl border border-border/60 bg-card/30 backdrop-blur-xl p-6 md:p-8">
              <dt className="text-[10px] md:text-[11px] font-mono uppercase tracking-[0.22em] text-muted-foreground">
                Paper account — not connected
              </dt>
              <dd className="mt-3 text-2xl md:text-3xl font-semibold tracking-[-0.02em] text-foreground">
                No snapshot found
              </dd>
              <p className="mt-4 text-sm md:text-base text-muted-foreground leading-relaxed">
                Generate the snapshot:
              </p>
              <code className="mt-3 block text-[11px] md:text-xs font-mono text-primary/90 bg-primary/5 border border-primary/20 rounded-lg px-3 py-2 leading-relaxed">
                quant paper status --json-out apps/web/.oracle-artifacts/paper-status.json
              </code>
              <p className="mt-3 text-[11px] md:text-xs font-mono text-muted-foreground/80 leading-relaxed">
                Requires Alpaca paper API keys in <code>.env.local</code>{" "}
                (ALPACA_API_KEY_ID + ALPACA_API_SECRET_KEY). The HTTP API at{" "}
                <code>/api/v1/paper/account</code> serves the same payload
                live; this static page is the disk-cached version.
              </p>
            </div>
          </div>
        </section>
      )}

      <section className="relative px-6 py-16 md:py-20 border-t border-border/40">
        <div className="container mx-auto max-w-3xl">
          <div className="rounded-2xl border border-primary/30 bg-primary/5 backdrop-blur-xl p-6 md:p-8 mb-8">
            <div className="text-[10px] md:text-[11px] font-mono uppercase tracking-[0.22em] text-primary mb-3">
              Brutal disclaimer
            </div>
            <p className="text-sm md:text-base text-foreground leading-relaxed">
              This is a paper account. The dollar amounts shown are play money.
              Live trading is gated behind{" "}
              <code className="font-mono text-primary">TRADING_ENABLED=true</code>{" "}
              and a triple-confirm safety check. The model that picks these
              positions is gradient-boosted decision trees — not AI, not a
              transformer, not magic. Past performance does not predict future
              returns.
            </p>
          </div>

          <div className="flex flex-wrap gap-3">
            <Link
              href="/results"
              className="inline-flex items-center gap-2 rounded-full border border-primary/40 px-5 py-3 text-sm font-mono uppercase tracking-[0.2em] text-primary hover:bg-primary/10 transition-colors"
            >
              View backtest results
              <ArrowUpRight className="w-4 h-4" />
            </Link>
            <Link
              href="https://github.com/ShAuRyA-Noodle/Shaurya-Stocks/blob/main/TRUST.md"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-2 rounded-full border border-border/60 px-5 py-3 text-sm font-mono uppercase tracking-[0.2em] text-muted-foreground hover:text-foreground hover:border-foreground/40 transition-colors"
            >
              Credibility contract
              <ArrowUpRight className="w-4 h-4" />
            </Link>
          </div>
        </div>
      </section>
    </main>
  )
}
