import Link from "next/link"
import { ArrowUpRight } from "lucide-react"

import type {
  BacktestSweepReport,
  PointInTimeComparison,
} from "@/lib/oracle/types"
import {
  formatPercent,
  formatRatio2,
  formatRatio3,
  formatSharpe,
} from "@/lib/oracle/format"

interface HonestyBlockProps {
  /**
   * Optional cross-config sweep report. When present, the panel surfaces
   * Probability of Backtest Overfitting (PBO) for the trial pool that
   * contained this run. When null, the brutal-disclaimer paragraph still
   * renders — the page never hides the caveats just because the sweep
   * artifact wasn't regenerated.
   */
  readonly sweep: BacktestSweepReport | null
  /**
   * Optional point-in-time-vs-raw comparison. When present, surfaces the
   * actual size of the survivorship-bias premium hidden in the headline
   * Sharpe.
   */
  readonly pitComparison: PointInTimeComparison | null
}

function pboReading(pbo: number): string {
  if (pbo > 0.6) return "Likely overfitting selection — treat headline Sharpe as suspect."
  if (pbo > 0.4) return "Borderline — the in-sample winner is roughly coin-flip out of sample."
  if (pbo > 0.2) return "Mild selection bias — the headline survives but is not pristine."
  return "Robust — the in-sample winner generally stays a winner out of sample."
}

export function HonestyBlock({ sweep, pitComparison }: HonestyBlockProps) {
  const pbo = sweep?.pbo ?? null

  return (
    <section
      id="honesty"
      className="relative px-6 py-20 md:py-28 border-t border-border/40"
      aria-labelledby="oracle-honesty-title"
    >
      <div className="container mx-auto max-w-5xl">
        <div className="text-[11px] font-mono tracking-[0.3em] uppercase text-primary mb-3">
          Honest reading
        </div>
        <h2
          id="oracle-honesty-title"
          className="text-3xl md:text-5xl font-semibold tracking-[-0.02em] mb-8"
        >
          What this is not.
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-6 mb-12">
          {pbo !== null && sweep ? (
            <div className="rounded-2xl border border-primary/40 bg-card/50 backdrop-blur-xl p-6 md:p-8">
              <dt className="text-[10px] md:text-[11px] font-mono uppercase tracking-[0.22em] text-muted-foreground">
                Probability of Backtest Overfitting
              </dt>
              <dd className="mt-3 text-4xl md:text-5xl font-semibold tracking-[-0.025em] text-primary tabular-nums">
                {formatRatio3(pbo)}
              </dd>
              <p className="mt-4 text-sm md:text-base text-foreground leading-relaxed">
                {pboReading(pbo)}
              </p>
              <p className="mt-3 text-[11px] md:text-xs font-mono text-muted-foreground/80 leading-relaxed">
                CSCV across <span className="tabular-nums">{sweep.n_configs}</span>{" "}
                configs · <span className="tabular-nums">{sweep.cscv_n_trials}</span>{" "}
                IS/OOS partitions ·{" "}
                <span className="tabular-nums">
                  {formatRatio2(sweep.n_observations_per_config)}
                </span>{" "}
                aligned rebalances per config
              </p>
            </div>
          ) : (
            <div className="rounded-2xl border border-border/60 bg-card/30 backdrop-blur-xl p-6 md:p-8">
              <dt className="text-[10px] md:text-[11px] font-mono uppercase tracking-[0.22em] text-muted-foreground">
                Probability of Backtest Overfitting
              </dt>
              <dd className="mt-3 text-2xl md:text-3xl font-semibold tracking-[-0.02em] text-muted-foreground">
                Not yet computed
              </dd>
              <p className="mt-4 text-sm text-muted-foreground leading-relaxed">
                Run the sweep to see the in-sample/out-of-sample selection-bias
                diagnostic for this trial pool:
              </p>
              <code className="mt-3 block text-[11px] md:text-xs font-mono text-primary/90 bg-primary/5 border border-primary/20 rounded-lg px-3 py-2 leading-relaxed">
                quant backtest sweep examples/backtest/sp500_momentum_sweep.yaml
              </code>
            </div>
          )}

          {pitComparison ? (
            <div className="rounded-2xl border border-primary/40 bg-card/50 backdrop-blur-xl p-6 md:p-8">
              <dt className="text-[10px] md:text-[11px] font-mono uppercase tracking-[0.22em] text-muted-foreground">
                Survivorship-bias premium
              </dt>
              <dd className="mt-3 text-4xl md:text-5xl font-semibold tracking-[-0.025em] text-primary tabular-nums">
                −{formatSharpe(pitComparison.raw.sharpe - pitComparison.pit.sharpe)} Sharpe
              </dd>
              <p className="mt-4 text-sm md:text-base text-foreground leading-relaxed">
                Same momentum strategy, two universes:
              </p>
              <dl className="mt-3 grid grid-cols-2 gap-3 text-[12px] md:text-[13px] font-mono">
                <div>
                  <dt className="text-muted-foreground">Survivors-only</dt>
                  <dd className="text-foreground tabular-nums">
                    Sharpe {formatSharpe(pitComparison.raw.sharpe)} ·{" "}
                    {formatPercent(pitComparison.raw.annualized_return)} AnnRet
                  </dd>
                </div>
                <div>
                  <dt className="text-muted-foreground">Point-in-time S&amp;P 500</dt>
                  <dd className="text-foreground tabular-nums">
                    Sharpe {formatSharpe(pitComparison.pit.sharpe)} ·{" "}
                    {formatPercent(pitComparison.pit.annualized_return)} AnnRet
                  </dd>
                </div>
              </dl>
              <p className="mt-3 text-[11px] md:text-xs font-mono text-muted-foreground/80 leading-relaxed">
                Gap = the &ldquo;joined-after&rdquo; forward-looking bias in the
                survivors-only dataset. Closing the &ldquo;exited-before&rdquo; gap
                requires delisted-name price coverage (a tracked data gap).
              </p>
            </div>
          ) : (
            <div className="rounded-2xl border border-border/60 bg-card/30 backdrop-blur-xl p-6 md:p-8">
              <dt className="text-[10px] md:text-[11px] font-mono uppercase tracking-[0.22em] text-muted-foreground">
                Survivorship bias
              </dt>
              <dd className="mt-3 text-2xl md:text-3xl font-semibold tracking-[-0.02em] text-foreground">
                Present, unfixed
              </dd>
              <p className="mt-4 text-sm md:text-base text-muted-foreground leading-relaxed">
                The S&amp;P 500 daily snapshot used here only contains names that
                survived to the dataset cut-off. Companies that went bust are
                silently absent. Momentum on a survivors-only universe overstates
                returns. Point-in-time membership history is a tracked gap.
              </p>
            </div>
          )}
        </div>

        <div className="rounded-2xl border border-primary/30 bg-primary/5 backdrop-blur-xl p-6 md:p-8 mb-10">
          <div className="text-[10px] md:text-[11px] font-mono uppercase tracking-[0.22em] text-primary mb-4">
            Brutal disclaimer — read once
          </div>
          <ul className="space-y-3 text-sm md:text-base text-foreground leading-relaxed">
            <li className="flex gap-3">
              <span className="text-primary mt-1">·</span>
              <span>
                These are <span className="text-primary">backtest</span> numbers,
                not live trading numbers. Live out-of-sample results are
                routinely lower than backtests for every strategy ever run.
              </span>
            </li>
            <li className="flex gap-3">
              <span className="text-primary mt-1">·</span>
              <span>
                Past performance does not predict future returns. Anyone who
                tells you a model can guarantee a profit on any specific stock
                tomorrow is selling you a lie.
              </span>
            </li>
            <li className="flex gap-3">
              <span className="text-primary mt-1">·</span>
              <span>
                This page is research output, not investment advice. No order
                routing, no real money, no recommendation to buy or sell any
                security. Treat it as a methodology demo.
              </span>
            </li>
            <li className="flex gap-3">
              <span className="text-primary mt-1">·</span>
              <span>
                The model is gradient-boosted decision trees (LightGBM). It is
                not &ldquo;AI&rdquo;, not a transformer, not a neural network. It
                outputs calibrated probabilities, not point price forecasts.
              </span>
            </li>
          </ul>
        </div>

        <div className="flex flex-wrap gap-3">
          <Link
            href="https://github.com/ShAuRyA-Noodle/Shaurya-Stocks/blob/main/TRUST.md"
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-2 rounded-full border border-primary/40 px-5 py-3 text-sm font-mono uppercase tracking-[0.2em] text-primary hover:bg-primary/10 transition-colors"
          >
            Full credibility contract
            <ArrowUpRight className="w-4 h-4" />
          </Link>
          <Link
            href="https://github.com/ShAuRyA-Noodle/Shaurya-Stocks/blob/main/REPRODUCE.md"
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-2 rounded-full border border-border/60 px-5 py-3 text-sm font-mono uppercase tracking-[0.2em] text-muted-foreground hover:text-foreground hover:border-foreground/40 transition-colors"
          >
            Reproduce these numbers
            <ArrowUpRight className="w-4 h-4" />
          </Link>
        </div>
      </div>
    </section>
  )
}
