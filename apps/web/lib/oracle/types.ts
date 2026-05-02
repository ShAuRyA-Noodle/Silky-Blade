/**
 * Typed shape of the on-disk backtest artifact bundle produced by
 * `quant backtest run`. Mirrors:
 *   apps/api/src/quant/backtest/{report,reproducibility}.py
 *
 * Field names, units, and ordering must match the artifact JSON exactly —
 * they are the source of truth, not these types.
 */

export interface BacktestWindow {
  readonly start: string // ISO date
  readonly end: string // ISO date
  readonly n_rebalances: number
}

export interface BacktestMetrics {
  readonly total_return: number // fraction (0.86 = +86%)
  readonly annualized_return: number // fraction
  readonly annualized_vol: number // fraction
  readonly sharpe: number
  readonly max_drawdown: number // fraction (positive)
  readonly turnover: number // total turnover ratio
  readonly deflated_sharpe_p: number // P(SR* > 0), in [0, 1]
  readonly dsr_n_trials: number
  readonly dsr_sharpes_std: number
  readonly return_skew: number
  readonly return_kurtosis: number
}

export interface BacktestWalkForward {
  readonly train_days: number
  readonly test_days: number
  readonly top_k: number
  readonly cost_bps: number
  readonly initial_capital: number
}

export interface BacktestSignal {
  readonly kind: string
  readonly params: Readonly<Record<string, number | string>>
}

export interface BacktestReport {
  readonly name: string
  readonly window: BacktestWindow
  readonly metrics: BacktestMetrics
  readonly walk_forward: BacktestWalkForward
  readonly signal: BacktestSignal
}

export interface BacktestManifest {
  readonly code_sha: string
  readonly config_hash: string
  readonly data_fingerprint: string
  readonly created_at: string // ISO timestamp
  readonly python_version: string
  readonly package_versions: Readonly<Record<string, string>>
}

export interface EquityPoint {
  readonly date: string // ISO date
  readonly equity: number // dollars
}

/**
 * One row of a backtest sweep — Sharpe + DSR for one config in a grid.
 * Mirrors the per-run shape in `quant.backtest.sweep.run_sweep`.
 */
export interface SweepRunSummary {
  readonly name: string
  readonly sharpe: number
  readonly deflated_sharpe_p: number
  readonly annualized_return: number
  readonly annualized_vol: number
  readonly max_drawdown: number
  readonly turnover: number
  readonly artifacts_dir: string
}

/**
 * Cross-config sweep report. PBO = Probability of Backtest Overfitting via
 * CSCV (López de Prado 2016). PBO ∈ [0, 1]; > 0.5 means the in-sample
 * winner is more likely than not below median out of sample.
 */
export interface BacktestSweepReport {
  readonly name: string
  readonly window: { readonly start: string; readonly end: string }
  readonly n_configs: number
  readonly n_observations_per_config: number
  readonly aligned_date_range: { readonly start: string; readonly end: string }
  readonly pbo: number
  readonly cscv_S: number
  readonly cscv_n_trials: number
  readonly runs: readonly SweepRunSummary[]
}

/**
 * Two-run comparison: the same strategy on the survivors-only universe vs.
 * point-in-time S&P 500 membership at each rebalance. The gap is the
 * "joined-after" forward-looking bias — non-zero by construction.
 */
export interface PointInTimeComparison {
  readonly raw: Pick<
    BacktestMetrics,
    "sharpe" | "annualized_return" | "max_drawdown" | "deflated_sharpe_p"
  >
  readonly pit: Pick<
    BacktestMetrics,
    "sharpe" | "annualized_return" | "max_drawdown" | "deflated_sharpe_p"
  >
}

/**
 * Bundle of everything the Oracle results page needs to render. The page
 * receives this as a single static prop assembled at build time. `sweep`
 * and `pitComparison` are optional — their absence does not break the
 * build, but the page always renders the brutal-disclaimer block whether
 * or not those diagnostics are present.
 */
export interface OracleArtifacts {
  readonly report: BacktestReport
  readonly manifest: BacktestManifest
  readonly equityCurve: readonly EquityPoint[]
  readonly sweep: BacktestSweepReport | null
  readonly pitComparison: PointInTimeComparison | null
}
