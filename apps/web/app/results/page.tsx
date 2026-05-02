import type { Metadata } from "next"

import { EquityCurve } from "@/components/oracle/equity-curve"
import { HonestyBlock } from "@/components/oracle/honesty-block"
import { KpiGrid } from "@/components/oracle/kpi-grid"
import { OracleHero } from "@/components/oracle/oracle-hero"
import { ReproBlock } from "@/components/oracle/repro-block"
import { TrustFootnote } from "@/components/oracle/trust-footnote"
import { loadOracleArtifacts } from "@/lib/oracle/load-artifacts"
import {
  formatPercent,
  formatRatio3,
  formatSharpe,
  formatYear,
} from "@/lib/oracle/format"

// Force static generation. The artifact bundle is read at build time; if it
// is missing the build fails (loadOracleArtifacts throws).
export const dynamic = "force-static"

export async function generateMetadata(): Promise<Metadata> {
  const { report } = loadOracleArtifacts()
  const m = report.metrics
  const startYear = formatYear(report.window.start)
  const endYear = formatYear(report.window.end)
  const title = `S&P 500 momentum · ${startYear}→${endYear} · Sharpe ${formatSharpe(m.sharpe)} · DSR P=${formatRatio3(m.deflated_sharpe_p)}`
  const description = `Walk-forward backtest. ${formatPercent(m.annualized_return)} annualized return, ${formatPercent(m.max_drawdown)} max drawdown. Real S&P 500 daily closes. Reproducible from the manifest in ORACLE.`
  return {
    title,
    description,
    openGraph: {
      title: `ORACLE — ${title}`,
      description,
      type: "article",
    },
    twitter: {
      card: "summary_large_image",
      title: `ORACLE — ${title}`,
      description,
    },
  }
}

export default function ResultsPage() {
  const { report, manifest, equityCurve, sweep } = loadOracleArtifacts()

  return (
    <main id="oracle-results" className="relative">
      <OracleHero report={report} />
      <KpiGrid report={report} />
      <HonestyBlock sweep={sweep} />
      <EquityCurve
        points={equityCurve}
        initialCapital={report.walk_forward.initial_capital}
      />
      <ReproBlock manifest={manifest} />
      <TrustFootnote report={report} />
    </main>
  )
}
