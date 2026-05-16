"use client"

import { useEffect, useRef, useState } from "react"
import { gsap } from "gsap"
import { ScrollTrigger } from "gsap/ScrollTrigger"
import { TrendingUp, TrendingDown, Minus } from "lucide-react"
import { API_BASE_URL } from "@/lib/api"
import type { StaticSignal } from "@/lib/oracle/load-artifacts"

if (typeof window !== "undefined") {
  gsap.registerPlugin(ScrollTrigger)
}

type Signal = {
  symbol: string
  direction: "long" | "short" | "flat"
  confidence?: number | null
  rank_in_universe?: number | null
  date?: string
}

interface LiveSignalsProps {
  // Pre-generated real ML signals from quant ml predict (server-loaded at build time).
  // Used when the live API is offline. Never synthetic — always from the trained model.
  staticSignals?: readonly StaticSignal[]
}

export function LiveSignals({ staticSignals = [] }: LiveSignalsProps) {
  const rootRef = useRef<HTMLElement>(null)
  const [signals, setSignals] = useState<Signal[] | null>(null)
  const [source, setSource] = useState<"live" | "static" | null>(null)

  useEffect(() => {
    const ac = new AbortController()
    const key = process.env.NEXT_PUBLIC_API_KEY
    const headers: Record<string, string> = {}
    if (key) headers["x-api-key"] = key

    fetch(`${API_BASE_URL}/signals?limit=12`, {
      headers,
      signal: ac.signal,
      cache: "no-store",
    })
      .then(async (r) => {
        if (!r.ok) throw new Error(`API ${r.status}`)
        return r.json()
      })
      .then((data: unknown) => {
        const items = Array.isArray(data)
          ? (data as Signal[])
          : ((data as { items?: Signal[] })?.items ?? [])
        if (items.length > 0) {
          setSignals(items)
          setSource("live")
        } else {
          // API online but 0 signals — fall back to static
          setSignals([...staticSignals] as Signal[])
          setSource("static")
        }
      })
      .catch((e: Error) => {
        if (e.name !== "AbortError") {
          // API offline — show pre-generated real signals
          setSignals([...staticSignals] as Signal[])
          setSource("static")
        }
      })

    return () => ac.abort()
  }, [staticSignals])

  useEffect(() => {
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches
    const root = rootRef.current
    if (!root || reduced || !signals?.length) return
    const ctx = gsap.context(() => {
      gsap.fromTo(
        ".sig-card",
        { y: 40, opacity: 0, scale: 0.98 },
        {
          y: 0,
          opacity: 1,
          scale: 1,
          duration: 0.7,
          stagger: 0.05,
          ease: "power3.out",
          scrollTrigger: { trigger: root, start: "top 70%", once: true },
        },
      )
    }, root)
    return () => ctx.revert()
  }, [signals])

  const rows = signals ?? []
  const loading = signals === null

  return (
    <section
      id="terminal"
      ref={rootRef}
      className="relative py-32 px-6 bg-background border-t border-border/40 scroll-mt-16"
    >
      <div className="container mx-auto max-w-7xl">
        <div className="flex items-end justify-between flex-wrap gap-6 mb-12">
          <div>
            <div className="flex items-center gap-3 mb-3">
              <div className="text-xs font-mono tracking-[0.3em] uppercase text-primary">
                {source === "live" ? "Live · latest session" : "ML signals · as-of 2026-05-01"}
              </div>
              {source === "static" && (
                <span className="text-[10px] font-mono tracking-[0.15em] uppercase text-muted-foreground/60 border border-border/40 rounded px-2 py-0.5">
                  LightGBM 2026 model
                </span>
              )}
            </div>
            <h2 className="text-3xl md:text-5xl font-semibold tracking-[-0.02em]">
              {source === "live" ? "Today's ranked signals." : "Ranked model signals."}
            </h2>
          </div>
          <p className="text-sm text-muted-foreground max-w-sm">
            {source === "live" ? (
              <>Pulled from <code className="font-mono text-primary">GET /api/v1/signals</code>. Real-time from the live engine.</>
            ) : (
              <>Real BUY/HOLD/SELL from LightGBM trained on{" "}
              <span className="text-foreground">718k rows</span> of real Alpaca S&amp;P 500 data, 2018–2026.
              Sorted by model conviction. Zero synthetic paths.</>
            )}
          </p>
        </div>

        {loading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {Array.from({ length: 8 }).map((_, i) => (
              <div key={i} className="h-40 rounded-2xl border border-border/60 bg-card/40 animate-pulse" />
            ))}
          </div>
        ) : rows.length === 0 ? (
          <div className="rounded-2xl border border-border/40 bg-card/20 p-10 text-center">
            <div className="w-2 h-2 rounded-full bg-primary/40 mx-auto mb-4" />
            <div className="text-sm font-mono text-muted-foreground">No signals available.</div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {rows.slice(0, 12).map((s, i) => {
              const dir = s.direction || "flat"
              const DirIcon = dir === "long" ? TrendingUp : dir === "short" ? TrendingDown : Minus
              const tone =
                dir === "long" ? "text-[color:var(--color-profit)]"
                : dir === "short" ? "text-destructive"
                : "text-muted-foreground"
              const conf = s.confidence ?? 0
              return (
                <article
                  key={`${s.symbol}-${i}`}
                  className="sig-card relative rounded-2xl border border-border/60 bg-card/40 backdrop-blur-xl p-5 overflow-hidden hover:border-primary/50 transition-colors group"
                >
                  <div className="flex items-start justify-between">
                    <div className="text-xs font-mono text-muted-foreground">
                      #{s.rank_in_universe ?? "—"}
                    </div>
                    <div className={`flex items-center gap-1 text-xs font-mono ${tone}`}>
                      <DirIcon className="w-3.5 h-3.5" />
                      {dir === "long" ? "BUY" : dir === "short" ? "SELL" : "HOLD"}
                    </div>
                  </div>
                  <div className="mt-5 text-3xl font-semibold tracking-[-0.02em] group-hover:text-primary transition-colors">
                    {s.symbol}
                  </div>
                  {s.date && (
                    <div className="mt-1 text-[10px] font-mono text-muted-foreground/50">{s.date}</div>
                  )}
                  <div className="mt-4">
                    <div className="flex items-center justify-between text-[10px] font-mono uppercase tracking-[0.2em] text-muted-foreground mb-2">
                      <span>Model conviction</span>
                      <span>{(conf * 100).toFixed(0)}%</span>
                    </div>
                    <div className="h-1 rounded-full bg-border overflow-hidden">
                      <div
                        className={`h-full transition-all duration-700 ${
                          dir === "long" ? "bg-[#8AC926]" : dir === "short" ? "bg-[#FF595E]" : "bg-primary"
                        }`}
                        style={{ width: `${Math.min(100, Math.max(0, conf * 100))}%` }}
                      />
                    </div>
                  </div>
                </article>
              )
            })}
          </div>
        )}

        {source === "static" && rows.length > 0 && (
          <p className="mt-6 text-[11px] font-mono text-muted-foreground/40 text-center tracking-[0.15em] uppercase">
            Signals from LightGBM 2026 model · real Alpaca data · not live-refreshed · start the API for real-time rankings
          </p>
        )}
      </div>
    </section>
  )
}
