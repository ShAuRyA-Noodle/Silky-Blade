"use client"

import { useEffect, useRef, useState } from "react"
import { gsap } from "gsap"
import { ScrollTrigger } from "gsap/ScrollTrigger"
import { TrendingUp, TrendingDown, Minus } from "lucide-react"
import { API_BASE_URL } from "@/lib/api"

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

export function LiveSignals() {
  const rootRef = useRef<HTMLElement>(null)
  const [signals, setSignals] = useState<Signal[] | null>(null)
  const [err, setErr] = useState<string | null>(null)

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
        setSignals(items)
      })
      .catch((e: Error) => {
        if (e.name !== "AbortError") setErr(e.message)
      })

    return () => ac.abort()
  }, [])

  useEffect(() => {
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches
    const root = rootRef.current
    if (!root || reduced) return
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
          scrollTrigger: { trigger: root, start: "top 70%" },
        },
      )
    }, root)
    return () => ctx.revert()
  }, [signals])

  const rows = signals ?? []
  const empty = !err && signals !== null && rows.length === 0
  const loading = signals === null && !err

  return (
    <section
      id="terminal"
      ref={rootRef}
      className="relative py-32 px-6 bg-background border-t border-border/40 scroll-mt-16"
    >
      <div className="container mx-auto max-w-7xl">
        <div className="flex items-end justify-between flex-wrap gap-6 mb-12">
          <div>
            <div className="text-xs font-mono tracking-[0.3em] uppercase text-primary mb-3">
              Live · latest session
            </div>
            <h2 className="text-3xl md:text-5xl font-semibold tracking-[-0.02em]">
              Today&apos;s ranked signals.
            </h2>
          </div>
          <p className="text-sm text-muted-foreground max-w-sm">
            Pulled from <code className="font-mono text-primary">GET /v1/signals</code>. If the
            API is offline the panel shows empty — we do not invent numbers.
          </p>
        </div>

        {err ? (
          <div className="rounded-2xl border border-destructive/40 bg-destructive/5 p-6 text-sm font-mono text-destructive">
            Signals endpoint unreachable: {err}. Start the API at{" "}
            <span className="text-foreground">{API_BASE_URL}</span>.
          </div>
        ) : loading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {Array.from({ length: 8 }).map((_, i) => (
              <div
                key={i}
                className="h-40 rounded-2xl border border-border/60 bg-card/40 animate-pulse"
              />
            ))}
          </div>
        ) : empty ? (
          <div className="rounded-2xl border border-border/60 bg-card/40 p-10 text-center">
            <div className="text-sm font-mono text-muted-foreground">
              No signals in the store yet. Run the training + signal-writer jobs.
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {rows.slice(0, 12).map((s, i) => {
              const dir = s.direction || "flat"
              const DirIcon =
                dir === "long" ? TrendingUp : dir === "short" ? TrendingDown : Minus
              const tone =
                dir === "long"
                  ? "text-[color:var(--color-profit)]"
                  : dir === "short"
                    ? "text-destructive"
                    : "text-muted-foreground"
              const conf = s.confidence ?? 0
              return (
                <article
                  key={`${s.symbol}-${i}`}
                  className="sig-card relative rounded-2xl border border-border/60 bg-card/40 backdrop-blur-xl p-5 overflow-hidden hover:border-primary/50 transition-colors"
                >
                  <div className="flex items-start justify-between">
                    <div className="text-xs font-mono text-muted-foreground">
                      #{s.rank_in_universe ?? "—"}
                    </div>
                    <div className={`flex items-center gap-1 text-xs font-mono ${tone}`}>
                      <DirIcon className="w-3.5 h-3.5" />
                      {dir.toUpperCase()}
                    </div>
                  </div>
                  <div className="mt-5 text-3xl font-semibold tracking-[-0.02em]">
                    {s.symbol}
                  </div>
                  <div className="mt-6">
                    <div className="flex items-center justify-between text-[10px] font-mono uppercase tracking-[0.2em] text-muted-foreground mb-2">
                      <span>Confidence</span>
                      <span>{(conf * 100).toFixed(0)}%</span>
                    </div>
                    <div className="h-1 rounded-full bg-border overflow-hidden">
                      <div
                        className="h-full bg-primary"
                        style={{ width: `${Math.min(100, Math.max(0, conf * 100))}%` }}
                      />
                    </div>
                  </div>
                </article>
              )
            })}
          </div>
        )}
      </div>
    </section>
  )
}
