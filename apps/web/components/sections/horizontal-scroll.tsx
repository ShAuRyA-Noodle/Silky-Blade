"use client"

import { useEffect, useRef } from "react"
import { gsap } from "gsap"
import { ScrollTrigger } from "gsap/ScrollTrigger"
import { Activity, Cpu, Gauge, Globe2, Radar, Workflow } from "lucide-react"

if (typeof window !== "undefined") {
  gsap.registerPlugin(ScrollTrigger)
}

const TILES = [
  {
    icon: Gauge,
    label: "Risk",
    title: "Pre-trade gate",
    detail: "Kill-switch · position caps · drawdown · max-positions.",
  },
  {
    icon: Workflow,
    label: "Orders",
    title: "State machine",
    detail: "Pending → submitted → filled / cancelled / rejected. Idempotent.",
  },
  {
    icon: Cpu,
    label: "Signals",
    title: "Ranked daily",
    detail: "Top-K per universe, ordered by rank then confidence.",
  },
  {
    icon: Radar,
    label: "Streaming",
    title: "Alpaca IEX → SSE",
    detail: "Reconnecting WS consumer, Redis pub/sub fan-out, 15 s heartbeats.",
  },
  {
    icon: Activity,
    label: "Backtest",
    title: "Walk-forward",
    detail: "Equal-weight top-K, bps cost model, Sharpe + drawdown + DSR.",
  },
  {
    icon: Globe2,
    label: "Data",
    title: "Real providers",
    detail: "11 APIs keyed. Canonical Postgres. Zero synthetic paths.",
  },
]

export function HorizontalScroll() {
  const rootRef = useRef<HTMLElement>(null)
  const trackRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches
    const root = rootRef.current
    const track = trackRef.current
    if (!root || !track) return

    const ctx = gsap.context(() => {
      if (reduced) return
      const distance = () => track.scrollWidth - window.innerWidth

      gsap.to(track, {
        x: () => -distance(),
        ease: "none",
        scrollTrigger: {
          trigger: root,
          start: "top top",
          end: () => `+=${distance()}`,
          scrub: 0.8,
          pin: true,
          anticipatePin: 1,
          invalidateOnRefresh: true,
        },
      })
    }, root)

    return () => ctx.revert()
  }, [])

  return (
    <section
      ref={rootRef}
      className="relative h-screen w-full overflow-hidden bg-background"
    >
      <div className="absolute top-16 left-0 right-0 px-8 flex items-end justify-between z-10">
        <div>
          <div className="text-xs font-mono tracking-[0.3em] uppercase text-primary">
            Architecture
          </div>
          <h2 className="mt-3 text-3xl md:text-5xl font-semibold tracking-[-0.02em]">
            Every layer, accounted for.
          </h2>
        </div>
      </div>

      <div
        ref={trackRef}
        className="absolute inset-0 flex items-center gap-8 pl-[12vw] pr-[50vw] will-change-transform"
      >
        {TILES.map((t, i) => {
          const Icon = t.icon
          return (
            <article
              key={t.title}
              className="group relative shrink-0 w-[78vw] md:w-[58vw] lg:w-[42vw] h-[60vh] rounded-3xl border border-border/60 bg-card/40 backdrop-blur-2xl overflow-hidden shadow-[0_0_40px_rgba(25,130,196,0.08)] hover:shadow-[0_0_60px_rgba(25,130,196,0.15)] transition-shadow duration-500"
            >
              <div
                className="absolute inset-0 opacity-[0.18]"
                style={{
                  background:
                    i % 3 === 0
                      ? "radial-gradient(circle at 20% 20%, rgba(25,130,196,0.3), transparent 55%)"
                      : i % 3 === 1
                        ? "radial-gradient(circle at 80% 80%, rgba(138,201,38,0.3), transparent 55%)"
                        : "radial-gradient(circle at 50% 20%, rgba(106,76,147,0.3), transparent 55%)",
                }}
              />
              <div className="relative h-full flex flex-col justify-between p-10">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-2xl bg-primary/10 border border-primary/30 flex items-center justify-center">
                    <Icon className="w-5 h-5 text-primary" />
                  </div>
                  <div className="text-xs font-mono tracking-[0.3em] uppercase text-primary">
                    {t.label}
                  </div>
                  <div className="ml-auto text-[10px] font-mono text-muted-foreground">
                    {String(i + 1).padStart(2, "0")} / {TILES.length}
                  </div>
                </div>
                <div>
                  <h3 className="text-3xl md:text-4xl font-semibold tracking-[-0.02em] mb-3">
                    {t.title}
                  </h3>
                  <p className="text-sm md:text-base text-muted-foreground max-w-md">
                    {t.detail}
                  </p>
                </div>
              </div>
            </article>
          )
        })}
      </div>
    </section>
  )
}
