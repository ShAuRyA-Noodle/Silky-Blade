"use client"

import { useEffect, useRef } from "react"
import { gsap } from "gsap"
import { ScrollTrigger } from "gsap/ScrollTrigger"
import { Brain, LineChart, ShieldAlert, Atom } from "lucide-react"

if (typeof window !== "undefined") {
  gsap.registerPlugin(ScrollTrigger)
}

const PANELS = [
  {
    k: "01",
    icon: Brain,
    eyebrow: "Signals",
    title: "LightGBM ensemble.\nRanked every morning.",
    body: "Triple-barrier labels (López de Prado). Gradient-boosted trees trained with purged K-Fold CV + embargo so future bars never leak into the past. Folds average into a single conviction score per name. No deep learning, no transformers — just disciplined gradient boosting.",
    stat: "SP500 + NDX100",
    statLabel: "Active universe",
  },
  {
    k: "02",
    icon: LineChart,
    eyebrow: "Execution",
    title: "Risk-gated by default.\nNo trade bypasses the checks.",
    body: "Every submission runs through a layered gate — kill-switch, position caps, drawdown kill, max-positions — before the broker ever hears about it. The terminal simply cannot bypass it.",
    stat: "4 layers",
    statLabel: "Pre-trade guards",
  },
  {
    k: "03",
    icon: ShieldAlert,
    eyebrow: "Ops",
    title: "One switch, zero ambiguity.\nFlip it and we stop trading.",
    body: "Kill-switch is a single Redis key, persisted with the reason and the timestamp. Gauged in Prometheus. Read on every submit, never cached. It's the big red button that actually works.",
    stat: "< 1 ms",
    statLabel: "Kill-switch read path",
  },
  {
    k: "04",
    icon: Atom,
    eyebrow: "Rigor",
    title: "Deflated Sharpe &\nBacktest Overfitting.",
    body: "Every published result ships with a reproducibility manifest: git sha, config hash, data fingerprint, environment snapshot. And its performance is adjusted for how many trials were burned finding it.",
    stat: "DSR + PBO",
    statLabel: "López de Prado",
  },
]

export function PinnedFeatures() {
  const rootRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const root = rootRef.current
    if (!root) return

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches

    // Query panels directly — do NOT use gsap.utils.toArray (scoping unreliable).
    // Direct DOM query guarantees we get exactly these 4 elements.
    const panelEls = Array.from(root.querySelectorAll<HTMLElement>(".pin-panel"))
    const total = panelEls.length
    if (total === 0) return

    // Set initial visibility entirely via direct style assignment.
    // This is the ONLY place opacity is set — no GSAP opacity tweens at all.
    const applyVisibility = (idx: number, animate: boolean) => {
      panelEls.forEach((el, i) => {
        el.style.transition = animate ? "opacity 0.45s ease, transform 0.45s ease" : "none"
        el.style.opacity = i === idx ? "1" : "0"
        el.style.transform = i === idx ? "translateY(0)" : i < idx ? "translateY(-24px)" : "translateY(24px)"
        el.style.pointerEvents = i === idx ? "auto" : "none"
      })
    }

    applyVisibility(0, false)

    // Reduced motion: show only first panel, no scroll magic.
    if (reduced) return

    let currentIdx = 0

    const ctx = gsap.context(() => {
      // Pin the section, use onUpdate to switch panels.
      // No GSAP opacity tweens — only CSS transitions via direct style.
      ScrollTrigger.create({
        trigger: root,
        start: "top top",
        end: () => `+=${total * 200}vh`,
        pin: true,
        pinType: "transform",
        anticipatePin: 1,
        invalidateOnRefresh: true,
        onUpdate: (self) => {
          const newIdx = Math.min(Math.floor(self.progress * total), total - 1)
          if (newIdx !== currentIdx) {
            currentIdx = newIdx
            applyVisibility(newIdx, true)
          }
        },
      })

      // Progress rail — separate scrubbed trigger.
      const rail = root.querySelector<HTMLElement>(".pin-progress-rail")
      if (rail) {
        gsap.to(rail, {
          scaleY: 1,
          ease: "none",
          scrollTrigger: {
            trigger: root,
            start: "top top",
            end: () => `+=${total * 200}vh`,
            scrub: true,
            pinType: "transform",
            invalidateOnRefresh: true,
          },
        })
      }
    }, root)

    return () => {
      ctx.revert()
      // Reset to initial state on cleanup so StrictMode remount is clean.
      applyVisibility(0, false)
    }
  }, [])

  return (
    <section
      id="philosophy"
      ref={rootRef}
      className="relative h-screen w-full overflow-hidden bg-background scroll-mt-16"
    >
      {/* Left rail */}
      <div className="absolute left-8 top-1/2 -translate-y-1/2 z-20 hidden lg:flex flex-col items-center gap-6">
        <div className="text-xs font-mono text-muted-foreground tracking-[0.3em] uppercase">Method</div>
        <div className="relative h-56 w-px bg-border overflow-hidden">
          <div
            className="pin-progress-rail absolute inset-0 origin-top bg-primary scale-y-0"
            style={{ willChange: "transform" }}
          />
        </div>
        <div className="flex flex-col gap-3 items-center">
          {PANELS.map((p) => (
            <span key={p.k} className="text-[10px] font-mono text-muted-foreground">{p.k}</span>
          ))}
        </div>
      </div>

      {/* Panels — initial opacity:0 via inline style; JS sets active panel */}
      <div className="relative h-full w-full">
        {PANELS.map((p, idx) => {
          const Icon = p.icon
          return (
            <article
              key={p.k}
              className="pin-panel absolute inset-0 flex items-center justify-center px-6"
              style={{
                zIndex: idx + 1,
                opacity: idx === 0 ? 1 : 0,
                transform: idx === 0 ? "translateY(0)" : "translateY(24px)",
                pointerEvents: idx === 0 ? "auto" : "none",
              }}
            >
              <div className="container mx-auto max-w-6xl grid lg:grid-cols-[1.2fr_1fr] gap-12 items-center">
                <div>
                  <div className="inline-flex items-center gap-3 mb-6">
                    <span className="text-xs font-mono text-primary tracking-[0.3em] uppercase">{p.eyebrow}</span>
                    <span className="h-px w-10 bg-primary/40" />
                    <span className="text-xs font-mono text-muted-foreground">{p.k}</span>
                  </div>
                  <h3 className="text-[clamp(2rem,5.5vw,4.5rem)] leading-[1.02] tracking-[-0.02em] font-semibold whitespace-pre-line text-foreground">
                    {p.title}
                  </h3>
                  <p className="mt-6 text-base md:text-lg text-muted-foreground max-w-xl leading-relaxed">
                    {p.body}
                  </p>
                </div>
                <div className="relative">
                  <div className="relative aspect-square max-w-md mx-auto">
                    <div className="pointer-events-none absolute inset-0 rounded-full bg-gradient-to-br from-primary/30 via-secondary/10 to-transparent blur-2xl" />
                    <div className="absolute inset-6 rounded-3xl border border-primary/20 backdrop-blur-xl bg-card/40 flex flex-col items-center justify-center text-center gap-4 shadow-[0_0_60px_rgba(25,130,196,0.2)]">
                      <Icon className="w-12 h-12 text-primary" />
                      <div className="text-5xl font-semibold tracking-tight text-primary font-mono tabular-nums">
                        {p.stat}
                      </div>
                      <div className="text-xs font-mono text-muted-foreground tracking-[0.25em] uppercase">
                        {p.statLabel}
                      </div>
                    </div>
                    <div className="pointer-events-none absolute inset-0 rounded-full border border-primary/10 animate-[spin_20s_linear_infinite]" />
                  </div>
                </div>
              </div>
            </article>
          )
        })}
      </div>
    </section>
  )
}
