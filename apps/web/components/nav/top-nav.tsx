"use client"

import { useEffect, useRef, useState } from "react"
import { gsap } from "gsap"
import { ScrollTrigger } from "gsap/ScrollTrigger"
import Link from "next/link"

if (typeof window !== "undefined") {
  gsap.registerPlugin(ScrollTrigger)
}

const LINKS = [
  { href: "/results", label: "Results" },
  { href: "/paper", label: "Paper" },
  { href: "/#philosophy", label: "Method" },
  { href: "/#terminal", label: "Signals" },
]

export function TopNav() {
  const ref = useRef<HTMLElement>(null)
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    const el = ref.current
    if (!el) return

    // Pure JS scroll progress — never uses GSAP cached scroll height.
    // GSAP end:"max" uses the home-page's 1120vh pinned height on shorter
    // pages, making the bar stop at ~50% on /results and /paper.
    const updateProgress = () => {
      const scrollTop = window.scrollY
      const docHeight = document.documentElement.scrollHeight - window.innerHeight
      setProgress(docHeight > 0 ? scrollTop / docHeight : 0)
    }
    window.addEventListener("scroll", updateProgress, { passive: true })
    updateProgress() // set immediately on mount

    // Stuck state still via GSAP (this is position-based, not height-based — safe)
    const topTrigger = ScrollTrigger.create({
      start: 40,
      end: 99999,
      onUpdate: (self) => {
        el.dataset.stuck = self.isActive ? "1" : "0"
      },
    })

    return () => {
      window.removeEventListener("scroll", updateProgress)
      topTrigger.kill()
    }
  }, [])

  return (
    <header
      ref={ref}
      data-stuck="0"
      className="fixed top-0 inset-x-0 z-50 transition-all duration-500
        data-[stuck=1]:bg-background/70 data-[stuck=1]:backdrop-blur-2xl
        data-[stuck=1]:border-b data-[stuck=1]:border-border/40
        data-[stuck=1]:shadow-[0_1px_0_rgba(25,130,196,0.08)]"
    >
      <div className="container mx-auto max-w-7xl px-6 h-16 flex items-center justify-between">
        <Link
          href="/"
          className="flex items-center gap-2.5 text-sm font-mono tracking-[0.3em] uppercase text-foreground hover:text-primary transition-colors duration-200"
        >
          <span className="relative inline-flex w-2 h-2 rounded-full bg-primary">
            <span className="absolute inset-0 rounded-full bg-primary animate-ping opacity-60" />
          </span>
          Oracle
        </Link>

        <nav className="hidden md:flex items-center gap-7 text-[11px] font-mono tracking-[0.18em] uppercase">
          {LINKS.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              className="relative text-muted-foreground hover:text-foreground transition-colors duration-200 group"
            >
              {l.label}
              <span className="absolute -bottom-0.5 left-0 h-px w-0 bg-primary group-hover:w-full transition-all duration-300" />
            </Link>
          ))}
        </nav>

        <Link
          href="/#terminal"
          className="rounded-full border border-primary/35 bg-primary/8 px-4 py-2 text-[11px] font-mono uppercase tracking-[0.2em] text-primary hover:bg-primary/15 hover:border-primary/60 transition-all duration-200"
        >
          Launch →
        </Link>
      </div>

      <div className="relative h-[1.5px] bg-border/30">
        <div
          className="absolute inset-y-0 left-0 bg-gradient-to-r from-primary via-[#8AC926] to-[#6A4C93] origin-left transition-none"
          style={{ width: `${progress * 100}%` }}
        />
      </div>
    </header>
  )
}
