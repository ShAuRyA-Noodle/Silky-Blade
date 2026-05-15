import type React from "react"
import type { Metadata } from "next"
import { Outfit, JetBrains_Mono } from "next/font/google"
import { Analytics } from "@vercel/analytics/next"
import "./globals.css"
import { SmoothScroll } from "@/components/providers/smooth-scroll"
import { TopNav } from "@/components/nav/top-nav"

const outfit = Outfit({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
  weight: ["300", "400", "500", "600", "700", "800"],
})

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
})

export const metadata: Metadata = {
  title: "ORACLE | Gradient-boosted trading signals on real market data",
  description:
    "Gradient-boosted trees (LightGBM) on triple-barrier labels, purged K-Fold CV with embargo, walk-forward backtests. Verified by Deflated Sharpe and Probability of Backtest Overfitting. Real providers, zero synthetic data.",
  icons: {
    icon: [{ url: "/icon.svg", type: "image/svg+xml" }],
    apple: "/apple-icon.png",
  },
  openGraph: {
    title: "ORACLE — Gradient-boosted signals. Verified.",
    description:
      "LightGBM · Purged K-Fold · DSR + PBO · Real data only. Walk-forward verified quantitative signals on S&P 500 + NDX 100.",
    images: [{ url: "/og-image.png", width: 1200, height: 630 }],
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "ORACLE — Gradient-boosted signals. Verified.",
    description:
      "LightGBM · Purged K-Fold · DSR + PBO · Real data only.",
    images: ["/og-image.png"],
  },
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${outfit.variable} ${jetbrainsMono.variable} font-sans antialiased bg-background text-foreground overflow-x-hidden`}
      >
        <SmoothScroll>
          <TopNav />
          {children}
        </SmoothScroll>
        <Analytics />
      </body>
    </html>
  )
}
