import { GeistMono } from "geist/font/mono";
import { GeistSans } from "geist/font/sans";
import type { Metadata, Viewport } from "next";
import Link from "next/link";
import type { ReactNode } from "react";
import { Nav } from "@/components/nav";
import { themeScript } from "@/components/theme-toggle";
import { Container } from "@/components/ui";
import { site } from "@/config/site";
import { overview } from "@/lib/data";
import "./globals.css";

export const metadata: Metadata = {
  title: { default: `${site.name} · ${site.tagline}`, template: `%s · ${site.name}` },
  description: site.description,
  icons: { icon: "/icon.svg" },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#fcfcfb" },
    { media: "(prefers-color-scheme: dark)", color: "#1a1a19" },
  ],
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${GeistSans.variable} ${GeistMono.variable}`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body className="min-h-screen antialiased">
        <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-surface focus:px-3 focus:py-2">
          Skip to content
        </a>
        <Nav />
        <main id="main">{children}</main>
        <footer className="no-print mt-20 border-t border-line bg-surface">
          <Container className="flex flex-col gap-4 py-8 text-sm text-ink-2 sm:flex-row sm:items-start sm:justify-between">
            <div className="max-w-xl space-y-2">
              <p className="font-medium text-ink">{site.name} · {site.tagline}</p>
              <p>
                Research data: Open University Learning Analytics Dataset (Kuzilek, Hlosta &amp; Zdrahal, 2017, CC BY 4.0).
                Sample upload data: UCI &ldquo;Predict Students&rsquo; Dropout and Academic Success&rdquo; (Realinho et al., 2022, CC BY 4.0).
              </p>
              <p className="text-xs text-muted">
                Results generated {overview.generated_at.slice(0, 10)} · pipeline v{overview.version}
                {overview.quick ? " · QUICK MODE (development data)" : ""}
              </p>
            </div>
            <div className="flex flex-wrap gap-x-5 gap-y-2">
              <Link href="/methodology/" className="hover:text-ink">Methodology</Link>
              <a href="/api/docs" className="hover:text-ink">API docs</a>
              {site.repoUrl && <a href={site.repoUrl} className="hover:text-ink" rel="noreferrer">Source code</a>}
            </div>
          </Container>
        </footer>
      </body>
    </html>
  );
}
