"use client";

import { Menu, X } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { site } from "@/config/site";
import { ThemeToggle } from "./theme-toggle";
import { cx } from "./ui";

const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/dataset/", label: "Dataset" },
  { href: "/models/", label: "Models" },
  { href: "/predict/", label: "Predict" },
  { href: "/train/", label: "Train on your data" },
  { href: "/methodology/", label: "Methodology" },
];

export function Logo() {
  return (
    <svg viewBox="0 0 32 32" className="h-7 w-7" aria-hidden>
      <rect width="32" height="32" rx="9" fill="var(--accent)" />
      <path d="M8 22.5 13.5 15l4 4.5L24 9.5" fill="none" stroke="#fff" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="24" cy="9.5" r="2.4" fill="#fff" />
    </svg>
  );
}

export function Nav() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const isActive = (href: string) => (href === "/" ? pathname === "/" : pathname?.startsWith(href.replace(/\/$/, "")));

  return (
    <header className="no-print sticky top-0 z-40 border-b border-line bg-surface/90 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-6xl items-center gap-4 px-4 sm:px-6">
        <Link href="/" className="flex items-center gap-2 font-semibold text-ink">
          <Logo />
          <span>{site.name}</span>
        </Link>
        <nav aria-label="Main" className="ml-4 hidden items-center gap-1 lg:flex">
          {LINKS.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              aria-current={isActive(l.href) ? "page" : undefined}
              className={cx(
                "rounded-lg px-3 py-1.5 text-sm font-medium transition",
                isActive(l.href) ? "bg-surface-2 text-ink" : "text-ink-2 hover:text-ink",
              )}
            >
              {l.label}
            </Link>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-1">
          <ThemeToggle />
          <button
            type="button"
            className="inline-flex h-9 w-9 items-center justify-center rounded-lg text-ink-2 hover:bg-surface-2 lg:hidden"
            aria-expanded={open}
            aria-controls="mobile-nav"
            aria-label={open ? "Close menu" : "Open menu"}
            onClick={() => setOpen((v) => !v)}
          >
            {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </div>
      </div>
      {open && (
        <nav id="mobile-nav" aria-label="Main" className="border-t border-line bg-surface px-4 py-2 lg:hidden">
          {LINKS.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              aria-current={isActive(l.href) ? "page" : undefined}
              onClick={() => setOpen(false)}
              className={cx("block rounded-lg px-3 py-2.5 text-base", isActive(l.href) ? "bg-surface-2 text-ink" : "text-ink-2")}
            >
              {l.label}
            </Link>
          ))}
        </nav>
      )}
    </header>
  );
}
