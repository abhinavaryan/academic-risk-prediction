"use client";

import { Monitor, Moon, Sun } from "lucide-react";
import { useSyncExternalStore } from "react";

type Mode = "system" | "light" | "dark";
const ORDER: Mode[] = ["system", "light", "dark"];
const listeners = new Set<() => void>();
let memory: Mode | null = null; // used when localStorage is blocked

function read(): Mode {
  if (memory) return memory;
  try {
    const m = localStorage.getItem("theme") as Mode | null;
    return m && ORDER.includes(m) ? m : "system";
  } catch {
    return "system";
  }
}

function subscribe(cb: () => void) {
  listeners.add(cb);
  window.addEventListener("storage", cb);
  return () => {
    listeners.delete(cb);
    window.removeEventListener("storage", cb);
  };
}

function apply(mode: Mode) {
  const dark = mode === "dark" || (mode === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.dataset.theme = dark ? "dark" : "light";
}

export function ThemeToggle() {
  const mode = useSyncExternalStore(subscribe, read, () => "system" as Mode);

  function next() {
    const m = ORDER[(ORDER.indexOf(mode) + 1) % ORDER.length] ?? "system";
    memory = m;
    try {
      localStorage.setItem("theme", m);
    } catch {
      /* storage blocked: theme still applies for this page view */
    }
    apply(m);
    listeners.forEach((l) => l());
  }

  const Icon = mode === "dark" ? Moon : mode === "light" ? Sun : Monitor;
  return (
    <button
      type="button"
      onClick={next}
      className="inline-flex h-9 w-9 items-center justify-center rounded-lg text-ink-2 hover:bg-surface-2 hover:text-ink"
      aria-label={`Theme: ${mode}. Click to change.`}
      title={`Theme: ${mode}`}
    >
      <Icon className="h-4 w-4" aria-hidden />
    </button>
  );
}

/** Runs before paint so the page never flashes the wrong theme. */
export const themeScript = `(function(){try{var m=localStorage.getItem('theme')||'system';var d=m==='dark'||(m==='system'&&matchMedia('(prefers-color-scheme: dark)').matches);document.documentElement.dataset.theme=d?'dark':'light';}catch(e){}})();`;
