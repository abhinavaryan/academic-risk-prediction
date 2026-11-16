import { AlertTriangle, CheckCircle2, CircleAlert, Info, OctagonAlert } from "lucide-react";
import type { HTMLAttributes, ReactNode } from "react";
import type { Band } from "@/lib/types";

export function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}

export function Container({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cx("mx-auto w-full max-w-6xl px-4 sm:px-6", className)}>{children}</div>;
}

export function Card({ children, className, ...rest }: { children: ReactNode; className?: string } & HTMLAttributes<HTMLElement>) {
  return (
    <section {...rest} className={cx("rounded-2xl border border-line bg-surface p-5 sm:p-6", className)}>
      {children}
    </section>
  );
}

export function SectionHeading({ eyebrow, title, children }: { eyebrow?: string; title: string; children?: ReactNode }) {
  return (
    <div className="mb-6 max-w-3xl">
      {eyebrow && <p className="mb-2 text-sm font-medium text-accent-ink">{eyebrow}</p>}
      <h2 className="text-2xl font-semibold tracking-tight text-ink sm:text-3xl">{title}</h2>
      {children && <div className="mt-3 text-base leading-relaxed text-ink-2">{children}</div>}
    </div>
  );
}

export function PageHeader({ eyebrow, title, children }: { eyebrow: string; title: string; children?: ReactNode }) {
  return (
    <header className="border-b border-line bg-surface">
      <Container className="py-10 sm:py-14">
        <p className="mb-3 text-sm font-medium text-accent-ink">{eyebrow}</p>
        <h1 className="max-w-3xl text-3xl font-semibold tracking-tight text-ink sm:text-4xl">{title}</h1>
        {children && <div className="mt-4 max-w-3xl text-lg leading-relaxed text-ink-2">{children}</div>}
      </Container>
    </header>
  );
}

export function StatTile({ label, value, hint }: { label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <div className="rounded-xl border border-line bg-surface p-4">
      <p className="text-sm text-ink-2">{label}</p>
      <p className="mt-1 text-2xl font-semibold tracking-tight text-ink">{value}</p>
      {hint && <p className="mt-1 text-xs leading-snug text-muted">{hint}</p>}
    </div>
  );
}

export function Pill({ children, tone = "neutral" }: { children: ReactNode; tone?: "neutral" | "accent" }) {
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium",
        tone === "accent" ? "bg-accent-wash text-accent-ink" : "bg-surface-2 text-ink-2",
      )}
    >
      {children}
    </span>
  );
}

const BAND_STYLE: Record<Band, { label: string; icon: typeof CheckCircle2; dot: string; text: string }> = {
  low: { label: "Low risk", icon: CheckCircle2, dot: "bg-good", text: "text-good-ink" },
  moderate: { label: "Moderate risk", icon: AlertTriangle, dot: "bg-warning", text: "text-warning-ink" },
  high: { label: "High risk", icon: OctagonAlert, dot: "bg-critical", text: "text-critical-ink" },
};

/** Status colour is never alone: icon + label always accompany it. */
export function RiskBadge({ band, size = "sm" }: { band: Band; size?: "sm" | "lg" }) {
  const s = BAND_STYLE[band];
  const Icon = s.icon;
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1.5 rounded-full border border-line bg-surface font-medium",
        s.text,
        size === "lg" ? "px-3 py-1 text-sm" : "px-2 py-0.5 text-xs",
      )}
    >
      <Icon aria-hidden className={size === "lg" ? "h-4 w-4" : "h-3.5 w-3.5"} />
      {s.label}
    </span>
  );
}

export function Callout({ tone = "info", title, children }: { tone?: "info" | "warn"; title?: string; children: ReactNode }) {
  const Icon = tone === "warn" ? CircleAlert : Info;
  return (
    <div
      role={tone === "warn" ? "alert" : "note"}
      className={cx(
        "flex gap-3 rounded-xl border p-4 text-sm leading-relaxed",
        tone === "warn" ? "border-warning/50 bg-warning/10 text-ink" : "border-line bg-accent-wash text-ink",
      )}
    >
      <Icon aria-hidden className={cx("mt-0.5 h-4 w-4 shrink-0", tone === "warn" ? "text-warning-ink" : "text-accent-ink")} />
      <div>
        {title && <p className="mb-1 font-medium">{title}</p>}
        {children}
      </div>
    </div>
  );
}

export const buttonClass = {
  primary:
    "inline-flex items-center justify-center gap-2 rounded-xl bg-accent px-4 py-2.5 text-sm font-medium text-white transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-50",
  secondary:
    "inline-flex items-center justify-center gap-2 rounded-xl border border-line bg-surface px-4 py-2.5 text-sm font-medium text-ink transition hover:bg-surface-2 disabled:cursor-not-allowed disabled:opacity-50",
  ghost:
    "inline-flex items-center gap-1.5 rounded-lg px-2 py-1 text-sm font-medium text-accent-ink hover:bg-accent-wash",
};

export function Segmented<T extends string | number>({
  options,
  value,
  onChange,
  label,
  format = String,
}: {
  options: T[];
  value: T;
  onChange: (v: T) => void;
  label: string;
  format?: (v: T) => string;
}) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex flex-wrap gap-1 rounded-xl border border-line bg-surface-2 p-1">
      {options.map((o) => (
        <button
          key={String(o)}
          type="button"
          role="radio"
          aria-checked={o === value}
          onClick={() => onChange(o)}
          className={cx(
            "rounded-lg px-3 py-1.5 text-sm font-medium transition",
            o === value ? "bg-surface text-ink shadow-sm" : "text-ink-2 hover:text-ink",
          )}
        >
          {format(o)}
        </button>
      ))}
    </div>
  );
}

export function Spinner({ className }: { className?: string }) {
  return (
    <span
      aria-hidden
      className={cx("inline-block h-4 w-4 animate-spin rounded-full border-2 border-current border-r-transparent", className)}
    />
  );
}
