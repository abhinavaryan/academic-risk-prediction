const DASH = "—";

const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

export function formatPct(v: number | null | undefined, digits = 1): string {
  if (!isNum(v)) return DASH;
  const pct = v * 100;
  return `${Number.isInteger(pct) ? pct.toFixed(0) : pct.toFixed(digits)}%`;
}

export function int(v: number | null | undefined): string {
  return isNum(v) ? Math.round(v).toLocaleString("en-US") : DASH;
}

export function compact(v: number | null | undefined): string {
  if (!isNum(v)) return DASH;
  return new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(v);
}

export function fixed(v: number | null | undefined, digits = 3): string {
  return isNum(v) ? v.toFixed(digits) : DASH;
}

export function signed(v: number, digits = 1): string {
  const s = (v * 100).toFixed(digits);
  return v > 0 ? `+${s}` : s;
}
