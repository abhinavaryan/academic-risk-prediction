"use client";

import { Table2 } from "lucide-react";
import { type ReactNode, useState } from "react";
import { cx } from "@/components/ui";

/** Categorical slots in fixed order (CVD-validated). Colour follows the entity, never its rank. */
export const SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)", "var(--series-5)"];
export const MUTED_MARK = "var(--axis)";

export const axis = {
  stroke: "var(--axis)",
  tick: { fill: "var(--muted)", fontSize: 12 },
  tickLine: false,
} as const;

export const grid = { stroke: "var(--grid)", strokeDasharray: undefined, vertical: false } as const;

export interface LegendItem {
  label: string;
  color: string;
  kind?: "line" | "box" | "dot";
}

export function Legend({ items }: { items: LegendItem[] }) {
  if (items.length < 2) return null;
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1.5 text-sm text-ink-2" aria-label="Legend">
      {items.map((it) => (
        <li key={it.label} className="flex items-center gap-2">
          {it.kind === "line" ? (
            <span aria-hidden className="inline-block h-0.5 w-4 rounded-full" style={{ background: it.color }} />
          ) : (
            <span aria-hidden className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: it.color }} />
          )}
          {it.label}
        </li>
      ))}
    </ul>
  );
}

export interface TableSpec {
  columns: string[];
  rows: (string | number)[][];
}

/** Every chart has a table-view twin, so no value is reachable only by hover or colour. */
export function ChartCard({
  title,
  subtitle,
  legend,
  table,
  children,
  className,
  footer,
}: {
  title: string;
  subtitle?: ReactNode;
  legend?: LegendItem[];
  table?: TableSpec;
  children: ReactNode;
  className?: string;
  footer?: ReactNode;
}) {
  const [showTable, setShowTable] = useState(false);
  return (
    <figure className={cx("flex min-w-0 flex-col rounded-2xl border border-line bg-surface p-5 sm:p-6", className)}>
      <div className="mb-4 flex items-start justify-between gap-3">
        <figcaption className="min-w-0">
          <h3 className="text-base font-semibold text-ink">{title}</h3>
          {subtitle && <p className="mt-1 text-sm leading-relaxed text-ink-2">{subtitle}</p>}
        </figcaption>
        {table && (
          <button
            type="button"
            onClick={() => setShowTable((v) => !v)}
            aria-pressed={showTable}
            className="no-print inline-flex shrink-0 items-center gap-1.5 rounded-lg border border-line px-2.5 py-1 text-xs font-medium text-ink-2 hover:bg-surface-2"
          >
            <Table2 className="h-3.5 w-3.5" aria-hidden />
            {showTable ? "Chart" : "Table"}
          </button>
        )}
      </div>
      {legend && <div className="mb-3">{<Legend items={legend} />}</div>}
      {showTable && table ? <DataTable spec={table} /> : <div className="min-w-0">{children}</div>}
      {footer && <div className="mt-3 text-xs leading-relaxed text-muted">{footer}</div>}
    </figure>
  );
}

export function DataTable({ spec }: { spec: TableSpec }) {
  return (
    <div className="max-h-80 overflow-auto rounded-lg border border-line">
      <table className="w-full text-left text-sm">
        <thead className="sticky top-0 bg-surface-2 text-ink-2">
          <tr>
            {spec.columns.map((c) => (
              <th key={c} scope="col" className="px-3 py-2 font-medium">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="tabular">
          {spec.rows.map((r, i) => (
            <tr key={i} className="border-t border-line">
              {r.map((v, j) => (
                <td key={j} className="px-3 py-1.5 text-ink">
                  {v}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export interface TooltipRow {
  label: string;
  value: string;
  color?: string;
}

export function TooltipBox({ title, rows }: { title: ReactNode; rows: TooltipRow[] }) {
  return (
    <div className="min-w-40 rounded-xl border border-line bg-surface px-3 py-2 text-sm shadow-lg">
      <p className="mb-1 font-medium text-ink">{title}</p>
      <ul className="space-y-0.5">
        {rows.map((r) => (
          <li key={r.label} className="flex items-center justify-between gap-4 text-ink-2">
            <span className="flex items-center gap-2">
              {r.color && <span aria-hidden className="inline-block h-2 w-2 rounded-full" style={{ background: r.color }} />}
              {r.label}
            </span>
            <span className="tabular font-medium text-ink">{r.value}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
