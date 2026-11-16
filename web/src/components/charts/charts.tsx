"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { formatPct } from "@/lib/format";
import { MUTED_MARK, TooltipBox, axis, grid } from "./common";

interface TipPayload {
  name?: string | number;
  value?: unknown;
  color?: string;
  stroke?: string;
  payload?: unknown;
}
interface TipProps {
  active?: boolean;
  payload?: readonly TipPayload[];
  label?: string | number;
}

export interface LineSeries {
  key: string;
  label: string;
  color: string;
}

/** Multi-series line chart with a crosshair tooltip. One y-axis, always. */
export function LineSeriesChart({
  data,
  xKey,
  series,
  height = 280,
  yFormat = (v: number) => String(v),
  xFormat = (v: number) => String(v),
  xLabel,
  yDomain,
  reference,
  xTicks,
}: {
  data: Record<string, number | null>[];
  xKey: string;
  series: LineSeries[];
  height?: number;
  yFormat?: (v: number) => string;
  xFormat?: (v: number) => string;
  xLabel?: string;
  yDomain?: [number | "auto" | "dataMin" | "dataMax", number | "auto" | "dataMin" | "dataMax"];
  reference?: { x: number; label: string };
  xTicks?: number[];
}) {
  return (
    <div style={{ height }} className="w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: xLabel ? 18 : 4, left: 0 }}>
          <CartesianGrid {...grid} />
          <XAxis
            dataKey={xKey}
            type="number"
            domain={["dataMin", "dataMax"]}
            ticks={xTicks}
            {...axis}
            tickFormatter={xFormat}
            label={xLabel ? { value: xLabel, position: "insideBottom", offset: -10, fill: "var(--muted)", fontSize: 12 } : undefined}
          />
          <YAxis {...axis} width={48} tickFormatter={yFormat} domain={yDomain ?? ["auto", "auto"]} />
          {reference && (
            <ReferenceLine
              x={reference.x}
              stroke="var(--ink-2)"
              strokeWidth={1}
              label={{ value: reference.label, position: "insideTopRight", fill: "var(--ink-2)", fontSize: 12 }}
            />
          )}
          <Tooltip
            cursor={{ stroke: "var(--axis)", strokeWidth: 1 }}
            content={(p: TipProps) =>
              p.active && p.payload?.length ? (
                <TooltipBox
                  title={`${xLabel ?? xKey} ${xFormat(Number(p.label))}`}
                  rows={p.payload.map((r) => ({
                    label: String(r.name),
                    value: typeof r.value === "number" ? yFormat(r.value) : "—",
                    color: r.stroke ?? r.color,
                  }))}
                />
              ) : null
            }
          />
          {series.map((s) => (
            <Line
              key={s.key}
              type="monotone"
              dataKey={s.key}
              name={s.label}
              stroke={s.color}
              strokeWidth={2}
              strokeLinecap="round"
              strokeLinejoin="round"
              dot={data.length <= 12 ? { r: 3.5, fill: s.color, stroke: "var(--surface)", strokeWidth: 2 } : false}
              activeDot={{ r: 5, fill: s.color, stroke: "var(--surface)", strokeWidth: 2 }}
              connectNulls
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export interface BarDatum {
  label: string;
  value: number;
  ci?: [number, number];
  highlight?: boolean;
  note?: string;
}

/** Horizontal bars: thin (≤24px), rounded data end, optional 95% CI whisker, emphasis colour. */
export function HBarChart({
  data,
  format = (v: number) => v.toFixed(3),
  domain,
  rowHeight = 34,
  highlightColor = "var(--series-1)",
  baseColor,
  valueLabel = "Value",
}: {
  data: BarDatum[];
  format?: (v: number) => string;
  domain?: [number, number];
  rowHeight?: number;
  highlightColor?: string;
  baseColor?: string;
  valueLabel?: string;
}) {
  const anyHighlight = data.some((d) => d.highlight);
  const rows = data.map((d) => ({
    ...d,
    err: d.ci ? [Math.max(0, d.value - d.ci[0]), Math.max(0, d.ci[1] - d.value)] : undefined,
  }));
  const color = (d: BarDatum) => (anyHighlight ? (d.highlight ? highlightColor : baseColor ?? MUTED_MARK) : highlightColor);
  return (
    <div style={{ height: Math.max(120, rows.length * rowHeight + 40) }} className="w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 56, bottom: 4, left: 4 }} barCategoryGap={6}>
          <CartesianGrid stroke="var(--grid)" horizontal={false} />
          <XAxis type="number" {...axis} domain={domain ?? [0, "auto"]} tickFormatter={format} />
          <YAxis
            type="category"
            dataKey="label"
            {...axis}
            width={190}
            interval={0}
            tick={{ fill: "var(--ink-2)", fontSize: 12 }}
          />
          <Tooltip
            cursor={{ fill: "var(--surface-2)" }}
            content={(p: TipProps) => {
              const d = p.payload?.[0]?.payload as (BarDatum & { err?: number[] }) | undefined;
              if (!p.active || !d) return null;
              const rowsTip = [{ label: valueLabel, value: format(d.value) }];
              if (d.ci) rowsTip.push({ label: "95% CI", value: `${format(d.ci[0])} – ${format(d.ci[1])}` });
              if (d.note) rowsTip.push({ label: "Note", value: d.note });
              return <TooltipBox title={d.label} rows={rowsTip} />;
            }}
          />
          <Bar
            dataKey="value"
            radius={[0, 4, 4, 0]}
            maxBarSize={20}
            isAnimationActive={false}
            label={{ position: "right", fill: "var(--ink-2)", fontSize: 12, formatter: (v: unknown) => format(Number(v)) }}
          >
            {rows.map((d) => (
              <Cell key={d.label} fill={color(d)} />
            ))}
            <ErrorBar dataKey="err" width={5} stroke="var(--ink)" strokeWidth={1} direction="x" />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/** ROC / PR / calibration style chart on the unit square, one line per model. */
export function UnitCurveChart({
  curves,
  xLabel,
  yLabel,
  diagonal = true,
  height = 300,
}: {
  curves: { name: string; color: string; points: [number, number][] }[];
  xLabel: string;
  yLabel: string;
  diagonal?: boolean;
  height?: number;
}) {
  return (
    <div style={{ height }} className="w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart margin={{ top: 8, right: 16, bottom: 18, left: 4 }}>
          <CartesianGrid {...grid} vertical />
          <XAxis
            type="number"
            dataKey="x"
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            {...axis}
            label={{ value: xLabel, position: "insideBottom", offset: -10, fill: "var(--muted)", fontSize: 12 }}
          />
          <YAxis
            type="number"
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            {...axis}
            width={44}
            label={{ value: yLabel, angle: -90, position: "insideLeft", offset: 10, fill: "var(--muted)", fontSize: 12 }}
          />
          {diagonal && (
            <ReferenceLine
              segment={[
                { x: 0, y: 0 },
                { x: 1, y: 1 },
              ]}
              stroke="var(--axis)"
              strokeWidth={1}
            />
          )}
          <Tooltip
            cursor={{ stroke: "var(--axis)" }}
            content={(p: TipProps) =>
              p.active && p.payload?.length ? (
                <TooltipBox
                  title={`${xLabel} ${Number((p.payload[0]?.payload as { x: number })?.x ?? 0).toFixed(2)}`}
                  rows={p.payload.map((r) => ({
                    label: String(r.name),
                    value: Number(r.value).toFixed(3),
                    color: r.stroke ?? r.color,
                  }))}
                />
              ) : null
            }
          />
          {curves.map((c) => (
            <Line
              key={c.name}
              data={c.points.map(([x, y]) => ({ x, y }))}
              dataKey="y"
              name={c.name}
              type="linear"
              stroke={c.color}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, stroke: "var(--surface)", strokeWidth: 2, fill: c.color }}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/** 2×2 confusion matrix; cell shade is a single-hue sequential ramp by share of all students. */
export function ConfusionMatrix({ tp, fp, tn, fn, positive = "At risk", negative = "Not at risk" }: {
  tp: number;
  fp: number;
  tn: number;
  fn: number;
  positive?: string;
  negative?: string;
}) {
  const total = tp + fp + tn + fn || 1;
  const steps = ["var(--seq-1)", "var(--seq-2)", "var(--seq-3)", "var(--seq-4)", "var(--seq-5)"];
  const cell = (n: number, title: string) => {
    const share = n / total;
    const idx = Math.min(4, Math.floor(share * 8));
    const darkText = idx <= 1;
    return (
      <div
        className="flex aspect-[4/3] flex-col items-center justify-center rounded-lg p-2 text-center"
        style={{ background: steps[idx], color: darkText ? "var(--ink)" : "#ffffff" }}
      >
        <span className="text-xl font-semibold">{n.toLocaleString("en-US")}</span>
        <span className="text-xs opacity-90">{title}</span>
        <span className="text-xs opacity-80">{formatPct(share)}</span>
      </div>
    );
  };
  return (
    <div className="grid grid-cols-[auto_1fr_1fr] gap-2 text-sm">
      <span />
      <span className="text-center text-xs font-medium text-ink-2">Predicted {positive.toLowerCase()}</span>
      <span className="text-center text-xs font-medium text-ink-2">Predicted {negative.toLowerCase()}</span>
      <span className="flex items-center text-xs font-medium text-ink-2 [writing-mode:vertical-rl] rotate-180">Actually {positive.toLowerCase()}</span>
      {cell(tp, "caught (TP)")}
      {cell(fn, "missed (FN)")}
      <span className="flex items-center text-xs font-medium text-ink-2 [writing-mode:vertical-rl] rotate-180">Actually {negative.toLowerCase()}</span>
      {cell(fp, "false alarm (FP)")}
      {cell(tn, "correctly safe (TN)")}
    </div>
  );
}

/** Diverging bars around zero: red pole raises risk, blue pole lowers it. */
export function DivergingBars({ items, format = (v: number) => `${v > 0 ? "+" : ""}${(v * 100).toFixed(1)} pts` }: {
  items: { label: string; value: number; detail?: string }[];
  format?: (v: number) => string;
}) {
  const max = Math.max(0.0001, ...items.map((i) => Math.abs(i.value)));
  return (
    <ul className="space-y-2.5">
      {items.map((it) => {
        const pct = (Math.abs(it.value) / max) * 50;
        const up = it.value > 0;
        return (
          <li key={it.label} className="text-sm">
            <div className="mb-1 flex items-baseline justify-between gap-3">
              <span className="text-ink">{it.label}</span>
              <span className="tabular shrink-0 text-ink-2">{format(it.value)}</span>
            </div>
            <div className="relative h-2.5 rounded-full bg-surface-2" aria-hidden>
              <div className="absolute inset-y-0 left-1/2 w-px bg-[var(--axis)]" />
              <div
                className="absolute inset-y-0 rounded-full"
                style={{
                  background: up ? "var(--risk-up)" : "var(--risk-down)",
                  left: up ? "50%" : `${50 - pct}%`,
                  width: `${pct}%`,
                }}
              />
            </div>
            {it.detail && <p className="mt-1 text-xs text-muted">{it.detail}</p>}
          </li>
        );
      })}
    </ul>
  );
}

/** Tall segmented bar: one whole split into parts (legend lists names in block order). */
export function SegmentedBar({ parts }: { parts: { label: string; value: number; color: string; dark?: boolean }[] }) {
  const total = parts.reduce((s, p) => s + p.value, 0) || 1;
  return (
    <div>
      <div className="flex h-7 w-full gap-0.5 overflow-hidden rounded-md" role="img"
        aria-label={parts.map((p) => `${p.label}: ${p.value.toLocaleString("en-US")} (${formatPct(p.value / total)})`).join(", ")}>
        {parts.map((p) =>
          p.value > 0 ? (
            <div
              key={p.label}
              title={`${p.label}: ${p.value.toLocaleString("en-US")} (${formatPct(p.value / total)})`}
              className="flex items-center rounded-[4px] px-2 text-xs font-medium"
              style={{ width: `${(p.value / total) * 100}%`, background: p.color, color: p.dark ? "#0b0b0b" : "#ffffff" }}
            >
              {p.value / total > 0.08 ? formatPct(p.value / total, 0) : ""}
            </div>
          ) : null,
        )}
      </div>
    </div>
  );
}

/** Probability meter with the model's decision threshold marked. */
export function RiskMeter({ probability, threshold }: { probability: number; threshold: number }) {
  const color = probability >= 0.66 ? "var(--critical)" : probability >= 0.33 ? "var(--warning)" : "var(--good)";
  return (
    <div>
      <div className="relative h-3 rounded-full bg-surface-2" role="meter" aria-valuemin={0} aria-valuemax={100}
        aria-valuenow={Math.round(probability * 100)} aria-label="Risk probability">
        <div className="absolute inset-y-0 left-0 rounded-full transition-[width]" style={{ width: `${probability * 100}%`, background: color }} />
        <div className="absolute -top-1 -bottom-1 w-0.5 rounded bg-[var(--ink)]" style={{ left: `${threshold * 100}%` }} title="Intervention threshold" />
      </div>
      <div className="mt-1.5 flex justify-between text-xs text-muted">
        <span>0%</span>
        <span>flag at {formatPct(threshold, 0)}</span>
        <span>100%</span>
      </div>
    </div>
  );
}
