import { describe, expect, it } from "vitest";
import { compact, fixed, formatPct, int } from "./format";

describe("format", () => {
  it("formats percentages", () => {
    expect(formatPct(0.1234)).toBe("12.3%");
    expect(formatPct(1)).toBe("100%");
    expect(formatPct(null)).toBe("—");
  });
  it("formats integers with grouping", () => {
    expect(int(32593)).toBe("32,593");
  });
  it("compacts large numbers", () => {
    expect(compact(10655280)).toBe("10.7M");
    expect(compact(950)).toBe("950");
  });
  it("formats fixed decimals and handles missing values", () => {
    expect(fixed(0.84567, 3)).toBe("0.846");
    expect(fixed(undefined)).toBe("—");
  });
});
