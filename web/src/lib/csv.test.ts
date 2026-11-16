import { describe, expect, it } from "vitest";
import { toCsv } from "./csv";

describe("toCsv", () => {
  it("writes a header and rows in column order", () => {
    expect(toCsv([{ a: 1, b: "x" }], ["b", "a"])).toBe("b,a\r\nx,1\r\n");
  });

  it("neutralises spreadsheet formulas (CSV injection)", () => {
    const out = toCsv([{ v: "=HYPERLINK(\"http://evil\")" }, { v: "+1" }, { v: "-2" }, { v: "@x" }], ["v"]);
    const lines = out.trim().split("\r\n").slice(1);
    expect(lines).toEqual(["\"'=HYPERLINK(\"\"http://evil\"\")\"", "'+1", "'-2", "'@x"]);
  });

  it("keeps real numbers numeric, including negatives", () => {
    expect(toCsv([{ v: -0.25 }], ["v"])).toBe("v\r\n-0.25\r\n");
  });

  it("quotes commas, quotes and newlines", () => {
    expect(toCsv([{ v: "a,b" }, { v: "say \"hi\"" }, { v: "l1\nl2" }], ["v"])).toBe(
      "v\r\n\"a,b\"\r\n\"say \"\"hi\"\"\"\r\n\"l1\nl2\"\r\n",
    );
  });

  it("renders null and undefined as empty cells", () => {
    expect(toCsv([{ a: null, b: undefined }], ["a", "b"])).toBe("a,b\r\n,\r\n");
  });
});
