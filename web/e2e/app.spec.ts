import { readFileSync } from "node:fs";
import { expect, test } from "@playwright/test";

const overview = JSON.parse(readFileSync(new URL("../src/generated/overview.json", import.meta.url), "utf8"));
const leaderboard = JSON.parse(readFileSync(new URL("../src/generated/leaderboard.json", import.meta.url), "utf8"));
const fmt = (n: number) => n.toLocaleString("en-US");

test("overview shows the real dataset counts from the pipeline", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("at-risk students");
  await expect(page.getByText(fmt(overview.dataset.registrations)).first()).toBeVisible();
  await expect(page.getByText(fmt(overview.dataset.students)).first()).toBeVisible();
  expect(errors).toEqual([]);
});

test("models page lists every trained model", async ({ page }) => {
  await page.goto("/models/");
  const rows = page.locator("table").first().locator("tbody tr");
  await expect(rows).toHaveCount(leaderboard.models.length);
  await expect(page.getByText("Deployed").first()).toBeVisible();
});

test("predict a real student and reveal the actual outcome", async ({ page }) => {
  await page.goto("/predict/");
  const risk = page.getByText(/^\d{1,3}%$/).first();
  await expect(risk).toBeVisible({ timeout: 30_000 });
  await page.getByRole("button", { name: "Random" }).click();
  await expect(risk).toBeVisible();
  await page.getByRole("button", { name: "Reveal actual outcome" }).click();
  await expect(page.getByText("Actual final result:")).toBeVisible();
});

test("what-if: more engagement does not raise risk for an inactive student", async ({ page }) => {
  await page.goto("/predict/");
  const value = page.getByText(/^\d{1,3}%$/).first();
  await expect(value).toBeVisible({ timeout: 30_000 });
  const clicks = page.getByLabel("Total LMS clicks", { exact: true });
  await clicks.fill("0");
  await expect(page.getByRole("meter")).toBeVisible();
  await page.waitForTimeout(800);
  const low = Number((await value.textContent())!.replace("%", ""));
  await clicks.fill("3000");
  await page.waitForTimeout(800);
  const high = Number((await value.textContent())!.replace("%", ""));
  expect(high).toBeLessThanOrEqual(low);
});

test("train on the sample dataset and download the report", async ({ page }) => {
  test.slow();
  await page.goto("/train/");
  await page.getByRole("button", { name: "Try the sample dataset" }).click();
  await expect(page.getByRole("combobox", { name: "Result column" })).toHaveValue("Target");
  await page.getByRole("button", { name: /Train & compare/ }).click();
  await expect(page.getByText(/Model comparison/)).toBeVisible({ timeout: 90_000 });
  await expect(page.getByText("Students scored")).toBeVisible();
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: /Download risk report/ }).click();
  const file = await download;
  const text = readFileSync((await file.path())!, "utf8");
  expect(text.replace(/^﻿/, "").startsWith("row,student,risk_probability")).toBe(true);
  expect(text.split("\r\n").length).toBeGreaterThan(4000);
});

test("rejects a file that is not a spreadsheet", async ({ page }) => {
  await page.goto("/train/");
  await page.setInputFiles("#file", { name: "notes.pdf", mimeType: "application/pdf", buffer: Buffer.from("%PDF-1.4") });
  await expect(page.getByText("Please upload a .csv or .xlsx file.")).toBeVisible();
});

test("API docs and health are served", async ({ request }) => {
  const health = await request.get("/api/health");
  expect(health.ok()).toBeTruthy();
  expect((await health.json()).status).toBe("ok");
  expect((await request.get("/api/docs")).ok()).toBeTruthy();
});
