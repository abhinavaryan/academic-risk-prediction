import { expect, test } from "@playwright/test";

test("mobile menu navigates and pages never scroll sideways", async ({ page, isMobile }) => {
  test.skip(!isMobile, "mobile only");
  await page.setViewportSize({ width: 360, height: 780 }); // narrowest common phone width
  await page.goto("/");
  await page.getByRole("button", { name: "Open menu" }).click();
  await page.locator("#mobile-nav").getByRole("link", { name: "Models" }).click();
  await expect(page).toHaveURL(/\/models\/$/);
  for (const path of ["/", "/dataset/", "/models/", "/predict/", "/train/", "/methodology/"]) {
    await page.goto(path);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    expect(overflow, path).toBeLessThanOrEqual(1);
  }
});
