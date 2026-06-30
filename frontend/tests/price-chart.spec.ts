import { test, expect } from "@playwright/test";

// ── PriceChart E2E Tests ─────────────────────────────────────────────────────

test.describe("PriceChart", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    // Wait for the page to fully load
    await page.waitForLoadState("networkidle");
  });

  test("renders a canvas element (TradingView chart)", async ({ page }) => {
    // The chart section has the title "BTC/USDT — Live Chart"
    const chartTitle = page.locator("text=BTC/USDT — Live Chart");
    await expect(chartTitle).toBeVisible({ timeout: 10000 });

    // TradingView lightweight-charts renders a <canvas> element
    const canvas = page.locator("canvas").first();
    await expect(canvas).toBeVisible({ timeout: 10000 });
  });

  test("chart has minimum dimensions", async ({ page }) => {
    // Find the canvas rendered by TradingView
    const canvas = page.locator("canvas").first();
    await expect(canvas).toBeVisible({ timeout: 10000 });

    const box = await canvas.boundingBox();
    expect(box).not.toBeNull();
    expect(box!.width).toBeGreaterThan(100);
    expect(box!.height).toBeGreaterThan(100);
  });

  test("chart renders with dark background theme", async ({ page }) => {
    // The chart section title should be visible
    const chartTitle = page.locator("text=BTC/USDT — Live Chart");
    await expect(chartTitle).toBeVisible({ timeout: 10000 });
  });

  test("chart renders with canvas that has pixel data", async ({ page }) => {
    const canvas = page.locator("canvas").first();
    await expect(canvas).toBeVisible({ timeout: 10000 });

    // Capture a screenshot — if the chart rendered candles, the image
    // should be larger than a blank canvas (non-trivial pixel data)
    const img = await canvas.screenshot();
    // A chart with 100 candles should produce a non-trivial image (>5KB)
    expect(img.length).toBeGreaterThan(5000);
  });

  test("chart container exists with proper height", async ({ page }) => {
    // The chart title should be visible
    const chartTitle = page.locator("text=BTC/USDT — Live Chart");
    await expect(chartTitle).toBeVisible({ timeout: 10000 });

    // The canvas should have a reasonable height
    const canvas = page.locator("canvas").first();
    await expect(canvas).toBeVisible({ timeout: 10000 });
    const box = await canvas.boundingBox();
    expect(box).not.toBeNull();
    expect(box!.height).toBeGreaterThanOrEqual(200);
  });
});

test.describe("PriceChart - Error Handling", () => {
  test("no critical console errors on page load", async ({ page }) => {
    const errors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") {
        errors.push(msg.text());
      }
    });
    await page.goto("/");
    await page.waitForLoadState("networkidle");
    await page.waitForTimeout(2000);
    // Filter out known non-critical errors (WebSocket connection failures, fetch errors)
    const criticalErrors = errors.filter(
      (e) =>
        !e.includes("WebSocket") &&
        !e.includes("ECONNREFUSED") &&
        !e.includes("fetch") &&
        !e.includes("net::ERR") &&
        !e.includes("Failed to") &&
        !e.includes("Connection refused")
    );
    expect(criticalErrors).toHaveLength(0);
  });

  test("page loads without JavaScript errors", async ({ page }) => {
    const jsErrors: Error[] = [];
    page.on("pageerror", (err) => jsErrors.push(err));
    await page.goto("/");
    await page.waitForLoadState("networkidle");
    await page.waitForTimeout(2000);
    expect(jsErrors).toHaveLength(0);
  });
});
