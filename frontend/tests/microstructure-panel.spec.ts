import { test, expect } from "@playwright/test";

// ── MicrostructurePanel E2E Tests ─────────────────────────────────────────────

test.describe("MicrostructurePanel", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await page.waitForLoadState("networkidle");
    // The microstructure tab needs to be clicked to show the panel
    const microTab = page.locator("button:has-text('microstructure')");
    await microTab.click();
    await page.waitForTimeout(1000);
  });

  test("renders all 4 sub-panels", async ({ page }) => {
    const orderbookLabel = page.locator("text=Orderbook Imbalance").first();
    const deltaLabel = page.locator("text=Delta / CVD").first();
    const spoofingLabel = page.locator("text=Spoofing Detection").first();
    const cascadeLabel = page.locator("text=Liquidation Cascade Risk").first();

    await expect(orderbookLabel).toBeVisible({ timeout: 5000 });
    await expect(deltaLabel).toBeVisible({ timeout: 5000 });
    await expect(spoofingLabel).toBeVisible({ timeout: 5000 });
    await expect(cascadeLabel).toBeVisible({ timeout: 5000 });
  });

  test("displays orderbook bid and ask volumes", async ({ page }) => {
    const bidsText = page.locator("text=/Bids: \\d+/").first();
    const asksText = page.locator("text=/Asks: \\d+/").first();
    await expect(bidsText).toBeVisible({ timeout: 5000 });
    await expect(asksText).toBeVisible({ timeout: 5000 });
  });

  test("displays CVD value", async ({ page }) => {
    const cvdText = page.locator("text=/CVD: /").first();
    await expect(cvdText).toBeVisible({ timeout: 5000 });
  });

  test("displays spoofing detection status", async ({ page }) => {
    const statusText = page.locator("text=/(ALERT: Spoofing|Clean)/").first();
    await expect(statusText).toBeVisible({ timeout: 5000 });
  });

  test("displays cascade risk severity", async ({ page }) => {
    const severityText = page.locator("text=/(CRITICAL|HIGH|ELEVATED|LOW|MINIMAL|NONE)/").first();
    await expect(severityText).toBeVisible({ timeout: 5000 });
  });

  test("displays cascade risk score", async ({ page }) => {
    const scoreText = page.locator("text=/Score: \\d+%/").first();
    await expect(scoreText).toBeVisible({ timeout: 5000 });
  });

  test("no JavaScript errors on microstructure tab", async ({ page }) => {
    const jsErrors: Error[] = [];
    page.on("pageerror", (err) => jsErrors.push(err));
    const microTab = page.locator("button:has-text('microstructure')");
    await microTab.click();
    await page.waitForTimeout(2000);
    expect(jsErrors).toHaveLength(0);
  });
});
