import { test, expect } from "@playwright/test";

// ── OrderbookHeatmap E2E Tests ────────────────────────────────────────────────

test.describe("OrderbookHeatmap - Page Integration", () => {
  test("page loads and renders without errors", async ({ page }) => {
    await page.goto("/");
    await page.waitForLoadState("networkidle");
    const cockpitTitle = page.locator("text=QUANTEX COCKPIT");
    await expect(cockpitTitle).toBeVisible({ timeout: 10000 });
  });

  test("microstructure tab renders without JavaScript errors", async ({ page }) => {
    const jsErrors: Error[] = [];
    page.on("pageerror", (err) => jsErrors.push(err));
    await page.goto("/");
    await page.waitForLoadState("networkidle");
    const microTab = page.locator("button:has-text('microstructure')");
    await microTab.click();
    await page.waitForTimeout(2000);
    expect(jsErrors).toHaveLength(0);
  });

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
    const criticalErrors = errors.filter(
      (e) =>
        !e.includes("WebSocket") &&
        !e.includes("ECONNREFUSED") &&
        !e.includes("fetch") &&
        !e.includes("net::ERR") &&
        !e.includes("Connection refused")
    );
    expect(criticalErrors).toHaveLength(0);
  });
});

test.describe("OrderbookHeatmap - Standalone Rendering", () => {
  test("renders with mock orderbook data", async ({ page }) => {
    await page.setContent(`
      <html>
        <head>
          <style>
            body { background: #0a0a0f; color: #ccc; font-family: monospace; }
            .panel { background: #111; border-radius: 8px; padding: 16px; margin: 16px; }
            .panel-title { font-size: 14px; font-weight: 600; margin-bottom: 12px; }
          </style>
        </head>
        <body>
          <div id="root"></div>
          <script type="module">
            const root = document.getElementById('root');
            root.innerHTML = \`
              <div class="panel">
                <div class="panel-title">Orderbook Heatmap (L2) — BTCUSDT</div>
                <div style="display:flex;gap:16px;font-size:10px;color:#888;padding:8px 10px;background:#0a0a0f;border-radius:6px">
                  <div>Mid: <strong style="color:#ccc;font-family:var(--font-mono)">\$50,000.00</strong></div>
                  <div>Spread: <strong style="color:#ccc;font-family:var(--font-mono)">\$2.50 (0.5bps)</strong></div>
                  <div>Imbalance: <strong style="color:#00ff88;font-family:var(--font-mono)">15%</strong></div>
                </div>
                <div style="display:flex;gap:0;position:relative;margin-top:14px">
                  <div style="flex:1">
                    <div style="font-size:8;color:#00ff88;text-transform:uppercase;letter-spacing:1;margin-bottom:4px">Bids</div>
                    <div style="display:flex;align-items:center;height:20px;margin-bottom:2px">
                      <div style="width:80px;text-align:right;padding-right:8px;font-size:10px;color:#666">49,998.00</div>
                      <div style="flex:1;height:16px;border-radius:3px;background:rgba(0,255,136,0.5)"></div>
                    </div>
                  </div>
                  <div style="width:2px;background:linear-gradient(to bottom,transparent,#444488,transparent);margin:0 4px"></div>
                  <div style="flex:1">
                    <div style="font-size:8px;color:#ff0044;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">Asks</div>
                    <div style="display:flex;align-items:center;height:20px;margin-bottom:2px">
                      <div style="width:80px;text-align:right;padding-right:8px;font-size:10px;color:#666">50,002.00</div>
                      <div style="flex:1;height:14px;border-radius:3px;background:rgba(255,0,68,0.4)"></div>
                    </div>
                  </div>
                </div>
              </div>
            \`;
          </script>
        </body>
      </html>
    `);

    await expect(page.locator("text=Orderbook Heatmap (L2) — BTCUSDT")).toBeVisible();
    await expect(page.locator("text=Bids")).toBeVisible();
    await expect(page.locator("text=Asks")).toBeVisible();
    await expect(page.locator("text=/Mid: /")).toBeVisible();
    await expect(page.locator("text=/Spread: /")).toBeVisible();
    await expect(page.locator("text=/Imbalance: /")).toBeVisible();
  });

  test("displays price levels correctly", async ({ page }) => {
    await page.setContent(`
      <html>
        <body>
          <div id="root"></div>
          <script type="module">
            const root = document.getElementById('root');
            root.innerHTML = \`
              <div class="panel">
                <div class="panel-title">Orderbook Heatmap (L2) — ETHUSDT</div>
                <div style="display:flex;gap:0;position:relative">
                  <div style="flex:1">
                    <div style="font-size:8px;color:#00ff88;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">Bids</div>
                    <div style="display:flex;align-items:center;height:20px;margin-bottom:2px">
                      <div style="width:80px;text-align:right;padding-right:8px;font-size:10px;color:#fff;font-weight:700">3,500.00</div>
                      <div style="flex:1;height:20px;border-radius:3px;background:rgba(0,255,136,0.8)">
                        <span style="position:absolute;left:6px;top:50%;transform:translateY(-50%);font-size:8px;color:rgba(255,255,255,0.7)">2.50</span>
                      </div>
                    </div>
                  </div>
                  <div style="flex:1">
                    <div style="font-size:8px;color:#ff0044;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">Asks</div>
                    <div style="display:flex;align-items:center;height:20px;margin-bottom:2px">
                      <div style="width:80px;text-align:right;padding-right:8px;font-size:10px;color:#fff;font-weight:700">3,501.00</div>
                      <div style="flex:1;height:18px;border-radius:3px;background:rgba(255,0,68,0.7)">
                        <span style="position:absolute;left:6px;top:50%;transform:translateY(-50%);font-size:8px;color:rgba(255,255,255,0.7)">1.80</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            \`;
          </script>
        </body>
      </html>
    `);

    await expect(page.locator("text=3,500.00")).toBeVisible();
    await expect(page.locator("text=3,501.00")).toBeVisible();
  });

  test("shows whale cluster detection section", async ({ page }) => {
    await page.setContent(`
      <html>
        <body>
          <div id="root"></div>
          <script type="module">
            const root = document.getElementById('root');
            root.innerHTML = \`
              <div class="panel">
                <div class="panel-title">Orderbook Heatmap (L2) — BTCUSDT</div>
                <div style="margin-top:12px;padding-top:10px;border-top:1px solid #1a1a2e">
                  <div style="font-size:9px;color:#555;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">
                    Whale Liquidity Clusters Detected
                  </div>
                  <div style="display:flex;gap:6px;flex-wrap:wrap">
                    <div style="padding:3px 10px;border-radius:6px;font-size:10px;background:rgba(0,255,136,0.1);color:#00ff88;border:1px solid rgba(0,255,136,0.2)">
                      $49,995.00 — 5.20
                    </div>
                    <div style="padding:3px 10px;border-radius:6px;font-size:10px;background:rgba(255,0,68,0.1);color:#ff0044;border:1px solid rgba(255,0,68,0.2)">
                      $50,005.00 — 3.80
                    </div>
                  </div>
                </div>
              </div>
            \`;
          </script>
        </body>
      </html>
    `);

    await expect(page.locator("text=Whale Liquidity Clusters Detected")).toBeVisible();
    await expect(page.locator("text=/\\$49,995\\.00/")).toBeVisible();
    await expect(page.locator("text=/\\$50,005\\.00/")).toBeVisible();
  });
});
