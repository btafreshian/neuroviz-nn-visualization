import { expect, test } from "@playwright/test"

test("trains, traces a sample, replays history, and resets cleanly", async ({ page }) => {
  const browserErrors: string[] = []
  page.on("console", (message) => {
    if (message.type() === "error") browserErrors.push(message.text())
  })
  page.on("pageerror", (error) => browserErrors.push(error.message))

  await page.goto("/")
  await expect(page.getByRole("heading", { name: "Neural Net Playground" })).toBeVisible()
  await expect(page.getByText("0/100", { exact: true }).first()).toBeVisible()
  await expect(page.getByText("Epoch replay", { exact: true })).toHaveCount(0)

  const boundary = page.getByLabel(/Decision boundary for/)
  await boundary.focus()
  await boundary.press("ArrowRight")
  await expect(page.getByText(/Sample #\d+ trace/)).toBeVisible()
  await expect(page.getByText(/Tracing sample #/)).toBeVisible()

  await page.getByRole("button", { name: "Train" }).click()
  await expect(page.getByText(/initializing/i)).toBeVisible()
  await expect.poll(async () => {
    const match = (await page.locator("body").innerText()).match(/epoch\s*(\d+)\/100/i)
    return Number(match?.[1] ?? 0)
  }).toBeGreaterThanOrEqual(3)

  await page.getByRole("button", { name: "Pause" }).click()
  await expect(page.getByText("Epoch replay", { exact: true })).toBeVisible()

  const replaySlider = page.getByRole("slider", { name: "Preview training epoch" })
  await replaySlider.focus()
  await replaySlider.press("Home")
  await expect(page.getByText(/Previewing epoch 0/)).toBeVisible()
  await expect(page.getByText("Replay epoch 0", { exact: true })).toBeVisible()

  await page.getByRole("button", { name: "Return to live" }).click()
  await expect(page.getByText(/Live epoch [1-9]\d*/)).toBeVisible()

  await page.getByRole("button", { name: "Reset", exact: true }).click()
  await expect(page.getByText("0/100", { exact: true }).first()).toBeVisible()
  await expect(page.getByText("Epoch replay", { exact: true })).toBeVisible()

  await page.reload()
  await expect(page.getByText("Epoch replay", { exact: true })).toHaveCount(0)
  await expect(page.getByText(/Sample #\d+ trace/)).toHaveCount(0)
  expect(browserErrors).toEqual([])
})

test("fits intermediate and mobile viewports without horizontal overflow", async ({ page }) => {
  for (const viewport of [{ width: 1024, height: 900 }, { width: 390, height: 844 }]) {
    await page.setViewportSize(viewport)
    await page.goto("/")
    const dimensions = await page.evaluate(() => ({
      viewportWidth: window.innerWidth,
      documentWidth: document.documentElement.scrollWidth,
    }))
    expect(dimensions.documentWidth).toBeLessThanOrEqual(dimensions.viewportWidth + 1)
  }
})
