import { describe, expect, it } from "vitest"
import { generateDataset, type DatasetConfig } from "./builtin-datasets"
import { createDatasetSplit } from "@/lib/nn/simple-trainer"

const config: DatasetConfig = {
  type: "circles",
  sampleCount: 241,
  noise: 0.08,
  seed: 42,
  normalize: true,
}

describe("generated datasets", () => {
  it("is deterministic and preserves an odd requested sample count", () => {
    const first = generateDataset(config)
    const second = generateDataset(config)

    expect(first.features).toEqual(second.features)
    expect(first.labels).toEqual(second.labels)
    expect(first.features).toHaveLength(241)
    expect(first.labels).toHaveLength(241)
    const classOneCount = first.labels.filter((label) => label === 1).length
    expect(Math.abs(classOneCount - (first.labels.length - classOneCount))).toBeLessThanOrEqual(1)
  })

  it("reports standardized feature statistics", () => {
    const dataset = generateDataset(config)

    for (const axis of dataset.featureStats?.processed ?? []) {
      expect(Math.abs(axis.mean)).toBeLessThan(1e-10)
      expect(axis.standardDeviation).toBeCloseTo(1, 8)
    }
  })

  it("creates exact, stratified train and test indices", () => {
    const dataset = generateDataset(config)
    const split = createDatasetSplit(dataset, 0.81)

    expect(split.trainIndices.length + split.testIndices.length).toBe(241)
    expect(new Set([...split.trainIndices, ...split.testIndices]).size).toBe(241)
    for (const indices of [split.trainIndices, split.testIndices]) {
      const labels = indices.map((index) => dataset.labels[index])
      expect(labels).toContain(0)
      expect(labels).toContain(1)
    }
  })
})
