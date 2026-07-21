import type { AxisStatistics, Dataset } from "@/lib/context/network-context"

export type DatasetType = "xor" | "moons" | "circles" | "spiral" | "blobs"

export interface DatasetConfig {
  type: DatasetType
  sampleCount: number
  noise: number
  seed: number
  normalize: boolean
}

interface DatasetDefinition {
  name: string
  description: string
  defaultNoise: number
}

export const DATASET_DEFINITIONS: Record<DatasetType, DatasetDefinition> = {
  xor: {
    name: "XOR quadrants",
    description: "Four alternating quadrants; a classic non-linear classification task.",
    defaultNoise: 0.08,
  },
  moons: {
    name: "Two moons",
    description: "Two interleaving arcs that require a curved decision boundary.",
    defaultNoise: 0.1,
  },
  circles: {
    name: "Concentric circles",
    description: "Nested classes that reveal learning gradually across many epochs.",
    defaultNoise: 0.08,
  },
  spiral: {
    name: "Two spirals",
    description: "A capacity challenge with a winding boundary and long training horizon.",
    defaultNoise: 0.04,
  },
  blobs: {
    name: "Gaussian blobs",
    description: "Two mostly separable clusters for a quick first training lesson.",
    defaultNoise: 0.12,
  },
}

export const DEFAULT_DATASET_CONFIG: DatasetConfig = {
  type: "circles",
  sampleCount: 240,
  noise: DATASET_DEFINITIONS.circles.defaultNoise,
  seed: 42,
  normalize: true,
}

export function generateDataset(config: DatasetConfig): Dataset & { labels: number[] } {
  const sampleCount = Math.max(20, Math.min(1000, Math.round(config.sampleCount)))
  const noise = Math.max(0, Math.min(0.5, config.noise))
  const random = createSeededRandom(config.seed)
  const gaussian = createGaussianRandom(random)
  const rawFeatures: number[][] = []
  const labels: number[] = []

  for (let index = 0; index < sampleCount; index += 1) {
    const label = index % 2
    const classIndex = Math.floor(index / 2)
    const classCount = Math.ceil(sampleCount / 2)
    let x = 0
    let y = 0

    switch (config.type) {
      case "xor": {
        const baseX = random() * 2 - 1
        const baseY = random() * 2 - 1
        x = baseX + gaussian() * noise
        y = baseY + gaussian() * noise
        labels.push(baseX * baseY >= 0 ? 1 : 0)
        rawFeatures.push([x, y])
        continue
      }
      case "moons": {
        const angle = random() * Math.PI
        if (label === 0) {
          x = Math.cos(angle)
          y = Math.sin(angle)
        } else {
          x = 1 - Math.cos(angle)
          y = 0.45 - Math.sin(angle)
        }
        break
      }
      case "circles": {
        const angle = random() * Math.PI * 2
        const radius = label === 0 ? 0.48 : 1
        x = radius * Math.cos(angle)
        y = radius * Math.sin(angle)
        break
      }
      case "spiral": {
        const progress = classIndex / Math.max(1, classCount - 1)
        const radius = 0.12 + progress * 0.95
        const angle = progress * Math.PI * 4.4 + label * Math.PI
        x = radius * Math.cos(angle)
        y = radius * Math.sin(angle)
        break
      }
      case "blobs": {
        const center = label === 0 ? -0.65 : 0.65
        const spread = 0.18 + noise * 0.7
        x = center + gaussian() * spread
        y = center * 0.65 + gaussian() * spread
        labels.push(label)
        rawFeatures.push([x, y])
        continue
      }
    }

    rawFeatures.push([x + gaussian() * noise, y + gaussian() * noise])
    labels.push(label)
  }

  const features = config.normalize ? standardizeFeatures(rawFeatures) : rawFeatures.map((feature) => [...feature])
  const definition = DATASET_DEFINITIONS[config.type]

  return {
    id: config.type,
    name: definition.name,
    description: definition.description,
    features,
    labels,
    inputSize: 2,
    outputSize: 1,
    noise,
    seed: config.seed,
    normalized: config.normalize,
    featureStats: {
      raw: calculateFeatureStatistics(rawFeatures),
      processed: calculateFeatureStatistics(features),
    },
  }
}

export function createSeededRandom(seed: number): () => number {
  let state = Math.trunc(seed) >>> 0
  return () => {
    state += 0x6d2b79f5
    let value = state
    value = Math.imul(value ^ (value >>> 15), value | 1)
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61)
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296
  }
}

function createGaussianRandom(random: () => number): () => number {
  let spare: number | null = null

  return () => {
    if (spare !== null) {
      const value = spare
      spare = null
      return value
    }

    const first = Math.max(Number.EPSILON, random())
    const second = random()
    const magnitude = Math.sqrt(-2 * Math.log(first))
    spare = magnitude * Math.sin(2 * Math.PI * second)
    return magnitude * Math.cos(2 * Math.PI * second)
  }
}

function standardizeFeatures(features: number[][]): number[][] {
  const statistics = calculateFeatureStatistics(features)
  return features.map((feature) =>
    feature.map((value, dimension) => (value - statistics[dimension].mean) / (statistics[dimension].standardDeviation || 1)),
  )
}

function calculateFeatureStatistics(features: number[][]): AxisStatistics[] {
  if (features.length === 0) return []
  const dimensionCount = features[0].length

  return Array.from({ length: dimensionCount }, (_, dimension) => {
    const values = features.map((feature) => feature[dimension])
    const mean = values.reduce((total, value) => total + value, 0) / values.length
    const variance = values.reduce((total, value) => total + (value - mean) ** 2, 0) / values.length
    return {
      mean,
      standardDeviation: Math.sqrt(variance),
      minimum: Math.min(...values),
      maximum: Math.max(...values),
    }
  })
}
