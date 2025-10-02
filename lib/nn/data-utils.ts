// Utilities for data preprocessing and batch handling

export interface DataSplit {
  trainFeatures: number[][]
  trainLabels: number[]
  valFeatures: number[][]
  valLabels: number[]
}

// Split data into training and validation sets
export function splitData(
  features: number[][],
  labels: number[],
  trainRatio = 0.8,
  shuffle = true,
  seed?: number,
): DataSplit {
  const indices = Array.from({ length: features.length }, (_, i) => i)

  if (shuffle) {
    // Simple seeded shuffle
    const rng = seed !== undefined ? createSeededRandom(seed) : Math.random
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1))
      ;[indices[i], indices[j]] = [indices[j], indices[i]]
    }
  }

  const trainSize = Math.floor(features.length * trainRatio)
  const trainIndices = indices.slice(0, trainSize)
  const valIndices = indices.slice(trainSize)

  return {
    trainFeatures: trainIndices.map((i) => features[i]),
    trainLabels: trainIndices.map((i) => labels[i]),
    valFeatures: valIndices.map((i) => features[i]),
    valLabels: valIndices.map((i) => labels[i]),
  }
}

// Create batches for mini-batch training
export function createBatches<T>(data: T[], batchSize: number): T[][] {
  const batches: T[][] = []
  for (let i = 0; i < data.length; i += batchSize) {
    batches.push(data.slice(i, i + batchSize))
  }
  return batches
}

// One-hot encode labels for classification
export function oneHotEncode(labels: number[], numClasses?: number): number[][] {
  const maxLabel = numClasses ?? Math.max(...labels) + 1
  return labels.map((label) => {
    const encoded = new Array(maxLabel).fill(0)
    encoded[label] = 1
    return encoded
  })
}

// Normalize features to [0, 1] range
export function normalizeFeatures(features: number[][]): { normalized: number[][]; min: number[]; max: number[] } {
  if (features.length === 0) return { normalized: [], min: [], max: [] }

  const numFeatures = features[0].length
  const min = new Array(numFeatures).fill(Number.POSITIVE_INFINITY)
  const max = new Array(numFeatures).fill(Number.NEGATIVE_INFINITY)

  // Find min and max for each feature
  for (const sample of features) {
    for (let i = 0; i < numFeatures; i++) {
      min[i] = Math.min(min[i], sample[i])
      max[i] = Math.max(max[i], sample[i])
    }
  }

  // Normalize
  const normalized = features.map((sample) =>
    sample.map((value, i) => {
      const range = max[i] - min[i]
      return range === 0 ? 0 : (value - min[i]) / range
    }),
  )

  return { normalized, min, max }
}

// Standardize features (zero mean, unit variance)
export function standardizeFeatures(features: number[][]): { standardized: number[][]; mean: number[]; std: number[] } {
  if (features.length === 0) return { standardized: [], mean: [], std: [] }

  const numFeatures = features[0].length
  const mean = new Array(numFeatures).fill(0)
  const std = new Array(numFeatures).fill(0)

  // Calculate mean
  for (const sample of features) {
    for (let i = 0; i < numFeatures; i++) {
      mean[i] += sample[i]
    }
  }
  for (let i = 0; i < numFeatures; i++) {
    mean[i] /= features.length
  }

  // Calculate standard deviation
  for (const sample of features) {
    for (let i = 0; i < numFeatures; i++) {
      const diff = sample[i] - mean[i]
      std[i] += diff * diff
    }
  }
  for (let i = 0; i < numFeatures; i++) {
    std[i] = Math.sqrt(std[i] / features.length)
  }

  // Standardize
  const standardized = features.map((sample) =>
    sample.map((value, i) => {
      return std[i] === 0 ? 0 : (value - mean[i]) / std[i]
    }),
  )

  return { standardized, mean, std }
}

// Simple seeded random number generator
function createSeededRandom(seed: number): () => number {
  let state = seed
  return () => {
    state = (state * 1664525 + 1013904223) % 2 ** 32
    return state / 2 ** 32
  }
}

// Shuffle array in place
export function shuffleArray<T>(array: T[], rng: () => number = Math.random): void {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    ;[array[i], array[j]] = [array[j], array[i]]
  }
}

// Generate synthetic 2D datasets for testing
export function generateCircularData(nSamples = 200, noise = 0.1): { features: number[][]; labels: number[] } {
  const features: number[][] = []
  const labels: number[] = []

  for (let i = 0; i < nSamples; i++) {
    const angle = (i / nSamples) * 2 * Math.PI
    const radius = 0.5 + 0.3 * Math.random()
    const x = radius * Math.cos(angle) + noise * (Math.random() - 0.5)
    const y = radius * Math.sin(angle) + noise * (Math.random() - 0.5)

    features.push([x, y])
    labels.push(radius > 0.6 ? 1 : 0)
  }

  return { features, labels }
}
