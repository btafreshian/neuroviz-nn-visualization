import type { Dataset } from "@/lib/types/neural-network"

// XOR dataset - classic non-linearly separable problem
export const xorDataset: Dataset = {
  name: "XOR",
  features: [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
  ],
  labels: [0, 1, 1, 0],
}

// Two moons dataset - curved decision boundary
export const moonsDataset: Dataset = {
  name: "Moons",
  features: [],
  labels: [],
}

// Spiral dataset - complex non-linear pattern
export const spiralDataset: Dataset = {
  name: "Spiral",
  features: [],
  labels: [],
}

// Generate moons dataset
function generateMoons(nSamples = 200, noise = 0.1): Dataset {
  const features: number[][] = []
  const labels: number[] = []

  const samplesPerMoon = Math.floor(nSamples / 2)

  for (let i = 0; i < samplesPerMoon; i++) {
    // First moon
    const t = (i / samplesPerMoon) * Math.PI
    const x1 = Math.cos(t) + noise * (Math.random() - 0.5)
    const y1 = Math.sin(t) + noise * (Math.random() - 0.5)
    features.push([x1, y1])
    labels.push(0)

    // Second moon
    const x2 = 1 - Math.cos(t) + noise * (Math.random() - 0.5)
    const y2 = 0.5 - Math.sin(t) + noise * (Math.random() - 0.5)
    features.push([x2, y2])
    labels.push(1)
  }

  return { name: "Moons", features, labels }
}

// Generate spiral dataset
function generateSpiral(nSamples = 200): Dataset {
  const features: number[][] = []
  const labels: number[] = []

  const samplesPerClass = Math.floor(nSamples / 2)

  for (let i = 0; i < samplesPerClass; i++) {
    const r = (i / samplesPerClass) * 5
    const t = ((1.75 * i) / samplesPerClass) * 2 * Math.PI

    // Class 0
    const x1 = r * Math.sin(t)
    const y1 = r * Math.cos(t)
    features.push([x1, y1])
    labels.push(0)

    // Class 1
    const x2 = r * Math.sin(t + Math.PI)
    const y2 = r * Math.cos(t + Math.PI)
    features.push([x2, y2])
    labels.push(1)
  }

  return { name: "Spiral", features, labels }
}

// Initialize generated datasets
moonsDataset.features = generateMoons().features
moonsDataset.labels = generateMoons().labels

spiralDataset.features = generateSpiral().features
spiralDataset.labels = generateSpiral().labels

export const builtinDatasets = {
  xor: xorDataset,
  moons: moonsDataset,
  spiral: spiralDataset,
}

export const BUILTIN_DATASETS = builtinDatasets
