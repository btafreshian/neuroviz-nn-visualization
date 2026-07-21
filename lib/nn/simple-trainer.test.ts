import { describe, expect, it } from "vitest"
import { DEFAULT_DATASET_CONFIG, generateDataset } from "@/lib/datasets/builtin-datasets"
import { DEFAULT_NETWORK, DEFAULT_TRAINING_CONFIG } from "@/lib/training/playground-defaults"
import {
  createDatasetSplit,
  createOptimizerBuffers,
  createZeroNetworkParameters,
  evaluateNetwork,
  initializeNetworkParameters,
  predictProbability,
  traceForwardPass,
  trainEpoch,
} from "./simple-trainer"

const DEFAULT_SETUP = {
  network: DEFAULT_NETWORK,
  dataset: DEFAULT_DATASET_CONFIG,
  training: DEFAULT_TRAINING_CONFIG,
}

describe("educational trainer", () => {
  it("represents the idle epoch-0 model with genuinely zero parameters", () => {
    const preset = DEFAULT_SETUP
    const dataset = generateDataset(preset.dataset)
    const parameters = createZeroNetworkParameters(preset.network.layers)
    const split = createDatasetSplit(dataset, preset.training.trainTestSplit)
    const baseline = evaluateNetwork(
      preset.network.layers,
      parameters.weights,
      parameters.biases,
      dataset,
      split.testIndices,
      preset.training.loss,
    )

    expect(parameters.weights.flatMap((matrix) => matrix.weights.flat()).every((value) => value === 0)).toBe(true)
    expect(parameters.biases.flatMap((vector) => vector.biases).every((value) => value === 0)).toBe(true)
    expect(baseline.loss).toBeCloseTo(Math.log(2), 10)
    expect(baseline.accuracy).toBeCloseTo(0.5, 10)
    expect(baseline.meanConfidence).toBeCloseTo(0.5, 10)
  })

  it("makes the default configuration improve gradually from near chance", () => {
    const preset = DEFAULT_SETUP
    const dataset = generateDataset(preset.dataset)
    const parameters = initializeNetworkParameters(preset.network.layers, 1337 + preset.dataset.seed)
    const split = createDatasetSplit(dataset, preset.training.trainTestSplit)
    const optimizer = createOptimizerBuffers(parameters.weights, parameters.biases)
    const baseline = evaluateNetwork(
      preset.network.layers,
      parameters.weights,
      parameters.biases,
      dataset,
      split.testIndices,
      preset.training.loss,
    )

    for (let epoch = 1; epoch <= 80; epoch += 1) {
      trainEpoch(
        preset.network.layers,
        parameters.weights,
        parameters.biases,
        dataset,
        split.trainIndices,
        preset.training,
        optimizer,
        epoch,
      )
    }

    const final = evaluateNetwork(
      preset.network.layers,
      parameters.weights,
      parameters.biases,
      dataset,
      split.testIndices,
      preset.training.loss,
    )
    expect(baseline.accuracy).toBeGreaterThanOrEqual(0.4)
    expect(baseline.accuracy).toBeLessThanOrEqual(0.65)
    expect(final.accuracy).toBeGreaterThanOrEqual(0.85)
    expect(final.loss).toBeLessThan(baseline.loss * 0.65)
  })

  it("returns an exact forward trace with additive edge contributions", () => {
    const preset = DEFAULT_SETUP
    const dataset = generateDataset(preset.dataset)
    const parameters = initializeNetworkParameters(preset.network.layers, 1337 + preset.dataset.seed)
    const features = dataset.features[0]
    const target = Number(dataset.labels[0])
    const trace = traceForwardPass(
      preset.network.layers,
      parameters.weights,
      parameters.biases,
      features,
      target,
      preset.training.loss,
    )

    expect(trace.outputProbability).toBeCloseTo(
      predictProbability(preset.network.layers, parameters.weights, parameters.biases, features),
      12,
    )
    for (let layerIndex = 1; layerIndex < trace.layers.length; layerIndex += 1) {
      for (let neuronIndex = 0; neuronIndex < trace.layers[layerIndex].preActivations.length; neuronIndex += 1) {
        const contributionTotal = trace.edgeContributions[layerIndex - 1]
          .reduce((total, row) => total + row[neuronIndex], 0)
        expect(contributionTotal + trace.layers[layerIndex].biases[neuronIndex]).toBeCloseTo(
          trace.layers[layerIndex].preActivations[neuronIndex],
          12,
        )
      }
    }
    expect(trace.sampleLoss).not.toBeNull()
    expect(Number.isFinite(trace.sampleLoss ?? Number.NaN)).toBe(true)
  })

  it("reports clipping, decay, and update diagnostics from real gradients", () => {
    const preset = DEFAULT_SETUP
    const config = {
      ...preset.training,
      learningRate: 0.01,
      learningRateDecay: 0.01,
      gradientClip: 0.001,
      weightDecay: 0.01,
    }
    const dataset = generateDataset(preset.dataset)
    const parameters = initializeNetworkParameters(preset.network.layers, 1337 + preset.dataset.seed)
    const split = createDatasetSplit(dataset, config.trainTestSplit)
    const optimizer = createOptimizerBuffers(parameters.weights, parameters.biases)
    const diagnostics = trainEpoch(
      preset.network.layers,
      parameters.weights,
      parameters.biases,
      dataset,
      split.trainIndices,
      config,
      optimizer,
      10,
    )

    expect(diagnostics.effectiveLearningRate).toBeCloseTo(0.01 / 1.09, 10)
    expect(diagnostics.clippedFraction).toBeGreaterThan(0)
    expect(diagnostics.appliedGradientNorm).toBeLessThanOrEqual(diagnostics.rawGradientNorm)
    expect(diagnostics.regularizationPenalty).toBeGreaterThan(0)
    expect(diagnostics.updateNorm).toBeGreaterThan(0)
  })
})
