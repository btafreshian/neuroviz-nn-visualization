import type {
  BiasVector,
  Dataset,
  NNLayer,
  TrainingConfig,
  WeightMatrix,
} from "@/lib/context/network-context"
import { createSeededRandom } from "@/lib/datasets/builtin-datasets"

export interface DatasetSplit {
  trainIndices: number[]
  testIndices: number[]
}

export interface OptimizerBuffers {
  step: number
  velocityWeights: number[][][]
  velocityBiases: number[][]
  firstMomentWeights: number[][][]
  firstMomentBiases: number[][]
  secondMomentWeights: number[][][]
  secondMomentBiases: number[][]
}

export interface EvaluationMetrics {
  loss: number
  accuracy: number
  meanConfidence: number
}

export interface ForwardTraceLayer {
  layerIndex: number
  activationType: NNLayer["activation"]
  weightedInputs: number[]
  biases: number[]
  preActivations: number[]
  activations: number[]
}

export interface ForwardTrace {
  input: number[]
  layers: ForwardTraceLayer[]
  edgeContributions: number[][][]
  outputProbability: number
  prediction: 0 | 1
  target: number | null
  sampleLoss: number | null
}

export interface EpochDiagnostics {
  effectiveLearningRate: number
  rawGradientNorm: number
  appliedGradientNorm: number
  updateNorm: number
  weightNorm: number
  regularizationPenalty: number
  clippedFraction: number
  batchLossVariance: number
  weightDeltas: number[][][]
}

export function initializeNetworkParameters(layers: NNLayer[], seed = 1337): {
  weights: WeightMatrix[]
  biases: BiasVector[]
} {
  const random = createSeededRandom(seed)
  const weights: WeightMatrix[] = []
  const biases: BiasVector[] = []

  for (let layerIndex = 0; layerIndex < layers.length - 1; layerIndex += 1) {
    const fromSize = layers[layerIndex].size
    const toSize = layers[layerIndex + 1].size
    const nextActivation = layers[layerIndex + 1].activation
    const limit = nextActivation === "relu" ? Math.sqrt(6 / fromSize) : Math.sqrt(6 / (fromSize + toSize))
    const matrix = Array.from({ length: fromSize }, () =>
      Array.from({ length: toSize }, () => (random() * 2 - 1) * limit),
    )
    weights.push({ weights: matrix, fromLayerIndex: layerIndex, toLayerIndex: layerIndex + 1 })
  }

  for (let layerIndex = 1; layerIndex < layers.length; layerIndex += 1) {
    biases.push({ biases: Array.from({ length: layers[layerIndex].size }, () => 0), layerIndex })
  }

  return { weights, biases }
}

export function createZeroNetworkParameters(layers: NNLayer[]): {
  weights: WeightMatrix[]
  biases: BiasVector[]
} {
  const weights = layers.slice(0, -1).map((layer, layerIndex) => ({
    fromLayerIndex: layerIndex,
    toLayerIndex: layerIndex + 1,
    weights: Array.from({ length: layer.size }, () =>
      Array.from({ length: layers[layerIndex + 1].size }, () => 0),
    ),
  }))
  const biases = layers.slice(1).map((layer, layerIndex) => ({
    layerIndex: layerIndex + 1,
    biases: Array.from({ length: layer.size }, () => 0),
  }))

  return { weights, biases }
}

export function createDatasetSplit(dataset: Dataset, trainRatio: number): DatasetSplit {
  const labels = dataset.labels.map(toNumericLabel)
  const firstClass: number[] = []
  const secondClass: number[] = []
  labels.forEach((label, index) => (label >= 0.5 ? secondClass : firstClass).push(index))

  const random = createSeededRandom((dataset.seed ?? 42) + 7919)
  shuffleInPlace(firstClass, random)
  shuffleInPlace(secondClass, random)

  const splitClass = (indices: number[]) => {
    const trainCount = Math.max(1, Math.min(indices.length - 1, Math.round(indices.length * trainRatio)))
    return { train: indices.slice(0, trainCount), test: indices.slice(trainCount) }
  }
  const firstSplit = splitClass(firstClass)
  const secondSplit = splitClass(secondClass)
  const trainIndices = [...firstSplit.train, ...secondSplit.train]
  const testIndices = [...firstSplit.test, ...secondSplit.test]
  shuffleInPlace(trainIndices, random)
  shuffleInPlace(testIndices, random)

  return { trainIndices, testIndices }
}

export function createOptimizerBuffers(weights: WeightMatrix[], biases: BiasVector[]): OptimizerBuffers {
  const zeroWeights = () => weights.map((matrix) => matrix.weights.map((row) => row.map(() => 0)))
  const zeroBiases = () => biases.map((vector) => vector.biases.map(() => 0))

  return {
    step: 0,
    velocityWeights: zeroWeights(),
    velocityBiases: zeroBiases(),
    firstMomentWeights: zeroWeights(),
    firstMomentBiases: zeroBiases(),
    secondMomentWeights: zeroWeights(),
    secondMomentBiases: zeroBiases(),
  }
}

export function trainEpoch(
  layers: NNLayer[],
  weights: WeightMatrix[],
  biases: BiasVector[],
  dataset: Dataset,
  trainIndices: number[],
  config: TrainingConfig,
  optimizer: OptimizerBuffers,
  epoch: number,
): EpochDiagnostics {
  const orderedIndices = [...trainIndices]
  if (config.shuffle) {
    shuffleInPlace(orderedIndices, createSeededRandom((dataset.seed ?? 42) + epoch * 104729))
  }

  const batchSize = Math.max(1, Math.min(config.batchSize, orderedIndices.length))
  const effectiveLearningRate = config.learningRate / (1 + config.learningRateDecay * Math.max(0, epoch - 1))
  const weightDeltas = weights.map((matrix) => matrix.weights.map((row) => row.map(() => 0)))
  const batchLosses: number[] = []
  let rawGradientSquares = 0
  let appliedGradientSquares = 0
  let updateSquares = 0
  let gradientCount = 0
  let clippedCount = 0

  for (let batchStart = 0; batchStart < orderedIndices.length; batchStart += batchSize) {
    const batchIndices = orderedIndices.slice(batchStart, batchStart + batchSize)
    const weightGradients = weights.map((matrix) => matrix.weights.map((row) => row.map(() => 0)))
    const biasGradients = biases.map((vector) => vector.biases.map(() => 0))
    let batchLoss = 0

    for (const sampleIndex of batchIndices) {
      const target = toNumericLabel(dataset.labels[sampleIndex])
      const pass = forwardPass(layers, weights, biases, dataset.features[sampleIndex])
      const prediction = clampProbability(pass.activations.at(-1)?.[0] ?? 0.5)
      batchLoss += calculateLoss(prediction, target, config.loss)

      const deltas: number[][] = Array.from({ length: layers.length }, () => [])
      const outputLayerIndex = layers.length - 1
      const outputActivation = layers[outputLayerIndex].activation
      const outputDerivative = activationDerivative(outputActivation, pass.preActivations[outputLayerIndex][0])
      deltas[outputLayerIndex] = [
        config.loss === "crossentropy" && outputActivation === "sigmoid"
          ? prediction - target
          : 2 * (prediction - target) * outputDerivative,
      ]

      for (let layerIndex = outputLayerIndex - 1; layerIndex >= 1; layerIndex -= 1) {
        deltas[layerIndex] = Array.from({ length: layers[layerIndex].size }, (_, neuronIndex) => {
          const downstream = deltas[layerIndex + 1].reduce(
            (sum, delta, nextNeuronIndex) =>
              sum + weights[layerIndex].weights[neuronIndex][nextNeuronIndex] * delta,
            0,
          )
          return downstream * activationDerivative(layers[layerIndex].activation, pass.preActivations[layerIndex][neuronIndex])
        })
      }

      for (let matrixIndex = 0; matrixIndex < weights.length; matrixIndex += 1) {
        for (let fromIndex = 0; fromIndex < weights[matrixIndex].weights.length; fromIndex += 1) {
          for (let toIndex = 0; toIndex < weights[matrixIndex].weights[fromIndex].length; toIndex += 1) {
            weightGradients[matrixIndex][fromIndex][toIndex] +=
              pass.activations[matrixIndex][fromIndex] * deltas[matrixIndex + 1][toIndex]
          }
        }
        for (let neuronIndex = 0; neuronIndex < biasGradients[matrixIndex].length; neuronIndex += 1) {
          biasGradients[matrixIndex][neuronIndex] += deltas[matrixIndex + 1][neuronIndex]
        }
      }
    }

    batchLosses.push(batchLoss / batchIndices.length)
    optimizer.step += 1
    const gradientScale = 1 / batchIndices.length

    for (let matrixIndex = 0; matrixIndex < weights.length; matrixIndex += 1) {
      for (let fromIndex = 0; fromIndex < weights[matrixIndex].weights.length; fromIndex += 1) {
        for (let toIndex = 0; toIndex < weights[matrixIndex].weights[fromIndex].length; toIndex += 1) {
          const parameter = weights[matrixIndex].weights[fromIndex][toIndex]
          const rawGradient =
            weightGradients[matrixIndex][fromIndex][toIndex] * gradientScale + config.weightDecay * parameter
          const gradient = clipGradient(rawGradient, config.gradientClip)
          const nextParameter = applyOptimizerUpdate(
            parameter,
            gradient,
            effectiveLearningRate,
            config,
            optimizer,
            "weight",
            matrixIndex,
            fromIndex,
            toIndex,
          )
          const update = nextParameter - parameter
          weights[matrixIndex].weights[fromIndex][toIndex] = nextParameter
          weightDeltas[matrixIndex][fromIndex][toIndex] += update
          rawGradientSquares += rawGradient * rawGradient
          appliedGradientSquares += gradient * gradient
          updateSquares += update * update
          gradientCount += 1
          if (gradient !== rawGradient) clippedCount += 1
        }
      }

      for (let neuronIndex = 0; neuronIndex < biases[matrixIndex].biases.length; neuronIndex += 1) {
        const parameter = biases[matrixIndex].biases[neuronIndex]
        const rawGradient = biasGradients[matrixIndex][neuronIndex] * gradientScale
        const gradient = clipGradient(rawGradient, config.gradientClip)
        const nextParameter = applyOptimizerUpdate(
          parameter,
          gradient,
          effectiveLearningRate,
          config,
          optimizer,
          "bias",
          matrixIndex,
          neuronIndex,
          0,
        )
        biases[matrixIndex].biases[neuronIndex] = nextParameter
        const update = nextParameter - parameter
        rawGradientSquares += rawGradient * rawGradient
        appliedGradientSquares += gradient * gradient
        updateSquares += update * update
        gradientCount += 1
        if (gradient !== rawGradient) clippedCount += 1
      }
    }
  }

  const weightNorm = calculateWeightNorm(weights)
  return {
    effectiveLearningRate,
    rawGradientNorm: Math.sqrt(rawGradientSquares / Math.max(1, gradientCount)),
    appliedGradientNorm: Math.sqrt(appliedGradientSquares / Math.max(1, gradientCount)),
    updateNorm: Math.sqrt(updateSquares),
    weightNorm,
    regularizationPenalty: 0.5 * config.weightDecay * weightNorm * weightNorm,
    clippedFraction: clippedCount / Math.max(1, gradientCount),
    batchLossVariance: calculateVariance(batchLosses),
    weightDeltas,
  }
}

export function evaluateNetwork(
  layers: NNLayer[],
  weights: WeightMatrix[],
  biases: BiasVector[],
  dataset: Dataset,
  indices: number[],
  loss: TrainingConfig["loss"],
): EvaluationMetrics {
  const evaluationIndices = indices.length > 0 ? indices : dataset.features.map((_, index) => index)
  let totalLoss = 0
  let totalConfidence = 0
  let correct = 0

  for (const index of evaluationIndices) {
    const target = toNumericLabel(dataset.labels[index])
    const prediction = predictProbability(layers, weights, biases, dataset.features[index])
    totalLoss += calculateLoss(prediction, target, loss)
    totalConfidence += target >= 0.5 ? prediction : 1 - prediction
    if ((prediction >= 0.5 ? 1 : 0) === (target >= 0.5 ? 1 : 0)) correct += 1
  }

  return {
    loss: totalLoss / evaluationIndices.length,
    accuracy: correct / evaluationIndices.length,
    meanConfidence: totalConfidence / evaluationIndices.length,
  }
}

export function predictProbability(
  layers: NNLayer[],
  weights: WeightMatrix[],
  biases: BiasVector[],
  features: number[],
): number {
  return clampProbability(forwardPass(layers, weights, biases, features).activations.at(-1)?.[0] ?? 0.5)
}

export function traceForwardPass(
  layers: NNLayer[],
  weights: WeightMatrix[],
  biases: BiasVector[],
  features: number[],
  target?: number,
  loss: TrainingConfig["loss"] = "crossentropy",
): ForwardTrace {
  const pass = forwardPass(layers, weights, biases, features)
  const edgeContributions = weights.map((matrix, matrixIndex) =>
    matrix.weights.map((row, sourceIndex) =>
      row.map((weight) => (pass.activations[matrixIndex]?.[sourceIndex] ?? 0) * weight),
    ),
  )
  const traceLayers = layers.map((layer, layerIndex) => {
    const layerBiases = layerIndex === 0
      ? Array.from({ length: layer.size }, () => 0)
      : [...(biases[layerIndex - 1]?.biases ?? Array.from({ length: layer.size }, () => 0))]
    const preActivations = layerIndex === 0
      ? [...(pass.activations[layerIndex] ?? [])]
      : [...(pass.preActivations[layerIndex] ?? [])]
    return {
      layerIndex,
      activationType: layer.activation,
      weightedInputs: preActivations.map((value, neuronIndex) => value - (layerBiases[neuronIndex] ?? 0)),
      biases: layerBiases,
      preActivations,
      activations: [...(pass.activations[layerIndex] ?? [])],
    }
  })
  const outputProbability = clampProbability(pass.activations.at(-1)?.[0] ?? 0.5)
  const normalizedTarget = target === undefined ? null : target >= 0.5 ? 1 : 0

  return {
    input: [...features],
    layers: traceLayers,
    edgeContributions,
    outputProbability,
    prediction: outputProbability >= 0.5 ? 1 : 0,
    target: normalizedTarget,
    sampleLoss: normalizedTarget === null ? null : calculateLoss(outputProbability, normalizedTarget, loss),
  }
}

export function calculateWeightNorm(weights: WeightMatrix[]): number {
  let sumOfSquares = 0
  for (const matrix of weights) {
    for (const row of matrix.weights) {
      for (const value of row) sumOfSquares += value * value
    }
  }
  return Math.sqrt(sumOfSquares)
}

function forwardPass(layers: NNLayer[], weights: WeightMatrix[], biases: BiasVector[], input: number[]) {
  const activations: number[][] = [input]
  const preActivations: number[][] = [[]]

  for (let layerIndex = 1; layerIndex < layers.length; layerIndex += 1) {
    const matrix = weights[layerIndex - 1]
    const biasVector = biases[layerIndex - 1]
    const values = Array.from({ length: layers[layerIndex].size }, (_, neuronIndex) => {
      let value = biasVector?.biases[neuronIndex] ?? 0
      for (let inputIndex = 0; inputIndex < activations[layerIndex - 1].length; inputIndex += 1) {
        value += activations[layerIndex - 1][inputIndex] * (matrix?.weights[inputIndex]?.[neuronIndex] ?? 0)
      }
      return value
    })
    preActivations.push(values)
    activations.push(values.map((value) => activate(layers[layerIndex].activation, value)))
  }

  return { activations, preActivations }
}

function applyOptimizerUpdate(
  parameter: number,
  gradient: number,
  learningRate: number,
  config: TrainingConfig,
  optimizer: OptimizerBuffers,
  kind: "weight" | "bias",
  groupIndex: number,
  firstIndex: number,
  secondIndex: number,
): number {
  const select = (weightValues: number[][][], biasValues: number[][]) =>
    kind === "weight" ? weightValues[groupIndex][firstIndex] : biasValues[groupIndex]
  const coordinate = kind === "weight" ? secondIndex : firstIndex

  if (config.optimizer === "sgd") return parameter - learningRate * gradient

  if (config.optimizer === "momentum") {
    const velocity = select(optimizer.velocityWeights, optimizer.velocityBiases)
    velocity[coordinate] = config.momentum * velocity[coordinate] + gradient
    return parameter - learningRate * velocity[coordinate]
  }

  const firstMoment = select(optimizer.firstMomentWeights, optimizer.firstMomentBiases)
  const secondMoment = select(optimizer.secondMomentWeights, optimizer.secondMomentBiases)
  firstMoment[coordinate] = config.momentum * firstMoment[coordinate] + (1 - config.momentum) * gradient
  secondMoment[coordinate] = 0.999 * secondMoment[coordinate] + 0.001 * gradient * gradient
  const correctedFirstMoment = firstMoment[coordinate] / (1 - config.momentum ** optimizer.step)
  const correctedSecondMoment = secondMoment[coordinate] / (1 - 0.999 ** optimizer.step)
  return parameter - (learningRate * correctedFirstMoment) / (Math.sqrt(correctedSecondMoment) + 1e-8)
}

function activate(type: NNLayer["activation"], value: number): number {
  switch (type) {
    case "relu":
      return Math.max(0, value)
    case "sigmoid":
      return 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, value))))
    case "tanh":
      return Math.tanh(value)
    case "linear":
      return value
  }
}

function activationDerivative(type: NNLayer["activation"], value: number): number {
  switch (type) {
    case "relu":
      return value > 0 ? 1 : 0
    case "sigmoid": {
      const activated = activate("sigmoid", value)
      return activated * (1 - activated)
    }
    case "tanh": {
      const activated = Math.tanh(value)
      return 1 - activated * activated
    }
    case "linear":
      return 1
  }
}

function calculateLoss(prediction: number, target: number, loss: TrainingConfig["loss"]): number {
  if (loss === "mse") return (prediction - target) ** 2
  const probability = clampProbability(prediction)
  return -(target * Math.log(probability) + (1 - target) * Math.log(1 - probability))
}

function clipGradient(value: number, limit: number): number {
  if (limit <= 0) return value
  return Math.max(-limit, Math.min(limit, value))
}

function calculateVariance(values: number[]): number {
  if (values.length < 2) return 0
  const mean = values.reduce((total, value) => total + value, 0) / values.length
  return values.reduce((total, value) => total + (value - mean) ** 2, 0) / values.length
}

function clampProbability(value: number): number {
  return Math.max(1e-7, Math.min(1 - 1e-7, value))
}

function toNumericLabel(label: number | number[]): number {
  return Array.isArray(label) ? (label[0] ?? 0) : label
}

function shuffleInPlace(values: number[], random: () => number): void {
  for (let index = values.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(random() * (index + 1))
    const currentValue = values[index]
    values[index] = values[swapIndex]
    values[swapIndex] = currentValue
  }
}
