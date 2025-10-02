// Web Worker for neural network training
// This runs in a separate thread to keep the UI responsive

import type { NetworkState, TrainConfig, Dataset, TrainingMetrics } from "@/lib/types/neural-network"
import { compileNetwork, forward, backward, updateParameters, predict } from "@/lib/nn/network"
import { splitData, oneHotEncode } from "@/lib/nn/data-utils"
import type { NetworkComputation } from "@/lib/nn/network"

// Worker state
let isTraining = false
let shouldPause = false
let currentStep = 0
let currentEpoch = 0
let compiledNetwork: NetworkComputation | null = null
let trainConfig: TrainConfig | null = null
let trainData: { features: number[][]; labels: number[] } | null = null
let valData: { features: number[][]; labels: number[] } | null = null

// Message types from main thread
type WorkerMessage =
  | { type: "START"; network: NetworkState; config: TrainConfig; dataset: Dataset }
  | { type: "PAUSE" }
  | { type: "RESUME" }
  | { type: "STEP" }
  | { type: "RESET" }
  | { type: "STOP" }

// Message types to main thread
type WorkerResponse =
  | {
      type: "UPDATE"
      weights: Array<[string, number]>
      biases: Array<[string, number]>
      gradients?: Array<[string, number]>
      activations?: Array<[string, number]>
      metrics: TrainingMetrics
    }
  | { type: "EPOCH_COMPLETE"; epoch: number; metrics: TrainingMetrics }
  | { type: "TRAINING_COMPLETE"; finalMetrics: TrainingMetrics }
  | { type: "ERROR"; message: string }
  | { type: "PAUSED" }
  | { type: "STOPPED" }

// Handle messages from main thread
self.onmessage = (event: MessageEvent<WorkerMessage>) => {
  const message = event.data

  try {
    switch (message.type) {
      case "START":
        startTraining(message.network, message.config, message.dataset)
        break
      case "PAUSE":
        pauseTraining()
        break
      case "RESUME":
        resumeTraining()
        break
      case "STEP":
        stepTraining()
        break
      case "RESET":
        resetTraining()
        break
      case "STOP":
        stopTraining()
        break
    }
  } catch (error) {
    postMessage({
      type: "ERROR",
      message: error instanceof Error ? error.message : "Unknown error occurred",
    } as WorkerResponse)
  }
}

function startTraining(network: NetworkState, config: TrainConfig, dataset: Dataset) {
  try {
    // Compile the network for computation
    compiledNetwork = compileNetwork(network)
    trainConfig = config

    // Prepare data
    const { features, labels } = dataset
    const split = splitData(features, labels, 0.8, true)

    trainData = { features: split.trainFeatures, labels: split.trainLabels }
    valData = { features: split.valFeatures, labels: split.valLabels }

    // Reset training state
    currentStep = 0
    currentEpoch = 0
    isTraining = true
    shouldPause = false

    // Start training loop
    requestAnimationFrame(trainingLoop)
  } catch (error) {
    postMessage({
      type: "ERROR",
      message: error instanceof Error ? error.message : "Failed to start training",
    } as WorkerResponse)
  }
}

function pauseTraining() {
  shouldPause = true
  postMessage({ type: "PAUSED" } as WorkerResponse)
}

function resumeTraining() {
  if (!isTraining) return
  shouldPause = false
  requestAnimationFrame(trainingLoop)
}

function stepTraining() {
  if (!compiledNetwork || !trainConfig || !trainData) return

  try {
    const metrics = trainSingleBatch()
    sendUpdate(metrics)
  } catch (error) {
    postMessage({
      type: "ERROR",
      message: error instanceof Error ? error.message : "Error during training step",
    } as WorkerResponse)
  }
}

function resetTraining() {
  isTraining = false
  shouldPause = false
  currentStep = 0
  currentEpoch = 0
  compiledNetwork = null
  trainConfig = null
  trainData = null
  valData = null
  postMessage({ type: "STOPPED" } as WorkerResponse)
}

function stopTraining() {
  isTraining = false
  postMessage({ type: "STOPPED" } as WorkerResponse)
}

// Main training loop
function trainingLoop() {
  if (!isTraining || shouldPause || !compiledNetwork || !trainConfig || !trainData) {
    return
  }

  try {
    // Train for a few batches per frame to balance responsiveness and speed
    const batchesPerFrame = 5
    let metrics: TrainingMetrics | null = null

    for (let i = 0; i < batchesPerFrame && isTraining && !shouldPause; i++) {
      metrics = trainSingleBatch()

      // Check if epoch is complete
      const batchSize = trainConfig.batchSize
      const totalBatches = Math.ceil(trainData.features.length / batchSize)

      if (currentStep % totalBatches === 0) {
        currentEpoch++
        postMessage({ type: "EPOCH_COMPLETE", epoch: currentEpoch, metrics } as WorkerResponse)

        // Check if training is complete
        if (currentEpoch >= trainConfig.epochs) {
          isTraining = false
          postMessage({ type: "TRAINING_COMPLETE", finalMetrics: metrics } as WorkerResponse)
          return
        }
      }
    }

    // Send periodic updates
    if (metrics && currentStep % 10 === 0) {
      sendUpdate(metrics)
    }

    // Continue training loop
    if (isTraining && !shouldPause) {
      requestAnimationFrame(trainingLoop)
    }
  } catch (error) {
    postMessage({
      type: "ERROR",
      message: error instanceof Error ? error.message : "Error in training loop",
    } as WorkerResponse)
  }
}

// Train a single batch
function trainSingleBatch(): TrainingMetrics {
  if (!compiledNetwork || !trainConfig || !trainData || !valData) {
    throw new Error("Training not properly initialized")
  }

  const batchSize = trainConfig.batchSize
  const totalSamples = trainData.features.length
  const batchStart = (currentStep * batchSize) % totalSamples
  const batchEnd = Math.min(batchStart + batchSize, totalSamples)

  // Get batch data
  const batchFeatures = trainData.features.slice(batchStart, batchEnd)
  const batchLabels = trainData.labels.slice(batchStart, batchEnd)

  // Convert labels for the task type
  const targetLabels = trainConfig.task === "classification" ? oneHotEncode(batchLabels) : batchLabels.map((l) => [l])

  let totalLoss = 0
  let correct = 0

  // Process each sample in the batch
  for (let i = 0; i < batchFeatures.length; i++) {
    const input = new Float32Array(batchFeatures[i])
    const target = new Float32Array(targetLabels[i])

    // Forward pass
    const output = forward(compiledNetwork, input)

    // Backward pass
    const loss = backward(compiledNetwork, output, target, trainConfig.loss)
    totalLoss += loss

    // Calculate accuracy for classification
    if (trainConfig.task === "classification") {
      const predicted = Array.from(output).indexOf(Math.max(...output))
      if (predicted === batchLabels[i]) {
        correct++
      }
    }
  }

  // Update parameters
  currentStep++
  updateParameters(compiledNetwork, trainConfig.optimizer, trainConfig.learningRate, currentStep, {
    momentum: trainConfig.momentum,
    beta1: trainConfig.momentum,
    beta2: trainConfig.beta2,
    weightDecay: trainConfig.weightDecay,
  })

  // Calculate validation metrics
  const valMetrics = calculateValidationMetrics()

  return {
    step: currentStep,
    epoch: currentEpoch,
    loss: totalLoss / batchFeatures.length,
    valLoss: valMetrics.loss,
    accuracy: trainConfig.task === "classification" ? correct / batchFeatures.length : undefined,
    valAccuracy: valMetrics.accuracy,
  }
}

// Calculate validation metrics
function calculateValidationMetrics(): { loss: number; accuracy?: number } {
  if (!compiledNetwork || !trainConfig || !valData || valData.features.length === 0) {
    return { loss: 0 }
  }

  const predictions = predict(compiledNetwork, valData.features)
  const targetLabels =
    trainConfig.task === "classification" ? oneHotEncode(valData.labels) : valData.labels.map((l) => [l])

  let totalLoss = 0
  let correct = 0

  for (let i = 0; i < predictions.length; i++) {
    const pred = new Float32Array(predictions[i])
    const target = new Float32Array(targetLabels[i])

    // Calculate loss (simplified - not using the full backward pass)
    if (trainConfig.loss === "mse") {
      for (let j = 0; j < pred.length; j++) {
        const diff = pred[j] - target[j]
        totalLoss += diff * diff
      }
    } else if (trainConfig.loss === "cross_entropy") {
      for (let j = 0; j < pred.length; j++) {
        const p = Math.max(1e-15, Math.min(1 - 1e-15, pred[j]))
        totalLoss -= target[j] * Math.log(p) + (1 - target[j]) * Math.log(1 - p)
      }
    }

    // Calculate accuracy for classification
    if (trainConfig.task === "classification") {
      const predicted = predictions[i].indexOf(Math.max(...predictions[i]))
      if (predicted === valData.labels[i]) {
        correct++
      }
    }
  }

  return {
    loss: totalLoss / predictions.length,
    accuracy: trainConfig.task === "classification" ? correct / predictions.length : undefined,
  }
}

// Send update to main thread with current network state
function sendUpdate(metrics: TrainingMetrics) {
  if (!compiledNetwork) return

  // Extract weights and biases with their IDs
  const weights: Array<[string, number]> = []
  const biases: Array<[string, number]> = []
  const gradients: Array<[string, number]> = []
  const activations: Array<[string, number]> = []

  // For now, send a subset of the data to avoid overwhelming the main thread
  // In a real implementation, you'd map these back to the original network structure

  // Send sample of weights (first 100)
  for (let i = 0; i < Math.min(100, compiledNetwork.weights.length); i++) {
    weights.push([`weight_${i}`, compiledNetwork.weights[i]])
    gradients.push([`weight_grad_${i}`, compiledNetwork.weightGradients[i]])
  }

  // Send all biases
  for (let i = 0; i < compiledNetwork.biases.length; i++) {
    biases.push([`bias_${i}`, compiledNetwork.biases[i]])
  }

  // Send current activations
  compiledNetwork.activations.forEach((layerActivations, layerIdx) => {
    layerActivations.forEach((activation, nodeIdx) => {
      activations.push([`activation_${layerIdx}_${nodeIdx}`, activation])
    })
  })

  postMessage({
    type: "UPDATE",
    weights,
    biases,
    gradients,
    activations,
    metrics,
  } as WorkerResponse)
}
