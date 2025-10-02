// Neural network implementation with forward and backward passes

import type { NetworkState } from "@/lib/types/neural-network"
import { activations, type ActivationType } from "./activations"
import { losses, type LossType } from "./losses"
import { optimizers, type OptimizerType, type OptimizerState } from "./optimizers"

export interface NetworkComputation {
  // Network structure
  layers: ComputationLayer[]
  totalParams: number

  // Buffers for computation
  activations: Float32Array[]
  gradients: Float32Array[]
  weights: Float32Array
  biases: Float32Array
  weightGradients: Float32Array
  biasGradients: Float32Array

  // Optimizer state
  optimizerState: OptimizerState
}

export interface ComputationLayer {
  id: string
  type: "input" | "dense" | "output"
  activation: ActivationType
  inputSize: number
  outputSize: number
  weightOffset: number // Offset in the weights array
  biasOffset: number // Offset in the biases array
  dropout?: number
}

// Convert NetworkState to computation-ready format
export function compileNetwork(network: NetworkState): NetworkComputation {
  const sortedLayers = [...network.layers].sort((a, b) => a.position.x - b.position.x)
  const computationLayers: ComputationLayer[] = []

  let weightOffset = 0
  let biasOffset = 0
  let totalParams = 0

  // Build computation layers
  for (let i = 0; i < sortedLayers.length; i++) {
    const layer = sortedLayers[i]
    const layerNodes = Object.values(network.nodes).filter((n) => n.layerId === layer.id)

    const inputSize = i === 0 ? layerNodes.length : computationLayers[i - 1].outputSize
    const outputSize = layerNodes.length

    const compLayer: ComputationLayer = {
      id: layer.id,
      type: layer.type,
      activation: layer.activation as ActivationType,
      inputSize,
      outputSize,
      weightOffset,
      biasOffset,
      dropout: layer.dropout,
    }

    computationLayers.push(compLayer)

    // Update offsets (skip weights for input layer)
    if (layer.type !== "input") {
      weightOffset += inputSize * outputSize
      totalParams += inputSize * outputSize
    }
    biasOffset += outputSize
    totalParams += outputSize
  }

  // Initialize buffers
  const activations = computationLayers.map((layer) => new Float32Array(layer.outputSize))
  const gradients = computationLayers.map((layer) => new Float32Array(layer.outputSize))
  const weights = new Float32Array(weightOffset)
  const biases = new Float32Array(biasOffset)
  const weightGradients = new Float32Array(weightOffset)
  const biasGradients = new Float32Array(biasOffset)

  // Initialize weights and biases from network state
  initializeFromNetworkState(network, computationLayers, weights, biases)

  return {
    layers: computationLayers,
    totalParams,
    activations,
    gradients,
    weights,
    biases,
    weightGradients,
    biasGradients,
    optimizerState: {},
  }
}

// Initialize weights and biases from the visual network
function initializeFromNetworkState(
  network: NetworkState,
  layers: ComputationLayer[],
  weights: Float32Array,
  biases: Float32Array,
) {
  const sortedLayers = [...network.layers].sort((a, b) => a.position.x - b.position.x)

  // Initialize biases
  let biasIdx = 0
  for (const layer of sortedLayers) {
    const layerNodes = Object.values(network.nodes)
      .filter((n) => n.layerId === layer.id)
      .sort((a, b) => (a.position?.y ?? 0) - (b.position?.y ?? 0))

    for (const node of layerNodes) {
      biases[biasIdx++] = node.bias
    }
  }

  // Initialize weights
  let weightIdx = 0
  for (let i = 1; i < sortedLayers.length; i++) {
    const currentLayer = sortedLayers[i]
    const prevLayer = sortedLayers[i - 1]

    const currentNodes = Object.values(network.nodes)
      .filter((n) => n.layerId === currentLayer.id)
      .sort((a, b) => (a.position?.y ?? 0) - (b.position?.y ?? 0))

    const prevNodes = Object.values(network.nodes)
      .filter((n) => n.layerId === prevLayer.id)
      .sort((a, b) => (a.position?.y ?? 0) - (b.position?.y ?? 0))

    // Fill weights matrix (row-major: input_size x output_size)
    for (let j = 0; j < prevNodes.length; j++) {
      for (let k = 0; k < currentNodes.length; k++) {
        const edge = Object.values(network.edges).find((e) => e.from === prevNodes[j].id && e.to === currentNodes[k].id)
        weights[weightIdx++] = edge?.weight ?? 0
      }
    }
  }
}

// Forward pass through the network
export function forward(net: NetworkComputation, input: Float32Array): Float32Array {
  // Set input layer activations
  net.activations[0].set(input)

  // Forward through each layer
  for (let i = 1; i < net.layers.length; i++) {
    const layer = net.layers[i]
    const prevActivations = net.activations[i - 1]
    const currentActivations = net.activations[i]

    // Linear transformation: y = Wx + b
    const weightStart = layer.weightOffset
    let weightIdx = weightStart

    for (let j = 0; j < layer.outputSize; j++) {
      let sum = net.biases[layer.biasOffset + j]

      for (let k = 0; k < layer.inputSize; k++) {
        sum += net.weights[weightIdx++] * prevActivations[k]
      }

      // Apply activation function
      const activation = activations[layer.activation]
      currentActivations[j] = activation.forward(sum)
    }

    // Apply dropout during training (not implemented in forward pass)
    // Dropout would be applied here during training
  }

  return net.activations[net.activations.length - 1]
}

// Backward pass (backpropagation)
export function backward(
  net: NetworkComputation,
  predictions: Float32Array,
  targets: Float32Array,
  lossType: LossType,
): number {
  const lossFunction = losses[lossType]

  // Compute loss
  const loss = lossFunction.forward(predictions, targets)

  // Clear gradients
  net.weightGradients.fill(0)
  net.biasGradients.fill(0)
  net.gradients.forEach((grad) => grad.fill(0))

  // Compute output layer gradients
  const outputLayerIdx = net.layers.length - 1
  lossFunction.backward(predictions, targets, net.gradients[outputLayerIdx])

  // Backpropagate through layers
    for (let i = outputLayerIdx; i >= 1; i--) {
      const layer = net.layers[i]

    const currentGradients = net.gradients[i]
    const prevGradients = net.gradients[i - 1]
    const prevActivations = net.activations[i - 1]

    // Compute gradients w.r.t. pre-activation (before activation function)
    const preActivationGrads = new Float32Array(layer.outputSize)
    const activation = activations[layer.activation]

    for (let j = 0; j < layer.outputSize; j++) {
      // Get the pre-activation value (we need to recompute or store it)
      let preActivation = net.biases[layer.biasOffset + j]
      let weightIdx = layer.weightOffset + j * layer.inputSize

      for (let k = 0; k < layer.inputSize; k++) {
        preActivation += net.weights[weightIdx++] * prevActivations[k]
      }

      preActivationGrads[j] = currentGradients[j] * activation.backward(preActivation)
    }

    // Compute weight gradients
    let weightIdx = layer.weightOffset
    for (let j = 0; j < layer.outputSize; j++) {
      for (let k = 0; k < layer.inputSize; k++) {
        net.weightGradients[weightIdx++] += preActivationGrads[j] * prevActivations[k]
      }
    }

    // Compute bias gradients
    for (let j = 0; j < layer.outputSize; j++) {
      net.biasGradients[layer.biasOffset + j] += preActivationGrads[j]
    }

    // Compute gradients w.r.t. previous layer activations
    if (i > 1) {
      weightIdx = layer.weightOffset
      for (let j = 0; j < layer.outputSize; j++) {
        for (let k = 0; k < layer.inputSize; k++) {
          prevGradients[k] += preActivationGrads[j] * net.weights[weightIdx++]
        }
      }
    }
  }

  return loss
}

// Update network parameters using optimizer
export function updateParameters(
  net: NetworkComputation,
  optimizerType: OptimizerType,
  learningRate: number,
  step: number,
  config?: { momentum?: number; beta1?: number; beta2?: number; weightDecay?: number },
) {
  const optimizer = optimizers[optimizerType]

  // Initialize optimizer state if needed
  if (Object.keys(net.optimizerState).length === 0) {
    net.optimizerState = optimizer.initialize(net.totalParams)
  }

  // Apply weight decay (L2 regularization)
  if (config?.weightDecay) {
    for (let i = 0; i < net.weights.length; i++) {
      net.weightGradients[i] += config.weightDecay * net.weights[i]
    }
  }

  // Combine weights and biases for optimizer update
  const allParams = new Float32Array(net.weights.length + net.biases.length)
  const allGradients = new Float32Array(net.weightGradients.length + net.biasGradients.length)

  allParams.set(net.weights, 0)
  allParams.set(net.biases, net.weights.length)
  allGradients.set(net.weightGradients, 0)
  allGradients.set(net.biasGradients, net.weights.length)

  // Update parameters
  optimizer.update(allParams, allGradients, net.optimizerState, learningRate, step)

  // Copy back to separate arrays
  net.weights.set(allParams.subarray(0, net.weights.length))
  net.biases.set(allParams.subarray(net.weights.length))
}

// Utility function to get network predictions
export function predict(net: NetworkComputation, inputs: number[][]): number[][] {
  const results: number[][] = []

  for (const input of inputs) {
    const inputArray = new Float32Array(input)
    const output = forward(net, inputArray)
    results.push(Array.from(output))
  }

  return results
}

// Calculate accuracy for classification tasks
export function calculateAccuracy(predictions: number[][], targets: number[]): number {
  let correct = 0

  for (let i = 0; i < predictions.length; i++) {
    const predicted = predictions[i].indexOf(Math.max(...predictions[i]))
    if (predicted === targets[i]) {
      correct++
    }
  }

  return correct / predictions.length
}
