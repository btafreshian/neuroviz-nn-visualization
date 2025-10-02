// Loss functions for training

export interface LossFunction {
  forward: (predictions: Float32Array, targets: Float32Array) => number
  backward: (predictions: Float32Array, targets: Float32Array, gradients: Float32Array) => void
}

// Mean Squared Error (for regression)
export const mse: LossFunction = {
  forward: (predictions: Float32Array, targets: Float32Array) => {
    let sum = 0
    for (let i = 0; i < predictions.length; i++) {
      const diff = predictions[i] - targets[i]
      sum += diff * diff
    }
    return sum / predictions.length
  },
  backward: (predictions: Float32Array, targets: Float32Array, gradients: Float32Array) => {
    const scale = 2 / predictions.length
    for (let i = 0; i < predictions.length; i++) {
      gradients[i] = scale * (predictions[i] - targets[i])
    }
  },
}

// Cross-Entropy Loss (for classification)
export const crossEntropy: LossFunction = {
  forward: (predictions: Float32Array, targets: Float32Array) => {
    let sum = 0
    const eps = 1e-15 // Small epsilon to prevent log(0)

    for (let i = 0; i < predictions.length; i++) {
      const p = Math.max(eps, Math.min(1 - eps, predictions[i]))
      sum -= targets[i] * Math.log(p) + (1 - targets[i]) * Math.log(1 - p)
    }
    return sum / predictions.length
  },
  backward: (predictions: Float32Array, targets: Float32Array, gradients: Float32Array) => {
    const eps = 1e-15
    const scale = 1 / predictions.length

    for (let i = 0; i < predictions.length; i++) {
      const p = Math.max(eps, Math.min(1 - eps, predictions[i]))
      gradients[i] = scale * ((p - targets[i]) / (p * (1 - p)))
    }
  },
}

export const losses = {
  mse,
  cross_entropy: crossEntropy,
} as const

export type LossType = keyof typeof losses
