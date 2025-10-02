// Activation functions and their derivatives

export interface ActivationFunction {
  forward: (x: number) => number
  backward: (x: number) => number // derivative
  forwardArray: (input: Float32Array, output: Float32Array) => void
  backwardArray: (input: Float32Array, gradOutput: Float32Array, gradInput: Float32Array) => void
}

// ReLU activation
export const relu: ActivationFunction = {
  forward: (x: number) => Math.max(0, x),
  backward: (x: number) => (x > 0 ? 1 : 0),
  forwardArray: (input: Float32Array, output: Float32Array) => {
    for (let i = 0; i < input.length; i++) {
      output[i] = Math.max(0, input[i])
    }
  },
  backwardArray: (input: Float32Array, gradOutput: Float32Array, gradInput: Float32Array) => {
    for (let i = 0; i < input.length; i++) {
      gradInput[i] = input[i] > 0 ? gradOutput[i] : 0
    }
  },
}

// Sigmoid activation
export const sigmoid: ActivationFunction = {
  forward: (x: number) => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x)))), // Clamp to prevent overflow
  backward: (x: number) => {
    const s = sigmoid.forward(x)
    return s * (1 - s)
  },
  forwardArray: (input: Float32Array, output: Float32Array) => {
    for (let i = 0; i < input.length; i++) {
      const clampedX = Math.max(-500, Math.min(500, input[i]))
      output[i] = 1 / (1 + Math.exp(-clampedX))
    }
  },
  backwardArray: (input: Float32Array, gradOutput: Float32Array, gradInput: Float32Array) => {
    for (let i = 0; i < input.length; i++) {
      const s = sigmoid.forward(input[i])
      gradInput[i] = gradOutput[i] * s * (1 - s)
    }
  },
}

// Tanh activation
export const tanh: ActivationFunction = {
  forward: (x: number) => Math.tanh(Math.max(-500, Math.min(500, x))),
  backward: (x: number) => {
    const t = tanh.forward(x)
    return 1 - t * t
  },
  forwardArray: (input: Float32Array, output: Float32Array) => {
    for (let i = 0; i < input.length; i++) {
      const clampedX = Math.max(-500, Math.min(500, input[i]))
      output[i] = Math.tanh(clampedX)
    }
  },
  backwardArray: (input: Float32Array, gradOutput: Float32Array, gradInput: Float32Array) => {
    for (let i = 0; i < input.length; i++) {
      const t = tanh.forward(input[i])
      gradInput[i] = gradOutput[i] * (1 - t * t)
    }
  },
}

// Linear activation (identity)
export const linear: ActivationFunction = {
  forward: (x: number) => x,
  backward: () => 1,
  forwardArray: (input: Float32Array, output: Float32Array) => {
    for (let i = 0; i < input.length; i++) {
      output[i] = input[i]
    }
  },
  backwardArray: (input: Float32Array, gradOutput: Float32Array, gradInput: Float32Array) => {
    for (let i = 0; i < input.length; i++) {
      gradInput[i] = gradOutput[i]
    }
  },
}

// Activation function registry
export const activations = {
  relu,
  sigmoid,
  tanh,
  linear,
} as const

export type ActivationType = keyof typeof activations
