// Optimizers for training neural networks

export interface OptimizerState {
  [key: string]: Float32Array // Store momentum, velocity, etc. for each parameter
}

export interface Optimizer {
  initialize: (paramCount: number) => OptimizerState
  update: (
    params: Float32Array,
    gradients: Float32Array,
    state: OptimizerState,
    learningRate: number,
    step: number,
  ) => void
}

// Stochastic Gradient Descent
export const sgd: Optimizer = {
  initialize: (paramCount: number) => ({}),
  update: (params: Float32Array, gradients: Float32Array, state: OptimizerState, learningRate: number) => {
    for (let i = 0; i < params.length; i++) {
      params[i] -= learningRate * gradients[i]
    }
  },
}

// SGD with Momentum
export const momentum: Optimizer = {
  initialize: (paramCount: number) => ({
    velocity: new Float32Array(paramCount),
  }),
  update: (
    params: Float32Array,
    gradients: Float32Array,
    state: OptimizerState,
    learningRate: number,
    step: number,
    beta = 0.9,
  ) => {
    const velocity = state.velocity as Float32Array

    for (let i = 0; i < params.length; i++) {
      velocity[i] = beta * velocity[i] + (1 - beta) * gradients[i]
      params[i] -= learningRate * velocity[i]
    }
  },
}

// Adam Optimizer
export const adam: Optimizer = {
  initialize: (paramCount: number) => ({
    m: new Float32Array(paramCount), // First moment
    v: new Float32Array(paramCount), // Second moment
  }),
  update: (
    params: Float32Array,
    gradients: Float32Array,
    state: OptimizerState,
    learningRate: number,
    step: number,
    beta1 = 0.9,
    beta2 = 0.999,
    epsilon = 1e-8,
  ) => {
    const m = state.m as Float32Array
    const v = state.v as Float32Array

    for (let i = 0; i < params.length; i++) {
      // Update biased first moment estimate
      m[i] = beta1 * m[i] + (1 - beta1) * gradients[i]

      // Update biased second raw moment estimate
      v[i] = beta2 * v[i] + (1 - beta2) * gradients[i] * gradients[i]

      // Compute bias-corrected first moment estimate
      const mHat = m[i] / (1 - Math.pow(beta1, step))

      // Compute bias-corrected second raw moment estimate
      const vHat = v[i] / (1 - Math.pow(beta2, step))

      // Update parameters
      params[i] -= (learningRate * mHat) / (Math.sqrt(vHat) + epsilon)
    }
  },
}

export const optimizers = {
  sgd,
  momentum,
  adam,
} as const

export type OptimizerType = keyof typeof optimizers
