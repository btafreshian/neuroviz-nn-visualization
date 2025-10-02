// Core type definitions for the neural network playground

export type Activation = "relu" | "sigmoid" | "tanh" | "linear"
export type LossFn = "mse" | "cross_entropy"
export type Optimizer = "sgd" | "momentum" | "adam"
export type LayerType = "input" | "dense" | "output"
export type TaskType = "classification" | "regression"

export interface NNNode {
  id: string
  layerId: string
  bias: number
  activation: Activation
  position?: { x: number; y: number }
  // Runtime values
  activationValue?: number
  gradient?: number
}

export interface NNEdge {
  id: string
  from: string // node id in layer L
  to: string // node id in layer L+1
  weight: number
  grad?: number
}

export interface NNLayer {
  id: string
  type: LayerType
  name: string
  activation: Activation
  units: number
  dropout?: number // 0..1
  position: { x: number; y: number }
}

export interface NetworkState {
  layers: NNLayer[]
  nodes: Record<string, NNNode>
  edges: Record<string, NNEdge>
}

export interface TrainConfig {
  task: TaskType
  loss: LossFn
  optimizer: Optimizer
  learningRate: number
  momentum?: number
  beta2?: number
  weightDecay?: number
  batchSize: number
  epochs: number
  gradientClip?: number
  earlyStopping?: {
    patience: number
    minDelta: number
  }
}

export interface Dataset {
  name: string
  features: number[][]
  labels: number[] | number[][]
  trainIndices?: number[]
  valIndices?: number[]
}

export interface TrainingMetrics {
  step: number
  epoch: number
  loss: number
  valLoss?: number
  accuracy?: number
  valAccuracy?: number
}

// Worker message types
export type WorkerMessage =
  | { type: "START"; net: NetworkState; config: TrainConfig; data: Dataset }
  | { type: "PAUSE" }
  | { type: "STEP" }
  | { type: "RESET" }

export type WorkerResponse =
  | {
      type: "UPDATE"
      weights: Array<[string, number]>
      grads?: Array<[string, number]>
      metrics: TrainingMetrics
    }
  | { type: "DONE" }
  | { type: "ERROR"; message: string }
