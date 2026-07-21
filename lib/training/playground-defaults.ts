import type { Network, TrainingConfig } from "@/lib/context/network-context"

export const DEFAULT_NETWORK: Network = {
  layers: [
    { id: "input", name: "Input Layer", type: "input", size: 2, activation: "linear" },
    { id: "hidden1", name: "Hidden Layer 1", type: "dense", size: 8, activation: "relu" },
    { id: "hidden2", name: "Hidden Layer 2", type: "dense", size: 6, activation: "relu" },
    { id: "output", name: "Output Layer", type: "output", size: 1, activation: "sigmoid" },
  ],
}

export const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
  learningRate: 0.001,
  epochs: 100,
  batchSize: 32,
  trainTestSplit: 0.8,
  optimizer: "adam",
  loss: "crossentropy",
  momentum: 0.9,
  weightDecay: 0.0001,
  learningRateDecay: 0,
  gradientClip: 5,
  shuffle: true,
}
