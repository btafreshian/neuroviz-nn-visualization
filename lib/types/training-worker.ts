import type { Dataset, NetworkState, TrainConfig, TrainingMetrics } from "@/lib/types/neural-network"

export type TrainingWorkerRequest =
  | { type: "START"; network: NetworkState; config: TrainConfig; dataset: Dataset }
  | { type: "PAUSE" }
  | { type: "RESUME" }
  | { type: "STEP" }
  | { type: "RESET" }
  | { type: "STOP" }

export type TrainingWorkerUpdate = {
  type: "UPDATE"
  weights: Array<[string, number]>
  biases: Array<[string, number]>
  gradients?: Array<[string, number]>
  activations?: Array<[string, number]>
  metrics: TrainingMetrics
}

export type TrainingWorkerResponse =
  | TrainingWorkerUpdate
  | { type: "EPOCH_COMPLETE"; epoch: number; metrics: TrainingMetrics }
  | { type: "TRAINING_COMPLETE"; finalMetrics: TrainingMetrics }
  | { type: "ERROR"; message: string }
  | { type: "PAUSED" }
  | { type: "STOPPED" }
