"use client"

import { useEffect, useState } from "react"
import { getTrainingManager, type TrainingState } from "@/lib/training/training-manager"
import type { NetworkState, TrainConfig, Dataset } from "@/lib/types/neural-network"

export function useTraining() {
  const [trainingState, setTrainingState] = useState<TrainingState>(() => getTrainingManager().getState())

  useEffect(() => {
    const manager = getTrainingManager()
    const unsubscribe = manager.subscribe(setTrainingState)
    return unsubscribe
  }, [])

  const startTraining = (network: NetworkState, config: TrainConfig, dataset: Dataset) => {
    getTrainingManager().startTraining(network, config, dataset)
  }

  const pauseTraining = () => {
    getTrainingManager().pauseTraining()
  }

  const resumeTraining = () => {
    getTrainingManager().resumeTraining()
  }

  const stepTraining = () => {
    getTrainingManager().stepTraining()
  }

  const stopTraining = () => {
    getTrainingManager().stopTraining()
  }

  const resetTraining = () => {
    getTrainingManager().resetTraining()
  }

  return {
    // State
    status: trainingState.status,
    metrics: trainingState.metrics,
    error: trainingState.error,
    weights: trainingState.weights,
    biases: trainingState.biases,
    gradients: trainingState.gradients,
    activations: trainingState.activations,

    // Actions
    startTraining,
    pauseTraining,
    resumeTraining,
    stepTraining,
    stopTraining,
    resetTraining,

    // Computed state
    isTraining: trainingState.status === "training",
    isPaused: trainingState.status === "paused",
    isComplete: trainingState.status === "complete",
    hasError: trainingState.status === "error",
  }
}
