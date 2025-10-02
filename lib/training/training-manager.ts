"use client"

import type { NetworkState, TrainConfig, Dataset, TrainingMetrics } from "@/lib/types/neural-network"

export type TrainingStatus = "idle" | "training" | "paused" | "error" | "complete"

export interface TrainingState {
  status: TrainingStatus
  metrics: TrainingMetrics | null
  error: string | null
  weights: Map<string, number>
  biases: Map<string, number>
  gradients: Map<string, number>
  activations: Map<string, number>
}

export class TrainingManager {
  private worker: Worker | null = null
  private listeners: Set<(state: TrainingState) => void> = new Set()
  private state: TrainingState = {
    status: "idle",
    metrics: null,
    error: null,
    weights: new Map(),
    biases: new Map(),
    gradients: new Map(),
    activations: new Map(),
  }

  constructor() {
    this.initializeWorker()
  }

  private initializeWorker() {
    try {
      // Create worker from the training worker file
      this.worker = new Worker(new URL("../workers/training.worker.ts", import.meta.url), {
        type: "module",
      })

      this.worker.onmessage = (event) => {
        this.handleWorkerMessage(event.data)
      }

      this.worker.onerror = (error) => {
        this.updateState({
          status: "error",
          error: `Worker error: ${error.message}`,
        })
      }
    } catch (error) {
      console.error("Failed to initialize training worker:", error)
      this.updateState({
        status: "error",
        error: "Failed to initialize training worker",
      })
    }
  }

  private handleWorkerMessage(message: any) {
    switch (message.type) {
      case "UPDATE":
        this.updateState({
          status: "training",
          metrics: message.metrics,
          weights: new Map(message.weights),
          biases: new Map(message.biases),
          gradients: new Map(message.gradients || []),
          activations: new Map(message.activations || []),
          error: null,
        })
        break

      case "EPOCH_COMPLETE":
        // Could trigger special epoch complete events here
        break

      case "TRAINING_COMPLETE":
        this.updateState({
          status: "complete",
          metrics: message.finalMetrics,
          error: null,
        })
        break

      case "PAUSED":
        this.updateState({
          status: "paused",
          error: null,
        })
        break

      case "STOPPED":
        this.updateState({
          status: "idle",
          metrics: null,
          error: null,
        })
        break

      case "ERROR":
        this.updateState({
          status: "error",
          error: message.message,
        })
        break
    }
  }

  private updateState(updates: Partial<TrainingState>) {
    this.state = { ...this.state, ...updates }
    this.listeners.forEach((listener) => listener(this.state))
  }

  // Public API
  startTraining(network: NetworkState, config: TrainConfig, dataset: Dataset) {
    if (!this.worker) {
      this.updateState({
        status: "error",
        error: "Training worker not available",
      })
      return
    }

    this.worker.postMessage({
      type: "START",
      network,
      config,
      dataset,
    })

    this.updateState({
      status: "training",
      error: null,
    })
  }

  pauseTraining() {
    if (!this.worker) return
    this.worker.postMessage({ type: "PAUSE" })
  }

  resumeTraining() {
    if (!this.worker) return
    this.worker.postMessage({ type: "RESUME" })
  }

  stepTraining() {
    if (!this.worker) return
    this.worker.postMessage({ type: "STEP" })
  }

  stopTraining() {
    if (!this.worker) return
    this.worker.postMessage({ type: "STOP" })
  }

  resetTraining() {
    if (!this.worker) return
    this.worker.postMessage({ type: "RESET" })
  }

  // State management
  getState(): TrainingState {
    return this.state
  }

  subscribe(listener: (state: TrainingState) => void): () => void {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  // Cleanup
  destroy() {
    if (this.worker) {
      this.worker.terminate()
      this.worker = null
    }
    this.listeners.clear()
  }
}

// Global training manager instance
let globalTrainingManager: TrainingManager | null = null

export function getTrainingManager(): TrainingManager {
  if (!globalTrainingManager) {
    globalTrainingManager = new TrainingManager()
  }
  return globalTrainingManager
}
