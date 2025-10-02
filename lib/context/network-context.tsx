"use client"

import { createContext, useContext, useState, useEffect, useRef, type ReactNode } from "react"

// Updated types to match what components expect
export interface NNLayer {
  id: string
  name: string
  type: "input" | "dense" | "output"
  size: number
  activation: "relu" | "sigmoid" | "tanh" | "linear"
}

export interface Network {
  layers: NNLayer[]
}

export interface TrainingConfig {
  learningRate: number
  epochs: number
  batchSize: number
  optimizer: "sgd" | "adam"
  loss: "mse" | "crossentropy"
}

export interface Dataset {
  name: string
  features: number[][]
  labels: number[]
  inputSize: number
  outputSize: number
}

export interface TrainingProgress {
  epoch: number
  loss: number
  accuracy: number
}

interface NetworkContextType {
  network: Network
  dataset: Dataset | null
  trainConfig: TrainingConfig
  isTraining: boolean
  trainingProgress: TrainingProgress
  loadTemplate: (template: string) => void
  updateLayer: (layerId: string, updates: Partial<NNLayer>) => void
  updateTrainingConfig: (config: Partial<TrainingConfig>) => void
  setDataset: (dataset: Dataset) => void
  startTraining: () => void
  stopTraining: () => void
  resetTraining: () => void
}

const NetworkContext = createContext<NetworkContextType | undefined>(undefined)

// Create classification network template
const createClassificationNetwork = (): Network => {
  const layers: NNLayer[] = [
    { id: "input", name: "Input Layer", type: "input", size: 2, activation: "linear" },
    { id: "hidden1", name: "Hidden Layer 1", type: "dense", size: 8, activation: "relu" },
    { id: "hidden2", name: "Hidden Layer 2", type: "dense", size: 6, activation: "relu" },
    { id: "output", name: "Output Layer", type: "output", size: 1, activation: "sigmoid" },
  ]

  return { layers }
}

// Simple XOR dataset
const createDefaultDataset = (): Dataset => ({
  name: "XOR Dataset",
  features: [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
  ],
  labels: [0, 1, 1, 0],
  inputSize: 2,
  outputSize: 1,
})

export function NetworkProvider({ children }: { children: ReactNode }) {
  const [network, setNetwork] = useState<Network>({ layers: [] })
  const [dataset, setDatasetState] = useState<Dataset | null>(createDefaultDataset())
  const [trainConfig, setTrainConfig] = useState<TrainingConfig>({
    learningRate: 0.01,
    epochs: 100,
    batchSize: 4,
    optimizer: "adam",
    loss: "mse",
  })
  const [isTraining, setIsTraining] = useState(false)
  const [trainingProgress, setTrainingProgress] = useState<TrainingProgress>({
    epoch: 0,
    loss: 0,
    accuracy: 0,
  })

  const trainingIntervalRef = useRef<NodeJS.Timeout | null>(null)

  useEffect(() => {
    console.log("[v0] NetworkProvider mounted, auto-loading classification network")
    loadTemplate("classification")
  }, [])

  const loadTemplate = (template: string) => {
    console.log("[v0] loadTemplate called with:", template)
    if (template === "classification") {
      const newNetwork = createClassificationNetwork()
      console.log("[v0] Created network with layers:", newNetwork.layers)
      setNetwork(newNetwork)
      setTrainingProgress({ epoch: 0, loss: 0, accuracy: 0 })
      console.log("[v0] Network state updated")
    }
  }

  const updateLayer = (layerId: string, updates: Partial<NNLayer>) => {
    setNetwork((prev) => ({
      ...prev,
      layers: prev.layers.map((layer) => (layer.id === layerId ? { ...layer, ...updates } : layer)),
    }))
  }

  const updateTrainingConfig = (config: Partial<TrainingConfig>) => {
    setTrainConfig((prev) => ({ ...prev, ...config }))
  }

  const setDataset = (newDataset: Dataset) => {
    setDatasetState(newDataset)
    // Reset training progress when dataset changes
    setTrainingProgress({ epoch: 0, loss: 0, accuracy: 0 })
  }

  const startTraining = () => {
    if (!dataset) {
      console.log("[v0] Cannot start training: no dataset")
      return
    }

    console.log("[v0] Starting training with config:", trainConfig)
    setIsTraining(true)
    setTrainingProgress({ epoch: 0, loss: 1.0, accuracy: 0 })

    if (trainingIntervalRef.current) {
      clearInterval(trainingIntervalRef.current)
    }

    // Simple training simulation
    let epoch = 0
    trainingIntervalRef.current = setInterval(() => {
      epoch++
      const progress = epoch / trainConfig.epochs
      const loss = Math.max(0.01, 1.0 * Math.exp(-progress * 3) + Math.random() * 0.05)
      const accuracy = Math.min(0.95, progress * 0.9 + Math.random() * 0.1)

      console.log(
        "[v0] Training epoch:",
        epoch,
        "loss:",
        loss.toFixed(4),
        "accuracy:",
        (accuracy * 100).toFixed(1) + "%",
      )
      setTrainingProgress({ epoch, loss, accuracy })

      if (epoch >= trainConfig.epochs) {
        console.log("[v0] Training completed")
        setIsTraining(false)
        if (trainingIntervalRef.current) {
          clearInterval(trainingIntervalRef.current)
          trainingIntervalRef.current = null
        }
      }
    }, 100)
  }

  const stopTraining = () => {
    console.log("[v0] Stopping training")
    if (trainingIntervalRef.current) {
      clearInterval(trainingIntervalRef.current)
      trainingIntervalRef.current = null
    }
    setIsTraining(false)
  }

  const resetTraining = () => {
    console.log("[v0] Resetting training")
    if (trainingIntervalRef.current) {
      clearInterval(trainingIntervalRef.current)
      trainingIntervalRef.current = null
    }
    setIsTraining(false)
    setTrainingProgress({ epoch: 0, loss: 0, accuracy: 0 })
  }

  useEffect(() => {
    return () => {
      if (trainingIntervalRef.current) {
        clearInterval(trainingIntervalRef.current)
      }
    }
  }, [])

  const value: NetworkContextType = {
    network,
    dataset,
    trainConfig,
    isTraining,
    trainingProgress,
    loadTemplate,
    updateLayer,
    updateTrainingConfig,
    setDataset,
    startTraining,
    stopTraining,
    resetTraining,
  }

  return <NetworkContext.Provider value={value}>{children}</NetworkContext.Provider>
}

export function useNetwork() {
  const context = useContext(NetworkContext)
  if (context === undefined) {
    throw new Error("useNetwork must be used within a NetworkProvider")
  }
  return context
}

// Keep the old export for backward compatibility
export function useNetworkStore() {
  return useNetwork()
}
