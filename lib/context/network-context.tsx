"use client"

import { createContext, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react"
import { DEFAULT_DATASET_CONFIG, generateDataset, type DatasetConfig } from "@/lib/datasets/builtin-datasets"
import {
  calculateWeightNorm,
  createDatasetSplit,
  createOptimizerBuffers,
  createZeroNetworkParameters,
  evaluateNetwork,
  initializeNetworkParameters,
  traceForwardPass,
  trainEpoch,
  type DatasetSplit,
  type ForwardTrace,
  type OptimizerBuffers,
} from "@/lib/nn/simple-trainer"
import {
  DEFAULT_NETWORK,
  DEFAULT_TRAINING_CONFIG,
} from "@/lib/training/playground-defaults"

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
  trainTestSplit: number
  optimizer: "sgd" | "momentum" | "adam"
  loss: "mse" | "crossentropy"
  momentum: number
  weightDecay: number
  learningRateDecay: number
  gradientClip: number
  shuffle: boolean
}

export interface AxisStatistics {
  mean: number
  standardDeviation: number
  minimum: number
  maximum: number
}

export interface Dataset {
  id?: string
  name: string
  description?: string
  features: number[][]
  labels: number[] | number[][]
  inputSize?: number
  outputSize?: number
  noise?: number
  seed?: number
  normalized?: boolean
  featureStats?: {
    raw: AxisStatistics[]
    processed: AxisStatistics[]
  }
}

export interface WeightMatrix {
  weights: number[][]
  fromLayerIndex: number
  toLayerIndex: number
}

export interface BiasVector {
  biases: number[]
  layerIndex: number
}

export type TrainingStatus = "idle" | "initializing" | "running" | "paused" | "stopped" | "complete"
export type TrainingSpeed = "slow" | "medium" | "fast"

export interface EpochTelemetry {
  epoch: number
  trainLoss: number
  validationLoss: number
  trainAccuracy: number
  validationAccuracy: number
  meanConfidence: number
  effectiveLearningRate: number
  rawGradientNorm: number
  appliedGradientNorm: number
  updateNorm: number
  weightNorm: number
  regularizationPenalty: number
  clippedFraction: number
  batchLossVariance: number
}

export interface EpochParameterSnapshot {
  epoch: number
  weights: WeightMatrix[]
  biases: BiasVector[]
  weightDeltas: number[][][]
}

export interface TrainingRunSnapshot {
  status: TrainingStatus
  config: TrainingConfig
  datasetSignature: string
  networkSignature: string
  initializationSeed: number
  history: EpochTelemetry[]
  parameterHistory: EpochParameterSnapshot[]
}

export interface VisualizationState {
  epoch: number
  liveEpoch: number
  telemetry: EpochTelemetry
  weights: WeightMatrix[]
  biases: BiasVector[]
  weightDeltas: number[][][]
  trace: ForwardTrace | null
  isHistorical: boolean
}

interface TrainingSession {
  status: TrainingStatus
  config: TrainingConfig
  network: Network
  dataset: Dataset
  split: DatasetSplit
  weights: WeightMatrix[]
  biases: BiasVector[]
  optimizer: OptimizerBuffers
  epoch: number
  initializationSeed: number
  datasetSignature: string
  networkSignature: string
  history: EpochTelemetry[]
  parameterHistory: EpochParameterSnapshot[]
}

interface NetworkContextType {
  network: Network
  dataset: Dataset | null
  datasetConfig: DatasetConfig
  datasetSplit: DatasetSplit
  trainConfig: TrainingConfig
  currentRun: TrainingRunSnapshot | null
  previousRun: TrainingRunSnapshot | null
  trainingSpeed: TrainingSpeed
  isTraining: boolean
  isPaused: boolean
  selectedSampleIndex: number | null
  replayEpoch: number | null
  visualizationState: VisualizationState
  updateTrainingConfig: (config: Partial<TrainingConfig>) => void
  resetTrainingConfig: () => void
  updateDatasetConfig: (config: Partial<DatasetConfig>) => void
  regenerateDataset: () => void
  setTrainingSpeed: (speed: TrainingSpeed) => void
  startTraining: () => void
  pauseTraining: () => void
  resumeTraining: () => void
  stepTraining: () => void
  stopTraining: () => void
  resetTraining: () => void
  selectSample: (sampleIndex: number | null) => void
  setReplayEpoch: (epoch: number) => void
  returnToLive: () => void
}

const NetworkContext = createContext<NetworkContextType | undefined>(undefined)
const MODEL_SEED_BASE = 1337
const INITIALIZATION_TRANSITION_MS = 220
const SPEED_SETTINGS: Record<TrainingSpeed, { delay: number; epochsPerFrame: number }> = {
  slow: { delay: 125, epochsPerFrame: 1 },
  medium: { delay: 50, epochsPerFrame: 1 },
  fast: { delay: 25, epochsPerFrame: 1 },
}
const initialNetwork = cloneNetwork(DEFAULT_NETWORK)
const initialDatasetConfig = { ...DEFAULT_DATASET_CONFIG }
const initialDataset = generateDataset(initialDatasetConfig)
const initialTrainingConfig = { ...DEFAULT_TRAINING_CONFIG }
const initialParameters = createZeroNetworkParameters(initialNetwork.layers)
const initialSplit = createDatasetSplit(initialDataset, initialTrainingConfig.trainTestSplit)
const initialProgress = createBaselineTelemetry(
  initialNetwork,
  initialDataset,
  initialTrainingConfig,
  initialParameters.weights,
  initialParameters.biases,
  initialSplit,
)

export function NetworkProvider({ children }: { children: ReactNode }) {
  const network = initialNetwork
  const [datasetConfig, setDatasetConfig] = useState<DatasetConfig>(() => ({ ...initialDatasetConfig }))
  const [dataset, setDatasetState] = useState<Dataset>(() => generateDataset(initialDatasetConfig))
  const [trainConfig, setTrainConfig] = useState<TrainingConfig>(() => ({ ...initialTrainingConfig }))
  const [weights, setWeights] = useState<WeightMatrix[]>(() => cloneWeights(initialParameters.weights))
  const [biases, setBiases] = useState<BiasVector[]>(() => cloneBiases(initialParameters.biases))
  const [weightDeltas, setWeightDeltas] = useState<number[][][]>(() => createZeroWeightDeltas(initialParameters.weights))
  const [trainingProgress, setTrainingProgress] = useState<EpochTelemetry>(initialProgress)
  const [currentRun, setCurrentRunState] = useState<TrainingRunSnapshot | null>(null)
  const [previousRun, setPreviousRun] = useState<TrainingRunSnapshot | null>(null)
  const [selectedSampleIndex, setSelectedSampleIndex] = useState<number | null>(null)
  const [replayEpoch, setReplayEpochState] = useState<number | null>(null)
  const [trainingSpeed, setTrainingSpeedState] = useState<TrainingSpeed>("medium")
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const sessionRef = useRef<TrainingSession | null>(null)
  const currentRunRef = useRef<TrainingRunSnapshot | null>(null)
  const speedRef = useRef<TrainingSpeed>("medium")

  const datasetSplit = useMemo(
    () => createDatasetSplit(dataset, trainConfig.trainTestSplit),
    [dataset, trainConfig.trainTestSplit],
  )
  const isTraining = currentRun?.status === "running" || currentRun?.status === "initializing"
  const isPaused = currentRun?.status === "paused"

  const visualizationState = useMemo<VisualizationState>(() => {
    const parameterSnapshot = replayEpoch === null
      ? null
      : currentRun?.parameterHistory.find((snapshot) => snapshot.epoch === replayEpoch) ?? null
    const telemetrySnapshot = parameterSnapshot
      ? currentRun?.history.find((item) => item.epoch === parameterSnapshot.epoch) ?? trainingProgress
      : trainingProgress
    const displayedWeights = parameterSnapshot?.weights ?? weights
    const displayedBiases = parameterSnapshot?.biases ?? biases
    const displayedDeltas = parameterSnapshot?.weightDeltas ?? weightDeltas
    const selectedFeature = selectedSampleIndex === null ? undefined : dataset.features[selectedSampleIndex]
    const selectedLabel = selectedSampleIndex === null ? undefined : dataset.labels[selectedSampleIndex]
    const numericLabel = selectedLabel === undefined
      ? undefined
      : Array.isArray(selectedLabel) ? (selectedLabel[0] ?? 0) : selectedLabel
    const trace = selectedFeature
      ? traceForwardPass(network.layers, displayedWeights, displayedBiases, selectedFeature, numericLabel, trainConfig.loss)
      : null

    return {
      epoch: telemetrySnapshot.epoch,
      liveEpoch: trainingProgress.epoch,
      telemetry: telemetrySnapshot,
      weights: displayedWeights,
      biases: displayedBiases,
      weightDeltas: displayedDeltas,
      trace,
      isHistorical: parameterSnapshot !== null,
    }
  }, [biases, currentRun, dataset.features, dataset.labels, network.layers, replayEpoch, selectedSampleIndex, trainConfig.loss, trainingProgress, weightDeltas, weights])

  const clearTimer = () => {
    if (timerRef.current) {
      clearTimeout(timerRef.current)
      timerRef.current = null
    }
  }

  const publishRun = (run: TrainingRunSnapshot | null) => {
    currentRunRef.current = run
    setCurrentRunState(run)
  }

  const invalidateActiveSession = () => {
    clearTimer()
    setReplayEpochState(null)
    if (sessionRef.current) {
      sessionRef.current.status = "stopped"
      const run = currentRunRef.current
      if (run) publishRun({ ...run, status: "stopped" })
      sessionRef.current = null
    }
  }

  const resetModel = (nextNetwork: Network, nextDataset: Dataset, nextConfig: TrainingConfig) => {
    const parameters = createZeroNetworkParameters(nextNetwork.layers)
    const split = createDatasetSplit(nextDataset, nextConfig.trainTestSplit)
    setWeights(parameters.weights)
    setBiases(parameters.biases)
    setWeightDeltas(createZeroWeightDeltas(parameters.weights))
    setTrainingProgress(
      createBaselineTelemetry(nextNetwork, nextDataset, nextConfig, parameters.weights, parameters.biases, split),
    )
  }

  const executeEpoch = (session: TrainingSession) => {
    if (session.epoch >= session.config.epochs) return
    session.epoch += 1
    const diagnostics = trainEpoch(
      session.network.layers,
      session.weights,
      session.biases,
      session.dataset,
      session.split.trainIndices,
      session.config,
      session.optimizer,
      session.epoch,
    )
    const trainMetrics = evaluateNetwork(
      session.network.layers,
      session.weights,
      session.biases,
      session.dataset,
      session.split.trainIndices,
      session.config.loss,
    )
    const validationMetrics = evaluateNetwork(
      session.network.layers,
      session.weights,
      session.biases,
      session.dataset,
      session.split.testIndices,
      session.config.loss,
    )
    const telemetry: EpochTelemetry = {
      epoch: session.epoch,
      trainLoss: trainMetrics.loss,
      validationLoss: validationMetrics.loss,
      trainAccuracy: trainMetrics.accuracy,
      validationAccuracy: validationMetrics.accuracy,
      meanConfidence: validationMetrics.meanConfidence,
      effectiveLearningRate: diagnostics.effectiveLearningRate,
      rawGradientNorm: diagnostics.rawGradientNorm,
      appliedGradientNorm: diagnostics.appliedGradientNorm,
      updateNorm: diagnostics.updateNorm,
      weightNorm: diagnostics.weightNorm,
      regularizationPenalty: diagnostics.regularizationPenalty,
      clippedFraction: diagnostics.clippedFraction,
      batchLossVariance: diagnostics.batchLossVariance,
    }

    session.history = [...session.history, telemetry]
    session.parameterHistory = [
      ...session.parameterHistory,
      createParameterSnapshot(session.epoch, session.weights, session.biases, diagnostics.weightDeltas),
    ]
    setWeights(cloneWeights(session.weights))
    setBiases(cloneBiases(session.biases))
    setWeightDeltas(diagnostics.weightDeltas)
    setTrainingProgress(telemetry)

    if (session.epoch >= session.config.epochs) {
      session.status = "complete"
      clearTimer()
    }
    publishRun(createRunSnapshot(session))
  }

  const scheduleSession = (session: TrainingSession) => {
    clearTimer()
    session.status = "running"
    publishRun(createRunSnapshot(session))
    const speed = SPEED_SETTINGS[speedRef.current]
    const renderNextFrame = () => {
      if (session.status !== "running") return
      for (let frameEpoch = 0; frameEpoch < speed.epochsPerFrame && session.epoch < session.config.epochs; frameEpoch += 1) {
        executeEpoch(session)
      }
      if (session.status === "running") {
        timerRef.current = setTimeout(renderNextFrame, speed.delay)
      }
    }
    timerRef.current = setTimeout(renderNextFrame, speed.delay)
  }

  const createSession = (status: TrainingStatus): TrainingSession | null => {
    if (!dataset || network.layers.length < 2) return null
    setReplayEpochState(null)
    const initializationSeed = MODEL_SEED_BASE + (dataset.seed ?? 0)
    const parameters = initializeNetworkParameters(network.layers, initializationSeed)
    const zeroParameters = createZeroNetworkParameters(network.layers)
    const split = createDatasetSplit(dataset, trainConfig.trainTestSplit)
    const baseline = createBaselineTelemetry(network, dataset, trainConfig, zeroParameters.weights, zeroParameters.biases, split)
    const datasetSignature = getDatasetSignature(dataset, trainConfig)
    const networkSignature = getNetworkSignature(network)
    const lastRun = currentRunRef.current

    if (
      lastRun &&
      lastRun.history.length > 1 &&
      lastRun.datasetSignature === datasetSignature &&
      lastRun.networkSignature === networkSignature &&
      lastRun.initializationSeed === initializationSeed
    ) {
      setPreviousRun(lastRun)
    } else {
      setPreviousRun(null)
    }

    const session: TrainingSession = {
      status,
      config: { ...trainConfig },
      network: cloneNetwork(network),
      dataset,
      split,
      weights: cloneWeights(parameters.weights),
      biases: cloneBiases(parameters.biases),
      optimizer: createOptimizerBuffers(parameters.weights, parameters.biases),
      epoch: 0,
      initializationSeed,
      datasetSignature,
      networkSignature,
      history: [baseline],
      parameterHistory: [createParameterSnapshot(0, zeroParameters.weights, zeroParameters.biases, createZeroWeightDeltas(zeroParameters.weights))],
    }
    sessionRef.current = session
    setWeights(cloneWeights(zeroParameters.weights))
    setBiases(cloneBiases(zeroParameters.biases))
    setWeightDeltas(createZeroWeightDeltas(zeroParameters.weights))
    setTrainingProgress(baseline)
    publishRun(createRunSnapshot(session))
    return session
  }

  const beginInitialization = (session: TrainingSession, nextAction: "run" | "step") => {
    clearTimer()
    session.status = "initializing"
    setWeights(cloneWeights(session.weights))
    setBiases(cloneBiases(session.biases))
    setWeightDeltas(createZeroWeightDeltas(session.weights))
    publishRun(createRunSnapshot(session))
    timerRef.current = setTimeout(() => {
      if (session.status !== "initializing") return
      if (nextAction === "run") {
        scheduleSession(session)
        return
      }
      executeEpoch(session)
      if (session.epoch < session.config.epochs) {
        session.status = "paused"
        publishRun(createRunSnapshot(session))
      }
    }, INITIALIZATION_TRANSITION_MS)
  }

  useEffect(() => () => clearTimer(), [])

  const updateTrainingConfig = (updates: Partial<TrainingConfig>) => {
    invalidateActiveSession()
    const provisional = sanitizeTrainingConfig({ ...trainConfig, ...updates }, trainConfig)
    const split = createDatasetSplit(dataset, provisional.trainTestSplit)
    const nextConfig = { ...provisional, batchSize: Math.min(provisional.batchSize, split.trainIndices.length) }
    setTrainConfig(nextConfig)
  }

  const resetTrainingConfig = () => {
    invalidateActiveSession()
    const split = createDatasetSplit(dataset, DEFAULT_TRAINING_CONFIG.trainTestSplit)
    setTrainConfig({
      ...DEFAULT_TRAINING_CONFIG,
      batchSize: Math.min(DEFAULT_TRAINING_CONFIG.batchSize, split.trainIndices.length),
    })
  }

  const applyDataset = (nextDatasetConfig: DatasetConfig) => {
    invalidateActiveSession()
    const sanitized = sanitizeDatasetConfig(nextDatasetConfig)
    const nextDataset = generateDataset(sanitized)
    const nextSplit = createDatasetSplit(nextDataset, trainConfig.trainTestSplit)
    const nextConfig = { ...trainConfig, batchSize: Math.min(trainConfig.batchSize, nextSplit.trainIndices.length) }
    setSelectedSampleIndex(null)
    setDatasetConfig(sanitized)
    setDatasetState(nextDataset)
    setTrainConfig(nextConfig)
    publishRun(null)
    setPreviousRun(null)
    resetModel(network, nextDataset, nextConfig)
  }

  const updateDatasetConfig = (updates: Partial<DatasetConfig>) => {
    applyDataset({ ...datasetConfig, ...updates })
  }

  const regenerateDataset = () => applyDataset({ ...datasetConfig, seed: datasetConfig.seed + 1 })

  const setTrainingSpeed = (speed: TrainingSpeed) => {
    speedRef.current = speed
    setTrainingSpeedState(speed)
    const session = sessionRef.current
    if (session?.status === "running") scheduleSession(session)
  }

  const startTraining = () => {
    setReplayEpochState(null)
    if (sessionRef.current?.status === "paused") {
      scheduleSession(sessionRef.current)
      return
    }
    const session = createSession("initializing")
    if (session) beginInitialization(session, "run")
  }

  const pauseTraining = () => {
    const session = sessionRef.current
    if (!session || (session.status !== "running" && session.status !== "initializing")) return
    clearTimer()
    session.status = "paused"
    publishRun(createRunSnapshot(session))
  }

  const resumeTraining = () => {
    const session = sessionRef.current
    if (session?.status === "paused") {
      setReplayEpochState(null)
      scheduleSession(session)
    }
  }

  const stepTraining = () => {
    setReplayEpochState(null)
    let session = sessionRef.current
    if (!session || session.status === "complete" || session.status === "stopped") {
      session = createSession("initializing")
      if (session) beginInitialization(session, "step")
      return
    } else {
      clearTimer()
      session.status = "paused"
    }
    if (!session) return
    executeEpoch(session)
    if (session.epoch < session.config.epochs) {
      session.status = "paused"
      publishRun(createRunSnapshot(session))
    }
  }

  const stopTraining = () => {
    const session = sessionRef.current
    if (!session) return
    clearTimer()
    session.status = "stopped"
    publishRun(createRunSnapshot(session))
    sessionRef.current = null
  }

  const resetTraining = () => {
    clearTimer()
    sessionRef.current = null
    setReplayEpochState(null)
    const retainedRun = currentRunRef.current
    publishRun(retainedRun ? { ...retainedRun, status: "idle" } : null)
    resetModel(network, dataset, trainConfig)
  }

  const selectSample = (sampleIndex: number | null) => {
    if (sampleIndex === null || sampleIndex < 0 || sampleIndex >= dataset.features.length) {
      setSelectedSampleIndex(null)
      return
    }
    setSelectedSampleIndex(sampleIndex)
  }

  const setReplayEpoch = (epoch: number) => {
    if (isTraining || !currentRun) return
    const nearest = currentRun.parameterHistory.reduce<EpochParameterSnapshot | null>((best, snapshot) =>
      !best || Math.abs(snapshot.epoch - epoch) < Math.abs(best.epoch - epoch) ? snapshot : best, null)
    if (nearest) setReplayEpochState(nearest.epoch)
  }

  const returnToLive = () => setReplayEpochState(null)

  const value: NetworkContextType = {
    network,
    dataset,
    datasetConfig,
    datasetSplit,
    trainConfig,
    currentRun,
    previousRun,
    trainingSpeed,
    isTraining,
    isPaused,
    selectedSampleIndex,
    replayEpoch,
    visualizationState,
    updateTrainingConfig,
    resetTrainingConfig,
    updateDatasetConfig,
    regenerateDataset,
    setTrainingSpeed,
    startTraining,
    pauseTraining,
    resumeTraining,
    stepTraining,
    stopTraining,
    resetTraining,
    selectSample,
    setReplayEpoch,
    returnToLive,
  }

  return <NetworkContext.Provider value={value}>{children}</NetworkContext.Provider>
}

export function useNetwork() {
  const context = useContext(NetworkContext)
  if (context === undefined) throw new Error("useNetwork must be used within a NetworkProvider")
  return context
}

function createBaselineTelemetry(
  network: Network,
  dataset: Dataset,
  config: TrainingConfig,
  weights: WeightMatrix[],
  biases: BiasVector[],
  split: DatasetSplit,
): EpochTelemetry {
  const trainMetrics = evaluateNetwork(network.layers, weights, biases, dataset, split.trainIndices, config.loss)
  const validationMetrics = evaluateNetwork(network.layers, weights, biases, dataset, split.testIndices, config.loss)
  const weightNorm = calculateWeightNorm(weights)
  return {
    epoch: 0,
    trainLoss: trainMetrics.loss,
    validationLoss: validationMetrics.loss,
    trainAccuracy: trainMetrics.accuracy,
    validationAccuracy: validationMetrics.accuracy,
    meanConfidence: validationMetrics.meanConfidence,
    effectiveLearningRate: config.learningRate,
    rawGradientNorm: 0,
    appliedGradientNorm: 0,
    updateNorm: 0,
    weightNorm,
    regularizationPenalty: 0.5 * config.weightDecay * weightNorm * weightNorm,
    clippedFraction: 0,
    batchLossVariance: 0,
  }
}

function createRunSnapshot(session: TrainingSession): TrainingRunSnapshot {
  return {
    status: session.status,
    config: { ...session.config },
    datasetSignature: session.datasetSignature,
    networkSignature: session.networkSignature,
    initializationSeed: session.initializationSeed,
    history: [...session.history],
    parameterHistory: [...session.parameterHistory],
  }
}

function createParameterSnapshot(
  epoch: number,
  weights: WeightMatrix[],
  biases: BiasVector[],
  weightDeltas: number[][][],
): EpochParameterSnapshot {
  return {
    epoch,
    weights: cloneWeights(weights),
    biases: cloneBiases(biases),
    weightDeltas: weightDeltas.map((matrix) => matrix.map((row) => [...row])),
  }
}

function sanitizeTrainingConfig(config: TrainingConfig, fallback: TrainingConfig): TrainingConfig {
  return {
    ...config,
    learningRate: finiteClamp(config.learningRate, 0.00001, 0.5, fallback.learningRate),
    epochs: Math.round(finiteClamp(config.epochs, 1, 2000, fallback.epochs)),
    batchSize: Math.round(finiteClamp(config.batchSize, 1, 1000, fallback.batchSize)),
    trainTestSplit: finiteClamp(config.trainTestSplit, 0.6, 0.9, fallback.trainTestSplit),
    momentum: finiteClamp(config.momentum, 0, 0.99, fallback.momentum),
    weightDecay: finiteClamp(config.weightDecay, 0, 0.1, fallback.weightDecay),
    learningRateDecay: finiteClamp(config.learningRateDecay, 0, 0.1, fallback.learningRateDecay),
    gradientClip: finiteClamp(config.gradientClip, 0, 10, fallback.gradientClip),
  }
}

function sanitizeDatasetConfig(config: DatasetConfig): DatasetConfig {
  return {
    ...config,
    sampleCount: Math.round(finiteClamp(config.sampleCount, 20, 1000, 240)),
    noise: finiteClamp(config.noise, 0, 0.5, 0.08),
    seed: Math.round(finiteClamp(config.seed, 1, 999999, 42)),
  }
}

function getDatasetSignature(dataset: Dataset, config: TrainingConfig): string {
  return [
    dataset.id,
    dataset.seed,
    dataset.features.length,
    dataset.noise,
    dataset.normalized,
    config.trainTestSplit,
  ].join("|")
}

function getNetworkSignature(network: Network): string {
  return network.layers.map((layer) => `${layer.size}:${layer.activation}`).join("|")
}

function finiteClamp(value: number, minimum: number, maximum: number, fallback: number): number {
  return Number.isFinite(value) ? Math.max(minimum, Math.min(maximum, value)) : fallback
}

function cloneNetwork(network: Network): Network {
  return { layers: network.layers.map((layer) => ({ ...layer })) }
}

function cloneWeights(weights: WeightMatrix[]): WeightMatrix[] {
  return weights.map((matrix) => ({ ...matrix, weights: matrix.weights.map((row) => [...row]) }))
}

function cloneBiases(biases: BiasVector[]): BiasVector[] {
  return biases.map((vector) => ({ ...vector, biases: [...vector.biases] }))
}

function createZeroWeightDeltas(weights: WeightMatrix[]): number[][][] {
  return weights.map((matrix) => matrix.weights.map((row) => row.map(() => 0)))
}
