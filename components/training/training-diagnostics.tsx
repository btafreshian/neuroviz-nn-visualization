"use client"

import { Activity } from "lucide-react"
import { useNetwork } from "@/lib/context/network-context"

export function TrainingDiagnostics() {
  const { visualizationState } = useNetwork()
  const trainingProgress = visualizationState.telemetry
  const generalizationGap = trainingProgress.trainAccuracy - trainingProgress.validationAccuracy
  const updateRatio = trainingProgress.weightNorm > 0 ? trainingProgress.updateNorm / trainingProgress.weightNorm : 0

  return (
    <div className="border-t bg-muted/20 px-4 py-3">
      <div className="mb-2 flex items-center gap-2 text-xs font-medium">
        <Activity className="h-3.5 w-3.5 text-muted-foreground" aria-hidden="true" />
        Training diagnostics
        <span className="font-normal text-muted-foreground">Measured from the current epoch</span>
      </div>
      <div className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border bg-border sm:grid-cols-4 xl:grid-cols-7">
        <Diagnostic label="Effective LR" value={formatCompact(trainingProgress.effectiveLearningRate)} />
        <Diagnostic label="Gradient RMS" value={formatCompact(trainingProgress.rawGradientNorm)} />
        <Diagnostic label="Update / weight" value={formatPercent(updateRatio)} />
        <Diagnostic label="Weight norm" value={formatCompact(trainingProgress.weightNorm)} />
        <Diagnostic label="Clipped" value={formatPercent(trainingProgress.clippedFraction)} />
        <Diagnostic label="Confidence" value={formatPercent(trainingProgress.meanConfidence)} />
        <Diagnostic label="Generalization gap" value={formatSignedPercent(generalizationGap)} />
      </div>
    </div>
  )
}

function Diagnostic({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0 bg-card px-2.5 py-2">
      <div className="truncate text-[9px] font-medium uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="truncate font-mono text-xs font-medium">{value}</div>
    </div>
  )
}

function formatCompact(value: number): string {
  if (!Number.isFinite(value) || value === 0) return "—"
  if (Math.abs(value) < 0.001) return value.toExponential(1)
  return value.toFixed(value < 0.1 ? 4 : 3)
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

function formatSignedPercent(value: number): string {
  const amount = (value * 100).toFixed(1)
  return `${value > 0 ? "+" : ""}${amount}%`
}
