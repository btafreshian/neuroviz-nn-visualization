"use client"

import { Pause, Play, RotateCcw, Square, StepForward } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { useNetwork, type TrainingSpeed } from "@/lib/context/network-context"

export function TrainingControls() {
  const {
    currentRun,
    isTraining,
    isPaused,
    visualizationState,
    trainConfig,
    dataset,
    trainingSpeed,
    setTrainingSpeed,
    startTraining,
    pauseTraining,
    resumeTraining,
    stepTraining,
    stopTraining,
    resetTraining,
  } = useNetwork()
  const displayedProgress = visualizationState.telemetry
  const canTrain = dataset !== null
  const totalEpochs = currentRun?.config.epochs ?? trainConfig.epochs
  const progress = totalEpochs > 0 ? (displayedProgress.epoch / totalEpochs) * 100 : 0
  const status = currentRun?.status ?? "idle"
  const hasCompletedWork = (currentRun?.history.length ?? 0) > 1

  const handlePrimaryAction = () => {
    if (isTraining) pauseTraining()
    else if (isPaused) resumeTraining()
    else startTraining()
  }

  return (
    <div className="p-3 sm:p-4">
      <div className="flex flex-col gap-4 xl:flex-row xl:items-center">
        <div className="flex flex-wrap items-center gap-2">
          <Button
            variant={isTraining ? "secondary" : "default"}
            size="sm"
            className="gap-2"
            onClick={handlePrimaryAction}
            disabled={!canTrain}
          >
            {isTraining ? <Pause className="h-3.5 w-3.5 fill-current" /> : <Play className="h-3.5 w-3.5 fill-current" />}
            {isTraining ? "Pause" : isPaused ? "Resume" : hasCompletedWork ? "New run" : "Train"}
          </Button>

          <Button variant="outline" size="sm" className="gap-2" onClick={stepTraining} disabled={!canTrain || isTraining}>
            <StepForward className="h-3.5 w-3.5" aria-hidden="true" />
            Step
          </Button>

          <Button variant="outline" size="sm" className="gap-2" onClick={stopTraining} disabled={!isTraining && !isPaused}>
            <Square className="h-3 w-3 fill-current" aria-hidden="true" />
            Stop
          </Button>

          <Button variant="outline" size="sm" className="gap-2" onClick={resetTraining}>
            <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
            Reset
          </Button>

          <div className="ml-1 flex items-center gap-2 text-xs text-muted-foreground">
            <span className={`h-2 w-2 rounded-full ${isTraining ? "animate-pulse bg-emerald-500" : isPaused ? "bg-amber-500" : "bg-muted-foreground/50"}`} />
            <span className="capitalize">{status}</span>
          </div>
        </div>

        <div className="flex min-w-0 flex-1 items-center gap-3 xl:max-w-xs">
          <Progress value={progress} className="h-2 flex-1" />
          <span className="w-9 text-right font-mono text-xs text-muted-foreground">{Math.round(progress)}%</span>
        </div>

        <Select value={trainingSpeed} onValueChange={(value) => setTrainingSpeed(value as TrainingSpeed)}>
          <SelectTrigger size="sm" className="w-full sm:w-28" aria-label="Visualization speed">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="slow">Slow</SelectItem>
            <SelectItem value="medium">Medium</SelectItem>
            <SelectItem value="fast">Fast</SelectItem>
          </SelectContent>
        </Select>

        <div className="grid grid-cols-4 divide-x overflow-hidden rounded-lg border bg-muted/30 text-center xl:min-w-[28rem]">
          <Metric label={visualizationState.isHistorical ? "Preview epoch" : "Epoch"} value={`${displayedProgress.epoch}/${totalEpochs}`} />
          <Metric label="Train loss" value={displayedProgress.trainLoss.toFixed(4)} />
          <Metric label="Validation" value={`${(displayedProgress.validationAccuracy * 100).toFixed(1)}%`} />
          <Metric label="Confidence" value={`${(displayedProgress.meanConfidence * 100).toFixed(1)}%`} />
        </div>
      </div>
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0 px-2 py-2">
      <div className="truncate text-[9px] font-medium uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="truncate font-mono text-xs font-medium text-foreground">{value}</div>
    </div>
  )
}
