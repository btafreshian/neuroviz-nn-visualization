"use client"

import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { useNetwork } from "@/lib/context/network-context"

export function TrainingControls() {
  const { isTraining, trainingProgress, trainConfig, dataset, startTraining, stopTraining, resetTraining } =
    useNetwork()

  const handleStartPause = () => {
    if (isTraining) {
      stopTraining()
    } else {
      startTraining()
    }
  }

  const handleReset = () => {
    resetTraining()
  }

  const canTrain = dataset !== null
  const progress = trainConfig.epochs > 0 ? (trainingProgress.epoch / trainConfig.epochs) * 100 : 0

  return (
    <div className="border-t bg-card p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Button
            variant={isTraining ? "secondary" : "default"}
            size="sm"
            onClick={handleStartPause}
            disabled={!canTrain}
          >
            {isTraining ? (
              <>
                <span className="mr-2">⏹</span>
                Stop
              </>
            ) : (
              <>
                <span className="mr-2">▶</span>
                Train
              </>
            )}
          </Button>

          <Button variant="outline" size="sm" onClick={handleReset}>
            <span className="mr-2">↻</span>
            Reset
          </Button>
        </div>

        <div className="flex items-center gap-4">
          {/* Progress bar */}
          {isTraining && (
            <div className="flex items-center gap-2">
              <Progress value={progress} className="w-32" />
              <span className="text-xs text-muted-foreground">{Math.round(progress)}%</span>
            </div>
          )}

          {/* Metrics */}
          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            <span>
              Epoch: {trainingProgress.epoch}/{trainConfig.epochs}
            </span>
            <span>Loss: {trainingProgress.loss.toFixed(4)}</span>
            <span>Accuracy: {(trainingProgress.accuracy * 100).toFixed(1)}%</span>
          </div>

          {/* Status indicator */}
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${isTraining ? "bg-green-500 animate-pulse" : "bg-gray-400"}`} />
            <span className="text-xs capitalize">{isTraining ? "training" : "idle"}</span>
          </div>
        </div>
      </div>

      {/* Training info */}
      {!canTrain && (
        <div className="mt-2 p-2 bg-muted/50 rounded text-sm text-muted-foreground">
          Load a dataset to start training
        </div>
      )}
    </div>
  )
}
