"use client"

import type { ReactNode } from "react"
import { RotateCcw, SlidersHorizontal } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { useNetwork, type TrainingConfig } from "@/lib/context/network-context"
import { ParameterEffect } from "./parameter-effect"

export function HyperparameterPanel() {
  const {
    trainConfig,
    visualizationState,
    currentRun,
    isTraining,
    isPaused,
    updateTrainingConfig,
    resetTrainingConfig,
  } = useNetwork()
  const displayedProgress = visualizationState.telemetry
  const settingsLocked = isTraining || isPaused
  const baselineLoss = currentRun?.history[0]?.validationLoss ?? displayedProgress.validationLoss
  const lossReduction = baselineLoss > 0 ? 1 - displayedProgress.validationLoss / baselineLoss : 0

  return (
    <Card className="gap-4 py-4 shadow-sm">
      <CardHeader className="px-4">
        <CardTitle className="flex items-center gap-2 text-sm">
          <SlidersHorizontal className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
          Hyperparameters
        </CardTitle>
        <CardAction>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={resetTrainingConfig}
            disabled={settingsLocked}
            title="Restore defaults"
            aria-label="Reset hyperparameters"
          >
            <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
          </Button>
        </CardAction>
      </CardHeader>

      <CardContent className="px-4">
        <fieldset disabled={settingsLocked} className="space-y-4 disabled:opacity-60">
          <Field label="Loss function">
            <Select
              value={trainConfig.loss}
              onValueChange={(loss) => updateTrainingConfig({ loss: loss as TrainingConfig["loss"] })}
              disabled={settingsLocked}
            >
              <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="crossentropy">Cross-entropy</SelectItem>
                <SelectItem value="mse">Mean squared error</SelectItem>
              </SelectContent>
            </Select>
            <ParameterEffect
              summary="Controls how prediction error becomes the learning signal."
              metric={`${Math.max(0, lossReduction * 100).toFixed(0)}% lower`}
            >
              {trainConfig.loss === "crossentropy"
                ? "Cross-entropy strongly penalizes confident incorrect predictions."
                : "Mean squared error applies a gentler squared penalty to probability errors."}
              <div className="mt-1 font-mono">
                Train {displayedProgress.trainLoss.toFixed(4)} · Validation {displayedProgress.validationLoss.toFixed(4)}
              </div>
            </ParameterEffect>
          </Field>

          <Field label="Learning rate">
            <Input
              type="number"
              value={trainConfig.learningRate}
              onChange={(event) => updateTrainingConfig({ learningRate: Number(event.target.value) })}
              step={0.0001}
              min={0.00001}
              max={0.5}
            />
            <ParameterEffect
              summary="Sets how far weights move during each learning update."
              metric={formatCompact(displayedProgress.effectiveLearningRate)}
            >
              Larger values can learn faster but may overshoot; smaller values are steadier but need more epochs.
            </ParameterEffect>
          </Field>

          <Field label="Max epochs">
            <Input
              type="number"
              value={trainConfig.epochs}
              onChange={(event) => updateTrainingConfig({ epochs: Number(event.target.value) })}
              min={1}
              max={2000}
              step={1}
            />
            <ParameterEffect
              summary="Limits the number of complete passes through the training data."
              metric={`${displayedProgress.epoch}/${trainConfig.epochs}`}
            >
              More epochs give the network additional chances to reduce loss and refine its decision boundary.
            </ParameterEffect>
          </Field>
        </fieldset>

        {settingsLocked && (
          <p className="mt-4 text-xs text-muted-foreground">
            Pause preserves the run; stop or finish it before changing these settings.
          </p>
        )}
      </CardContent>
    </Card>
  )
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return <div className="space-y-1.5"><Label className="text-xs">{label}</Label>{children}</div>
}

function formatCompact(value: number): string {
  if (!Number.isFinite(value) || value === 0) return "0"
  if (Math.abs(value) < 0.001) return value.toExponential(1)
  return value.toFixed(value < 0.1 ? 4 : 3)
}
