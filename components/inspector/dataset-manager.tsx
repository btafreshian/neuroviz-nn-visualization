"use client"

import type { KeyboardEvent } from "react"
import { Database, RefreshCw } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { useNetwork } from "@/lib/context/network-context"
import { DATASET_DEFINITIONS, type DatasetType } from "@/lib/datasets/builtin-datasets"
import { ParameterEffect } from "./parameter-effect"

export function DatasetManager() {
  const {
    datasetConfig,
    datasetSplit,
    trainConfig,
    isTraining,
    isPaused,
    updateDatasetConfig,
    regenerateDataset,
    updateTrainingConfig,
  } = useNetwork()
  const settingsLocked = isTraining || isPaused
  const definition = DATASET_DEFINITIONS[datasetConfig.type]
  const trainingSamples = datasetSplit.trainIndices.length
  const testingSamples = datasetSplit.testIndices.length
  const batchesPerEpoch = Math.ceil(trainingSamples / Math.max(1, trainConfig.batchSize))

  const handleDatasetChange = (value: string) => {
    const type = value as DatasetType
    updateDatasetConfig({ type, noise: DATASET_DEFINITIONS[type].defaultNoise })
  }

  const commitNumericInput = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter") event.currentTarget.blur()
  }

  return (
    <Card className="gap-4 py-4 shadow-sm">
      <CardHeader className="px-4">
        <CardTitle className="flex items-center gap-2 text-sm">
          <Database className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
          Dataset
        </CardTitle>
        <CardAction>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={regenerateDataset}
            disabled={settingsLocked}
            title="Generate a new sample with the next seed"
            aria-label="Regenerate dataset"
          >
            <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" />
          </Button>
        </CardAction>
      </CardHeader>

      <CardContent className="space-y-4 px-4">
        <fieldset disabled={settingsLocked} className="space-y-4 disabled:opacity-60">
          <div>
            <Label className="text-xs">Problem shape</Label>
            <Select value={datasetConfig.type} onValueChange={handleDatasetChange} disabled={settingsLocked}>
              <SelectTrigger className="mt-1 w-full"><SelectValue /></SelectTrigger>
              <SelectContent>
                {Object.entries(DATASET_DEFINITIONS).map(([key, item]) => (
                  <SelectItem key={key} value={key}>{item.name}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="mt-1.5 text-xs leading-relaxed text-muted-foreground">{definition.description}</p>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <Field label="Samples">
              <Input
                key={datasetConfig.sampleCount}
                type="number"
                defaultValue={datasetConfig.sampleCount}
                onBlur={(event) => updateDatasetConfig({ sampleCount: Number(event.target.value) })}
                onKeyDown={commitNumericInput}
                min={20}
                max={1000}
                step={1}
              />
            </Field>
            <Field label="Data seed">
              <Input
                key={datasetConfig.seed}
                type="number"
                defaultValue={datasetConfig.seed}
                onBlur={(event) => updateDatasetConfig({ seed: Number(event.target.value) })}
                onKeyDown={commitNumericInput}
                min={1}
                max={999999}
                step={1}
              />
            </Field>
          </div>
          <ParameterEffect
            summary="Every sample participates in the exact stratified split below."
            metric={`${batchesPerEpoch} batches/epoch`}
          >
            More samples usually make validation more reliable but increase the work in each epoch. Counts can change one sample at a time.
          </ParameterEffect>

          <div>
            <div className="flex items-center justify-between">
              <Label className="text-xs">Noise</Label>
              <span className="font-mono text-xs text-muted-foreground">σ {datasetConfig.noise.toFixed(2)}</span>
            </div>
            <Slider
              value={[datasetConfig.noise]}
              onValueChange={([noise]) => updateDatasetConfig({ noise })}
              min={0}
              max={0.4}
              step={0.01}
              className="mt-3"
              disabled={settingsLocked}
            />
          </div>
          <ParameterEffect summary="Noise perturbs each generated coordinate before scaling." metric={`σ ${datasetConfig.noise.toFixed(2)}`}>
            Higher noise increases class overlap, usually raising validation loss and making a perfectly sharp boundary impossible.
          </ParameterEffect>

          <div>
            <div className="flex items-center justify-between">
              <Label className="text-xs">Training split</Label>
              <span className="font-mono text-xs text-muted-foreground">{(trainConfig.trainTestSplit * 100).toFixed(0)}%</span>
            </div>
            <Slider
              value={[trainConfig.trainTestSplit * 100]}
              onValueChange={([value]) => updateTrainingConfig({ trainTestSplit: value / 100 })}
              min={60}
              max={90}
              step={1}
              className="mt-3"
              disabled={settingsLocked}
            />
          </div>
          <ParameterEffect summary="The test set stays unseen during parameter updates." metric={`${trainingSamples}/${testingSamples}`}>
            Stratification keeps both classes represented. The displayed counts come from the same indices used by training and validation.
          </ParameterEffect>
        </fieldset>

        <div className="grid grid-cols-3 divide-x overflow-hidden rounded-lg border bg-muted/30 text-center">
          <Stat label="Features" value="2" />
          <Stat label="Train" value={trainingSamples.toLocaleString()} />
          <Stat label="Test" value={testingSamples.toLocaleString()} />
        </div>

        {settingsLocked && <p className="text-xs text-muted-foreground">Stop or finish the current run to change dataset settings.</p>}
      </CardContent>
    </Card>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <div className="space-y-1"><Label className="text-xs">{label}</Label>{children}</div>
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0 px-2 py-2">
      <div className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="truncate font-mono text-xs font-medium">{value}</div>
    </div>
  )
}
