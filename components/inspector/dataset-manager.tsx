"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { useNetworkStore } from "@/lib/context/network-context"
import { BUILTIN_DATASETS } from "@/lib/datasets/builtin-datasets"

export function DatasetManager() {
  const { dataset, setDataset, trainConfig, updateTrainingConfig } = useNetworkStore()

  const handleDatasetChange = (datasetName: string) => {
    const selectedDataset = BUILTIN_DATASETS[datasetName]
    if (selectedDataset) {
      setDataset(selectedDataset)
    }
  }

  const handleTrainTestSplit = ([value]: number[]) => {
    updateTrainingConfig({ trainTestSplit: value / 100 })
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm flex items-center gap-2">
          <span>📊</span>
          Dataset
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <Label className="text-xs">Built-in Datasets</Label>
          <Select onValueChange={handleDatasetChange} defaultValue="xor">
            <SelectTrigger className="mt-1">
              <SelectValue placeholder="Select dataset" />
            </SelectTrigger>
            <SelectContent>
              {Object.entries(BUILTIN_DATASETS).map(([key, dataset]) => (
                <SelectItem key={key} value={key}>
                  {dataset.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {dataset && (
          <div className="space-y-3">
            <div>
              <Label className="text-xs">Current Dataset</Label>
              <div className="mt-1 p-2 bg-muted rounded">
                <div className="text-sm font-medium">{dataset.name}</div>
                <div className="text-xs text-muted-foreground mt-1">
                  {dataset.features?.length || 0} samples, {dataset.features?.[0]?.length || 0} features
                </div>
              </div>
            </div>

            <div>
              <Label className="text-xs">Train/Test Split</Label>
              <Slider
                value={[(trainConfig?.trainTestSplit || 0.8) * 100]}
                onValueChange={handleTrainTestSplit}
                min={50}
                max={95}
                step={5}
                className="mt-2"
              />
              <div className="flex justify-between text-xs text-muted-foreground mt-1">
                <span>Train: {((trainConfig?.trainTestSplit || 0.8) * 100).toFixed(0)}%</span>
                <span>Test: {(100 - (trainConfig?.trainTestSplit || 0.8) * 100).toFixed(0)}%</span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="p-2 bg-muted rounded">
                <div className="font-medium">Training</div>
                <div className="text-muted-foreground">
                  {Math.floor((dataset.features?.length || 0) * (trainConfig?.trainTestSplit || 0.8))} samples
                </div>
              </div>
              <div className="p-2 bg-muted rounded">
                <div className="font-medium">Testing</div>
                <div className="text-muted-foreground">
                  {Math.ceil((dataset.features?.length || 0) * (1 - (trainConfig?.trainTestSplit || 0.8)))} samples
                </div>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
