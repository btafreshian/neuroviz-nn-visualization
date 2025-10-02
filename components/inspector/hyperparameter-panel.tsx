"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Slider } from "@/components/ui/slider"
import { useNetwork } from "@/lib/context/network-context"

export function HyperparameterPanel() {
  const { trainConfig, updateTrainingConfig } = useNetwork()

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm flex items-center gap-2">⚙️ Hyperparameters</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <Label className="text-xs">Learning Rate</Label>
          <Input
            type="number"
            value={trainConfig.learningRate}
            onChange={(e) => updateTrainingConfig({ learningRate: Number.parseFloat(e.target.value) })}
            step="0.001"
            min="0.0001"
            max="1"
            className="mt-1"
          />
        </div>

        <div>
          <Label className="text-xs">Batch Size</Label>
          <Slider
            value={[trainConfig.batchSize]}
            onValueChange={([value]) => updateTrainingConfig({ batchSize: value })}
            min={1}
            max={128}
            step={1}
            className="mt-2"
          />
          <div className="text-xs text-muted-foreground mt-1">{trainConfig.batchSize} samples</div>
        </div>

        <div>
          <Label className="text-xs">Max Epochs</Label>
          <Input
            type="number"
            value={trainConfig.epochs}
            onChange={(e) => updateTrainingConfig({ epochs: Number.parseInt(e.target.value) })}
            min="1"
            max="10000"
            className="mt-1"
          />
        </div>
      </CardContent>
    </Card>
  )
}
