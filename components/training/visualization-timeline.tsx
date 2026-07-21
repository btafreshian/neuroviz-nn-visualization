"use client"

import { Eye, RotateCcw } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { useNetwork } from "@/lib/context/network-context"

export function VisualizationTimeline() {
  const { currentRun, isTraining, replayEpoch, visualizationState, setReplayEpoch, returnToLive } = useNetwork()
  const snapshots = currentRun?.parameterHistory ?? []
  if (isTraining || snapshots.length <= 1) return null

  const maximumEpoch = snapshots.at(-1)?.epoch ?? 0
  const displayedEpoch = replayEpoch ?? visualizationState.liveEpoch

  return (
    <div className="border-t bg-muted/15 px-4 py-3 lg:px-6">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2 text-xs">
        <div className="flex items-center gap-2">
          <Eye className="h-3.5 w-3.5 text-muted-foreground" aria-hidden="true" />
          <span className="font-medium">Epoch replay</span>
          <span className="text-muted-foreground">
            {visualizationState.isHistorical
              ? `Previewing epoch ${visualizationState.epoch} · live model remains at epoch ${visualizationState.liveEpoch}`
              : `Live model · epoch ${visualizationState.liveEpoch}`}
          </span>
        </div>
        {visualizationState.isHistorical && (
          <Button variant="ghost" size="sm" className="h-7 gap-1.5 px-2 text-xs" onClick={returnToLive}>
            <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
            Return to live
          </Button>
        )}
      </div>
      <div className="grid grid-cols-[auto_1fr_auto] items-center gap-3">
        <span className="w-5 text-right font-mono text-[10px] text-muted-foreground">0</span>
        <Slider
          aria-label="Preview training epoch"
          value={[Math.min(maximumEpoch, displayedEpoch)]}
          min={0}
          max={maximumEpoch}
          step={1}
          onValueChange={([epoch]) => setReplayEpoch(epoch)}
        />
        <span className="w-8 font-mono text-[10px] text-muted-foreground">{maximumEpoch}</span>
      </div>
    </div>
  )
}
