"use client"

import { Button } from "@/components/ui/button"

interface ViewTransform {
  x: number
  y: number
  scale: number
}

interface CanvasControlsProps {
  viewTransform: ViewTransform
  onFitToNetwork: () => void
  onResetView: () => void
}

export function CanvasControls({ viewTransform, onFitToNetwork, onResetView }: CanvasControlsProps) {
  return (
    <div className="absolute bottom-4 right-4 flex flex-col gap-2">
      <Button variant="outline" size="sm" onClick={onFitToNetwork} title="Fit to network (F)">
        <span>⛶</span>
      </Button>

      <Button variant="outline" size="sm" onClick={onResetView} title="Reset view">
        <span>↻</span>
      </Button>

      <div className="text-xs text-muted-foreground text-center bg-card px-2 py-1 rounded border">
        {Math.round(viewTransform.scale * 100)}%
      </div>
    </div>
  )
}
