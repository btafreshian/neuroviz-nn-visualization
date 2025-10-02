"use client"

import type React from "react"
import type { NNLayer } from "@/lib/types/neural-network"
import { useNetwork } from "@/lib/context/network-context"
import { cn } from "@/lib/utils"

interface LayerColumnProps {
  layer: NNLayer
  isSelected?: boolean
  onSelect?: () => void
}

export function LayerColumn({ layer, isSelected = false, onSelect }: LayerColumnProps) {
  const { updateLayer } = useNetwork()

  const handleAddNode = (e: React.MouseEvent) => {
    e.stopPropagation()
    updateLayer(layer.id, { size: layer.size + 1 })
  }

  const handleRemoveNode = (e: React.MouseEvent) => {
    e.stopPropagation()
    if (layer.size > 1) {
      updateLayer(layer.id, { size: layer.size - 1 })
    }
  }

  return (
    <div
      className={cn("relative pointer-events-auto", isSelected && "ring-2 ring-blue-400 ring-opacity-50 rounded-lg")}
      onClick={onSelect}
    >
      {/* Layer controls (only show when selected and not input/output) */}
      {isSelected && layer.type === "dense" && (
        <div className="absolute -right-12 top-1/2 transform -translate-y-1/2 flex flex-col gap-1">
          <button
            onClick={handleAddNode}
            className="w-6 h-6 bg-primary text-primary-foreground rounded-full flex items-center justify-center hover:bg-primary/90 transition-colors"
            title="Add neuron"
          >
            <span className="text-sm">+</span>
          </button>
          <button
            onClick={handleRemoveNode}
            disabled={layer.size <= 1}
            className="w-6 h-6 bg-destructive text-destructive-foreground rounded-full flex items-center justify-center hover:bg-destructive/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            title="Remove neuron"
          >
            <span className="text-sm">-</span>
          </button>
        </div>
      )}
    </div>
  )
}
