"use client"

import { useEffect, useState } from "react"
import type { NNEdge, NNNode } from "@/lib/types/neural-network"
import { useTraining } from "@/hooks/use-training"
import { cn } from "@/lib/utils"

interface EdgeConnectionProps {
  edge: NNEdge
  fromNode: NNNode
  toNode: NNNode
  isSelected: boolean
  onSelect: () => void
}

export function EdgeConnection({ edge, fromNode, toNode, isSelected, onSelect }: EdgeConnectionProps) {
  const [animatedWeight, setAnimatedWeight] = useState(edge.weight)
  const [animatedGradient, setAnimatedGradient] = useState(0)

  const { weights, gradients, isTraining } = useTraining()

  useEffect(() => {
    const weightKey = `weight_${edge.id}`
    const gradientKey = `weight_grad_${edge.id}`

    const currentWeight = weights.get(weightKey) ?? edge.weight
    const currentGradient = gradients.get(gradientKey) ?? 0

    if (isTraining) {
      // Smooth animation towards new values
      const animate = () => {
        setAnimatedWeight((prev) => {
          const diff = currentWeight - prev
          if (Math.abs(diff) < 0.001) return currentWeight
          return prev + diff * 0.15
        })

        setAnimatedGradient((prev) => {
          const diff = currentGradient - prev
          if (Math.abs(diff) < 0.001) return currentGradient
          return prev + diff * 0.2
        })
      }

      const interval = setInterval(animate, 16)
      return () => clearInterval(interval)
    } else {
      setAnimatedWeight(currentWeight)
      setAnimatedGradient(currentGradient)
    }
  }, [weights, gradients, isTraining, edge.id, edge.weight])

  if (!fromNode || !toNode) return null

  const fromX = fromNode.position?.x ?? 0
  const fromY = fromNode.position?.y ?? 0
  const toX = toNode.position?.x ?? 0
  const toY = toNode.position?.y ?? 0

  // Calculate control points for bezier curve
  const controlOffset = Math.abs(toX - fromX) * 0.5
  const controlX1 = fromX + controlOffset
  const controlY1 = fromY
  const controlX2 = toX - controlOffset
  const controlY2 = toY

  const pathData = `M ${fromX + 24} ${fromY} C ${controlX1} ${controlY1}, ${controlX2} ${controlY2}, ${toX - 24} ${toY}`

  const weightMagnitude = Math.abs(animatedWeight)
  const gradientMagnitude = Math.abs(animatedGradient)

  // Dynamic stroke width based on weight magnitude
  const strokeWidth = Math.max(1, Math.min(8, weightMagnitude * 15))

  // Dynamic opacity based on weight magnitude
  const opacity = Math.max(0.2, Math.min(1, weightMagnitude * 3))

  // Color based on weight sign, with gradient influence
  const getStrokeColor = () => {
    if (isTraining && gradientMagnitude > 0.01) {
      // Show gradient activity with pulsing colors
      return animatedWeight >= 0 ? "#22c55e" : "#ef4444" // Brighter during active learning
    }
    return animatedWeight >= 0 ? "#10b981" : "#dc2626" // Standard colors
  }

  const shouldPulse = isTraining && gradientMagnitude > 0.01

  return (
    <svg className="absolute inset-0 pointer-events-none" style={{ zIndex: 1 }}>
      <defs>
        <linearGradient id={`gradient-${edge.id}`} x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor={getStrokeColor()} stopOpacity={opacity * 0.5} />
          <stop offset="50%" stopColor={getStrokeColor()} stopOpacity={opacity} />
          <stop offset="100%" stopColor={getStrokeColor()} stopOpacity={opacity * 0.5} />
        </linearGradient>
      </defs>

      <path
        d={pathData}
        fill="none"
        stroke={isSelected ? "#facc15" : `url(#gradient-${edge.id})`}
        strokeWidth={strokeWidth}
        opacity={isSelected ? 1 : opacity}
        className={cn(
          "transition-all duration-200 pointer-events-auto cursor-pointer",
          shouldPulse && "animate-pulse",
          isSelected && "stroke-yellow-400",
        )}
        onClick={(e) => {
          e.stopPropagation()
          onSelect()
        }}
        title={`Weight: ${animatedWeight.toFixed(4)}\nGradient: ${animatedGradient.toFixed(4)}`}
        style={{
          filter: shouldPulse ? `drop-shadow(0 0 ${gradientMagnitude * 10}px ${getStrokeColor()})` : undefined,
        }}
      />

      {/* Weight label */}
      {isSelected && (
        <text
          x={(fromX + toX) / 2}
          y={(fromY + toY) / 2 - 10}
          textAnchor="middle"
          className="fill-foreground text-xs font-mono pointer-events-none"
          style={{ fontSize: "10px" }}
        >
          {animatedWeight.toFixed(3)}
        </text>
      )}

      {isSelected && isTraining && gradientMagnitude > 0.001 && (
        <text
          x={(fromX + toX) / 2}
          y={(fromY + toY) / 2 + 15}
          textAnchor="middle"
          className="fill-muted-foreground text-xs font-mono pointer-events-none"
          style={{ fontSize: "8px" }}
        >
          ∇{animatedGradient.toFixed(3)}
        </text>
      )}
    </svg>
  )
}
