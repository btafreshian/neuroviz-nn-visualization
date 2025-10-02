"use client"

import type React from "react"

import { useState, useEffect } from "react"
import type { NNNode, NNLayer } from "@/lib/types/neural-network"
import { useTraining } from "@/hooks/use-training"
import { cn } from "@/lib/utils"

interface NeuronNodeProps {
  node: NNNode
  layer: NNLayer
  isSelected: boolean
  onSelect: () => void
  isConnecting: boolean
  onStartConnection: () => void
  onEndConnection: (nodeId: string) => void
}

export function NeuronNode({
  node,
  layer,
  isSelected,
  onSelect,
  isConnecting,
  onStartConnection,
  onEndConnection,
}: NeuronNodeProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [animatedActivation, setAnimatedActivation] = useState(0)

  const { activations, isTraining } = useTraining()

  useEffect(() => {
    const activationKey = `activation_${layer.id}_${node.id}`
    const currentActivation = activations.get(activationKey) ?? 0

    if (isTraining) {
      // Smooth animation towards new activation value
      const animate = () => {
        setAnimatedActivation((prev) => {
          const diff = currentActivation - prev
          if (Math.abs(diff) < 0.01) return currentActivation
          return prev + diff * 0.1 // Smooth interpolation
        })
      }

      const interval = setInterval(animate, 16) // ~60fps
      return () => clearInterval(interval)
    } else {
      setAnimatedActivation(currentActivation)
    }
  }, [activations, isTraining, layer.id, node.id])

  const handleMouseDown = (e: React.MouseEvent) => {
    e.stopPropagation()

    if (e.button === 0) {
      if (e.shiftKey && !isConnecting) {
        onStartConnection()
      } else if (isConnecting) {
        onEndConnection(node.id)
      } else {
        onSelect()
        setIsDragging(true)
      }
    }
  }

  const getNodeColor = () => {
    switch (layer.type) {
      case "input":
        return "bg-blue-500"
      case "output":
        return "bg-green-500"
      default:
        return "bg-purple-500"
    }
  }

  const getActivationIntensity = () => {
    return Math.min(Math.abs(animatedActivation), 1)
  }

  const getActivationGlow = () => {
    const intensity = getActivationIntensity()
    const baseColor = layer.type === "input" ? "59, 130, 246" : layer.type === "output" ? "34, 197, 94" : "168, 85, 247"
    return `0 0 ${intensity * 20}px rgba(${baseColor}, ${intensity * 0.8})`
  }

  const x = node.position?.x ?? layer.position.x
  const y = node.position?.y ?? layer.position.y

  return (
    <div
      className={cn(
        "absolute w-12 h-12 rounded-full border-2 cursor-pointer transition-all duration-200",
        "flex items-center justify-center text-xs font-medium text-white",
        "hover:scale-110 hover:shadow-lg",
        getNodeColor(),
        isSelected && "ring-4 ring-yellow-400 ring-opacity-60",
        isConnecting && "ring-2 ring-blue-400 animate-pulse",
        isDragging && "scale-110 shadow-xl",
        isTraining && "animate-pulse",
      )}
      style={{
        left: x,
        top: y,
        transform: "translate(-50%, -50%)",
        opacity: 0.7 + getActivationIntensity() * 0.3,
        boxShadow: getActivationGlow(),
        scale: 1 + getActivationIntensity() * 0.1,
      }}
      onMouseDown={handleMouseDown}
      title={`Node ${node.id.slice(0, 8)}\nBias: ${node.bias.toFixed(3)}\nActivation: ${animatedActivation.toFixed(3)}`}
    >
      {/* Node content */}
      <div className="text-center">
        <div className="text-[10px] leading-none">
          {layer.type === "input" ? "I" : layer.type === "output" ? "O" : "H"}
        </div>
        <div className="text-[8px] leading-none opacity-80">{animatedActivation.toFixed(1)}</div>
      </div>

      {/* Connection points */}
      {layer.type !== "input" && (
        <div className="absolute -left-1 top-1/2 w-2 h-2 bg-gray-400 rounded-full transform -translate-y-1/2 opacity-60" />
      )}
      {layer.type !== "output" && (
        <div className="absolute -right-1 top-1/2 w-2 h-2 bg-gray-400 rounded-full transform -translate-y-1/2 opacity-60" />
      )}
    </div>
  )
}
