"use client"

import { useEffect, useState } from "react"
import { useNetwork } from "@/lib/context/network-context"

interface HoverState {
  layerIndex: number
  neuronIndex: number
  x: number
  y: number
}

export function GraphCanvas() {
  const { network, loadTemplate, trainingProgress, isTraining, weights, biases } = useNetwork()
  const [hoveredNeuron, setHoveredNeuron] = useState<HoverState | null>(null)

  useEffect(() => {
    if (network.layers.length === 0) {
      loadTemplate("classification")
    }
  }, [network.layers.length, loadTemplate])

  const getNeuronPosition = (layerIndex: number, neuronIndex: number, layerSize: number) => {
    const layerSpacing = 180
    const neuronSpacing = 40
    const neuronRadius = 18

    const x = layerIndex * layerSpacing + neuronRadius + 60
    const y = (neuronIndex - (layerSize - 1) / 2) * neuronSpacing

    return { x, y }
  }

  const getEdgeProperties = (weight: number) => {
    const absWeight = Math.abs(weight)

    // At epoch 0, transitionFactor = 0 (uniform appearance), at later epochs it increases
    const transitionFactor = Math.min(1, trainingProgress.epoch / 20) // Gradually transition over first 20 epochs

    const targetStrokeWidth = Math.max(0.5, Math.min(3.6, absWeight * 3))
    const strokeWidth = 2 + (targetStrokeWidth - 2) * transitionFactor

    const targetColor = weight >= 0 ? { r: 59, g: 130, b: 246 } : { r: 251, g: 191, b: 36 }
    const greyColor = { r: 156, g: 163, b: 175 } // grey-400

    // Interpolate between grey and target color
    const r = Math.round(greyColor.r + (targetColor.r - greyColor.r) * transitionFactor)
    const g = Math.round(greyColor.g + (targetColor.g - greyColor.g) * transitionFactor)
    const b = Math.round(greyColor.b + (targetColor.b - greyColor.b) * transitionFactor)

    const color = `rgb(${r}, ${g}, ${b})`

    const targetOpacity = Math.max(0.2, Math.min(0.9, absWeight * 0.8 + 0.2))
    const opacity = 0.5 + (targetOpacity - 0.5) * transitionFactor

    return { strokeWidth, color, opacity }
  }

  const getIncomingWeights = (layerIndex: number, neuronIndex: number): number[] => {
    if (layerIndex === 0) return [] // Input layer has no incoming weights

    const weightMatrix = weights[layerIndex - 1]
    if (!weightMatrix) return []

    // Get all weights from previous layer neurons to this neuron
    const incomingWeights: number[] = []
    for (let i = 0; i < weightMatrix.weights.length; i++) {
      incomingWeights.push(weightMatrix.weights[i][neuronIndex] || 0)
    }

    return incomingWeights
  }

  const getBias = (layerIndex: number, neuronIndex: number): number => {
    if (layerIndex === 0) return 0 // Input layer has no bias

    const biasVector = biases.find((b) => b.layerIndex === layerIndex)
    return biasVector?.biases[neuronIndex] || 0
  }

  if (network.layers.length === 0) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center text-muted-foreground">
          <p className="text-sm">Loading network...</p>
        </div>
      </div>
    )
  }

  const networkWidth = (network.layers.length - 1) * 180 + 160
  const maxLayerSize = Math.max(...network.layers.map((l) => l.size))
  const networkHeight = maxLayerSize * 40 + 120

  return (
    <div className="h-full flex items-center justify-center overflow-hidden p-4">
      <div className="relative" style={{ width: networkWidth, height: networkHeight }}>
        {/* SVG for edges */}
        <svg
          className="absolute inset-0 pointer-events-none"
          width={networkWidth}
          height={networkHeight}
          viewBox={`0 0 ${networkWidth} ${networkHeight}`}
          style={{ width: networkWidth, height: networkHeight }}
        >
          <defs>
            <linearGradient id="blueGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="rgb(59, 130, 246)" stopOpacity="0.6" />
              <stop offset="100%" stopColor="rgb(59, 130, 246)" stopOpacity="0.9" />
            </linearGradient>
            <linearGradient id="amberGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="rgb(251, 191, 36)" stopOpacity="0.6" />
              <stop offset="100%" stopColor="rgb(251, 191, 36)" stopOpacity="0.9" />
            </linearGradient>
            <linearGradient id="greyGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="rgb(156, 163, 175)" stopOpacity="0.5" />
              <stop offset="100%" stopColor="rgb(156, 163, 175)" stopOpacity="0.5" />
            </linearGradient>
          </defs>

          {network.layers[0] && (
            <>
              {Array.from({ length: network.layers[0].size }).map((_, i) => {
                const pos = getNeuronPosition(0, i, network.layers[0].size)
                const edgeLength = 50
                // Simulate input edge weight
                const weight = 0.5 + Math.random() * 0.3
                const { strokeWidth, color, opacity } = getEdgeProperties(weight)

                return (
                  <line
                    key={`input-${i}`}
                    x1={pos.x - edgeLength}
                    y1={pos.y + networkHeight / 2}
                    x2={pos.x}
                    y2={pos.y + networkHeight / 2}
                    stroke={color}
                    strokeWidth={strokeWidth}
                    opacity={opacity}
                    className="transition-all duration-300"
                  />
                )
              })}
            </>
          )}

          {network.layers.slice(0, -1).map((layer, layerIndex) => {
            const nextLayer = network.layers[layerIndex + 1]
            const connections = []
            const weightMatrix = weights[layerIndex]

            for (let i = 0; i < layer.size; i++) {
              for (let j = 0; j < nextLayer.size; j++) {
                const pos1 = getNeuronPosition(layerIndex, i, layer.size)
                const pos2 = getNeuronPosition(layerIndex + 1, j, nextLayer.size)

                // Get weight value from matrix
                const weight = weightMatrix?.weights[i]?.[j] ?? 0
                const { strokeWidth, color, opacity } = getEdgeProperties(weight)

                const x1 = pos1.x
                const y1 = pos1.y + networkHeight / 2
                const x2 = pos2.x
                const y2 = pos2.y + networkHeight / 2

                connections.push(
                  <line
                    key={`${layerIndex}-${i}-${j}`}
                    x1={x1}
                    y1={y1}
                    x2={x2}
                    y2={y2}
                    stroke={color}
                    strokeWidth={strokeWidth}
                    opacity={opacity}
                    className="transition-all duration-300"
                  />,
                )
              }
            }
            return connections
          })}

          {network.layers.length > 0 && (
            <>
              {Array.from({ length: network.layers[network.layers.length - 1].size }).map((_, i) => {
                const pos = getNeuronPosition(
                  network.layers.length - 1,
                  i,
                  network.layers[network.layers.length - 1].size,
                )
                const edgeLength = 50
                // Simulate output edge weight
                const weight = 0.5 + Math.random() * 0.3
                const { strokeWidth, color, opacity } = getEdgeProperties(weight)

                return (
                  <line
                    key={`output-${i}`}
                    x1={pos.x}
                    y1={pos.y + networkHeight / 2}
                    x2={pos.x + edgeLength}
                    y2={pos.y + networkHeight / 2}
                    stroke={color}
                    strokeWidth={strokeWidth}
                    opacity={opacity}
                    className="transition-all duration-300"
                  />
                )
              })}
            </>
          )}
        </svg>

        {/* Neurons */}
        {network.layers.map((layer, layerIndex) => (
          <div key={layer.id}>
            {Array.from({ length: layer.size }).map((_, nodeIndex) => {
              const pos = getNeuronPosition(layerIndex, nodeIndex, layer.size)

              let neuronColor = "bg-blue-500 border-blue-600"
              if (layer.type === "input") neuronColor = "bg-green-500 border-green-600"
              if (layer.type === "output") neuronColor = "bg-amber-400 border-amber-500"

              const brightness = isTraining ? 0.7 + trainingProgress.accuracy * 0.3 : 1

              return (
                <div
                  key={nodeIndex}
                  className={`absolute w-9 h-9 rounded-full ${neuronColor} border-2 flex items-center justify-center text-white text-xs font-medium shadow-md transition-opacity cursor-pointer`}
                  style={{
                    left: `${pos.x - 18}px`,
                    top: `${pos.y + networkHeight / 2 - 18}px`,
                    opacity: brightness,
                  }}
                  onMouseEnter={(e) => {
                    const rect = e.currentTarget.getBoundingClientRect()
                    setHoveredNeuron({
                      layerIndex,
                      neuronIndex: nodeIndex,
                      x: pos.x + 18,
                      y: pos.y + networkHeight / 2,
                    })
                  }}
                  onMouseLeave={() => setHoveredNeuron(null)}
                  title={`${layer.name} - Neuron ${nodeIndex + 1}`}
                >
                  {nodeIndex + 1}
                </div>
              )
            })}

            {/* Layer label */}
            <div
              className="absolute text-center text-xs"
              style={{
                left: `${layerIndex * 180 + 20}px`,
                top: `${networkHeight / 2 + ((layer.size - 1) / 2) * 40 + 40}px`,
                width: "120px",
              }}
            >
              <div className="font-medium text-foreground">{layer.name}</div>
              <div className="text-muted-foreground">{layer.size} neurons</div>
            </div>
          </div>
        ))}

        {hoveredNeuron && (
          <div
            className="absolute bg-background border border-border rounded-lg shadow-lg p-3 text-xs z-50 pointer-events-none"
            style={{
              left: `${hoveredNeuron.x + 10}px`,
              top: `${hoveredNeuron.y - 40}px`,
              minWidth: "180px",
            }}
          >
            <div className="font-semibold mb-2 text-foreground">
              {network.layers[hoveredNeuron.layerIndex].name} - Neuron {hoveredNeuron.neuronIndex + 1}
            </div>

            {hoveredNeuron.layerIndex > 0 && (
              <>
                <div className="text-muted-foreground mb-1">Incoming Weights:</div>
                <div className="max-h-32 overflow-y-auto mb-2 space-y-0.5">
                  {getIncomingWeights(hoveredNeuron.layerIndex, hoveredNeuron.neuronIndex).map((weight, idx) => (
                    <div key={idx} className="flex justify-between font-mono">
                      <span className="text-muted-foreground">w{idx + 1}:</span>
                      <span className={weight >= 0 ? "text-blue-500" : "text-amber-400"}>{weight.toFixed(4)}</span>
                    </div>
                  ))}
                </div>

                <div className="border-t border-border pt-2 mt-2">
                  <div className="flex justify-between font-mono">
                    <span className="text-muted-foreground">Bias:</span>
                    <span
                      className={
                        getBias(hoveredNeuron.layerIndex, hoveredNeuron.neuronIndex) >= 0
                          ? "text-blue-500"
                          : "text-amber-400"
                      }
                    >
                      {getBias(hoveredNeuron.layerIndex, hoveredNeuron.neuronIndex).toFixed(4)}
                    </span>
                  </div>
                </div>
              </>
            )}

            {hoveredNeuron.layerIndex === 0 && (
              <div className="text-muted-foreground italic">Input neuron (no weights/bias)</div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
