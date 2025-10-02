"use client"

import type React from "react"
import { useRef, useState, useCallback, useEffect } from "react"
import { useNetwork } from "@/lib/context/network-context"

interface ViewTransform {
  x: number
  y: number
  scale: number
}

export function GraphCanvas() {
  const canvasRef = useRef<HTMLDivElement>(null)
  const [viewTransform, setViewTransform] = useState<ViewTransform>({ x: 0, y: 0, scale: 1 })
  const [isPanning, setIsPanning] = useState(false)
  const [panStart, setPanStart] = useState({ x: 0, y: 0 })

  const { network, loadTemplate, trainingProgress, isTraining } = useNetwork()

  useEffect(() => {
    console.log("[v0] GraphCanvas: network layers changed:", network.layers.length)
    if (network.layers.length > 0) {
      console.log("[v0] GraphCanvas: fitting network to view")
      fitToNetwork()
    }
  }, [fitToNetwork, network.layers.length])

  // Handle canvas interactions
  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.button === 1 || (e.button === 0 && e.metaKey)) {
        // Middle mouse or Cmd+click for panning
        setIsPanning(true)
        setPanStart({ x: e.clientX - viewTransform.x, y: e.clientY - viewTransform.y })
        e.preventDefault()
      }
    },
    [viewTransform],
  )

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (isPanning) {
        setViewTransform((prev) => ({
          ...prev,
          x: e.clientX - panStart.x,
          y: e.clientY - panStart.y,
        }))
      }
    },
    [isPanning, panStart],
  )

  const handleMouseUp = useCallback(() => {
    setIsPanning(false)
  }, [])

  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      e.preventDefault()
      const rect = canvasRef.current?.getBoundingClientRect()
      if (!rect) return

      const mouseX = e.clientX - rect.left
      const mouseY = e.clientY - rect.top

      const scaleFactor = e.deltaY > 0 ? 0.9 : 1.1
      const newScale = Math.max(0.1, Math.min(3, viewTransform.scale * scaleFactor))

      // Zoom towards mouse position
      const scaleRatio = newScale / viewTransform.scale
      setViewTransform((prev) => ({
        x: mouseX - (mouseX - prev.x) * scaleRatio,
        y: mouseY - (mouseY - prev.y) * scaleRatio,
        scale: newScale,
      }))
    },
    [viewTransform.scale],
  )

  // Auto-fit network on load
  const fitToNetwork = useCallback(() => {
    if (network.layers.length === 0) return

    const rect = canvasRef.current?.getBoundingClientRect()
    if (!rect) return

    console.log("[v0] Fitting network to canvas, rect:", rect.width, "x", rect.height)
    const networkWidth = (network.layers.length - 1) * 200
    setViewTransform({
      x: (rect.width - networkWidth) / 2,
      y: rect.height / 2,
      scale: 1,
    })
  }, [network.layers])

  console.log("[v0] GraphCanvas render: network has", network.layers.length, "layers")

  return (
    <div className="flex-1 relative bg-muted/20 overflow-hidden">
      <div
        ref={canvasRef}
        className="absolute inset-0 cursor-grab active:cursor-grabbing"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onWheel={handleWheel}
      >
        {/* Canvas content with transform */}
        <div
          className="absolute inset-0 origin-top-left"
          style={{
            transform: `translate(${viewTransform.x}px, ${viewTransform.y}px) scale(${viewTransform.scale})`,
          }}
        >
          {/* Neural network visualization */}
          {network.layers.length > 0 ? (
            <div className="absolute top-1/2 left-0 transform -translate-y-1/2">
              {(() => {
                const maxLayerSize = Math.max(...network.layers.map((l) => l.size))
                const svgHeight = maxLayerSize * 60 + 100
                const svgWidth = (network.layers.length - 1) * 200 + 100
                const neuronRadius = 20 // Half of 40px neuron width

                return (
                  <svg
                    className="absolute pointer-events-none"
                    style={{
                      width: `${svgWidth}px`,
                      height: `${svgHeight}px`,
                      left: "0px",
                      top: `${-svgHeight / 2}px`,
                    }}
                  >
                    {network.layers.slice(0, -1).map((layer, layerIndex) => {
                      const nextLayer = network.layers[layerIndex + 1]
                      const connections = []

                      for (let i = 0; i < layer.size; i++) {
                        for (let j = 0; j < nextLayer.size; j++) {
                          // Calculate neuron center positions
                          // X position: layer index * 200 (spacing) + 20 (neuron radius)
                          const x1 = layerIndex * 200 + neuronRadius
                          // Y position: center of SVG + offset based on neuron index
                          const y1 = svgHeight / 2 + (i - (layer.size - 1) / 2) * 60

                          const x2 = (layerIndex + 1) * 200 + neuronRadius
                          const y2 = svgHeight / 2 + (j - (nextLayer.size - 1) / 2) * 60

                          const opacity = isTraining ? 0.2 + trainingProgress.accuracy * 0.5 : 0.3

                          connections.push(
                            <line
                              key={`${layerIndex}-${i}-${j}`}
                              x1={x1}
                              y1={y1}
                              x2={x2}
                              y2={y2}
                              stroke="#64748b"
                              strokeWidth="1"
                              opacity={opacity}
                              className="transition-opacity duration-300"
                            />,
                          )
                        }
                      }
                      return connections
                    })}
                  </svg>
                )
              })()}

              {/* Render neurons */}
              {network.layers.map((layer, layerIndex) => (
                <div
                  key={layer.id}
                  className="absolute"
                  style={{
                    left: `${layerIndex * 200}px`,
                    top: "0px",
                    transform: "translateY(-50%)",
                  }}
                >
                  {/* Layer label */}
                  <div
                    className="text-center mb-4"
                    style={{ transform: `translateY(${(layer.size - 1) * 30 + 60}px)` }}
                  >
                    <div className="text-sm font-semibold text-foreground">{layer.name}</div>
                    <div className="text-xs text-muted-foreground">{layer.size} neurons</div>
                  </div>

                  {/* Neurons */}
                  <div className="flex flex-col items-center">
                    {Array.from({ length: layer.size }).map((_, nodeIndex) => {
                      let neuronColor = "bg-blue-500 border-blue-600"
                      if (layer.type === "input") neuronColor = "bg-green-500 border-green-600"
                      if (layer.type === "output") neuronColor = "bg-red-500 border-red-600"

                      const brightness = isTraining ? 0.7 + trainingProgress.accuracy * 0.3 : 1

                      return (
                        <div
                          key={nodeIndex}
                          className={`w-10 h-10 rounded-full ${neuronColor} border-2 flex items-center justify-center text-white text-xs font-bold shadow-lg hover:scale-110 transition-all cursor-pointer mb-2`}
                          style={{
                            transform: `translateY(${(nodeIndex - (layer.size - 1) / 2) * 60}px)`,
                            opacity: brightness,
                          }}
                          title={`${layer.name} - Neuron ${nodeIndex + 1}`}
                        >
                          {nodeIndex + 1}
                        </div>
                      )
                    })}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center space-y-4">
                <h2 className="text-lg font-medium">Loading Neural Network...</h2>
                <button
                  onClick={() => loadTemplate("classification")}
                  className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90"
                >
                  Load Classification Network
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Simple controls */}
      <div className="absolute bottom-4 right-4 flex gap-2">
        <button onClick={fitToNetwork} className="px-3 py-2 bg-white/90 rounded-md text-sm hover:bg-white shadow-md">
          Center Network
        </button>
        <button
          onClick={() => setViewTransform({ x: 0, y: 0, scale: 1 })}
          className="px-3 py-2 bg-white/90 rounded-md text-sm hover:bg-white shadow-md"
        >
          Reset View
        </button>
      </div>
    </div>
  )
}
