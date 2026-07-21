"use client"

import { useEffect, useRef, useState } from "react"
import { useNetwork } from "@/lib/context/network-context"

interface HoverState {
  layerIndex: number
  neuronIndex: number
  x: number
  y: number
}

const LAYER_SPACING = 180
const NEURON_SPACING = 40
const NEURON_RADIUS = 18

export function GraphCanvas() {
  const { network, visualizationState, selectedSampleIndex } = useNetwork()
  const { weights, biases, weightDeltas, trace, telemetry, isHistorical } = visualizationState
  const [hoveredNeuron, setHoveredNeuron] = useState<HoverState | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [viewportSize, setViewportSize] = useState({ width: 0, height: 0 })

  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    const updateSize = () => {
      const bounds = container.getBoundingClientRect()
      setViewportSize({ width: bounds.width, height: bounds.height })
    }
    updateSize()
    const observer = new ResizeObserver(updateSize)
    observer.observe(container)
    return () => observer.disconnect()
  }, [])

  if (network.layers.length === 0) {
    return <div ref={containerRef} className="flex h-full items-center justify-center text-sm text-muted-foreground">Loading network…</div>
  }

  const getNeuronPosition = (layerIndex: number, neuronIndex: number, layerSize: number) => ({
    x: layerIndex * LAYER_SPACING + NEURON_RADIUS + 60,
    y: (neuronIndex - (layerSize - 1) / 2) * NEURON_SPACING,
  })
  const networkWidth = (network.layers.length - 1) * LAYER_SPACING + 210
  const maxLayerSize = Math.max(...network.layers.map((layer) => layer.size))
  const networkHeight = maxLayerSize * NEURON_SPACING + 120
  const stageScale = viewportSize.width > 0 && viewportSize.height > 0
    ? Math.min(1, Math.max(0.2, Math.min((viewportSize.width - 8) / networkWidth, (viewportSize.height - 8) / networkHeight)))
    : 1
  const maximumUpdate = Math.max(0, ...weightDeltas.flat(2).map((value) => Math.abs(value)))
  const maximumContribution = Math.max(0, ...(trace?.edgeContributions.flat(2).map((value) => Math.abs(value)) ?? [0]))

  const isRelatedToHover = (matrixIndex: number, sourceIndex: number, targetIndex: number) =>
    !hoveredNeuron ||
    (hoveredNeuron.layerIndex === matrixIndex && hoveredNeuron.neuronIndex === sourceIndex) ||
    (hoveredNeuron.layerIndex === matrixIndex + 1 && hoveredNeuron.neuronIndex === targetIndex)

  return (
    <div ref={containerRef} className="relative flex h-full w-full items-center justify-center overflow-hidden">
      <div className="absolute left-0 top-0 z-10 flex flex-wrap gap-3 rounded-md border bg-background/90 px-2.5 py-1.5 text-[10px] text-muted-foreground shadow-sm">
        <LegendLine color="border-blue-500" label="positive" />
        <LegendLine color="border-amber-500" label="negative" />
        <LegendLine color="border-slate-400" label="zero" thin />
        <span>{trace ? "brightness = activation · emphasis = contribution" : "glow = latest update"}</span>
      </div>

      <div className="relative shrink-0" style={{ width: networkWidth * stageScale, height: networkHeight * stageScale }}>
        <div className="absolute left-0 top-0 origin-top-left" style={{ width: networkWidth, height: networkHeight, transform: `scale(${stageScale})` }}>
          <svg className="pointer-events-none absolute inset-0" width={networkWidth} height={networkHeight} viewBox={`0 0 ${networkWidth} ${networkHeight}`}>
            {network.layers[0] && Array.from({ length: network.layers[0].size }).map((_, index) => {
              const position = getNeuronPosition(0, index, network.layers[0].size)
              return <line key={`input-${index}`} x1={position.x - 50} y1={position.y + networkHeight / 2} x2={position.x} y2={position.y + networkHeight / 2} stroke="rgb(148, 163, 184)" strokeWidth={1.25} opacity={0.45} />
            })}

            {network.layers.slice(0, -1).map((layer, matrixIndex) => {
              const nextLayer = network.layers[matrixIndex + 1]
              return Array.from({ length: layer.size }).flatMap((_, sourceIndex) =>
                Array.from({ length: nextLayer.size }).map((__, targetIndex) => {
                  const source = getNeuronPosition(matrixIndex, sourceIndex, layer.size)
                  const target = getNeuronPosition(matrixIndex + 1, targetIndex, nextLayer.size)
                  const weight = weights[matrixIndex]?.weights[sourceIndex]?.[targetIndex] ?? 0
                  const update = weightDeltas[matrixIndex]?.[sourceIndex]?.[targetIndex] ?? 0
                  const contribution = trace?.edgeContributions[matrixIndex]?.[sourceIndex]?.[targetIndex] ?? 0
                  const sourceActivation = trace?.layers[matrixIndex]?.activations[sourceIndex] ?? 0
                  const absoluteWeight = Math.abs(weight)
                  const contributionStrength = maximumContribution > 0 ? Math.min(1, Math.abs(contribution) / maximumContribution) : 0
                  const updateStrength = maximumUpdate > 0 ? Math.min(1, Math.abs(update) / maximumUpdate) : 0
                  const related = isRelatedToHover(matrixIndex, sourceIndex, targetIndex)
                  const color = absoluteWeight < 1e-12 ? "rgb(148, 163, 184)" : weight >= 0 ? "rgb(59, 130, 246)" : "rgb(245, 158, 11)"
                  const baseOpacity = absoluteWeight < 1e-12 ? 0.45 : Math.max(0.2, Math.min(0.92, 0.2 + absoluteWeight * 0.55))
                  const opacity = hoveredNeuron
                    ? related ? Math.max(0.84, baseOpacity) : 0.06
                    : trace ? 0.1 + contributionStrength * 0.82 : baseOpacity
                  const strokeWidth = absoluteWeight < 1e-12 ? 1 : Math.max(0.65, Math.min(4, 0.65 + absoluteWeight * 1.5))
                  const lineKey = `${matrixIndex}-${sourceIndex}-${targetIndex}`
                  const sign = absoluteWeight < 1e-12 ? "zero" : weight > 0 ? "positive" : "negative"
                  const tooltip = `Weight ${formatValue(weight)} · Sign ${sign} · Update ${formatValue(update)} · Source activation ${formatValue(sourceActivation)} · Contribution ${formatValue(contribution)}`

                  return (
                    <g key={lineKey}>
                      <line
                        x1={source.x}
                        y1={source.y + networkHeight / 2}
                        x2={target.x}
                        y2={target.y + networkHeight / 2}
                        stroke={color}
                        strokeWidth={strokeWidth}
                        opacity={opacity}
                        className="pointer-events-auto cursor-help transition-all duration-200 ease-linear"
                      >
                        <title>{tooltip}</title>
                      </line>
                      {!isHistorical && telemetry.epoch > 0 && updateStrength > 0.08 && related && (
                        <line
                          key={`pulse-${telemetry.epoch}-${lineKey}`}
                          x1={source.x}
                          y1={source.y + networkHeight / 2}
                          x2={target.x}
                          y2={target.y + networkHeight / 2}
                          stroke={color}
                          strokeWidth={strokeWidth + 1.5}
                          opacity={Math.min(0.9, updateStrength)}
                          className="pointer-events-none"
                          style={{ filter: `drop-shadow(0 0 ${2 + updateStrength * 4}px ${color})` }}
                        >
                          <animate attributeName="opacity" from={String(Math.min(0.9, updateStrength))} to="0" dur="320ms" fill="freeze" />
                        </line>
                      )}
                    </g>
                  )
                }),
              )
            })}

            {Array.from({ length: network.layers.at(-1)?.size ?? 0 }).map((_, index) => {
              const lastIndex = network.layers.length - 1
              const position = getNeuronPosition(lastIndex, index, network.layers[lastIndex].size)
              return <line key={`output-${index}`} x1={position.x} y1={position.y + networkHeight / 2} x2={position.x + 50} y2={position.y + networkHeight / 2} stroke="rgb(148, 163, 184)" strokeWidth={1.25} opacity={0.45} />
            })}
          </svg>

          {network.layers.map((layer, layerIndex) => {
            const layerTrace = trace?.layers[layerIndex]
            const maximumActivation = Math.max(1e-9, ...(layerTrace?.activations.map((value) => Math.abs(value)) ?? [0]))
            const biasValues = layerTrace?.biases ?? biases[layerIndex - 1]?.biases ?? []
            const maximumBias = Math.max(1e-9, ...biasValues.map((value) => Math.abs(value)))

            return (
              <div key={layer.id}>
                {Array.from({ length: layer.size }).map((_, neuronIndex) => {
                  const position = getNeuronPosition(layerIndex, neuronIndex, layer.size)
                  const activation = layerTrace?.activations[neuronIndex]
                  const bias = biasValues[neuronIndex] ?? 0
                  const activationStrength = activation === undefined ? 1 : Math.min(1, Math.abs(activation) / maximumActivation)
                  const inactiveRelu = layer.activation === "relu" && activation === 0
                  const saturated = (layer.activation === "tanh" && Math.abs(activation ?? 0) >= 0.95) ||
                    (layer.activation === "sigmoid" && activation !== undefined && (activation <= 0.05 || activation >= 0.95))
                  const identity = getLayerIdentity(layer.type)
                  const background = activation === undefined ? identity.fill : mixWithWhite(identity.rgb, 0.18 + activationStrength * 0.82)
                  const biasStrength = Math.min(1, Math.abs(bias) / maximumBias)

                  return (
                    <div key={neuronIndex}>
                      <button
                        type="button"
                        className={`absolute flex h-9 w-9 cursor-help items-center justify-center rounded-full border-2 text-xs font-medium shadow-md transition-all duration-200 ease-linear ${identity.border}`}
                        style={{ left: position.x - NEURON_RADIUS, top: position.y + networkHeight / 2 - NEURON_RADIUS, backgroundColor: background, color: activationStrength > 0.48 || activation === undefined ? "white" : "rgb(15, 23, 42)" }}
                        onMouseEnter={() => setHoveredNeuron({ layerIndex, neuronIndex, x: position.x + NEURON_RADIUS, y: position.y + networkHeight / 2 })}
                        onMouseLeave={() => setHoveredNeuron(null)}
                        onFocus={() => setHoveredNeuron({ layerIndex, neuronIndex, x: position.x + NEURON_RADIUS, y: position.y + networkHeight / 2 })}
                        onBlur={() => setHoveredNeuron(null)}
                        aria-label={`${layer.name} neuron ${neuronIndex + 1}${activation === undefined ? "" : `, activation ${formatValue(activation)}`}`}
                      >
                        {neuronIndex + 1}
                        {layerIndex > 0 && (
                          <span
                            className={`absolute -right-1 -top-1 rounded-full border border-background transition-all duration-200 ease-linear ${Math.abs(bias) < 1e-12 ? "bg-slate-400" : bias > 0 ? "bg-blue-500" : "bg-amber-500"}`}
                            style={{ width: 5 + biasStrength * 5, height: 5 + biasStrength * 5 }}
                            title={`Bias ${formatValue(bias)}`}
                          />
                        )}
                        {(inactiveRelu || saturated) && (
                          <span className={`absolute -bottom-1 -right-1 h-2.5 w-2.5 rounded-full border-2 border-background ${inactiveRelu ? "bg-slate-400" : "bg-rose-500"}`} title={inactiveRelu ? "Inactive ReLU" : "Saturated activation"} />
                        )}
                      </button>
                    </div>
                  )
                })}

                <div className="absolute text-center text-xs" style={{ left: layerIndex * LAYER_SPACING + 20, top: networkHeight / 2 + ((layer.size - 1) / 2) * NEURON_SPACING + 40, width: 120 }}>
                  <div className="font-medium text-foreground">{layer.name}</div>
                  <div className="text-muted-foreground">{layer.size} neurons</div>
                </div>
              </div>
            )
          })}

          {trace && (
            <div
              className="absolute rounded-md border bg-background/95 px-2 py-1 text-[10px] shadow-sm"
              style={{ left: (network.layers.length - 1) * LAYER_SPACING + 108, top: networkHeight / 2 - 18 }}
            >
              <div className="font-semibold">Class {trace.prediction === 1 ? "B" : "A"}</div>
              <div className="font-mono text-muted-foreground">p(B) {(trace.outputProbability * 100).toFixed(1)}%</div>
            </div>
          )}

          {hoveredNeuron && (() => {
            const layer = network.layers[hoveredNeuron.layerIndex]
            const layerTrace = trace?.layers[hoveredNeuron.layerIndex]
            const activation = layerTrace?.activations[hoveredNeuron.neuronIndex]
            const preActivation = layerTrace?.preActivations[hoveredNeuron.neuronIndex]
            const weightedInput = layerTrace?.weightedInputs[hoveredNeuron.neuronIndex]
            const bias = layerTrace?.biases[hoveredNeuron.neuronIndex] ?? (hoveredNeuron.layerIndex > 0 ? biases[hoveredNeuron.layerIndex - 1]?.biases[hoveredNeuron.neuronIndex] ?? 0 : 0)
            const incomingWeights = hoveredNeuron.layerIndex > 0
              ? weights[hoveredNeuron.layerIndex - 1]?.weights.map((row) => row[hoveredNeuron.neuronIndex] ?? 0) ?? []
              : []
            const incomingContributions = hoveredNeuron.layerIndex > 0
              ? trace?.edgeContributions[hoveredNeuron.layerIndex - 1]?.map((row) => row[hoveredNeuron.neuronIndex] ?? 0) ?? []
              : []
            const tooltipLeft = hoveredNeuron.x > networkWidth - 220 ? hoveredNeuron.x - 220 : hoveredNeuron.x + 10

            return (
              <div className="pointer-events-none absolute z-50 w-52 rounded-lg border bg-background/97 p-3 text-xs shadow-xl" style={{ left: tooltipLeft, top: Math.max(8, hoveredNeuron.y - 70) }}>
                <div className="mb-2 font-semibold">{layer.name} · neuron {hoveredNeuron.neuronIndex + 1}</div>
                <MetricRow label="Activation" value={activation === undefined ? "Select a sample" : formatValue(activation)} />
                {hoveredNeuron.layerIndex > 0 && (
                  <>
                    <MetricRow label="Weighted sum" value={weightedInput === undefined ? "—" : formatValue(weightedInput)} />
                    <MetricRow label="Bias" value={formatValue(bias)} />
                    <MetricRow label="Pre-activation" value={preActivation === undefined ? "—" : formatValue(preActivation)} />
                    <MetricRow label="Function" value={layer.activation} />
                    <div className="mt-2 max-h-28 space-y-1 overflow-y-auto border-t pt-2 font-mono text-[10px]">
                      {incomingWeights.map((weight, index) => (
                        <div key={index} className="grid grid-cols-[1fr_auto] gap-2">
                          <span className="text-muted-foreground">a{index + 1} × {formatValue(weight)}</span>
                          <span>{trace ? formatValue(incomingContributions[index] ?? 0) : "—"}</span>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </div>
            )
          })()}
        </div>
      </div>

      {selectedSampleIndex !== null && (
        <div className="absolute bottom-0 left-0 rounded-md border bg-background/90 px-2 py-1 text-[10px] text-muted-foreground shadow-sm">
          Tracing sample #{selectedSampleIndex + 1}
        </div>
      )}
    </div>
  )
}

function LegendLine({ color, label, thin = false }: { color: string; label: string; thin?: boolean }) {
  return <span><span className={`mr-1 inline-block w-3 border-t ${thin ? "border-t" : "border-t-2"} ${color}`} />{label}</span>
}

function MetricRow({ label, value }: { label: string; value: string }) {
  return <div className="flex justify-between gap-3"><span className="text-muted-foreground">{label}</span><span className="font-mono">{value}</span></div>
}

function getLayerIdentity(type: string) {
  if (type === "input") return { border: "border-emerald-600", fill: "rgb(34, 197, 94)", rgb: [34, 197, 94] as const }
  if (type === "output") return { border: "border-amber-500", fill: "rgb(251, 191, 36)", rgb: [251, 191, 36] as const }
  return { border: "border-blue-600", fill: "rgb(59, 130, 246)", rgb: [59, 130, 246] as const }
}

function mixWithWhite(color: readonly [number, number, number], strength: number) {
  const channel = (value: number) => Math.round(255 - (255 - value) * strength)
  return `rgb(${channel(color[0])}, ${channel(color[1])}, ${channel(color[2])})`
}

function formatValue(value: number) {
  if (!Number.isFinite(value)) return "—"
  if (Math.abs(value) < 0.001 && value !== 0) return value.toExponential(2)
  return value.toFixed(4)
}
