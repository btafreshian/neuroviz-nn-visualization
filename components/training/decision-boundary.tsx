"use client"

import { useEffect, useRef, useState, type KeyboardEvent, type MouseEvent } from "react"
import { X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useNetwork } from "@/lib/context/network-context"
import { predictProbability } from "@/lib/nn/simple-trainer"

interface Bounds {
  xMin: number
  xMax: number
  yMin: number
  yMax: number
}

interface RenderMeta {
  bounds: Bounds
  width: number
  height: number
  visibleIndices: number[]
}

interface CanvasSize {
  width: number
  height: number
  deviceScale: number
}

const NEUTRAL = [248, 250, 252] as const
const CLASS_A = [191, 219, 254] as const
const CLASS_B = [253, 230, 138] as const
const SURFACE_TRANSITION_MS = 180

export function DecisionBoundary() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number | null>(null)
  const displayedProbabilitiesRef = useRef<Float32Array | null>(null)
  const displayedSampleProbabilitiesRef = useRef<Float32Array | null>(null)
  const surfaceShapeRef = useRef({ width: 0, height: 0 })
  const renderMetaRef = useRef<RenderMeta | null>(null)
  const { dataset, network, selectedSampleIndex, selectSample, visualizationState } = useNetwork()
  const { weights, biases, telemetry, trace, isHistorical } = visualizationState
  const datasetRef = useRef(dataset)
  const [canvasSize, setCanvasSize] = useState<CanvasSize>({ width: 0, height: 0, deviceScale: 1 })

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const measure = () => {
      const rect = canvas.getBoundingClientRect()
      const nextSize = {
        width: Math.max(1, Math.round(rect.width)),
        height: Math.max(1, Math.round(rect.height)),
        deviceScale: window.devicePixelRatio || 1,
      }
      setCanvasSize((current) =>
        current.width === nextSize.width && current.height === nextSize.height && current.deviceScale === nextSize.deviceScale
          ? current
          : nextSize,
      )
    }

    measure()
    const resizeObserver = new ResizeObserver(measure)
    resizeObserver.observe(canvas)
    return () => resizeObserver.disconnect()
  }, [dataset])

  useEffect(() => {
    if (!dataset || dataset.features[0]?.length !== 2 || !canvasRef.current || canvasSize.width === 0 || canvasSize.height === 0) return
    const canvas = canvasRef.current

    const render = () => {
      const context = canvas.getContext("2d")
      if (!context) return
      const { width, height, deviceScale } = canvasSize
      const bounds = getBounds(dataset.features)
      const xRange = bounds.xMax - bounds.xMin
      const yRange = bounds.yMax - bounds.yMin
      const surfaceWidth = Math.min(140, Math.max(90, Math.round(width / 4)))
      const surfaceHeight = Math.max(72, Math.round((surfaceWidth * height) / width))
      const targetProbabilities = new Float32Array(surfaceWidth * surfaceHeight)
      const pointStep = Math.max(1, Math.ceil(dataset.features.length / 240))
      const visibleIndices = dataset.features.map((_, index) => index).filter((index) => index % pointStep === 0)
      const targetSampleProbabilities = new Float32Array(visibleIndices.length)

      const backingWidth = Math.round(width * deviceScale)
      const backingHeight = Math.round(height * deviceScale)
      if (canvas.width !== backingWidth || canvas.height !== backingHeight) {
        canvas.width = backingWidth
        canvas.height = backingHeight
      }
      context.setTransform(deviceScale, 0, 0, deviceScale, 0, 0)

      for (let pixelY = 0; pixelY < surfaceHeight; pixelY += 1) {
        for (let pixelX = 0; pixelX < surfaceWidth; pixelX += 1) {
          const dataX = bounds.xMin + (pixelX / Math.max(1, surfaceWidth - 1)) * xRange
          const dataY = bounds.yMax - (pixelY / Math.max(1, surfaceHeight - 1)) * yRange
          targetProbabilities[pixelY * surfaceWidth + pixelX] = weights.length === network.layers.length - 1
            ? predictProbability(network.layers, weights, biases, [dataX, dataY])
            : 0.5
        }
      }

      visibleIndices.forEach((sampleIndex, position) => {
        targetSampleProbabilities[position] = weights.length === network.layers.length - 1
          ? predictProbability(network.layers, weights, biases, dataset.features[sampleIndex])
          : 0.5
      })

      const sameDataset = datasetRef.current === dataset
      const sameSurfaceShape = surfaceShapeRef.current.width === surfaceWidth && surfaceShapeRef.current.height === surfaceHeight
      const previous = sameDataset && sameSurfaceShape && displayedProbabilitiesRef.current?.length === targetProbabilities.length
        ? displayedProbabilitiesRef.current
        : targetProbabilities
      const previousSampleProbabilities = sameDataset && displayedSampleProbabilitiesRef.current?.length === targetSampleProbabilities.length
        ? displayedSampleProbabilitiesRef.current
        : targetSampleProbabilities
      datasetRef.current = dataset
      surfaceShapeRef.current = { width: surfaceWidth, height: surfaceHeight }
      if (animationRef.current !== null) cancelAnimationFrame(animationRef.current)
      const startedAt = performance.now()

      const drawFrame = (probabilities: Float32Array, sampleProbabilities: Float32Array) => {
        context.clearRect(0, 0, width, height)
        drawSurface(context, probabilities, surfaceWidth, surfaceHeight, width, height)
        drawGrid(context, width, height)
        drawSamples(context, visibleIndices, dataset.features, dataset.labels, bounds, width, height, sampleProbabilities, selectedSampleIndex)
        displayedProbabilitiesRef.current = probabilities
        displayedSampleProbabilitiesRef.current = sampleProbabilities
        renderMetaRef.current = { bounds, width, height, visibleIndices }
      }

      const animate = (now: number) => {
        const progress = Math.min(1, (now - startedAt) / SURFACE_TRANSITION_MS)
        const eased = progress * (2 - progress)
        const interpolated = new Float32Array(targetProbabilities.length)
        const interpolatedSampleProbabilities = new Float32Array(targetSampleProbabilities.length)
        for (let index = 0; index < targetProbabilities.length; index += 1) {
          interpolated[index] = previous[index] + (targetProbabilities[index] - previous[index]) * eased
        }
        for (let index = 0; index < targetSampleProbabilities.length; index += 1) {
          interpolatedSampleProbabilities[index] = previousSampleProbabilities[index] + (targetSampleProbabilities[index] - previousSampleProbabilities[index]) * eased
        }
        drawFrame(interpolated, interpolatedSampleProbabilities)
        if (progress < 1) animationRef.current = requestAnimationFrame(animate)
        else animationRef.current = null
      }

      drawFrame(previous, previousSampleProbabilities)
      if (previous !== targetProbabilities || previousSampleProbabilities !== targetSampleProbabilities) {
        animationRef.current = requestAnimationFrame(animate)
      }
    }

    render()
    return () => {
      if (animationRef.current !== null) cancelAnimationFrame(animationRef.current)
    }
  }, [biases, canvasSize, dataset, network.layers, selectedSampleIndex, weights])

  if (!dataset || dataset.features[0]?.length !== 2) {
    return <div className="flex h-[340px] items-center justify-center rounded-lg border border-dashed bg-muted/20 px-6 text-center text-sm text-muted-foreground">Decision boundary visualization is only available for 2D datasets.</div>
  }

  const chooseNearestSample = (clientX: number, clientY: number) => {
    const canvas = canvasRef.current
    const meta = renderMetaRef.current
    if (!canvas || !meta) return
    const rect = canvas.getBoundingClientRect()
    const pointerX = clientX - rect.left
    const pointerY = clientY - rect.top
    const xRange = meta.bounds.xMax - meta.bounds.xMin
    const yRange = meta.bounds.yMax - meta.bounds.yMin
    let nearestIndex: number | null = null
    let nearestDistance = 13

    for (const index of meta.visibleIndices) {
      const feature = dataset.features[index]
      const x = ((feature[0] - meta.bounds.xMin) / xRange) * meta.width
      const y = meta.height - ((feature[1] - meta.bounds.yMin) / yRange) * meta.height
      const distance = Math.hypot(x - pointerX, y - pointerY)
      if (distance < nearestDistance) {
        nearestIndex = index
        nearestDistance = distance
      }
    }
    selectSample(nearestIndex)
  }

  const handleCanvasClick = (event: MouseEvent<HTMLCanvasElement>) => chooseNearestSample(event.clientX, event.clientY)
  const handleCanvasKeyDown = (event: KeyboardEvent<HTMLCanvasElement>) => {
    const visibleIndices = renderMetaRef.current?.visibleIndices ?? []
    if (event.key === "Escape") {
      selectSample(null)
      return
    }
    if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key) || visibleIndices.length === 0) return
    event.preventDefault()
    const currentPosition = selectedSampleIndex === null ? -1 : visibleIndices.indexOf(selectedSampleIndex)
    const direction = event.key === "ArrowLeft" || event.key === "ArrowUp" ? -1 : 1
    const nextPosition = (currentPosition + direction + visibleIndices.length) % visibleIndices.length
    selectSample(visibleIndices[nextPosition])
  }

  return (
    <div className="space-y-3">
      <canvas
        ref={canvasRef}
        className="h-[340px] w-full cursor-crosshair rounded-lg border bg-slate-50 shadow-inner focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        aria-label={`Decision boundary for ${dataset.name}. Click a sample to trace it through the network.`}
        tabIndex={0}
        onClick={handleCanvasClick}
        onKeyDown={handleCanvasKeyDown}
      >
        Decision boundary visualization for {dataset.name}.
      </canvas>

      {selectedSampleIndex !== null && trace && (
        <div className="rounded-lg border bg-muted/25 p-3 text-xs">
          <div className="mb-2 flex items-center justify-between gap-3">
            <div className="font-medium">Sample #{selectedSampleIndex + 1} trace</div>
            <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => selectSample(null)} aria-label="Clear selected sample"><X className="h-3.5 w-3.5" /></Button>
          </div>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 sm:grid-cols-3">
            <TraceMetric label="Coordinates" value={`${trace.input[0].toFixed(2)}, ${trace.input[1].toFixed(2)}`} />
            <TraceMetric label="True class" value={trace.target === 1 ? "Class B" : "Class A"} />
            <TraceMetric label="Prediction" value={trace.prediction === 1 ? "Class B" : "Class A"} />
            <TraceMetric label="p(Class B)" value={`${(trace.outputProbability * 100).toFixed(1)}%`} />
            <TraceMetric label="Sample loss" value={trace.sampleLoss?.toFixed(4) ?? "—"} />
            <TraceMetric label="State" value={isHistorical ? `Replay epoch ${telemetry.epoch}` : `Live epoch ${telemetry.epoch}`} />
          </div>
        </div>
      )}

      <div className="space-y-2 text-xs text-muted-foreground">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-4">
            <LegendDot className="bg-blue-500" label="Class A" />
            <LegendDot className="bg-amber-500" label="Class B" />
            <LegendDot className="border-2 border-red-600" label="Misclassified" />
            <LegendDot className="border-2 border-violet-600" label="Selected" />
          </div>
          <div className="font-mono">Epoch {telemetry.epoch} · {(telemetry.validationAccuracy * 100).toFixed(1)}% accuracy</div>
        </div>
        <div className="grid grid-cols-[auto_1fr_auto] items-center gap-2 text-[10px]"><span>Confident A</span><div className="h-2 rounded-full border bg-gradient-to-r from-blue-300 via-slate-50 to-amber-200" /><span>Confident B</span></div>
        <div className="text-center text-[10px]">Pale center = uncertain prediction near 50%</div>
      </div>
    </div>
  )
}

function drawSurface(context: CanvasRenderingContext2D, probabilities: Float32Array, surfaceWidth: number, surfaceHeight: number, width: number, height: number) {
  const surface = document.createElement("canvas")
  surface.width = surfaceWidth
  surface.height = surfaceHeight
  const surfaceContext = surface.getContext("2d")
  if (!surfaceContext) return
  const image = surfaceContext.createImageData(surfaceWidth, surfaceHeight)
  for (let index = 0; index < probabilities.length; index += 1) {
    const probability = probabilities[index]
    const target = probability < 0.5 ? CLASS_A : CLASS_B
    const strength = Math.min(1, Math.abs(probability - 0.5) * 2) * 0.9
    const offset = index * 4
    image.data[offset] = mixChannel(NEUTRAL[0], target[0], strength)
    image.data[offset + 1] = mixChannel(NEUTRAL[1], target[1], strength)
    image.data[offset + 2] = mixChannel(NEUTRAL[2], target[2], strength)
    image.data[offset + 3] = 255
  }
  surfaceContext.putImageData(image, 0, 0)
  context.imageSmoothingEnabled = true
  context.imageSmoothingQuality = "high"
  context.drawImage(surface, 0, 0, width, height)
}

function drawGrid(context: CanvasRenderingContext2D, width: number, height: number) {
  context.strokeStyle = "rgba(148, 163, 184, 0.16)"
  context.lineWidth = 1
  for (let step = 1; step < 5; step += 1) {
    const x = (width / 5) * step
    const y = (height / 5) * step
    context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke()
    context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke()
  }
}

function drawSamples(
  context: CanvasRenderingContext2D,
  indices: number[],
  features: number[][],
  labels: Array<number | number[]>,
  bounds: Bounds,
  width: number,
  height: number,
  probabilities: Float32Array,
  selectedSampleIndex: number | null,
) {
  const xRange = bounds.xMax - bounds.xMin
  const yRange = bounds.yMax - bounds.yMin
  for (let position = 0; position < indices.length; position += 1) {
    const index = indices[position]
    const feature = features[index]
    const x = ((feature[0] - bounds.xMin) / xRange) * width
    const y = height - ((feature[1] - bounds.yMin) / yRange) * height
    const classB = getNumericLabel(labels[index] ?? 0) >= 0.5
    const probability = probabilities[position] ?? 0.5
    const misclassified = (probability >= 0.5) !== classB
    const selected = index === selectedSampleIndex
    if (selected) {
      context.beginPath(); context.arc(x, y, 8, 0, Math.PI * 2); context.strokeStyle = "#7c3aed"; context.lineWidth = 3; context.stroke()
    }
    context.beginPath(); context.arc(x, y, 4.25, 0, Math.PI * 2)
    context.fillStyle = classB ? "#f59e0b" : "#3b82f6"; context.fill()
    context.strokeStyle = misclassified ? "#dc2626" : "rgba(255, 255, 255, 0.95)"; context.lineWidth = misclassified ? 2.5 : 2; context.stroke()
  }
}

function TraceMetric({ label, value }: { label: string; value: string }) {
  return <div><div className="text-[9px] font-medium uppercase tracking-wide text-muted-foreground">{label}</div><div className="font-mono text-foreground">{value}</div></div>
}

function LegendDot({ className, label }: { className: string; label: string }) {
  return <div className="flex items-center gap-1.5"><span className={`h-2.5 w-2.5 rounded-full ${className}`} /><span>{label}</span></div>
}

function getBounds(features: number[][]): Bounds {
  const xValues = features.map((feature) => feature[0])
  const yValues = features.map((feature) => feature[1])
  const rawXMin = Math.min(...xValues); const rawXMax = Math.max(...xValues)
  const rawYMin = Math.min(...yValues); const rawYMax = Math.max(...yValues)
  const xRange = rawXMax - rawXMin || 1; const yRange = rawYMax - rawYMin || 1
  return { xMin: rawXMin - xRange * 0.14, xMax: rawXMax + xRange * 0.14, yMin: rawYMin - yRange * 0.14, yMax: rawYMax + yRange * 0.14 }
}

function getNumericLabel(label: number | number[]) { return Array.isArray(label) ? (label[0] ?? 0) : label }
function mixChannel(from: number, to: number, amount: number) { return Math.round(from + (to - from) * amount) }
