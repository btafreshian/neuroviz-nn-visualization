"use client"

import { useEffect, useRef, useState, type MouseEvent } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useNetwork, type EpochTelemetry, type TrainingConfig } from "@/lib/context/network-context"

interface ChartPoint {
  epoch: number
  value: number
}

interface ChartSeries {
  label: string
  color: string
  points: ChartPoint[]
  dashed?: boolean
  muted?: boolean
}

export function TrainingCharts() {
  const { currentRun, previousRun, visualizationState } = useNetwork()
  const currentHistory = currentRun?.history ?? []
  const previousHistory = previousRun?.history ?? []
  const currentConfig = currentRun?.config
  const differentLossScale = Boolean(previousRun && currentConfig && previousRun.config.loss !== currentConfig.loss)
  const currentBaseline = currentHistory[0]
  const previousBaseline = previousHistory[0]
  const currentPoint = visualizationState.telemetry
  const isResetView = currentRun?.status === "idle" && currentRun.history.length > 1 && visualizationState.liveEpoch === 0 && !visualizationState.isHistorical

  const lossValue = (point: EpochTelemetry, baseline: EpochTelemetry | undefined, key: "trainLoss" | "validationLoss") =>
    differentLossScale ? point[key] / Math.max(1e-9, baseline?.[key] ?? 1) : point[key]

  const lossSeries: ChartSeries[] = [
    {
      label: "Current validation",
      color: "#ef4444",
      points: currentHistory.map((point) => ({ epoch: point.epoch, value: lossValue(point, currentBaseline, "validationLoss") })),
    },
    {
      label: "Current train",
      color: "#f97316",
      muted: true,
      points: currentHistory.map((point) => ({ epoch: point.epoch, value: lossValue(point, currentBaseline, "trainLoss") })),
    },
  ]
  if (previousHistory.length > 0) {
    lossSeries.push({
      label: "Previous validation",
      color: "#64748b",
      dashed: true,
      points: previousHistory.map((point) => ({ epoch: point.epoch, value: lossValue(point, previousBaseline, "validationLoss") })),
    })
  }

  const accuracySeries: ChartSeries[] = [
    {
      label: "Current validation",
      color: "#22c55e",
      points: currentHistory.map((point) => ({ epoch: point.epoch, value: point.validationAccuracy })),
    },
    {
      label: "Current train",
      color: "#14b8a6",
      muted: true,
      points: currentHistory.map((point) => ({ epoch: point.epoch, value: point.trainAccuracy })),
    },
  ]
  if (previousHistory.length > 0) {
    accuracySeries.push({
      label: "Previous validation",
      color: "#64748b",
      dashed: true,
      points: previousHistory.map((point) => ({ epoch: point.epoch, value: point.validationAccuracy })),
    })
  }

  const changedParameters = currentConfig && previousRun ? getChangedParameters(previousRun.config, currentConfig) : []

  return (
    <div className="space-y-3">
      {isResetView && (
        <div className="rounded-lg border bg-muted/25 px-3 py-2 text-xs text-muted-foreground">
          Run history retained · live model reset to epoch 0
        </div>
      )}
      {previousRun && (
        <div className="flex flex-wrap items-center gap-2 rounded-lg border bg-muted/25 px-3 py-2 text-xs">
          <span className="font-medium">Previous run</span>
          <span className="text-muted-foreground">shown dashed</span>
          {changedParameters.map((change) => (
            <span key={change} className="rounded-full border bg-background px-2 py-0.5 font-mono text-[10px]">{change}</span>
          ))}
        </div>
      )}

      <div className="grid gap-3 sm:grid-cols-2">
        <ChartCard
          label={differentLossScale ? "Loss remaining" : "Loss"}
          primaryValue={differentLossScale ? `${(lossValue(currentPoint, currentBaseline, "validationLoss") * 100).toFixed(1)}%` : currentPoint.validationLoss.toFixed(4)}
          secondaryValue={`train ${currentPoint.trainLoss.toFixed(4)}`}
          dotClassName="bg-red-500"
          series={lossSeries}
          fixedRange={differentLossScale ? [0, Math.max(1.05, getSeriesMaximum(lossSeries) * 1.05)] : [0, Math.max(0.1, getSeriesMaximum(lossSeries) * 1.08)]}
          valueFormatter={differentLossScale ? (value) => `${(value * 100).toFixed(0)}%` : formatLoss}
          cursorEpoch={visualizationState.isHistorical ? visualizationState.epoch : null}
        />
        <ChartCard
          label="Accuracy"
          primaryValue={`${(currentPoint.validationAccuracy * 100).toFixed(1)}%`}
          secondaryValue={`train ${(currentPoint.trainAccuracy * 100).toFixed(1)}%`}
          dotClassName="bg-green-500"
          series={accuracySeries}
          fixedRange={[0, 1]}
          valueFormatter={(value) => `${(value * 100).toFixed(0)}%`}
          cursorEpoch={visualizationState.isHistorical ? visualizationState.epoch : null}
        />
      </div>
    </div>
  )
}

interface ChartCardProps {
  label: string
  primaryValue: string
  secondaryValue: string
  dotClassName: string
  series: ChartSeries[]
  fixedRange: [number, number]
  valueFormatter: (value: number) => string
  cursorEpoch: number | null
}

function ChartCard({ label, primaryValue, secondaryValue, dotClassName, series, fixedRange, valueFormatter, cursorEpoch }: ChartCardProps) {
  return (
    <Card className="gap-3 border-0 bg-muted/30 py-4 shadow-none">
      <CardHeader className="px-4">
        <CardTitle className="flex flex-wrap items-center gap-2 text-sm">
          <span className={`h-2 w-2 rounded-full ${dotClassName}`} aria-hidden="true" />
          <span>{label}</span>
          <span className="font-mono text-xs font-normal text-foreground">{primaryValue}</span>
          <span className="font-mono text-[10px] font-normal text-muted-foreground">{secondaryValue}</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="px-4">
        <MiniChart series={series} height={240} fixedRange={fixedRange} valueFormatter={valueFormatter} label={label} cursorEpoch={cursorEpoch} />
        <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[9px] text-muted-foreground">
          {series.map((item) => (
            <div key={item.label} className="flex items-center gap-1">
              <span className={`block w-4 border-t-2 ${item.dashed ? "border-dashed" : ""}`} style={{ borderColor: item.color, opacity: item.muted ? 0.65 : 1 }} />
              {item.label}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

interface MiniChartProps {
  series: ChartSeries[]
  height: number
  fixedRange: [number, number]
  valueFormatter: (value: number) => string
  label: string
  cursorEpoch: number | null
}

function MiniChart({ series, height, fixedRange, valueFormatter, label, cursorEpoch }: MiniChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [hoveredEpoch, setHoveredEpoch] = useState<number | null>(null)
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 })
  const allPoints = series.flatMap((item) => item.points)
  const maximumEpoch = Math.max(1, ...allPoints.map((point) => point.epoch))

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const renderChart = () => {
      const ctx = canvas.getContext("2d")
      if (!ctx) return
      const rect = canvas.getBoundingClientRect()
      const dpr = window.devicePixelRatio || 1
      const padding = { left: 36, right: 10, top: 10, bottom: 24 }
      const chartWidth = Math.max(1, rect.width - padding.left - padding.right)
      const chartHeight = Math.max(1, rect.height - padding.top - padding.bottom)
      const [minimumValue, maximumValue] = fixedRange
      const valueRange = maximumValue - minimumValue || 1

      canvas.width = Math.max(1, Math.round(rect.width * dpr))
      canvas.height = Math.max(1, Math.round(rect.height * dpr))
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, rect.width, rect.height)
      ctx.font = "9px sans-serif"
      ctx.textBaseline = "middle"

      for (let index = 0; index <= 4; index += 1) {
        const ratio = index / 4
        const y = padding.top + chartHeight * ratio
        const value = maximumValue - valueRange * ratio
        ctx.strokeStyle = "rgba(100, 116, 139, 0.15)"
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(padding.left, y)
        ctx.lineTo(rect.width - padding.right, y)
        ctx.stroke()
        ctx.fillStyle = "#64748b"
        ctx.textAlign = "right"
        ctx.fillText(valueFormatter(value), padding.left - 5, y)
      }

      for (const epoch of [0, Math.round(maximumEpoch / 2), maximumEpoch]) {
        const x = padding.left + (epoch / maximumEpoch) * chartWidth
        ctx.fillStyle = "#64748b"
        ctx.textAlign = epoch === 0 ? "left" : epoch === maximumEpoch ? "right" : "center"
        ctx.fillText(String(epoch), x, rect.height - 7)
      }

      const toCanvasPoint = (point: ChartPoint) => ({
        x: padding.left + (point.epoch / maximumEpoch) * chartWidth,
        y: padding.top + chartHeight - ((point.value - minimumValue) / valueRange) * chartHeight,
      })

      for (const item of series) {
        if (item.points.length < 2) continue
        ctx.strokeStyle = item.color
        ctx.globalAlpha = item.muted ? 0.62 : 1
        ctx.lineWidth = item.dashed ? 1.5 : 2
        ctx.setLineDash(item.dashed ? [5, 4] : [])
        ctx.lineJoin = "round"
        ctx.lineCap = "round"
        ctx.beginPath()
        item.points.forEach((point, index) => {
          const canvasPoint = toCanvasPoint(point)
          if (index === 0) ctx.moveTo(canvasPoint.x, canvasPoint.y)
          else ctx.lineTo(canvasPoint.x, canvasPoint.y)
        })
        ctx.stroke()
      }
      ctx.globalAlpha = 1
      ctx.setLineDash([])

      if (cursorEpoch !== null) {
        const x = padding.left + (cursorEpoch / maximumEpoch) * chartWidth
        ctx.strokeStyle = "rgba(124, 58, 237, 0.85)"
        ctx.lineWidth = 1.5
        ctx.setLineDash([3, 3])
        ctx.beginPath()
        ctx.moveTo(x, padding.top)
        ctx.lineTo(x, padding.top + chartHeight)
        ctx.stroke()
        ctx.setLineDash([])
      }

      if (hoveredEpoch !== null) {
        const x = padding.left + (hoveredEpoch / maximumEpoch) * chartWidth
        ctx.strokeStyle = "rgba(15, 23, 42, 0.35)"
        ctx.beginPath()
        ctx.moveTo(x, padding.top)
        ctx.lineTo(x, padding.top + chartHeight)
        ctx.stroke()
      }
    }

    renderChart()
    const observer = new ResizeObserver(renderChart)
    observer.observe(canvas)
    return () => observer.disconnect()
  }, [series, fixedRange, valueFormatter, maximumEpoch, hoveredEpoch, cursorEpoch])

  const handleMouseMove = (event: MouseEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect()
    const leftPadding = 36
    const rightPadding = 10
    const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left - leftPadding) / (rect.width - leftPadding - rightPadding)))
    setHoveredEpoch(Math.round(ratio * maximumEpoch))
    setTooltipPosition({ x: event.clientX - rect.left, y: event.clientY - rect.top })
  }

  const tooltipValues = hoveredEpoch === null
    ? []
    : series.map((item) => ({ item, point: getNearestPoint(item.points, hoveredEpoch) })).filter((entry) => entry.point)

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        className="block w-full"
        style={{ height: `${height}px` }}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoveredEpoch(null)}
        aria-label={`${label} by epoch. Current run is solid and previous run is dashed.`}
      />
      {hoveredEpoch !== null && tooltipValues.length > 0 && (
        <div
          className="pointer-events-none absolute z-10 min-w-32 rounded-md border bg-background/95 p-2 text-[10px] shadow-md"
          style={{ left: Math.min(tooltipPosition.x + 10, 145), top: Math.max(4, tooltipPosition.y - 30) }}
        >
          <div className="mb-1 font-medium">Epoch {hoveredEpoch}</div>
          {tooltipValues.map(({ item, point }) => (
            <div key={item.label} className="flex justify-between gap-3 text-muted-foreground">
              <span>{item.label}</span><span className="font-mono text-foreground">{valueFormatter(point!.value)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function getNearestPoint(points: ChartPoint[], epoch: number): ChartPoint | undefined {
  return points.reduce<ChartPoint | undefined>((nearest, point) =>
    !nearest || Math.abs(point.epoch - epoch) < Math.abs(nearest.epoch - epoch) ? point : nearest, undefined)
}

function getSeriesMaximum(series: ChartSeries[]): number {
  return Math.max(...series.flatMap((item) => item.points.map((point) => point.value)), 0)
}

function formatLoss(value: number): string {
  return value >= 1 ? value.toFixed(1) : value.toFixed(2)
}

function getChangedParameters(previous: TrainingConfig, current: TrainingConfig): string[] {
  const labels: Record<keyof TrainingConfig, string> = {
    learningRate: "LR",
    epochs: "epochs",
    batchSize: "batch",
    trainTestSplit: "split",
    optimizer: "optimizer",
    loss: "loss",
    momentum: "β1",
    weightDecay: "decay",
    learningRateDecay: "LR decay",
    gradientClip: "clip",
    shuffle: "shuffle",
  }

  return (Object.keys(labels) as Array<keyof TrainingConfig>)
    .filter((key) => previous[key] !== current[key])
    .map((key) => `${labels[key]} ${formatConfigValue(previous[key])} → ${formatConfigValue(current[key])}`)
}

function formatConfigValue(value: TrainingConfig[keyof TrainingConfig]): string {
  if (typeof value === "boolean") return value ? "on" : "off"
  return String(value)
}
