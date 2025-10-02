"use client"

import { useEffect, useState, useRef } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useNetwork } from "@/lib/context/network-context"

interface DataPoint {
  step: number
  value: number
}

export function TrainingCharts() {
  const { trainingProgress, isTraining } = useNetwork()
  const [lossHistory, setLossHistory] = useState<DataPoint[]>([])
  const [accuracyHistory, setAccuracyHistory] = useState<DataPoint[]>([])
  const lastTrainingState = useRef(false)

  useEffect(() => {
    if (isTraining && !lastTrainingState.current) {
      console.log("[v0] Training started - resetting charts")
      setLossHistory([])
      setAccuracyHistory([])
    }

    if (trainingProgress.epoch === 0 && !isTraining && (lossHistory.length > 0 || accuracyHistory.length > 0)) {
      console.log("[v0] Training reset - clearing charts")
      setLossHistory([])
      setAccuracyHistory([])
    }

    lastTrainingState.current = isTraining

    if (trainingProgress.epoch > 0 && isTraining) {
      setLossHistory((prev) => {
        const newPoint = { step: trainingProgress.epoch, value: trainingProgress.loss }
        const updated = [...prev.slice(-100), newPoint]
        return updated
      })

      setAccuracyHistory((prev) => {
        const newPoint = { step: trainingProgress.epoch, value: trainingProgress.accuracy }
        const updated = [...prev.slice(-100), newPoint]
        return updated
      })
    }
  }, [trainingProgress, isTraining])

  return (
    <div className="grid grid-cols-2 gap-4">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Loss</CardTitle>
        </CardHeader>
        <CardContent>
          <MiniChart data={lossHistory} color="#ef4444" height={120} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Accuracy</CardTitle>
        </CardHeader>
        <CardContent>
          <MiniChart
            data={accuracyHistory}
            color="#22c55e"
            height={120}
            formatValue={(v) => `${(v * 100).toFixed(1)}%`}
          />
        </CardContent>
      </Card>
    </div>
  )
}

interface MiniChartProps {
  data: DataPoint[]
  color: string
  height: number
  formatValue?: (value: number) => string
}

function MiniChart({ data, color, height, formatValue }: MiniChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const rect = canvas.getBoundingClientRect()
    const dpr = window.devicePixelRatio || 1

    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    ctx.scale(dpr, dpr)

    // Clear canvas
    ctx.clearRect(0, 0, rect.width, rect.height)

    if (data.length < 2) {
      ctx.fillStyle = "#64748b"
      ctx.font = "12px sans-serif"
      ctx.textAlign = "center"
      ctx.fillText("No data yet", rect.width / 2, rect.height / 2)
      return
    }

    // Find data bounds
    const minValue = Math.min(...data.map((d) => d.value))
    const maxValue = Math.max(...data.map((d) => d.value))
    const minStep = Math.min(...data.map((d) => d.step))
    const maxStep = Math.max(...data.map((d) => d.step))

    const valueRange = maxValue - minValue || 1
    const stepRange = maxStep - minStep || 1

    // Helper function to convert data point to canvas coordinates
    const toCanvasCoords = (point: DataPoint) => ({
      x: ((point.step - minStep) / stepRange) * rect.width,
      y: rect.height - ((point.value - minValue) / valueRange) * rect.height,
    })

    // Draw training data
    ctx.strokeStyle = color
    ctx.lineWidth = 2
    ctx.beginPath()

    const firstPoint = toCanvasCoords(data[0])
    ctx.moveTo(firstPoint.x, firstPoint.y)

    for (let i = 1; i < data.length; i++) {
      const point = toCanvasCoords(data[i])
      ctx.lineTo(point.x, point.y)
    }

    ctx.stroke()

    // Draw current value point
    const lastPoint = toCanvasCoords(data[data.length - 1])
    ctx.fillStyle = color
    ctx.beginPath()
    ctx.arc(lastPoint.x, lastPoint.y, 3, 0, 2 * Math.PI)
    ctx.fill()
  }, [data, color, height])

  const currentValue = data[data.length - 1]?.value

  return (
    <div className="relative">
      <canvas ref={canvasRef} className="w-full" style={{ height: `${height}px` }} />

      {/* Current value overlay */}
      {currentValue !== undefined && (
        <div className="absolute top-2 right-2 text-xs">
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
            <span>{formatValue ? formatValue(currentValue) : currentValue.toFixed(4)}</span>
          </div>
        </div>
      )}
    </div>
  )
}
