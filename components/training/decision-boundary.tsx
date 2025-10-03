"use client"

import { useEffect, useRef } from "react"
import { useNetwork } from "@/lib/context/network-context"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

function mockNetworkPrediction(x: number, y: number, accuracy: number, epoch: number, datasetName: string): number {
  const trainingProgress = Math.min(1, epoch / 200)

  // Dataset-specific decision boundaries
  if (datasetName === "XOR") {
    // XOR pattern: diagonal quadrants
    const xorPattern = (x > 0.5 ? 1 : 0) ^ (y > 0.5 ? 1 : 0)
    const noise = (Math.random() - 0.5) * (1 - accuracy) * 0.5
    return xorPattern + noise
  } else if (datasetName === "Moons") {
    // Curved boundary for moons
    const centerX = 0.5
    const centerY = 0.25
    const radius = 0.8
    const dist = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2)
    const moonBoundary = dist < radius ? 1 : 0
    const noise = (Math.random() - 0.5) * (1 - accuracy) * 0.3
    const result = moonBoundary + noise + (y - 0.25) * 0.5
    return Math.max(0, Math.min(1, result))
  } else if (datasetName === "Spiral") {
    // Spiral pattern
    const angle = Math.atan2(y, x)
    const radius = Math.sqrt(x * x + y * y)
    const spiralValue = Math.sin(angle - radius * 0.5 + trainingProgress * Math.PI)
    const noise = (Math.random() - 0.5) * (1 - accuracy) * 0.4
    return (spiralValue + 1) / 2 + noise
  }

  // Default linear boundary
  const linear = x + y - 1
  const noise = (Math.random() - 0.5) * (1 - accuracy) * 0.5
  return 1 / (1 + Math.exp(-(linear + noise)))
}

export function DecisionBoundary() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const { dataset, trainingProgress, network, isTraining } = useNetwork()

  useEffect(() => {
    if (!dataset || !canvasRef.current) return

    // Only show decision boundary for 2D datasets
    if (dataset.features[0]?.length !== 2) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height

    // Clear canvas
    ctx.clearRect(0, 0, width, height)

    // Find data bounds with padding
    const xValues = dataset.features.map((f) => f[0])
    const yValues = dataset.features.map((f) => f[1])
    const xMin = Math.min(...xValues) - 0.5
    const xMax = Math.max(...xValues) + 0.5
    const yMin = Math.min(...yValues) - 0.5
    const yMax = Math.max(...yValues) + 0.5

    if (network.layers.length > 0 && trainingProgress.epoch > 0) {
      const imageData = ctx.createImageData(width, height)
      const data = imageData.data

      for (let py = 0; py < height; py++) {
        for (let px = 0; px < width; px++) {
          const dataX = xMin + (px / width) * (xMax - xMin)
          const dataY = yMax - (py / height) * (yMax - yMin)

          const prediction = mockNetworkPrediction(
            dataX,
            dataY,
            trainingProgress.accuracy,
            trainingProgress.epoch,
            dataset.name,
          )

          const idx = (py * width + px) * 4

          if (prediction > 0.5) {
            data[idx] = 251
            data[idx + 1] = 191
            data[idx + 2] = 36
          } else {
            data[idx] = 59
            data[idx + 1] = 130
            data[idx + 2] = 246
          }

          const confidence = Math.abs(prediction - 0.5) * 2
          const baseAlpha = 0.1 + trainingProgress.accuracy * 0.2
          data[idx + 3] = Math.floor((baseAlpha + confidence * 0.15) * 255)
        }
      }

      ctx.putImageData(imageData, 0, 0)
    } else {
      ctx.fillStyle = "rgba(100, 116, 139, 0.1)"
      ctx.fillRect(0, 0, width, height)
    }

    dataset.features.forEach((feature, idx) => {
      const x = ((feature[0] - xMin) / (xMax - xMin)) * width
      const y = height - ((feature[1] - yMin) / (yMax - yMin)) * height

      const isClass1 = dataset.labels[idx] === 1

      // Draw glow effect
      ctx.beginPath()
      ctx.arc(x, y, 5, 0, 2 * Math.PI)
      ctx.fillStyle = isClass1 ? "rgba(251, 191, 36, 0.3)" : "rgba(59, 130, 246, 0.3)"
      ctx.fill()

      // Draw main point
      ctx.beginPath()
      ctx.arc(x, y, 3.5, 0, 2 * Math.PI)
      ctx.fillStyle = isClass1 ? "#fbbf24" : "#3b82f6"
      ctx.fill()

      // Draw border for contrast
      ctx.strokeStyle = "#ffffff"
      ctx.lineWidth = 1.5
      ctx.stroke()
    })
  }, [dataset, trainingProgress, network, isTraining])

  if (!dataset || dataset.features[0]?.length !== 2) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Decision Boundary</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-sm text-muted-foreground text-center py-8">
            Decision boundary visualization is only available for 2D datasets
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">Decision Boundary</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <canvas ref={canvasRef} width={280} height={200} className="border rounded bg-muted/20" />

        <div className="flex justify-between text-xs text-muted-foreground">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full bg-blue-500" />
            <span>Class A</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full bg-amber-400" />
            <span>Class B</span>
          </div>
        </div>

        {trainingProgress.epoch > 0 && (
          <div className="text-xs text-muted-foreground text-center">
            Epoch {trainingProgress.epoch} • Accuracy: {(trainingProgress.accuracy * 100).toFixed(1)}%
          </div>
        )}
      </CardContent>
    </Card>
  )
}
