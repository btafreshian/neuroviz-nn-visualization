"use client"

import { Button } from "@/components/ui/button"
import { GraphCanvas } from "./graph-canvas/graph-canvas"
import { TrainingControls } from "./training/training-controls"
import { TrainingCharts } from "./training/training-charts"
import { DecisionBoundary } from "./training/decision-boundary"
import { useNetwork, NetworkProvider } from "@/lib/context/network-context"
import { NetworkInspector } from "./inspector/network-inspector"
import { DatasetManager } from "./inspector/dataset-manager"
import { HyperparameterPanel } from "./inspector/hyperparameter-panel"

function PlaygroundContent() {
  const { network } = useNetwork()

  return (
    <div className="h-screen flex flex-col bg-background">
      {/* Header */}
      <header className="border-b bg-card px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <h1 className="text-xl font-semibold">Neural Net Playground</h1>
            <p className="text-sm text-muted-foreground">
              Design, train, and visualize tiny neural networks in your browser
            </p>
          </div>

          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm">
              <span className="mr-2">↑</span>
              Import
            </Button>
            <Button variant="outline" size="sm">
              <span className="mr-2">↓</span>
              Export
            </Button>
            <Button variant="outline" size="sm">
              <span className="mr-2">↗</span>
              Share
            </Button>
          </div>
        </div>
      </header>

      <div className="flex-1 flex">
        {/* Main Canvas Area */}
        <main className="flex-1 flex flex-col">
          <GraphCanvas />
          <TrainingControls />
        </main>

        {/* Right Inspector Panel */}
        <aside className="w-80 border-l bg-card overflow-y-auto">
          <div className="p-4 space-y-4">
            <NetworkInspector />
            <DatasetManager />
            <HyperparameterPanel />

            <div>
              <h4 className="font-medium mb-3">Training Progress</h4>
              <TrainingCharts />
            </div>

            <div>
              <h4 className="font-medium mb-3">Visualization</h4>
              <DecisionBoundary />
            </div>

            {/* Network stats */}
            <div className="bg-muted/50 p-3 rounded-lg">
              <h4 className="text-sm font-medium mb-2">Network Stats</h4>
              <div className="space-y-1 text-xs">
                <div>Layers: {network.layers.length}</div>
                <div>
                  Total Parameters:{" "}
                  {network.layers.reduce((total, layer, i) => {
                    if (i === 0) return total
                    return total + network.layers[i - 1].size * layer.size + layer.size
                  }, 0)}
                </div>
              </div>
            </div>
          </div>
        </aside>
      </div>
    </div>
  )
}

export function NeuralNetworkPlayground() {
  return (
    <NetworkProvider>
      <PlaygroundContent />
    </NetworkProvider>
  )
}
