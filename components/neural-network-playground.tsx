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

      <div className="flex-1 overflow-hidden p-4 gap-4 grid grid-cols-[1fr_320px]">
        {/* Left side - Network visualization and controls */}
        <div className="flex flex-col gap-4 min-h-0 overflow-y-auto">
          {/* Network visualization - fixed height with padding */}
          <div className="bg-card border rounded-lg p-8 h-[450px] flex-shrink-0">
            <GraphCanvas />
          </div>

          {/* Training controls */}
          <div className="bg-card border rounded-lg flex-shrink-0">
            <TrainingControls />
          </div>

          {/* Training Progress and Decision Boundary independently scrollable */}
          <div className="grid grid-cols-2 gap-4 flex-shrink-0">
            <div className="bg-card border rounded-lg flex flex-col h-[500px]">
              <h4 className="font-medium p-4 pb-2 flex-shrink-0">Training Progress</h4>
              <div className="overflow-y-auto flex-1 px-4 pb-4">
                <TrainingCharts />
              </div>
            </div>

            <div className="bg-card border rounded-lg flex flex-col h-[500px]">
              <h4 className="font-medium p-4 pb-2 flex-shrink-0">Decision Boundary</h4>
              <div className="overflow-y-auto flex-1 px-4 pb-4">
                <DecisionBoundary />
              </div>
            </div>
          </div>
        </div>

        {/* Right side - Inspector panels */}
        <div className="flex flex-col gap-4 overflow-y-auto">
          <div className="bg-card border rounded-lg p-4">
            <NetworkInspector />
          </div>

          <div className="bg-card border rounded-lg p-4">
            <DatasetManager />
          </div>

          <div className="bg-card border rounded-lg p-4">
            <HyperparameterPanel />
          </div>
        </div>
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
