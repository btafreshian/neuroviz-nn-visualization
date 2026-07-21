"use client"

import { BrainCircuit } from "lucide-react"
import { GraphCanvas } from "./graph-canvas/graph-canvas"
import { TrainingControls } from "./training/training-controls"
import { TrainingDiagnostics } from "./training/training-diagnostics"
import { VisualizationTimeline } from "./training/visualization-timeline"
import { TrainingCharts } from "./training/training-charts"
import { DecisionBoundary } from "./training/decision-boundary"
import { NetworkProvider } from "@/lib/context/network-context"
import { NetworkInspector } from "./inspector/network-inspector"
import { DatasetManager } from "./inspector/dataset-manager"
import { HyperparameterPanel } from "./inspector/hyperparameter-panel"

function PlaygroundContent() {
  return (
    <div className="min-h-screen bg-muted/30">
      <header className="sticky top-0 z-30 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/80">
        <div className="flex min-h-16 items-center px-4 lg:px-6">
          <div className="flex min-w-0 items-center gap-3">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground shadow-sm">
              <BrainCircuit className="h-5 w-5" aria-hidden="true" />
            </div>
            <div className="min-w-0">
              <h1 className="truncate text-base font-semibold tracking-tight sm:text-lg">Neural Net Playground</h1>
              <p className="hidden truncate text-xs text-muted-foreground md:block">Train, inspect, and replay a compact neural network</p>
            </div>
          </div>
        </div>
      </header>

      <main className="grid gap-4 p-4 lg:p-6 xl:grid-cols-[minmax(0,1fr)_20rem]">
        <section className="min-w-0 space-y-4">
          <section className="overflow-hidden rounded-xl border bg-card shadow-sm">
            <div className="h-[380px] p-4 sm:h-[420px] lg:p-6">
              <GraphCanvas />
            </div>
            <VisualizationTimeline />
          </section>

          <section className="overflow-hidden rounded-xl border bg-card shadow-sm">
            <TrainingControls />
            <TrainingDiagnostics />
          </section>

          <div className="grid items-stretch gap-4 lg:grid-cols-2">
            <section className="flex flex-col rounded-xl border bg-card shadow-sm">
              <div className="border-b px-5 py-4">
                <h2 className="text-sm font-semibold">Training Progress</h2>
                <p className="mt-1 text-xs text-muted-foreground">Train, validation, and previous-run comparison</p>
              </div>
              <div className="flex-1 p-4">
                <TrainingCharts />
              </div>
            </section>

            <section className="flex flex-col rounded-xl border bg-card shadow-sm">
              <div className="border-b px-5 py-4">
                <h2 className="text-sm font-semibold">Decision Boundary</h2>
                <p className="mt-1 text-xs text-muted-foreground">Model regions and dataset samples</p>
              </div>
              <div className="flex-1 p-4">
                <DecisionBoundary />
              </div>
            </section>
          </div>
        </section>

        <aside className="grid items-start gap-4 md:grid-cols-2 lg:grid-cols-3 xl:sticky xl:top-20 xl:grid-cols-1 xl:self-start">
          <NetworkInspector />
          <HyperparameterPanel />
          <DatasetManager />
        </aside>

      </main>
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
