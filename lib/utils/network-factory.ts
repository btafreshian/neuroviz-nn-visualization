import { nanoid } from "nanoid"
import type { NNLayer, NNNode, NNEdge, NetworkState, Activation, LayerType } from "@/lib/types/neural-network"

// Factory functions for creating network components
export function createLayer(type: LayerType, units: number, activation: Activation = "relu", name?: string): NNLayer {
  return {
    id: nanoid(),
    type,
    name: name || `${type.charAt(0).toUpperCase() + type.slice(1)} Layer`,
    activation,
    units,
    position: { x: 0, y: 0 },
    dropout: type === "dense" ? 0.0 : undefined,
  }
}

export function createNode(layerId: string, activation: Activation = "relu"): NNNode {
  return {
    id: nanoid(),
    layerId,
    bias: Math.random() * 0.2 - 0.1, // Small random bias
    activation,
  }
}

export function createEdge(fromNodeId: string, toNodeId: string): NNEdge {
  return {
    id: nanoid(),
    from: fromNodeId,
    to: toNodeId,
    weight: Math.random() * 0.4 - 0.2, // Xavier-like initialization
  }
}

// Create a complete network with layers, nodes, and edges
export function createNetwork(
  layerConfigs: Array<{ type: LayerType; units: number; activation?: Activation }>,
): NetworkState {
  const layers: NNLayer[] = []
  const nodes: Record<string, NNNode> = {}
  const edges: Record<string, NNEdge> = {}

  // Create layers and nodes
  layerConfigs.forEach((config, layerIndex) => {
    const layer = createLayer(config.type, config.units, config.activation, `Layer ${layerIndex + 1}`)
    layer.position = { x: layerIndex * 200, y: 0 }
    layers.push(layer)

    // Create nodes for this layer
    for (let i = 0; i < config.units; i++) {
      const node = createNode(layer.id, layer.activation)
      node.position = { x: layerIndex * 200, y: i * 60 }
      nodes[node.id] = node
    }
  })

  // Create edges between consecutive layers
  for (let i = 0; i < layers.length - 1; i++) {
    const currentLayer = layers[i]
    const nextLayer = layers[i + 1]

    const currentNodes = Object.values(nodes).filter((n) => n.layerId === currentLayer.id)
    const nextNodes = Object.values(nodes).filter((n) => n.layerId === nextLayer.id)

    // Fully connect layers
    currentNodes.forEach((fromNode) => {
      nextNodes.forEach((toNode) => {
        const edge = createEdge(fromNode.id, toNode.id)
        edges[edge.id] = edge
      })
    })
  }

  return { layers, nodes, edges }
}

// Predefined network architectures
export const networkTemplates = {
  xor: () =>
    createNetwork([
      { type: "input", units: 2, activation: "linear" },
      { type: "dense", units: 4, activation: "relu" },
      { type: "output", units: 1, activation: "sigmoid" },
    ]),

  classification: () =>
    createNetwork([
      { type: "input", units: 2, activation: "linear" },
      { type: "dense", units: 8, activation: "relu" },
      { type: "dense", units: 4, activation: "relu" },
      { type: "output", units: 2, activation: "sigmoid" },
    ]),

  regression: () =>
    createNetwork([
      { type: "input", units: 2, activation: "linear" },
      { type: "dense", units: 6, activation: "relu" },
      { type: "dense", units: 3, activation: "relu" },
      { type: "output", units: 1, activation: "linear" },
    ]),
}
