import type { NetworkState } from "@/lib/types/neural-network"

export interface ValidationError {
  type: "error" | "warning"
  message: string
  layerId?: string
  nodeId?: string
  edgeId?: string
}

export function validateNetwork(network: NetworkState): ValidationError[] {
  const errors: ValidationError[] = []
  const { layers, nodes, edges } = network

  // Check if network has at least input and output layers
  const inputLayers = layers.filter((l) => l.type === "input")
  const outputLayers = layers.filter((l) => l.type === "output")

  if (inputLayers.length === 0) {
    errors.push({ type: "error", message: "Network must have at least one input layer" })
  }
  if (inputLayers.length > 1) {
    errors.push({ type: "error", message: "Network can only have one input layer" })
  }
  if (outputLayers.length === 0) {
    errors.push({ type: "error", message: "Network must have at least one output layer" })
  }
  if (outputLayers.length > 1) {
    errors.push({ type: "error", message: "Network can only have one output layer" })
  }

  // Check layer ordering
  const sortedLayers = [...layers].sort((a, b) => a.position.x - b.position.x)
  if (sortedLayers[0]?.type !== "input") {
    errors.push({ type: "error", message: "First layer must be input layer" })
  }
  if (sortedLayers[sortedLayers.length - 1]?.type !== "output") {
    errors.push({ type: "error", message: "Last layer must be output layer" })
  }

  // Check each layer has nodes
  layers.forEach((layer) => {
    const layerNodes = Object.values(nodes).filter((n) => n.layerId === layer.id)
    if (layerNodes.length === 0) {
      errors.push({
        type: "error",
        message: `Layer "${layer.name}" has no nodes`,
        layerId: layer.id,
      })
    }
    if (layerNodes.length !== layer.units) {
      errors.push({
        type: "warning",
        message: `Layer "${layer.name}" units (${layer.units}) doesn't match node count (${layerNodes.length})`,
        layerId: layer.id,
      })
    }
  })

  // Check connectivity between consecutive layers
  for (let i = 0; i < sortedLayers.length - 1; i++) {
    const currentLayer = sortedLayers[i]
    const nextLayer = sortedLayers[i + 1]

    const currentNodes = Object.values(nodes).filter((n) => n.layerId === currentLayer.id)
    const nextNodes = Object.values(nodes).filter((n) => n.layerId === nextLayer.id)

    const layerEdges = Object.values(edges).filter(
      (e) => currentNodes.some((n) => n.id === e.from) && nextNodes.some((n) => n.id === e.to),
    )

    if (layerEdges.length === 0) {
      errors.push({
        type: "error",
        message: `No connections between "${currentLayer.name}" and "${nextLayer.name}"`,
        layerId: currentLayer.id,
      })
    }

    // Check for full connectivity (warning only)
    const expectedConnections = currentNodes.length * nextNodes.length
    if (layerEdges.length < expectedConnections) {
      errors.push({
        type: "warning",
        message: `Incomplete connectivity between "${currentLayer.name}" and "${nextLayer.name}" (${layerEdges.length}/${expectedConnections})`,
        layerId: currentLayer.id,
      })
    }
  }

  // Check for orphaned nodes
  Object.values(nodes).forEach((node) => {
    const layer = layers.find((l) => l.id === node.layerId)
    if (!layer) {
      errors.push({
        type: "error",
        message: "Node belongs to non-existent layer",
        nodeId: node.id,
      })
    }
  })

  // Check for invalid edges
  Object.values(edges).forEach((edge) => {
    const fromNode = nodes[edge.from]
    const toNode = nodes[edge.to]

    if (!fromNode) {
      errors.push({
        type: "error",
        message: "Edge references non-existent source node",
        edgeId: edge.id,
      })
    }
    if (!toNode) {
      errors.push({
        type: "error",
        message: "Edge references non-existent target node",
        edgeId: edge.id,
      })
    }
  })

  return errors
}

export function isNetworkValid(network: NetworkState): boolean {
  const errors = validateNetwork(network)
  return !errors.some((e) => e.type === "error")
}
