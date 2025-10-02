import type { NetworkState, NNNode, NNEdge } from "@/lib/types/neural-network"
import { createLayer, createNode, createEdge } from "./network-factory"

// Layer operations
export function addLayer(
  network: NetworkState,
  layerConfig: { type: "dense"; units: number; activation?: string },
  insertIndex?: number,
): NetworkState {
  const newLayer = createLayer(layerConfig.type, layerConfig.units, layerConfig.activation as any)

  const layers = [...network.layers]
  const nodes = { ...network.nodes }
  const edges = { ...network.edges }

  // Determine position
  const index = insertIndex ?? layers.length
  newLayer.position = { x: index * 200, y: 0 }

  // Insert layer
  layers.splice(index, 0, newLayer)

  // Update positions of subsequent layers
  for (let i = index + 1; i < layers.length; i++) {
    layers[i].position.x = i * 200
  }

  // Create nodes for the new layer
  for (let i = 0; i < layerConfig.units; i++) {
    const node = createNode(newLayer.id, newLayer.activation)
    node.position = { x: newLayer.position.x, y: i * 60 }
    nodes[node.id] = node
  }

  // Connect to adjacent layers
  const newNodes = Object.values(nodes).filter((n) => n.layerId === newLayer.id)

  // Connect from previous layer
  if (index > 0) {
    const prevLayer = layers[index - 1]
    const prevNodes = Object.values(nodes).filter((n) => n.layerId === prevLayer.id)

    prevNodes.forEach((fromNode) => {
      newNodes.forEach((toNode) => {
        const edge = createEdge(fromNode.id, toNode.id)
        edges[edge.id] = edge
      })
    })
  }

  // Connect to next layer
  if (index < layers.length - 1) {
    const nextLayer = layers[index + 1]
    const nextNodes = Object.values(nodes).filter((n) => n.layerId === nextLayer.id)

    newNodes.forEach((fromNode) => {
      nextNodes.forEach((toNode) => {
        const edge = createEdge(fromNode.id, toNode.id)
        edges[edge.id] = edge
      })
    })
  }

  return { layers, nodes, edges }
}

export function removeLayer(network: NetworkState, layerId: string): NetworkState {
  const layers = network.layers.filter((l) => l.id !== layerId)
  const nodes = { ...network.nodes }
  const edges = { ...network.edges }

  // Remove nodes belonging to this layer
  Object.keys(nodes).forEach((nodeId) => {
    if (nodes[nodeId].layerId === layerId) {
      delete nodes[nodeId]
    }
  })

  // Remove edges connected to removed nodes
  Object.keys(edges).forEach((edgeId) => {
    const edge = edges[edgeId]
    if (!nodes[edge.from] || !nodes[edge.to]) {
      delete edges[edgeId]
    }
  })

  // Update layer positions
  layers.forEach((layer, index) => {
    layer.position.x = index * 200
  })

  return { layers, nodes, edges }
}

export function updateLayerUnits(network: NetworkState, layerId: string, newUnits: number): NetworkState {
  const layers = network.layers.map((l) => (l.id === layerId ? { ...l, units: newUnits } : l))
  const nodes = { ...network.nodes }
  const edges = { ...network.edges }

  const layer = layers.find((l) => l.id === layerId)!
  const currentNodes = Object.values(nodes).filter((n) => n.layerId === layerId)
  const currentCount = currentNodes.length

  if (newUnits > currentCount) {
    // Add nodes
    for (let i = currentCount; i < newUnits; i++) {
      const node = createNode(layerId, layer.activation)
      node.position = { x: layer.position.x, y: i * 60 }
      nodes[node.id] = node

      // Connect to adjacent layers
      const prevLayerNodes = Object.values(nodes).filter((n) => {
        const nodeLayer = layers.find((l) => l.id === n.layerId)
        return nodeLayer && nodeLayer.position.x < layer.position.x
      })
      const nextLayerNodes = Object.values(nodes).filter((n) => {
        const nodeLayer = layers.find((l) => l.id === n.layerId)
        return nodeLayer && nodeLayer.position.x > layer.position.x
      })

      // Connect from previous layer
      prevLayerNodes.forEach((fromNode) => {
        const edge = createEdge(fromNode.id, node.id)
        edges[edge.id] = edge
      })

      // Connect to next layer
      nextLayerNodes.forEach((toNode) => {
        const edge = createEdge(node.id, toNode.id)
        edges[edge.id] = edge
      })
    }
  } else if (newUnits < currentCount) {
    // Remove nodes
    const nodesToRemove = currentNodes.slice(newUnits)
    nodesToRemove.forEach((node) => {
      delete nodes[node.id]

      // Remove connected edges
      Object.keys(edges).forEach((edgeId) => {
        const edge = edges[edgeId]
        if (edge.from === node.id || edge.to === node.id) {
          delete edges[edgeId]
        }
      })
    })
  }

  return { layers, nodes, edges }
}

// Node operations
export function addNodeToLayer(network: NetworkState, layerId: string): NetworkState {
  const layer = network.layers.find((l) => l.id === layerId)
  if (!layer) return network

  return updateLayerUnits(network, layerId, layer.units + 1)
}

export function removeNodeFromLayer(network: NetworkState, nodeId: string): NetworkState {
  const node = network.nodes[nodeId]
  if (!node) return network

  const layer = network.layers.find((l) => l.id === node.layerId)
  if (!layer || layer.units <= 1) return network

  return updateLayerUnits(network, layer.id, layer.units - 1)
}

// Edge operations
export function addEdge(network: NetworkState, fromNodeId: string, toNodeId: string): NetworkState {
  const edges = { ...network.edges }

  // Check if edge already exists
  const existingEdge = Object.values(edges).find((e) => e.from === fromNodeId && e.to === toNodeId)
  if (existingEdge) return network

  const edge = createEdge(fromNodeId, toNodeId)
  edges[edge.id] = edge

  return { ...network, edges }
}

export function removeEdge(network: NetworkState, edgeId: string): NetworkState {
  const edges = { ...network.edges }
  delete edges[edgeId]
  return { ...network, edges }
}

// Utility functions
export function getLayerNodes(network: NetworkState, layerId: string): NNNode[] {
  return Object.values(network.nodes).filter((n) => n.layerId === layerId)
}

export function getNodeEdges(network: NetworkState, nodeId: string): { incoming: NNEdge[]; outgoing: NNEdge[] } {
  const edges = Object.values(network.edges)
  return {
    incoming: edges.filter((e) => e.to === nodeId),
    outgoing: edges.filter((e) => e.from === nodeId),
  }
}

export function getNetworkStats(network: NetworkState) {
  const totalNodes = Object.keys(network.nodes).length
  const totalEdges = Object.keys(network.edges).length
  const totalParams = totalEdges + totalNodes // weights + biases

  return {
    layers: network.layers.length,
    nodes: totalNodes,
    edges: totalEdges,
    parameters: totalParams,
  }
}
