import type { Metadata } from "next"
import { NeuralNetworkPlayground } from "@/components/neural-network-playground"

export const metadata: Metadata = {
  title: "Neural Net Playground",
  description: "Design, train, and visualize tiny neural networks in your browser.",
}

export default function HomePage() {
  return <NeuralNetworkPlayground />
}
