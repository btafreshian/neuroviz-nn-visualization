"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useNetwork } from "@/lib/context/network-context"

export function NetworkInspector() {
  const { network } = useNetwork()

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">Network Inspector</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          <div>
            <div className="text-xs text-muted-foreground">Architecture</div>
            <div className="text-sm font-mono">{network.layers.map((layer) => layer.size).join(" → ")}</div>
          </div>

          <div>
            <div className="text-xs text-muted-foreground">Total Parameters</div>
            <div className="text-sm font-mono">
              {network.layers.reduce((total, layer, i) => {
                if (i === 0) return total
                return total + network.layers[i - 1].size * layer.size + layer.size
              }, 0)}
            </div>
          </div>

          <div>
            <div className="text-xs text-muted-foreground">Layers</div>
            <div className="space-y-1">
              {network.layers.map((layer, i) => (
                <div key={layer.id} className="text-xs bg-muted/50 p-2 rounded">
                  <div className="font-medium">{layer.name}</div>
                  <div className="text-muted-foreground">{layer.size} neurons</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
