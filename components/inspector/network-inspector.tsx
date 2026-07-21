"use client"

import { Network as NetworkIcon } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useNetwork } from "@/lib/context/network-context"

export function NetworkInspector() {
  const { network } = useNetwork()
  const totalParameters = network.layers.reduce((total, layer, index) => {
    if (index === 0) return total
    return total + network.layers[index - 1].size * layer.size + layer.size
  }, 0)

  return (
    <Card className="gap-4 py-4 shadow-sm">
      <CardHeader className="px-4">
        <CardTitle className="flex items-center gap-2 text-sm">
          <NetworkIcon className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
          Network Inspector
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 px-4">
        <div className="grid grid-cols-[1fr_auto] gap-2">
          <Summary label="Architecture" value={network.layers.map((layer) => layer.size).join(" → ")} />
          <Summary label="Parameters" value={totalParameters.toLocaleString()} align="right" />
        </div>

        <div>
          <div className="mb-2 text-xs font-medium text-muted-foreground">Layers</div>
          <div className="space-y-1.5">
            {network.layers.map((layer) => (
              <div key={layer.id} className="flex items-center justify-between rounded-lg border bg-muted/30 px-3 py-2 text-xs">
                <span className="font-medium">{layer.name}</span>
                <span className="text-muted-foreground">{layer.size} neurons</span>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function Summary({ label, value, align = "left" }: { label: string; value: string; align?: "left" | "right" }) {
  return (
    <div className={`min-w-0 rounded-lg border bg-muted/30 px-3 py-2 ${align === "right" ? "text-right" : ""}`}>
      <div className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="truncate font-mono text-sm font-medium">{value}</div>
    </div>
  )
}
