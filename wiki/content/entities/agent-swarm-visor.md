---
title: AgentSwarmVisor
type: entity
tags:
  - frontend
  - threejs
  - brain
  - visualization
created: 2026-06-30
updated: 2026-06-30
source_file: frontend/src/components/AgentSwarmVisor.tsx
---

# AgentSwarmVisor

**File:** `frontend/src/components/AgentSwarmVisor.tsx`
**Type:** React component (Three.js 3D visualization)
**Purpose:** Real-time 3D visualization of the brain agent swarm — renders each brain as a glowing sphere in a circular layout with inter-agent connections and particle background.

## Architecture

```
Agent[] (props)
    ↓
THREE.WebGLRenderer
    ├── Background particle field (300 points, ambient rotation)
    ├── Agent nodes (spheres in circle, radius=2)
    │   ├── MeshPhysicalMaterial (emissive intensity ∝ confidence)
    │   └── Glow sphere (transparent overlay)
    ├── Connections (LineSegments between active agents, 70% density)
    └── HTML overlay labels (brain_id text positioned over 3D nodes)
```

## Agent Interface

```typescript
interface Agent {
  id: string;          // brain_id (e.g., "timesfm", "freqai")
  label: string;       // display name
  color: string;       // hex color (#00ff88, #ff0044, etc.)
  signal: "long" | "short" | "hold";
  confidence: number;  // 0.0–1.0 → drives emissive intensity
  active: boolean;     // false → dimmed (opacity 0.4, smaller sphere)
}
```

## Visual Mapping

| Agent Property | 3D Effect |
|----------------|-----------|
| `active=true` | Sphere radius 0.25, full opacity, emissive intensity = confidence × 0.5 |
| `active=false` | Sphere radius 0.15, opacity 0.4, emissive 0.05 |
| `color` | Hex → THREE color (emissive + glow + label text shadow) |
| Connection | Drawn between two active agents with 70% probability |
| Background | 300-point particle field, slow Y-rotation (0.001 rad/frame) |

## Animation

- **Node pulse:** Each sphere scales 1 ± 0.1 using `sin(time * 2 + position.x)` — creates wave effect
- **Particle rotation:** Entire background rotates at 0.001 rad/frame
- **Resize:** Responsive — listens to `window.resize`, updates camera aspect + renderer size

## Performance

- **WebGL renderer** with `antialias: true`, `alpha: true`
- **Pixel ratio clamped** to `min(devicePixelRatio, 2)` — prevents GPU overload on retina displays
- **Cleanup:** Disposes renderer, removes DOM element, cancels animation frame on unmount

## Dependencies

| Package | Purpose |
|---------|---------|
| `three` (THREE) | 3D scene, camera, renderer, materials, geometry |
| `react` (useRef, useEffect) | DOM mount, lifecycle management |

## Related

- [[brain-ecosystem]] — Brain weights and tier distribution
- [[real-time-trading-dashboard]] — Dashboard layout where this component is used
- [[signal-aggregation-logic]] — How brain signals are aggregated before visualization
