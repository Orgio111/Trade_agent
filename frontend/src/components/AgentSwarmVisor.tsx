"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";

interface Agent {
  id: string;
  label: string;
  color: string;
  signal: "long" | "short" | "hold";
  confidence: number;
  active: boolean;
}

interface AgentSwarmVisorProps {
  agents: Agent[];
}

export default function AgentSwarmVisor({ agents }: AgentSwarmVisorProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<{
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    renderer: THREE.WebGLRenderer;
    particles: THREE.Points;
    connections: THREE.LineSegments;
    agentNodes: { mesh: THREE.Mesh; label: THREE.Sprite }[];
    animationId: number;
  } | null>(null);

  useEffect(() => {
    if (!mountRef.current || agents.length === 0) return;

    const width = mountRef.current.clientWidth;
    const height = mountRef.current.clientHeight;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0f);

    // Camera
    const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 100);
    camera.position.set(0, 2, 6);
    camera.lookAt(0, 0, 0);

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mountRef.current.appendChild(renderer.domElement);

    // Lights
    const ambient = new THREE.AmbientLight(0x222244, 0.5);
    scene.add(ambient);
    const dirLight = new THREE.DirectionalLight(0xffffff, 1);
    dirLight.position.set(5, 10, 5);
    scene.add(dirLight);

    // Background particle field
    const particleCount = 300;
    const particleGeo = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount * 3; i++) {
      positions[i] = (Math.random() - 0.5) * 20;
    }
    particleGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const particleMat = new THREE.PointsMaterial({
      color: 0x444488,
      size: 0.03,
      transparent: true,
      opacity: 0.6,
    });
    const particles = new THREE.Points(particleGeo, particleMat);
    scene.add(particles);

    // Agent nodes in a circle
    const radius = 2;
    const agentNodes: { mesh: THREE.Mesh; label: THREE.Sprite }[] = [];
    const nodePositions: THREE.Vector3[] = [];

    agents.forEach((agent, i) => {
      const angle = (i / agents.length) * Math.PI * 2;
      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;
      const pos = new THREE.Vector3(x, 0, z);
      nodePositions.push(pos);

      // Sphere
      const colorHex = agent.color === "#00ff88" ? 0x00ff88
        : agent.color === "#ff0044" ? 0xff0044
        : agent.color === "#00aaff" ? 0x00aaff
        : agent.color === "#aa00ff" ? 0xaa00ff
        : agent.color === "#ffaa00" ? 0xffaa00
        : agent.color === "#e0e0e0" ? 0xe0e0e0
        : 0x888888;

      const sphereGeo = new THREE.SphereGeometry(agent.active ? 0.25 : 0.15, 16, 16);
      const sphereMat = new THREE.MeshPhysicalMaterial({
        color: colorHex,
        emissive: colorHex,
        emissiveIntensity: agent.active ? (agent.confidence * 0.5) : 0.05,
        metalness: 0.3,
        roughness: 0.4,
        transparent: true,
        opacity: agent.active ? 1 : 0.4,
      });
      const sphere = new THREE.Mesh(sphereGeo, sphereMat);
      sphere.position.copy(pos);
      scene.add(sphere);

      // Glow
      const glowGeo = new THREE.SphereGeometry(agent.active ? 0.35 : 0.2, 16, 16);
      const glowMat = new THREE.MeshBasicMaterial({
        color: colorHex,
        transparent: true,
        opacity: agent.active ? 0.15 : 0.05,
      });
      const glow = new THREE.Mesh(glowGeo, glowMat);
      glow.position.copy(pos);
      scene.add(glow);

      agentNodes.push({ mesh: sphere, label: null as unknown as THREE.Sprite });
    });

    // Connections between agents
    const connectionPairs: { start: THREE.Vector3; end: THREE.Vector3 }[] = [];
    for (let i = 0; i < nodePositions.length; i++) {
      for (let j = i + 1; j < nodePositions.length; j++) {
        // Not all agents are connected — connect based on active status
        if (agents[i].active && agents[j].active && Math.random() > 0.3) {
          connectionPairs.push({ start: nodePositions[i], end: nodePositions[j] });
        }
      }
    }

    const connPositions: number[] = [];
    connectionPairs.forEach((pair) => {
      connPositions.push(pair.start.x, pair.start.y, pair.start.z);
      connPositions.push(pair.end.x, pair.end.y, pair.end.z);
    });

    const connGeo = new THREE.BufferGeometry();
    connGeo.setAttribute("position", new THREE.Float32BufferAttribute(connPositions, 3));
    const connMat = new THREE.LineBasicMaterial({
      color: 0x444488,
      transparent: true,
      opacity: 0.2,
    });
    const connections = new THREE.LineSegments(connGeo, connMat);
    scene.add(connections);

    // Animation
    let time = 0;
    const animate = () => {
      time += 0.005;
      // Rotate entire scene slowly
      particles.rotation.y += 0.001;
      // Pulse nodes
      agentNodes.forEach((node) => {
        const scale = 1 + Math.sin(time * 2 + node.mesh.position.x) * 0.1;
        node.mesh.scale.set(scale, scale, scale);
      });
      renderer.render(scene, camera);
      sceneRef.current!.animationId = requestAnimationFrame(animate);
    };

    sceneRef.current = { scene, camera, renderer, particles, connections, agentNodes, animationId: 0 };
    animate();

    // Resize
    const handleResize = () => {
      if (!mountRef.current || !sceneRef.current) return;
      const w = mountRef.current.clientWidth;
      const h = mountRef.current.clientHeight;
      sceneRef.current.camera.aspect = w / h;
      sceneRef.current.camera.updateProjectionMatrix();
      sceneRef.current.renderer.setSize(w, h);
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      if (sceneRef.current) {
        cancelAnimationFrame(sceneRef.current.animationId);
        renderer.dispose();
        if (mountRef.current?.contains(renderer.domElement)) {
          mountRef.current.removeChild(renderer.domElement);
        }
      }
      sceneRef.current = null;
    };
  }, [agents]);

  return (
    <div
      ref={mountRef}
      style={{
        width: "100%",
        height: "100%",
        minHeight: 280,
        borderRadius: 8,
        overflow: "hidden",
        position: "relative",
      }}
    >
      {/* Overlay labels */}
      {agents.map((agent, i) => {
        const angle = (i / agents.length) * Math.PI * 2;
        const radius = 2;
        const x = Math.cos(angle) * radius;
        const z = Math.sin(angle) * radius;
        return (
          <div
            key={agent.id}
            style={{
              position: "absolute",
              left: `${50 + (x / 4) * 50}%`,
              top: `${50 - (z / 4) * 50}%`,
              transform: "translate(-50%, -50%)",
              color: agent.active ? agent.color : "#444",
              fontSize: 9,
              fontWeight: 600,
              textTransform: "uppercase",
              letterSpacing: 1,
              pointerEvents: "none",
              textShadow: `0 0 8px ${agent.color}44`,
              opacity: agent.active ? 1 : 0.4,
            }}
          >
            {agent.id}
          </div>
        );
      })}
    </div>
  );
}
