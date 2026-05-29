"use client";

import { useEffect, useRef, useCallback, useState } from "react";
import type { DashboardSnapshot } from "@/lib/types";

export type ConnectionStatus = "connecting" | "connected" | "disconnected" | "error";

interface UseTradingSocketOptions {
  /** WebSocket URL — defaults to the same host /ws */
  url?: string;
  /** Reconnect delay in ms (default: 2000) */
  reconnectDelay?: number;
  /** Maximum reconnection attempts (default: Infinity) */
  maxRetries?: number;
  /** Called with each snapshot update */
  onMessage?: (data: DashboardSnapshot) => void;
}

interface UseTradingSocketReturn {
  /** Latest snapshot received */
  data: DashboardSnapshot | null;
  /** Connection status */
  status: ConnectionStatus;
  /** Manually reconnect */
  reconnect: () => void;
}

export function useTradingSocket(
  options: UseTradingSocketOptions = {}
): UseTradingSocketReturn {
  const {
    reconnectDelay = 2000,
    maxRetries = Infinity,
    onMessage,
  } = options;

  const [data, setData] = useState<DashboardSnapshot | null>(null);
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const wsRef = useRef<WebSocket | null>(null);
  const retryCountRef = useRef(0);
  const retryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;

  const getUrl = useCallback(() => {
    if (options.url) return options.url;
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${protocol}//${window.location.host}/ws`;
  }, [options.url]);

  const connect = useCallback(() => {
    if (!mountedRef.current) return;

    // Clean up any existing connection
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setStatus("connecting");
    const url = getUrl();
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      if (!mountedRef.current) {
        ws.close();
        return;
      }
      setStatus("connected");
      retryCountRef.current = 0;
    };

    ws.onmessage = (event: MessageEvent) => {
      try {
        const snapshot = JSON.parse(event.data) as DashboardSnapshot;
        if (mountedRef.current) {
          setData(snapshot);
          onMessageRef.current?.(snapshot);
        }
      } catch {
        // Ignore malformed messages
      }
    };

    ws.onclose = () => {
      if (!mountedRef.current) return;
      setStatus("disconnected");
      wsRef.current = null;

      // Schedule reconnection
      if (retryCountRef.current < maxRetries) {
        retryCountRef.current += 1;
        retryTimerRef.current = setTimeout(connect, reconnectDelay);
      } else {
        setStatus("error");
      }
    };

    ws.onerror = () => {
      // onclose will fire after this
    };
  }, [getUrl, reconnectDelay, maxRetries]);

  const reconnect = useCallback(() => {
    retryCountRef.current = 0;
    connect();
  }, [connect]);

  // Connect on mount, disconnect on unmount
  useEffect(() => {
    mountedRef.current = true;
    connect();

    return () => {
      mountedRef.current = false;
      if (retryTimerRef.current) clearTimeout(retryTimerRef.current);
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [connect]);

  // Reconnect when tab becomes visible and we're disconnected
  useEffect(() => {
    const handleVisibility = () => {
      if (
        !document.hidden &&
        (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN)
      ) {
        reconnect();
      }
    };
    document.addEventListener("visibilitychange", handleVisibility);
    return () => document.removeEventListener("visibilitychange", handleVisibility);
  }, [reconnect]);

  return { data, status, reconnect };
}
