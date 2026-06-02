"use client";

import { useEffect, useRef } from "react";
import {
  createChart, IChartApi, ISeriesApi, Time,
  CandlestickData, LineData, HistogramData,
  CandlestickSeries, LineSeries, HistogramSeries,
} from "lightweight-charts";

interface Candle {
  time: number | string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface PriceChartProps {
  data: Candle[];
}

function computeEMA(closes: number[], period: number): number[] {
  if (closes.length < period) return [];
  const k = 2 / (period + 1);
  const result: number[] = [];
  let ema = closes.slice(0, period).reduce((a, b) => a + b, 0) / period;
  for (let i = period - 1; i < closes.length; i++) {
    if (i > period - 1) {
      ema = closes[i] * k + ema * (1 - k);
    }
    result.push(ema);
  }
  return result;
}

/**
 * Safe helper to convert candle data to CandlestickData[].
 * Filters out any items with null/NaN values to avoid lightweight-charts
 * "Value is null" errors during rendering.
 */
function toCandleData(candles: Candle[]): CandlestickData[] {
  return candles
    .filter((d) => {
      if (d.open == null || isNaN(d.open)) return false;
      if (d.high == null || isNaN(d.high)) return false;
      if (d.low == null || isNaN(d.low)) return false;
      if (d.close == null || isNaN(d.close)) return false;
      if (d.time == null) return false;
      return true;
    })
    .map((d) => ({
      time: d.time as Time,
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
    }));
}

function toVolumeData(candles: Candle[]): HistogramData[] {
  return candles
    .filter((d) => d.volume != null && !isNaN(d.volume) && d.time != null)
    .map((d) => ({
      time: d.time as Time,
      value: d.volume,
      color: d.close >= d.open ? "#00ff8822" : "#ff004422",
    }));
}

function toEmaData(
  candles: Candle[],
  values: number[]
): LineData[] {
  if (values.length === 0 || candles.length < values.length) return [];
  return candles
    .slice(candles.length - values.length)
    .filter((d) => d.time != null)
    .map((d, i) => ({
      time: d.time as Time,
      value: values[i],
    }));
}

export default function PriceChart({ data }: PriceChartProps) {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartApiRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<"Histogram"> | null>(null);
  const ema9Ref = useRef<ISeriesApi<"Line"> | null>(null);
  const ema21Ref = useRef<ISeriesApi<"Line"> | null>(null);
  const hasFitRef = useRef(false);

  // ── Chart creation (runs once) ───────────────────────────
  useEffect(() => {
    if (!chartRef.current) return;

    const chart = createChart(chartRef.current, {
      layout: {
        background: { color: "#0a0a0f" },
        textColor: "#888",
        fontSize: 10,
        fontFamily: "'Courier New', monospace",
      },
      grid: {
        vertLines: { color: "#1a1a2e" },
        horzLines: { color: "#1a1a2e" },
      },
      width: chartRef.current.clientWidth,
      height: 320,
      crosshair: {
        mode: 0,
        vertLine: { color: "#444488", width: 1, style: 2, labelBackgroundColor: "#444488" },
        horzLine: { color: "#444488", width: 1, style: 2, labelBackgroundColor: "#444488" },
      },
      timeScale: {
        borderColor: "#1a1a2e",
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: "#1a1a2e",
      },
    });

    // Candlestick series
    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#00ff88",
      downColor: "#ff0044",
      borderUpColor: "#00ff88",
      borderDownColor: "#ff0044",
      wickUpColor: "#00ff88",
      wickDownColor: "#ff0044",
    });
    candleSeriesRef.current = candlestickSeries;

    // Volume histogram
    const volumeSeries = chart.addSeries(HistogramSeries, {
      priceFormat: { type: "volume" },
      priceScaleId: "volume",
    });
    chart.priceScale("volume").applyOptions({
      scaleMargins: { top: 0.85, bottom: 0 },
    });
    volumeSeriesRef.current = volumeSeries;

    // EMA 9
    const ema9 = chart.addSeries(LineSeries, {
      color: "#00aaff",
      lineWidth: 1,
      title: "EMA 9",
      priceLineVisible: false,
    });
    ema9Ref.current = ema9;

    // EMA 21
    const ema21 = chart.addSeries(LineSeries, {
      color: "#ffaa00",
      lineWidth: 1,
      lineStyle: 2,
      title: "EMA 21",
      priceLineVisible: false,
    });
    ema21Ref.current = ema21;

    chartApiRef.current = chart;

    // Debounced resize handler
    let resizeTimer: ReturnType<typeof setTimeout>;
    const handleResize = () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        if (chartRef.current && chartApiRef.current) {
          chartApiRef.current.applyOptions({ width: chartRef.current.clientWidth });
        }
      }, 100);
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      clearTimeout(resizeTimer);
      chart.remove();
      chartApiRef.current = null;
      candleSeriesRef.current = null;
      volumeSeriesRef.current = null;
      ema9Ref.current = null;
      ema21Ref.current = null;
    };
  }, []); // ⬅️ Empty deps — chart created once

  // ── Data updates (runs every time data changes) ──────────
  useEffect(() => {
    if (!candleSeriesRef.current || !volumeSeriesRef.current) return;

    if (data.length === 0) return;

    // Candles
    const candleData = toCandleData(data);
    if (candleData.length > 0) {
      try {
        candleSeriesRef.current.setData(candleData);
      } catch (err) {
        console.warn("PriceChart: candle setData error", err);
      }
    }

    // Volume
    const volData = toVolumeData(data);
    if (volData.length > 0) {
      try {
        volumeSeriesRef.current.setData(volData);
      } catch (err) {
        console.warn("PriceChart: volume setData error", err);
      }
    }

    // EMAs
    const closes = data.map((d) => d.close);
    const ema9Values = computeEMA(closes, 9);
    const ema21Values = computeEMA(closes, 21);

    if (ema9Ref.current && ema9Values.length > 0) {
      const ema9Data = toEmaData(data, ema9Values);
      if (ema9Data.length > 0) {
        try {
          ema9Ref.current.setData(ema9Data);
        } catch (err) {
          console.warn("PriceChart: EMA9 setData error", err);
        }
      }
    }

    if (ema21Ref.current && ema21Values.length > 0) {
      const ema21Data = toEmaData(data, ema21Values);
      if (ema21Data.length > 0) {
        try {
          ema21Ref.current.setData(ema21Data);
        } catch (err) {
          console.warn("PriceChart: EMA21 setData error", err);
        }
      }
    }

    // Fit content on first load only — prevents snapping back on every tick
    if (!hasFitRef.current && chartApiRef.current && candleData.length > 0) {
      try {
        chartApiRef.current.timeScale().fitContent();
        hasFitRef.current = true;
      } catch { /* ignore */ }
    }
  }, [data]);

  return (
    <div
      ref={chartRef}
      style={{
        width: "100%",
        height: 320,
        borderRadius: 8,
        overflow: "hidden",
      }}
    />
  );
}
