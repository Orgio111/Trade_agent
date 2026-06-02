"use client";

import { useEffect, useRef } from "react";
import {
  createChart, IChartApi, ISeriesApi,
  CandlestickData, LineData, HistogramData,
  CandlestickSeries, LineSeries, HistogramSeries,
} from "lightweight-charts";

interface Candle {
  time: string;
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

export default function PriceChart({ data }: PriceChartProps) {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartApiRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<"Histogram"> | null>(null);
  const ema9Ref = useRef<ISeriesApi<"Line"> | null>(null);
  const ema21Ref = useRef<ISeriesApi<"Line"> | null>(null);

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

    // v5 API: use addSeries with series definition
    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#00ff88",
      downColor: "#ff0044",
      borderUpColor: "#00ff88",
      borderDownColor: "#ff0044",
      wickUpColor: "#00ff88",
      wickDownColor: "#ff0044",
    });

    const volumeSeries = chart.addSeries(HistogramSeries, {
      priceFormat: { type: "volume" },
      priceScaleId: "volume",
    });
    chart.priceScale("volume").applyOptions({
      scaleMargins: { top: 0.85, bottom: 0 },
    });

    const ema9 = chart.addSeries(LineSeries, {
      color: "#00aaff",
      lineWidth: 1,
      title: "EMA 9",
      priceLineVisible: false,
    });

    const ema21 = chart.addSeries(LineSeries, {
      color: "#ffaa00",
      lineWidth: 1,
      lineStyle: 2,
      title: "EMA 21",
      priceLineVisible: false,
    });

    candleSeriesRef.current = candlestickSeries;
    volumeSeriesRef.current = volumeSeries;
    ema9Ref.current = ema9;
    ema21Ref.current = ema21;
    chartApiRef.current = chart;

    // Set data
    if (data.length > 0) {
      const candleData: CandlestickData[] = data.map((d) => ({
        time: d.time as any,
        open: d.open,
        high: d.high,
        low: d.low,
        close: d.close,
      }));
      candlestickSeries.setData(candleData);

      const volData: HistogramData[] = data.map((d) => ({
        time: d.time as any,
        value: d.volume,
        color: d.close >= d.open ? "#00ff8822" : "#ff004422",
      }));
      volumeSeries.setData(volData);

      // True EMA computation
      const closes = data.map((d) => d.close);
      const ema9Values = computeEMA(closes, 9);
      const ema21Values = computeEMA(closes, 21);

      if (ema9Values.length > 0) {
        const ema9Data: LineData[] = data.slice(data.length - ema9Values.length).map((d, i) => ({
          time: d.time as any,
          value: ema9Values[i],
        }));
        ema9.setData(ema9Data);
      }

      if (ema21Values.length > 0) {
        const ema21Data: LineData[] = data.slice(data.length - ema21Values.length).map((d, i) => ({
          time: d.time as any,
          value: ema21Values[i],
        }));
        ema21.setData(ema21Data);
      }
    }

    chart.timeScale().fitContent();

    // Debounced resize
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
    };
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
