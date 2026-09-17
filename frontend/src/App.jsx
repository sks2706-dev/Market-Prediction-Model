import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { createChart, ColorType } from 'lightweight-charts';
import { ArrowUpRight, ArrowDownRight, Clock, Activity } from 'lucide-react';

export default function App() {
  const todayStr = new Date().toISOString().split('T')[0];
  
  const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

  const [date, setDate] = useState(todayStr);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const areaSeriesRef = useRef(null);

  const fetchPrediction = async (selectedDate = date) => {
    setLoading(true);
    setError('');
    try {
      const response = await axios.get(`${API_BASE_URL}/api/predict`, {
        params: { target_date: selectedDate }
      });
      setData(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to connect to backend server.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!chartContainerRef.current) return;

    chartContainerRef.current.innerHTML = '';

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#a8a29e',
        fontFamily: "'Plus Jakarta Sans', sans-serif",
      },
      grid: {
        vertLines: { color: '#1c1917' },
        horzLines: { color: '#1c1917' },
      },
      width: chartContainerRef.current.clientWidth || 800,
      height: 420,
      timeScale: {
        borderColor: '#292524',
        fixLeftEdge: true,  
        fixRightEdge: true
      },
      rightPriceScale: { borderColor: '#292524' },
    });

    const areaSeries = chart.addAreaSeries({
      lineColor: '#10b981',
      topColor: 'rgba(16, 185, 129, 0.2)',
      bottomColor: 'rgba(16, 185, 129, 0.0)',
      lineWidth: 2,
    });

    chartRef.current = chart;
    areaSeriesRef.current = areaSeries;

    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };

    window.addEventListener('resize', handleResize);

    fetchPrediction(todayStr);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
      chartRef.current = null;
      areaSeriesRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (data && data.chart_data && areaSeriesRef.current) {
      const isUp = data.signal === 'UP';

      areaSeriesRef.current.applyOptions({
        lineColor: isUp ? '#10b981' : '#f43f5e',
        topColor: isUp ? 'rgba(16, 185, 129, 0.25)' : 'rgba(244, 63, 94, 0.25)',
      });

      areaSeriesRef.current.setData(data.chart_data);

      const targetDate = data.data_date_used;
      areaSeriesRef.current.setMarkers([
        {
          time: targetDate,
          position: isUp ? 'belowBar' : 'aboveBar',
          color: isUp ? '#10b981' : '#f43f5e',
          shape: isUp ? 'arrowUp' : 'arrowDown',
          text: `Signal: ${data.signal} (${data.confidence}%)`,
        },
      ]);

      // focus zoom window on 90 bars surrounding/ending at target date
      const totalBars = data.chart_data.length;
      const targetIndex = data.chart_data.findIndex(d => d.time === targetDate);

      if (targetIndex !== -1) {
        const fromIdx = Math.max(0, targetIndex - 60);
        const toIdx = Math.min(totalBars - 1, targetIndex + 30);
        chartRef.current?.timeScale().setVisibleLogicalRange({
          from: fromIdx,
          to: toIdx,
        });
      } else {
        chartRef.current?.timeScale().setVisibleLogicalRange({
          from: Math.max(0, totalBars - 90),
          to: totalBars - 1,
        });
      }
    }
  }, [data]);

  const handleTodayClick = () => {
    setDate(todayStr);
    fetchPrediction(todayStr);
  };

  return (
    <div className="min-h-screen bg-stone-950 text-stone-200 p-6 md:p-12">
      <div className="max-w-7xl mx-auto space-y-10">        
        <header className="border-b border-stone-800 pb-8 flex flex-col md:flex-row justify-between items-start md:items-end gap-6">
          <div className="space-y-2 max-w-xl">
            <div className="flex items-center gap-2 text-stone-400 text-xs tracking-widest uppercase font-semibold">
              <Activity className="w-4 h-4 text-emerald-500" />
              KNN Model Implementation
            </div>
            <h1 className="text-4xl md:text-5xl font-extrabold text-stone-100 tracking-tight font-serif-headline">
              Stock Signal Prediction Engine
            </h1>
            <p className="text-stone-400 text-sm leading-relaxed">
              Quantitative KNN prediction engine evaluating technical indicators on NSE: TATACONSUM.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <input
              type="date"
              min="2010-07-21"
              max={todayStr}
              value={date}
              onChange={(e) => setDate(e.target.value)}
              className="bg-stone-900 border border-stone-800 text-stone-200 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:border-stone-500 font-medium"
            />

            <button
              type="button"
              onClick={handleTodayClick}
              className="bg-stone-900 hover:bg-stone-800 text-stone-300 px-3.5 py-2.5 rounded-lg text-sm border border-stone-800 transition flex items-center gap-1.5"
            >
              <Clock className="w-4 h-4 text-stone-400" /> Today
            </button>

            <button
              onClick={() => fetchPrediction()}
              disabled={loading}
              className="bg-stone-100 hover:bg-white text-stone-950 font-semibold px-6 py-2.5 rounded-lg text-sm transition disabled:opacity-50"
            >
              {loading ? 'Evaluating...' : 'Run Prediction'}
            </button>
          </div>
        </header>

        {error && (
          <div className="bg-rose-950/40 border border-rose-800/50 text-rose-300 p-4 rounded-lg text-sm">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 items-start">
          <main className="lg:col-span-8 bg-stone-900/40 border border-stone-800/80 rounded-xl p-6 space-y-4">
            <div className="flex justify-between items-baseline border-b border-stone-800/60 pb-4">
              <h2 className="text-xl font-semibold text-stone-200 font-serif-headline italic">
                Market Trajectory
              </h2>
              {data && (
                <span className="text-xs text-stone-500 font-mono">
                  TARGET DATE: {data.data_date_used}
                </span>
              )}
            </div>
            <div ref={chartContainerRef} className="w-full min-h-[420px]" />
          </main>

          <aside className="lg:col-span-4 space-y-6">
            {data ? (
              <>
                <div className="bg-stone-900/60 border border-stone-800 rounded-xl p-6 space-y-6">
                  <div className="flex justify-between items-center">
                    <span className="text-xs uppercase tracking-widest text-stone-500 font-semibold">Signal Verdict</span>
                    <span className="text-xs text-stone-500 font-mono">{data.symbol}</span>
                  </div>

                  <div className="flex items-baseline justify-between">
                    <div className="flex items-center gap-3">
                      {data.signal === 'UP' ? (
                        <ArrowUpRight className="w-10 h-10 text-emerald-500 stroke-[2.5]" />
                      ) : (
                        <ArrowDownRight className="w-10 h-10 text-rose-500 stroke-[2.5]" />
                      )}
                      <span className={`text-4xl font-extrabold tracking-tight font-serif-headline ${
                        data.signal === 'UP' ? 'text-emerald-400' : 'text-rose-400'
                      }`}>
                        {data.signal}
                      </span>
                    </div>

                    <div className="text-right">
                      <p className="text-xs text-stone-500">Close Price</p>
                      <p className="text-2xl font-bold text-stone-100 font-mono mt-0.5">₹{data.current_price}</p>
                    </div>
                  </div>
                </div>

                <div className="bg-stone-900/60 border border-stone-800 rounded-xl p-6 space-y-4">
                  <div className="flex justify-between items-baseline">
                    <span className="text-xs uppercase tracking-widest text-stone-500 font-semibold">Model Confidence</span>
                    <span className="text-2xl font-bold text-stone-100 font-mono">{data.confidence}%</span>
                  </div>

                  <div className="w-full bg-stone-800 rounded-full h-2 overflow-hidden">
                    <div
                      className={`h-full transition-all duration-700 ${
                        data.signal === 'UP' ? 'bg-emerald-500' : 'bg-rose-500'
                      }`}
                      style={{ width: `${data.confidence}%` }}
                    />
                  </div>
                  <p className="text-xs text-stone-500 text-right">KNN Probability Density</p>
                </div>

                <div className="bg-stone-900/60 border border-stone-800 rounded-xl p-6 space-y-4">
                  <span className="text-xs uppercase tracking-widest text-stone-500 font-semibold block">
                    Session Metrics
                  </span>

                  <div className="space-y-2.5 text-sm font-mono">
                    <div className="flex justify-between py-1 border-b border-stone-800/50">
                      <span className="text-stone-400 font-sans">Open</span>
                      <span className="text-stone-200">₹{data.original_features.Open ?? 'N/A'}</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-stone-800/50">
                      <span className="text-stone-400 font-sans">High</span>
                      <span className="text-stone-200">₹{data.original_features.High ?? 'N/A'}</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-stone-800/50">
                      <span className="text-stone-400 font-sans">Low</span>
                      <span className="text-stone-200">₹{data.original_features.Low ?? 'N/A'}</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-stone-800/50">
                      <span className="text-stone-400 font-sans">Close</span>
                      <span className="text-stone-200">₹{data.original_features.Close ?? 'N/A'}</span>
                    </div>
                    <div className="flex justify-between py-1">
                      <span className="text-stone-400 font-sans">Last</span>
                      <span className="text-stone-200">₹{data.original_features.Last ?? 'N/A'}</span>
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <div className="bg-stone-900/30 border border-dashed border-stone-800 rounded-xl p-8 text-center text-stone-500 space-y-2">
                <p className="font-serif-headline italic text-stone-400 text-lg">Awaiting Evaluation</p>
                <p className="text-xs max-w-xs mx-auto">
                  Select a date and click "Run Prediction" to compute KNN inference signals.
                </p>
              </div>
            )}
          </aside>
        </div>
      </div>
    </div>
  );
}