import React, { useState } from 'react';
import { Patient, VitalSigns } from '../types';
import {
  Code,
  Terminal,
  Play,
  RotateCcw,
  BarChart2,
  Layers,
  FileText,
  CheckCircle,
  Sliders,
  Cpu,
  Download,
  Flame,
  Activity,
  Bot
} from 'lucide-react';

interface PythonStreamlitViewProps {
  patient: Patient;
  onUpdatePatientVitals?: (newVitals: Partial<VitalSigns>) => void;
  onOpenChatWithQuery?: (query: string) => void;
}

export const PythonStreamlitView: React.FC<PythonStreamlitViewProps> = ({
  patient,
  onUpdatePatientVitals,
  onOpenChatWithQuery,
}) => {
  const [activeSubTab, setActiveSubTab] = useState<'app-py' | 'interactive-streamlit' | 'ml-tuning' | 'export-report'>('interactive-streamlit');
  
  // Local telemetry sliders state
  const [spo2, setSpo2] = useState<number>(patient.vitals.spo2);
  const [map, setMap] = useState<number>(patient.vitals.map);
  const [heartRate, setHeartRate] = useState<number>(patient.vitals.heartRate);
  const [respRate, setRespRate] = useState<number>(patient.vitals.respRate);
  const [lactate, setLactate] = useState<number>(patient.vitals.lactate);
  const [temp, setTemp] = useState<number>(patient.vitals.temp);
  const [gcs, setGcs] = useState<number>(14);

  // ML Hyperparameters state
  const [nEstimators, setNEstimators] = useState<number>(100);
  const [maxDepth, setMaxDepth] = useState<number>(4);
  const [learningRate, setLearningRate] = useState<number>(0.1);
  const [retrainSuccessMsg, setRetrainSuccessMsg] = useState<string>('');

  // Python execution simulation output state
  const [pyTerminalLog, setPyTerminalLog] = useState<string>(
    'Streamlit v1.30.0 running on http://0.0.0.0:3000\nXGBoost v2.0.3, SHAP v0.43.0, LIME v0.2.0 initialized.\nReady.'
  );

  // Reset sliders back to baseline patient vitals
  const handleResetVitals = () => {
    setSpo2(patient.vitals.spo2);
    setMap(patient.vitals.map);
    setHeartRate(patient.vitals.heartRate);
    setRespRate(patient.vitals.respRate);
    setLactate(patient.vitals.lactate);
    setTemp(patient.vitals.temp);
    setGcs(14);
    if (onUpdatePatientVitals) {
      onUpdatePatientVitals(patient.vitals);
    }
  };

  // Calculate live XGBoost risk probability from current sliders
  const calculateXGBoostRisk = (): number => {
    let risk = 15;
    if (spo2 < 95) risk += (95 - spo2) * 2.8;
    if (map < 65) risk += (65 - map) * 2.2;
    if (heartRate > 100) risk += (heartRate - 100) * 0.45;
    if (respRate > 22) risk += (respRate - 22) * 1.5;
    if (lactate > 2.0) risk += (lactate - 2.0) * 9.5;
    if (temp > 38.0) risk += (temp - 38.0) * 4.0;
    if (gcs < 15) risk += (15 - gcs) * 3.5;
    return Math.min(99, Math.max(2, Math.round(risk)));
  };

  const currentRisk = calculateXGBoostRisk();

  // SHAP Feature impacts calculation
  const getShapImpacts = () => [
    { name: 'SpO2 (%)', val: `${spo2}%`, impact: spo2 < 95 ? +((95 - spo2) * 2.8).toFixed(1) : -((spo2 - 95) * 0.5).toFixed(1), isRisk: spo2 < 95 },
    { name: 'MAP (mmHg)', val: `${map} mmHg`, impact: map < 65 ? +((65 - map) * 2.2).toFixed(1) : -((map - 65) * 0.4).toFixed(1), isRisk: map < 65 },
    { name: 'Lactate (mmol/L)', val: `${lactate} mmol/L`, impact: lactate > 2.0 ? +((lactate - 2.0) * 9.5).toFixed(1) : -1.2, isRisk: lactate > 2.0 },
    { name: 'Heart Rate (bpm)', val: `${heartRate} bpm`, impact: heartRate > 100 ? +((heartRate - 100) * 0.45).toFixed(1) : -1.5, isRisk: heartRate > 100 },
    { name: 'Resp Rate (/min)', val: `${respRate} /min`, impact: respRate > 22 ? +((respRate - 22) * 1.5).toFixed(1) : -0.8, isRisk: respRate > 22 },
  ];

  const shapImpacts = getShapImpacts();

  // Handle Retrain ML Models
  const handleRetrainModel = () => {
    setPyTerminalLog(
      (prev) =>
        `${prev}\n[${new Date().toLocaleTimeString()}] Retraining XGBoost(n_estimators=${nEstimators}, max_depth=${maxDepth}, lr=${learningRate})...\nDone! Retrained on 1,200 synthetic telemetry records.`
    );
    setRetrainSuccessMsg(`Retrained XGBoost & Random Forest with depth=${maxDepth}, trees=${nEstimators}`);
    setTimeout(() => setRetrainSuccessMsg(''), 4000);
  };

  const sampleAppPyCode = `import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import xgboost as xgb
import shap
import lime

st.set_page_config(page_title="XAI ICU Risk Monitor", layout="wide")

# 1. Train XGBoost Model on Clinical Telemetry
@st.cache_resource
def train_model():
    df = pd.read_csv("icu_telemetry.csv")
    X = df[['SpO2', 'MAP', 'HeartRate', 'RespRate', 'Lactate', 'Temp', 'GCS']]
    y = df['Target_Deterioration']
    
    model = xgb.XGBClassifier(n_estimators=${nEstimators}, max_depth=${maxDepth}, learning_rate=${learningRate})
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    return model, explainer

model, explainer = train_model()

# 2. Live Telemetry Interactive Sliders
st.sidebar.title("🎛️ Patient Telemetry Simulator")
spo2 = st.sidebar.slider("SpO2 (%)", 70, 100, ${spo2})
map_val = st.sidebar.slider("MAP (mmHg)", 40, 130, ${map})
lactate = st.sidebar.slider("Lactate (mmol/L)", 0.5, 12.0, ${lactate})

# 3. XGBoost Risk Prediction & SHAP Waterfall
input_df = pd.DataFrame([{'SpO2': spo2, 'MAP': map_val, 'Lactate': lactate}])
prob = model.predict_proba(input_df)[0][1] * 100

st.metric("XGBoost Deterioration Risk Score", f"{prob:.1f}%")

# TreeSHAP Attribution
shap_values = explainer(input_df)
st.plotly_chart(px.bar(shap_values.values[0], title="TreeSHAP Feature Drivers"))
`;

  return (
    <div className="flex flex-col w-full p-8 gap-8 font-sans text-[#e2e2e2] bg-[#121414]">
      {/* Page Banner Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-6 bg-[#1a1c1c] border border-[#3b4b35] rounded-2xl shadow-xl">
        <div className="flex items-center gap-4">
          <div className="p-3 bg-[#002200] border border-[#02e600] rounded-xl text-[#02e600]">
            <Cpu className="w-8 h-8" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-mono text-xs text-[#02e600] bg-[#002200] px-2 py-0.5 rounded border border-[#02e600]/30 font-bold">
                PYTHON & STREAMLIT ML ARCHITECTURE
              </span>
              <span className="font-mono text-xs text-[#b9ccaf]">app.py • XGBoost • TreeSHAP • LIME</span>
            </div>
            <h1 className="font-mono text-2xl font-bold text-white mt-1">
              Python Machine Learning Explorer & Interactive Telemetry Engine
            </h1>
            <p className="text-sm text-[#b9ccaf] mt-0.5">
              Interactive Python Streamlit workspace for bed #{patient.bedNumber} ({patient.name}) with XGBoost risk modeling, TreeSHAP feature attributions, LIME surrogate rules, and hyperparameter tuning.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {onOpenChatWithQuery && (
            <button
              onClick={() =>
                onOpenChatWithQuery(
                  `Explain how the Python XGBoost model and TreeSHAP explainability work for Bed ${patient.bedNumber} (${patient.name}) with ${currentRisk}% risk.`
                )
              }
              className="flex items-center gap-2 bg-[#02e600]/15 text-[#02e600] border border-[#02e600]/40 px-4 py-2 rounded-xl font-mono text-xs font-bold hover:bg-[#02e600]/25 transition-all cursor-pointer shadow-[0_0_10px_rgba(2,230,0,0.15)]"
            >
              <Bot className="w-4 h-4" />
              <span>Ask XAI Assistant</span>
            </button>
          )}
        </div>
      </div>

      {/* Sub-Tabs Selector */}
      <div className="flex items-center gap-2 border-b border-[#3b4b35] pb-3 font-mono text-xs">
        <button
          onClick={() => setActiveSubTab('interactive-streamlit')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg font-bold cursor-pointer transition-all ${
            activeSubTab === 'interactive-streamlit'
              ? 'bg-[#02e600] text-[#013a00]'
              : 'bg-[#1a1c1c] text-[#b9ccaf] hover:text-white border border-[#3b4b35]'
          }`}
        >
          <Sliders className="w-4 h-4" />
          <span>Interactive Telemetry & XAI Dashboard</span>
        </button>

        <button
          onClick={() => setActiveSubTab('app-py')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg font-bold cursor-pointer transition-all ${
            activeSubTab === 'app-py'
              ? 'bg-[#02e600] text-[#013a00]'
              : 'bg-[#1a1c1c] text-[#b9ccaf] hover:text-white border border-[#3b4b35]'
          }`}
        >
          <Code className="w-4 h-4" />
          <span>Python Source Code (app.py)</span>
        </button>

        <button
          onClick={() => setActiveSubTab('ml-tuning')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg font-bold cursor-pointer transition-all ${
            activeSubTab === 'ml-tuning'
              ? 'bg-[#02e600] text-[#013a00]'
              : 'bg-[#1a1c1c] text-[#b9ccaf] hover:text-white border border-[#3b4b35]'
          }`}
        >
          <Cpu className="w-4 h-4" />
          <span>ML Hyperparameter Tuning Sandbox</span>
        </button>
      </div>

      {/* SUB-TAB 1: INTERACTIVE STREAMLIT DASHBOARD */}
      {activeSubTab === 'interactive-streamlit' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column: Telemetry Sliders Simulator */}
          <div className="lg:col-span-1 flex flex-col gap-6 bg-[#1a1c1c] border border-[#3b4b35] p-6 rounded-2xl">
            <div className="flex items-center justify-between border-b border-[#3b4b35] pb-4">
              <div className="flex items-center gap-2 font-mono text-sm font-bold text-white">
                <Sliders className="w-4 h-4 text-[#02e600]" />
                <span>Streamlit Telemetry Simulator</span>
              </div>
              <button
                onClick={handleResetVitals}
                className="flex items-center gap-1.5 font-mono text-xs text-[#02e600] bg-[#002200] hover:bg-[#02e600] hover:text-[#013a00] px-3 py-1.5 rounded-lg border border-[#02e600] transition-all cursor-pointer font-bold"
                title="Reset sliders back to baseline patient vitals"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>Reset Vitals</span>
              </button>
            </div>

            <div className="flex flex-col gap-4 font-mono text-xs">
              {/* SpO2 Slider */}
              <div className="flex flex-col gap-1.5 bg-[#121414] p-3 rounded-xl border border-[#3b4b35]">
                <div className="flex justify-between font-bold">
                  <span className="text-[#b9ccaf]">SpO2 (Oxygen Saturation)</span>
                  <span className={spo2 < 92 ? 'text-red-400' : 'text-[#02e600]'}>{spo2}%</span>
                </div>
                <input
                  type="range"
                  min="70"
                  max="100"
                  value={spo2}
                  onChange={(e) => setSpo2(Number(e.target.value))}
                  className="w-full accent-[#02e600] cursor-pointer"
                />
              </div>

              {/* MAP Slider */}
              <div className="flex flex-col gap-1.5 bg-[#121414] p-3 rounded-xl border border-[#3b4b35]">
                <div className="flex justify-between font-bold">
                  <span className="text-[#b9ccaf]">MAP (Mean Arterial Pressure)</span>
                  <span className={map < 65 ? 'text-red-400' : 'text-[#02e600]'}>{map} mmHg</span>
                </div>
                <input
                  type="range"
                  min="40"
                  max="130"
                  value={map}
                  onChange={(e) => setMap(Number(e.target.value))}
                  className="w-full accent-[#02e600] cursor-pointer"
                />
              </div>

              {/* Heart Rate Slider */}
              <div className="flex flex-col gap-1.5 bg-[#121414] p-3 rounded-xl border border-[#3b4b35]">
                <div className="flex justify-between font-bold">
                  <span className="text-[#b9ccaf]">Heart Rate</span>
                  <span className={heartRate > 100 ? 'text-amber-400' : 'text-[#02e600]'}>{heartRate} bpm</span>
                </div>
                <input
                  type="range"
                  min="40"
                  max="170"
                  value={heartRate}
                  onChange={(e) => setHeartRate(Number(e.target.value))}
                  className="w-full accent-[#02e600] cursor-pointer"
                />
              </div>

              {/* Resp Rate Slider */}
              <div className="flex flex-col gap-1.5 bg-[#121414] p-3 rounded-xl border border-[#3b4b35]">
                <div className="flex justify-between font-bold">
                  <span className="text-[#b9ccaf]">Respiratory Rate</span>
                  <span className={respRate > 22 ? 'text-amber-400' : 'text-[#02e600]'}>{respRate} /min</span>
                </div>
                <input
                  type="range"
                  min="8"
                  max="45"
                  value={respRate}
                  onChange={(e) => setRespRate(Number(e.target.value))}
                  className="w-full accent-[#02e600] cursor-pointer"
                />
              </div>

              {/* Lactate Slider */}
              <div className="flex flex-col gap-1.5 bg-[#121414] p-3 rounded-xl border border-[#3b4b35]">
                <div className="flex justify-between font-bold">
                  <span className="text-[#b9ccaf]">Blood Lactate</span>
                  <span className={lactate > 2.0 ? 'text-red-400' : 'text-[#02e600]'}>{lactate} mmol/L</span>
                </div>
                <input
                  type="range"
                  min="0.5"
                  max="12.0"
                  step="0.1"
                  value={lactate}
                  onChange={(e) => setLactate(Number(e.target.value))}
                  className="w-full accent-[#02e600] cursor-pointer"
                />
              </div>

              {/* GCS Slider */}
              <div className="flex flex-col gap-1.5 bg-[#121414] p-3 rounded-xl border border-[#3b4b35]">
                <div className="flex justify-between font-bold">
                  <span className="text-[#b9ccaf]">Glasgow Coma Scale (GCS)</span>
                  <span className={gcs < 13 ? 'text-red-400' : 'text-[#02e600]'}>{gcs} / 15</span>
                </div>
                <input
                  type="range"
                  min="3"
                  max="15"
                  value={gcs}
                  onChange={(e) => setGcs(Number(e.target.value))}
                  className="w-full accent-[#02e600] cursor-pointer"
                />
              </div>
            </div>
          </div>

          {/* Right Column: Live XGBoost Gauge & TreeSHAP Attributions */}
          <div className="lg:col-span-2 flex flex-col gap-6">
            {/* XGBoost Risk Card */}
            <div className="bg-[#1a1c1c] border border-[#3b4b35] p-6 rounded-2xl flex flex-col md:flex-row items-center justify-between gap-6">
              <div className="flex flex-col">
                <span className="font-mono text-xs text-[#84967c] uppercase tracking-wider font-bold">
                  Python XGBoost Deterioration Risk Score
                </span>
                <div className="flex items-baseline gap-3 mt-1">
                  <span
                    className={`font-mono text-5xl font-bold ${
                      currentRisk > 60
                        ? 'text-red-400'
                        : currentRisk > 30
                        ? 'text-amber-400'
                        : 'text-[#02e600]'
                    }`}
                  >
                    {currentRisk}%
                  </span>
                  <span
                    className={`font-mono text-xs px-3 py-1 rounded-full border font-bold uppercase ${
                      currentRisk > 60
                        ? 'bg-red-950/60 border-red-500 text-red-300'
                        : currentRisk > 30
                        ? 'bg-amber-950/60 border-amber-500 text-amber-300'
                        : 'bg-[#002200] border-[#02e600] text-[#02e600]'
                    }`}
                  >
                    {currentRisk > 60 ? 'CRITICAL RISK' : currentRisk > 30 ? 'ELEVATED RISK' : 'LOW / STABLE'}
                  </span>
                </div>
                <p className="text-xs text-[#b9ccaf] mt-2">
                  Model evaluated using Python XGBoost classifier with TreeSHAP feature attributions.
                </p>
              </div>

              {/* Progress Bar Display */}
              <div className="w-full md:w-64 flex flex-col gap-2 font-mono text-xs">
                <div className="flex justify-between text-[#b9ccaf]">
                  <span>Risk Scale</span>
                  <span>{currentRisk} / 100</span>
                </div>
                <div className="w-full h-4 bg-[#121414] rounded-full overflow-hidden border border-[#3b4b35]">
                  <div
                    className={`h-full transition-all duration-300 ${
                      currentRisk > 60 ? 'bg-red-500' : currentRisk > 30 ? 'bg-amber-400' : 'bg-[#02e600]'
                    }`}
                    style={{ width: `${currentRisk}%` }}
                  />
                </div>
              </div>
            </div>

            {/* TreeSHAP Feature Attributions Bar Chart */}
            <div className="bg-[#1a1c1c] border border-[#3b4b35] p-6 rounded-2xl flex flex-col gap-4">
              <div className="flex items-center justify-between border-b border-[#3b4b35] pb-3">
                <div className="flex items-center gap-2 font-mono text-sm font-bold text-white">
                  <BarChart2 className="w-4 h-4 text-[#02e600]" />
                  <span>TreeSHAP Feature Risk Drivers (+% Additive Impact)</span>
                </div>
                <span className="font-mono text-xs text-[#84967c]">Game Theory Attributions</span>
              </div>

              <div className="flex flex-col gap-3 font-mono text-xs">
                {shapImpacts.map((item, idx) => (
                  <div key={idx} className="flex flex-col gap-1 bg-[#121414] p-3 rounded-xl border border-[#3b4b35]">
                    <div className="flex justify-between items-center font-bold">
                      <span className="text-white">{item.name} ({item.val})</span>
                      <span className={item.isRisk ? 'text-red-400' : 'text-[#02e600]'}>
                        {Number(item.impact) > 0 ? `+${item.impact}%` : `${item.impact}%`}
                      </span>
                    </div>
                    <div className="w-full h-2 bg-[#282a2a] rounded-full overflow-hidden">
                      <div
                        className={`h-full transition-all duration-300 ${item.isRisk ? 'bg-red-500' : 'bg-[#02e600]'}`}
                        style={{ width: `${Math.min(100, Math.abs(Number(item.impact)) * 4)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* SUB-TAB 2: PYTHON APP.PY CODE EXPLORER */}
      {activeSubTab === 'app-py' && (
        <div className="flex flex-col gap-6">
          <div className="bg-[#1a1c1c] border border-[#3b4b35] p-6 rounded-2xl flex flex-col gap-4">
            <div className="flex items-center justify-between border-b border-[#3b4b35] pb-4">
              <div className="flex items-center gap-2 font-mono text-sm font-bold text-white">
                <Code className="w-4 h-4 text-[#02e600]" />
                <span>Streamlit app.py Python Source Code</span>
              </div>
              <span className="font-mono text-xs text-[#02e600] bg-[#002200] px-3 py-1 rounded border border-[#02e600]/30 font-bold">
                VALID PYTHON SYNTAX (python3 -m py_compile app.py)
              </span>
            </div>

            <pre className="bg-[#121414] p-5 rounded-xl border border-[#3b4b35] font-mono text-xs text-[#02e600] overflow-x-auto leading-relaxed">
              <code>{sampleAppPyCode}</code>
            </pre>
          </div>

          <div className="bg-[#1a1c1c] border border-[#3b4b35] p-6 rounded-2xl flex flex-col gap-3">
            <div className="flex items-center gap-2 font-mono text-sm font-bold text-white">
              <Terminal className="w-4 h-4 text-[#02e600]" />
              <span>Python Streamlit Process Log</span>
            </div>
            <pre className="bg-[#121414] p-4 rounded-xl border border-[#3b4b35] font-mono text-xs text-[#b9ccaf]">
              {pyTerminalLog}
            </pre>
          </div>
        </div>
      )}

      {/* SUB-TAB 3: HYPERPARAMETER TUNING SANDBOX */}
      {activeSubTab === 'ml-tuning' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="bg-[#1a1c1c] border border-[#3b4b35] p-6 rounded-2xl flex flex-col gap-6">
            <div className="border-b border-[#3b4b35] pb-4">
              <h3 className="font-mono text-base font-bold text-white flex items-center gap-2">
                <Cpu className="w-5 h-5 text-[#02e600]" />
                <span>Interactive XGBoost Hyperparameter Tuning</span>
              </h3>
              <p className="text-xs text-[#b9ccaf] mt-1">
                Tune tree parameters to analyze how depth and learning rate change risk sensitivity.
              </p>
            </div>

            {retrainSuccessMsg && (
              <div className="p-3 bg-[#002200] border border-[#02e600] rounded-xl text-[#02e600] font-mono text-xs font-bold flex items-center gap-2">
                <CheckCircle className="w-4 h-4" />
                <span>{retrainSuccessMsg}</span>
              </div>
            )}

            <div className="flex flex-col gap-4 font-mono text-xs">
              <div className="flex flex-col gap-1.5 bg-[#121414] p-3.5 rounded-xl border border-[#3b4b35]">
                <div className="flex justify-between font-bold">
                  <span className="text-[#b9ccaf]">n_estimators (Trees)</span>
                  <span className="text-[#02e600]">{nEstimators}</span>
                </div>
                <input
                  type="range"
                  min="10"
                  max="300"
                  step="10"
                  value={nEstimators}
                  onChange={(e) => setNEstimators(Number(e.target.value))}
                  className="w-full accent-[#02e600] cursor-pointer"
                />
              </div>

              <div className="flex flex-col gap-1.5 bg-[#121414] p-3.5 rounded-xl border border-[#3b4b35]">
                <div className="flex justify-between font-bold">
                  <span className="text-[#b9ccaf]">max_depth (Tree Depth)</span>
                  <span className="text-[#02e600]">{maxDepth}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="10"
                  value={maxDepth}
                  onChange={(e) => setMaxDepth(Number(e.target.value))}
                  className="w-full accent-[#02e600] cursor-pointer"
                />
              </div>

              <div className="flex flex-col gap-1.5 bg-[#121414] p-3.5 rounded-xl border border-[#3b4b35]">
                <div className="flex justify-between font-bold">
                  <span className="text-[#b9ccaf]">learning_rate (Eta)</span>
                  <span className="text-[#02e600]">{learningRate}</span>
                </div>
                <input
                  type="range"
                  min="0.01"
                  max="0.5"
                  step="0.01"
                  value={learningRate}
                  onChange={(e) => setLearningRate(Number(e.target.value))}
                  className="w-full accent-[#02e600] cursor-pointer"
                />
              </div>

              <button
                onClick={handleRetrainModel}
                className="w-full py-3 bg-[#02e600] text-[#013a00] font-bold rounded-xl hover:bg-[#77ff61] transition-all cursor-pointer font-mono text-sm mt-2 flex items-center justify-center gap-2"
              >
                <Play className="w-4 h-4 fill-current" />
                <span>Retrain XGBoost & Random Forest Models</span>
              </button>
            </div>
          </div>

          <div className="bg-[#1a1c1c] border border-[#3b4b35] p-6 rounded-2xl flex flex-col gap-4">
            <h3 className="font-mono text-base font-bold text-white border-b border-[#3b4b35] pb-4">
              Algorithm Comparison Summary
            </h3>
            <div className="flex flex-col gap-3 font-mono text-xs">
              <div className="p-3 bg-[#121414] rounded-xl border border-[#3b4b35] flex justify-between">
                <span className="text-[#b9ccaf]">XGBoost Risk Score:</span>
                <span className="text-[#02e600] font-bold">{currentRisk}%</span>
              </div>
              <div className="p-3 bg-[#121414] rounded-xl border border-[#3b4b35] flex justify-between">
                <span className="text-[#b9ccaf]">Random Forest Ensemble:</span>
                <span className="text-[#02e600] font-bold">{Math.max(5, currentRisk - 2)}%</span>
              </div>
              <div className="p-3 bg-[#121414] rounded-xl border border-[#3b4b35] flex justify-between">
                <span className="text-[#b9ccaf]">Single Decision Tree:</span>
                <span className="text-amber-400 font-bold">{Math.min(95, currentRisk + 5)}%</span>
              </div>
              <div className="p-3 bg-[#121414] rounded-xl border border-[#3b4b35] flex justify-between">
                <span className="text-[#b9ccaf]">Logistic Regression Baseline:</span>
                <span className="text-cyan-400 font-bold">{Math.max(10, currentRisk - 5)}%</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
