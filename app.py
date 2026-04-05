"""
FlexFlowSim — Interactive Dashboard v3
========================================
Features: Configure, Train, Evaluate, Simulate, Sensitivity, Compare
With: flow diagram, auto-calibration, JSON import/export, Excel export,
      scenario queue, Gantt chart, sensitivity sweep, config comparison,
      dark/light theme.

Launch:  streamlit run app.py
"""

import hashlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from env import FlexFlowSimEnv, load_config
from baselines import BASELINE_POLICIES, run_episode

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(page_title="FlexFlowSim", page_icon="🏭", layout="wide", initial_sidebar_state="expanded")

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════

st.sidebar.title("🏭 FlexFlowSim")
st.sidebar.caption("DES + RL Benchmark for Manufacturing Routing")

page = st.sidebar.radio("Navigation", [
    "⚙️ Configure", "🎓 Train", "📊 Evaluate",
    "🔬 Simulate", "📈 Sensitivity", "🔄 Compare",
], index=0)

is_dark = st.sidebar.toggle("🌙 Dark Mode", value=True, key="dark_toggle")

# Inject theme CSS based on toggle
if is_dark:
    st.markdown("""<style>
        .stApp { background-color: #0a0f1a; }
        [data-testid="stSidebar"] { background-color: #111827; }
        .stMarkdown, .stText, h1, h2, h3, p, span, label, .stMetricValue, .stMetricLabel { color: #e2e8f0 !important; }
        [data-testid="stMetricValue"] { color: #e2e8f0 !important; }
        [data-testid="stExpander"] { background-color: #1e293b; border-color: #1e293b; }
        .stDataFrame { color: #e2e8f0; }
        input, select, textarea { background-color: #1e293b !important; color: #e2e8f0 !important; border-color: #374151 !important; }
        .stSelectbox > div > div { background-color: #1e293b; color: #e2e8f0; }
        [data-testid="stNumberInput"] input { background-color: #1e293b !important; color: #e2e8f0 !important; }
        .stProgress > div > div { background-color: #1e293b; }
        .stAlert { background-color: #1e293b; }
        hr { border-color: #1e293b; }
        .stTabs [data-baseweb="tab-list"] { background-color: #111827; }
        .stCaption, .stCaptionContainer { color: #94a3b8 !important; }
    </style>""", unsafe_allow_html=True)
else:
    st.markdown("""<style>
        .stApp { background-color: #f8fafc; }
        [data-testid="stSidebar"] { background-color: #ffffff; }
        .stMarkdown, .stText, h1, h2, h3, p, span, label { color: #1e293b !important; }
        [data-testid="stMetricValue"] { color: #1e293b !important; }
        [data-testid="stExpander"] { background-color: #f1f5f9; border-color: #e2e8f0; }
        input, select, textarea { background-color: #ffffff !important; color: #1e293b !important; border-color: #cbd5e1 !important; }
        .stSelectbox > div > div { background-color: #ffffff; color: #1e293b; }
        [data-testid="stNumberInput"] input { background-color: #ffffff !important; color: #1e293b !important; }
        .stProgress > div > div { background-color: #e2e8f0; }
        hr { border-color: #e2e8f0; }
        .stCaption, .stCaptionContainer { color: #64748b !important; }
    </style>""", unsafe_allow_html=True)

# Set Plotly chart theme to match
import plotly.io as pio
if is_dark:
    pio.templates.default = "plotly_dark"
    CHART_BG = "#111827"
    CHART_PAPER = "#0a0f1a"
    CHART_FONT = "#e2e8f0"
    CHART_GRID = "#1e293b"
else:
    pio.templates.default = "plotly_white"
    CHART_BG = "#ffffff"
    CHART_PAPER = "#ffffff"
    CHART_FONT = "#000000"
    CHART_GRID = "#d0d0d0"

def apply_theme(fig):
    """Force chart colors to match current theme."""
    fig.update_layout(
        paper_bgcolor=CHART_PAPER,
        plot_bgcolor=CHART_BG,
        font=dict(color=CHART_FONT, size=12),
        title_font=dict(color=CHART_FONT, size=14),
    )
    fig.update_xaxes(
        gridcolor=CHART_GRID, zerolinecolor=CHART_GRID,
        tickfont=dict(color=CHART_FONT, size=11),
        title_font=dict(color=CHART_FONT, size=12),
    )
    fig.update_yaxes(
        gridcolor=CHART_GRID, zerolinecolor=CHART_GRID,
        tickfont=dict(color=CHART_FONT, size=11),
        title_font=dict(color=CHART_FONT, size=12),
    )
    # Force subplot titles (annotations) to match
    for ann in fig.layout.annotations:
        ann.font.color = CHART_FONT
        ann.font.size = 13
    return fig

CONFIG_DIR = Path("configs")
config_files = sorted(CONFIG_DIR.glob("*.json")) if CONFIG_DIR.exists() else []
config_names = [f.stem for f in config_files]

selected_config_path = None
selected_config_name = None
if config_names:
    options = ["— Select a config —"] + config_names
    prev = st.session_state.get("_selected_cfg", "— Select a config —")
    default_idx = options.index(prev) if prev in options else 0
    choice = st.sidebar.selectbox("Active Config", options, index=default_idx)
    st.session_state["_selected_cfg"] = choice
    if choice != "— Select a config —":
        selected_config_name = choice
        selected_config_path = CONFIG_DIR / f"{selected_config_name}.json"
    if st.sidebar.button("🔄 Reload Config"):
        for key in list(st.session_state.keys()):
            if key not in ["_selected_cfg"]:
                del st.session_state[key]
        st.rerun()
else:
    st.sidebar.warning("No configs in configs/")

def load_active_config():
    if selected_config_path and selected_config_path.exists():
        with open(selected_config_path) as f:
            return json.load(f)
    return None

# Detect config change and FLUSH all stale state
if selected_config_name and st.session_state.get("_loaded_cfg_name") != selected_config_name:
    keys_to_clear = [k for k in st.session_state.keys()
                     if k not in ("_selected_cfg", "_loaded_cfg_name")]
    for k in keys_to_clear:
        del st.session_state[k]
    st.session_state["_loaded_cfg_name"] = selected_config_name

cfg_preview = load_active_config()
if cfg_preview:
    sps = [len(s.get("servers", [])) for s in cfg_preview.get("stages", [])]
    na = 1
    for m in sps: na *= m
    st.sidebar.markdown("---")
    st.sidebar.caption(f"**Topology:** {len(sps)} stages · {sum(sps)} servers · {na} actions")

DISTRIBUTIONS = ["normal", "exponential", "lognormal", "uniform"]
SCENARIOS = {
    "CostFocus": {"weights": (0.8, 0.1, 0.1), "label": "Cost-Dominant"},
    "ThroughputFocus": {"weights": (0.1, 0.8, 0.1), "label": "Throughput-Dominant"},
    "LeadTimeFocus": {"weights": (0.1, 0.1, 0.8), "label": "Lead Time-Dominant"},
    "Balanced": {"weights": (0.33, 0.33, 0.34), "label": "Balanced"},
}

# Custom weight scenarios stored in session state
if "custom_scenarios" not in st.session_state:
    st.session_state["custom_scenarios"] = {}

def get_all_scenarios():
    """Return merged preset + custom scenarios."""
    merged = dict(SCENARIOS)
    for name, data in st.session_state["custom_scenarios"].items():
        merged[name] = data
    return merged

def render_recommendation(all_results, scenario_name=""):
    """Render a recommendation panel showing best method per objective."""
    if not all_results:
        return
    
    objectives = [
        ("totalCost", "Lowest Cost", "min", "💰"),
        ("totalDeparted", "Highest Throughput", "max", "📦"),
        ("avgLeadTime", "Lowest Lead Time", "min", "⏱️"),
    ]
    
    st.markdown("---")
    st.subheader(f"📋 Recommendation{f' — {scenario_name}' if scenario_name else ''}")
    
    cols = st.columns(3)
    pareto_candidates = set()
    
    for ci, (metric, label, direction, icon) in enumerate(objectives):
        scores = {}
        for name, df in all_results.items():
            if metric in df.columns:
                scores[name] = df[metric].mean()
        
        if not scores:
            continue
            
        if direction == "min":
            best_name = min(scores, key=scores.get)
        else:
            best_name = max(scores, key=scores.get)
        
        best_val = scores[best_name]
        best_std = all_results[best_name][metric].std()
        
        # Find runner-up
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=(direction == "max"))
        runner = sorted_items[1] if len(sorted_items) > 1 else None
        
        with cols[ci]:
            st.markdown(f"**{icon} {label}**")
            is_rl = any(tag in best_name for tag in ["DQN", "PPO"])
            badge = "🤖 RL" if is_rl else "📐 Rule"
            st.success(f"**{best_name}**\n\n{best_val:.1f} ± {best_std:.1f}  ({badge})")
            if runner:
                st.caption(f"Runner-up: {runner[0]} ({runner[1]:.1f})")
        
        pareto_candidates.add(best_name)
    
    # Pareto-efficient methods (non-dominated)
    methods = list(all_results.keys())
    dominated = set()
    for a in methods:
        for b in methods:
            if a == b: continue
            a_cost = all_results[a]["totalCost"].mean()
            b_cost = all_results[b]["totalCost"].mean()
            a_dep = all_results[a]["totalDeparted"].mean()
            b_dep = all_results[b]["totalDeparted"].mean()
            a_lt = all_results[a]["avgLeadTime"].mean()
            b_lt = all_results[b]["avgLeadTime"].mean()
            # b dominates a if b is better or equal on all, strictly better on at least one
            if b_cost <= a_cost and b_dep >= a_dep and b_lt <= a_lt:
                if b_cost < a_cost or b_dep > a_dep or b_lt < a_lt:
                    dominated.add(a)
    
    pareto = [m for m in methods if m not in dominated]
    if pareto:
        st.markdown("**🏆 Pareto-Efficient Methods** (non-dominated across all 3 objectives):")
        pareto_info = []
        for m in pareto:
            is_rl = any(tag in m for tag in ["DQN", "PPO"])
            pareto_info.append(f"{'🤖' if is_rl else '📐'} **{m}** — "
                              f"Cost: {all_results[m]['totalCost'].mean():.0f}, "
                              f"Dep: {all_results[m]['totalDeparted'].mean():.1f}, "
                              f"LT: {all_results[m]['avgLeadTime'].mean():.1f}")
        for p in pareto_info:
            st.markdown(f"- {p}")

def render_weight_editor():
    """Render custom weight scenario editor in sidebar or inline."""
    st.markdown("---")
    st.subheader("🎛️ Custom Weight Scenarios")
    
    # Show existing custom scenarios
    if st.session_state["custom_scenarios"]:
        for name, data in list(st.session_state["custom_scenarios"].items()):
            w = data["weights"]
            st.caption(f"**{name}**: cost={w[0]}, tp={w[1]}, lt={w[2]}")
            if st.button(f"🗑️ Remove {name}", key=f"rm_custom_{name}"):
                del st.session_state["custom_scenarios"][name]
                st.rerun()
    
    # Add new
    with st.expander("➕ Add Custom Scenario", expanded=False):
        cn = st.text_input("Scenario name", placeholder="MyScenario", key="cust_name")
        cc1, cc2, cc3 = st.columns(3)
        with cc1: wc = st.slider("w_cost", 0.0, 1.0, 0.33, 0.01, key="cw_cost")
        with cc2: wt = st.slider("w_throughput", 0.0, 1.0, 0.33, 0.01, key="cw_tp")
        with cc3: wl = st.slider("w_leadtime", 0.0, 1.0, 0.34, 0.01, key="cw_lt")
        
        total_w = wc + wt + wl
        if abs(total_w - 1.0) > 0.01:
            st.warning(f"Weights sum to {total_w:.2f} — should be 1.0")
        
        if st.button("✅ Add Scenario", disabled=(not cn or abs(total_w - 1.0) > 0.05)):
            # Normalise to exactly 1.0
            s = wc + wt + wl
            st.session_state["custom_scenarios"][cn] = {
                "weights": (round(wc/s, 3), round(wt/s, 3), round(1.0 - round(wc/s, 3) - round(wt/s, 3), 3)),
                "label": f"Custom ({wc:.2f}/{wt:.2f}/{wl:.2f})"
            }
            st.success(f"Added: {cn}")
            st.rerun()

# ═══════════════════════════════════════════════════════════════════
# FLOW DIAGRAM
# ═══════════════════════════════════════════════════════════════════

def render_flow_diagram(stages_cfg):
    fig = go.Figure()
    colors = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"]
    n_st = len(stages_cfg)
    fig.add_shape(type="rect", x0=0.02, y0=0.4, x1=0.12, y1=0.6, fillcolor="#f59e0b", line=dict(color="#f59e0b"), opacity=0.8)
    fig.add_annotation(x=0.07, y=0.5, text="Arrivals", showarrow=False, font=dict(color="white", size=11, family="Arial Black"))
    sx_start = 0.18
    sw = (0.82 - sx_start) / n_st if n_st else 0.3
    for si, stage in enumerate(stages_cfg):
        srvs = stage.get("servers", [])
        sx = sx_start + si * sw
        col = colors[si % len(colors)]
        fig.add_annotation(x=sx + sw * 0.4, y=0.95, text=f"<b>{stage.get('name', f'Stage {si+1}')}</b>", showarrow=False, font=dict(color=col, size=12))
        for sj, srv in enumerate(srvs):
            cy = 0.5 if len(srvs) == 1 else 0.2 + 0.6 * sj / (len(srvs) - 1)
            bx0, bx1, by0, by1 = sx + 0.02, sx + sw * 0.75, cy - 0.08, cy + 0.08
            fig.add_shape(type="rect", x0=bx0, y0=by0, x1=bx1, y1=by1, fillcolor=col, line=dict(color=col, width=2), opacity=0.15)
            svc = srv.get("service_time", {})
            fig.add_annotation(x=(bx0+bx1)/2, y=cy, text=f"<b>{srv.get('name','S')}</b><br>{svc.get('distribution','?')}(μ={svc.get('mean','?')})", showarrow=False, font=dict(size=9), align="center")
            if si == 0:
                fig.add_annotation(x=bx0, y=cy, ax=0.12, ay=0.5, showarrow=True, arrowhead=2, arrowcolor="#f59e0b", arrowwidth=1.5)
            if si > 0:
                prev_x = sx_start + (si-1)*sw + sw*0.75 + 0.02
                fig.add_annotation(x=bx0, y=cy, ax=prev_x, ay=cy, showarrow=True, arrowhead=2, arrowcolor=colors[(si-1)%len(colors)], arrowwidth=1)
    last_x = sx_start + (n_st-1)*sw + sw*0.75 + 0.02
    fig.add_shape(type="rect", x0=0.88, y0=0.4, x1=0.98, y1=0.6, fillcolor="#10b981", line=dict(color="#10b981"), opacity=0.8)
    fig.add_annotation(x=0.93, y=0.5, text="Departures", showarrow=False, font=dict(color="white", size=11, family="Arial Black"))
    fig.add_annotation(x=0.88, y=0.5, ax=last_x, ay=0.5, showarrow=True, arrowhead=2, arrowcolor=colors[(n_st-1)%len(colors)], arrowwidth=1.5)
    fig.update_layout(xaxis=dict(visible=False, range=[0,1]), yaxis=dict(visible=False, range=[0,1]),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=10), height=250, showlegend=False)
    return fig

# ═══════════════════════════════════════════════════════════════════
# PAGE 1: CONFIGURE
# ═══════════════════════════════════════════════════════════════════

def page_configure():
    import hashlib
    st.title("⚙️ Environment Configuration")
    cfg = load_active_config()
    if not cfg: st.error("No config loaded."); return

    ic, ec, _ = st.columns([1, 1, 2])
    with ic:
        uploaded = st.file_uploader("📂 Import JSON", type=["json"], key="cfg_up")
        if uploaded:
            try:
                imp = json.load(uploaded)
                CONFIG_DIR.mkdir(exist_ok=True)
                with open(CONFIG_DIR / uploaded.name, "w") as f: json.dump(imp, f, indent=2)
                st.success(f"Imported {uploaded.name}"); st.rerun()
            except Exception as e: st.error(f"Invalid: {e}")
    with ec:
        st.download_button("📥 Export Config", json.dumps(cfg, indent=2), f"{selected_config_name}.json", "application/json")

    st.subheader("Flow Diagram")
    flow_fig = render_flow_diagram(cfg.get("stages", []))
    apply_theme(flow_fig)
    st.plotly_chart(flow_fig, width="stretch")

    col1, col2 = st.columns([1, 1])
    ck = selected_config_name or "none"  # config key prefix
    with col1:
        st.subheader("Arrival & Simulation")
        arr = cfg.get("arrival", {})
        arr_dist = st.selectbox("Arrival distribution", DISTRIBUTIONS, index=DISTRIBUTIONS.index(arr.get("distribution", "exponential")), key=f"ad_{ck}")
        arr_mean = st.number_input("Mean IAT (min)", value=float(arr.get("mean", 10)), min_value=0.1, step=0.5, key=f"am_{ck}")
        max_time = st.number_input("Shift (min)", value=float(cfg.get("max_time", 480)), min_value=10.0, step=10.0, key=f"mt_{ck}")
        dt = st.number_input("dt (min)", value=float(cfg.get("dt", 1.0)), min_value=0.01, step=0.1, key=f"dt_{ck}")
        max_queue = st.number_input("Max queue", value=float(cfg.get("max_queue", 50)), min_value=1.0, step=5.0, key=f"mq_{ck}")
        waiting_cost = st.number_input("Waiting cost", value=float(cfg.get("waiting_cost", 0.1)), min_value=0.0, step=0.01, format="%.3f", key=f"wc_{ck}")

        st.subheader("Normalisation Constants")
        # Read DIRECTLY from disk config every time — bypass all caching
        disk_cfg = load_active_config() or {}
        disk_norms = disk_cfg.get("norm_constants", [1,1,1])
        cal_ver = st.session_state.get("_cal_ver", 0)
        nc1 = st.number_input("Cost", value=float(disk_norms[0]), min_value=0.1, step=10.0, key=f"n1_{ck}_v{cal_ver}")
        nc2 = st.number_input("Throughput", value=float(disk_norms[1]), min_value=0.1, step=1.0, key=f"n2_{ck}_v{cal_ver}")
        nc3 = st.number_input("WIP", value=float(disk_norms[2]), min_value=0.1, step=10.0, key=f"n3_{ck}_v{cal_ver}")

        cc1, cc2 = st.columns(2)
        with cc1: cal_eps = st.number_input("Cal episodes", value=10, min_value=3, max_value=50)
        with cc2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🎯 Auto-Calibrate"):
                with st.spinner("Calibrating..."):
                    # ALWAYS read stages from the JSON file, not session state
                    fresh_cfg = load_active_config()
                    if fresh_cfg is None:
                        st.error("No config loaded"); st.stop()
                    es = fresh_cfg.get("stages", [])
                    tc = {"stages": es, "arrival": fresh_cfg.get("arrival", {"distribution": arr_dist, "mean": arr_mean}),
                          "waiting_cost": fresh_cfg.get("waiting_cost", waiting_cost),
                          "max_time": fresh_cfg.get("max_time", max_time),
                          "dt": fresh_cfg.get("dt", dt),
                          "max_queue": fresh_cfg.get("max_queue", max_queue),
                          "norm_constants": [1,1,1]}
                    try:
                        ce = FlexFlowSimEnv(config=tc, weights=(.33,.33,.34), seed=42)
                        rng = np.random.default_rng(42); cs = rng.integers(0, 2**31, size=cal_eps)
                        co, de, lt = [], [], []
                        p = st.progress(0.0)
                        for i in range(cal_eps):
                            o, inf = ce.reset(seed=int(cs[i]))
                            while True:
                                o,_,_,tr,inf = ce.step(ce.action_space.sample())
                                if tr: break
                            co.append(inf["total_cost"]); de.append(inf["total_departed"]); lt.append(inf["avg_lead_time"])
                            p.progress((i+1)/cal_eps)
                        mc,md,ml = np.mean(co),np.mean(de),np.mean(lt)
                        w = md*ml
                        r1,r2,r3 = max(round(float(mc),-1),1), max(round(float(md),0),1), max(round(float(w),-1),1)
                        # Auto-save to JSON file
                        fresh_cfg["norm_constants"] = [r1, r2, r3]
                        with open(selected_config_path, "w") as f:
                            json.dump(fresh_cfg, f, indent=2)
                        # Clear ALL stale widget keys so new values display
                        ck = selected_config_name or "none"
                        st.session_state["_cal_ver"] = st.session_state.get("_cal_ver", 0) + 1
                        st.success(f"Calibrated & saved: [{r1:.0f}, {r2:.0f}, {r3:.0f}]"); st.rerun()
                    except Exception as e: st.error(f"Failed: {e}")

    with col2:
        st.subheader("Stages & Servers")
        # Always compare against actual file content, not just name
        cfg_hash = hashlib.md5(json.dumps(cfg.get("stages", []), sort_keys=True).encode()).hexdigest()
        if "edit_stages" not in st.session_state or st.session_state.get("_cfg_hash") != cfg_hash:
            st.session_state["edit_stages"] = [{"name": s.get("name", f"Stage {i+1}"), "servers": [dict(sv) for sv in s.get("servers", [])]} for i, s in enumerate(cfg.get("stages", []))]
            st.session_state["_cfg_hash"] = cfg_hash
        stages = st.session_state["edit_stages"]
        if st.button("➕ Add Stage"):
            stages.append({"name": f"Stage {len(stages)+1}", "servers": [{"name": "Server 1", "service_time": {"distribution": "normal", "mean": 10.0, "std": 2.0, "min": 0.1}, "processing_cost": 1.0, "idle_cost": 0.5}]})
            st.rerun()
        updated = []
        for si, stg in enumerate(stages):
            st.markdown("---")
            h1, h2 = st.columns([3, 1])
            with h1: sn = st.text_input(f"Stage {si+1}", value=stg["name"], key=f"sn{si}")
            with h2:
                st.markdown("<br>", unsafe_allow_html=True)
                if len(stages) > 1 and st.button("🗑️", key=f"rs{si}"): stages.pop(si); st.rerun()
            srvs = stg.get("servers", []); us = []
            for sj, sv in enumerate(srvs):
                with st.expander(f"{sn} → {sv.get('name', f'S{sj+1}')}", expanded=False):
                    e1, e2 = st.columns([3, 1])
                    with e1: svn = st.text_input("Name", value=sv.get("name", f"S{sj+1}"), key=f"vn{si}{sj}")
                    with e2:
                        if len(srvs) > 1 and st.button("🗑️", key=f"rv{si}{sj}"): stg["servers"].pop(sj); st.rerun()
                    svc = sv.get("service_time", {})
                    sd = st.selectbox("Dist", DISTRIBUTIONS, index=DISTRIBUTIONS.index(svc.get("distribution", "normal")), key=f"d{si}{sj}")
                    sn_d = {"distribution": sd}
                    if sd in ("normal", "lognormal"):
                        sn_d["mean"] = st.number_input("Mean", value=float(svc.get("mean", 10)), min_value=0.01, step=0.5, key=f"m{si}{sj}")
                        sn_d["std"] = st.number_input("Std", value=float(svc.get("std", 2)), min_value=0.01, step=0.5, key=f"s{si}{sj}")
                        if sd == "normal": sn_d["min"] = st.number_input("Min", value=float(svc.get("min", 0.1)), min_value=0.0, step=0.1, key=f"mn{si}{sj}")
                    elif sd == "exponential": sn_d["mean"] = st.number_input("Mean", value=float(svc.get("mean", 5)), min_value=0.01, step=0.5, key=f"m{si}{sj}")
                    elif sd == "uniform":
                        sn_d["low"] = st.number_input("Low", value=float(svc.get("low", 1)), min_value=0.0, step=0.5, key=f"l{si}{sj}")
                        sn_d["high"] = st.number_input("High", value=float(svc.get("high", 10)), min_value=0.01, step=0.5, key=f"h{si}{sj}")
                    pc = st.number_input("Proc cost", value=float(sv.get("processing_cost", 1)), min_value=0.0, step=0.1, key=f"p{si}{sj}")
                    ic = st.number_input("Idle cost", value=float(sv.get("idle_cost", 0.5)), min_value=0.0, step=0.1, key=f"i{si}{sj}")
                    us.append({"name": svn, "service_time": sn_d, "processing_cost": pc, "idle_cost": ic})
            if st.button(f"➕ Server → {sn}", key=f"as{si}"):
                stg["servers"].append({"name": f"Server {len(srvs)+1}", "service_time": {"distribution": "normal", "mean": 10.0, "std": 2.0, "min": 0.1}, "processing_cost": 1.0, "idle_cost": 0.5}); st.rerun()
            updated.append({"name": sn, "servers": us})
        st.session_state["edit_stages"] = updated
        sps = [len(s["servers"]) for s in updated]; na = 1
        for m in sps: na *= m
        st.caption(f"Topology: {len(updated)} stages × {sps} = {na} actions")

    st.divider()
    s1, s2, _ = st.columns([1, 1, 2])
    with s1:
        if st.button("💾 Save Config", type="primary"):
            nc = {"_metadata": cfg.get("_metadata", {}), "stages": updated, "arrival": {"distribution": arr_dist, "mean": arr_mean},
                  "waiting_cost": waiting_cost, "max_time": max_time, "dt": dt, "max_queue": max_queue, "norm_constants": [nc1, nc2, nc3]}
            with open(selected_config_path, "w") as f: json.dump(nc, f, indent=2)
            st.success(f"Saved to {selected_config_path}")
    with s2:
        nn = st.text_input("New name", placeholder="my_config")
        if st.button("📄 Save As") and nn:
            with open(CONFIG_DIR / f"{nn}.json", "w") as f:
                json.dump({"stages": updated, "arrival": {"distribution": arr_dist, "mean": arr_mean},
                           "waiting_cost": waiting_cost, "max_time": max_time, "dt": dt, "max_queue": max_queue, "norm_constants": [nc1, nc2, nc3]}, f, indent=2)
            st.success(f"Saved {nn}.json")


# ═══════════════════════════════════════════════════════════════════
# PAGE 2: TRAIN
# ═══════════════════════════════════════════════════════════════════

def page_train():
    st.title("🎓 Train RL Agents")
    if not selected_config_path: st.error("No config."); return
    c1, c2, c3 = st.columns(3)
    with c1: algos = st.multiselect("Algorithms", ["DQN", "PPO"], default=["DQN", "PPO"])
    with c2:
        all_scens = get_all_scenarios()
        scens = st.multiselect("Scenarios", list(all_scens.keys()), default=["Balanced"])
    with c3:
        ss = st.text_input("Seeds", value="42,123,256,512,1024")
        seeds = [int(x.strip()) for x in ss.split(",") if x.strip()]
    episodes = st.slider("Episodes", 50, 2000, 500, 50)
    odir = st.text_input("Output", value="results")
    tr = len(algos) * len(scens) * len(seeds)
    st.info(f"**{tr} runs** = {len(algos)} × {len(scens)} × {len(seeds)} × {episodes} eps")

    if st.button("🚀 Launch Training", type="primary", disabled=(tr == 0)):
        os.makedirs(odir, exist_ok=True)
        from stable_baselines3 import DQN as D, PPO as P
        from stable_baselines3.common.callbacks import BaseCallback
        AM = {"DQN": D, "PPO": P}
        class CB(BaseCallback):
            def __init__(s, pb, tx, et): super().__init__(verbose=0); s.pb=pb; s.tx=tx; s.et=et; s.ec=0; s.er=[]; s._c=0; s.b=-np.inf
            def _on_step(s):
                s._c += s.locals.get("rewards",[0])[0]
                if s.locals.get("dones",[False])[0]:
                    s.ec+=1; s.er.append(s._c)
                    if s._c > s.b: s.b = s._c
                    s.pb.progress(min(s.ec/max(s.et,1),1.0))
                    if s.ec%10==0: s.tx.text(f"Ep {s.ec}/{s.et} | Avg: {np.mean(s.er[-20:]):.3f} | Best: {s.b:.3f}")
                    s._c=0
                return True
        logs=[]; ri=0; op=st.progress(0.0)
        for an in algos:
            for sn in scens:
                w=all_scens[sn]["weights"]
                for sd in seeds:
                    ri+=1; tag=f"{an}_{sn}_seed{sd}"; st.subheader(f"Run {ri}/{tr}: {tag}")
                    pb=st.progress(0.0); tx=st.empty()
                    env=FlexFlowSimEnv(config=str(selected_config_path),weights=w,seed=sd)
                    spe=int(env._max_time/env._dt); A=AM[an]
                    hp=dict(learning_rate=5e-4,buffer_size=100000,learning_starts=max(1000,spe*2),batch_size=256,gamma=0.95,target_update_interval=500,exploration_fraction=0.5,exploration_initial_eps=1.0,exploration_final_eps=0.05,train_freq=4,gradient_steps=1,policy_kwargs={"net_arch":[64,64]}) if an=="DQN" else dict(learning_rate=3e-4,n_steps=min(2048,spe),batch_size=64,n_epochs=10,gamma=0.95,gae_lambda=0.95,clip_range=0.2,ent_coef=0.01,policy_kwargs={"net_arch":[64,64]})
                    m=A("MlpPolicy",env,verbose=0,seed=sd,**hp); cb=CB(pb,tx,episodes)
                    t0=time.time(); m.learn(total_timesteps=episodes*spe,callback=cb); el=time.time()-t0
                    m.save(os.path.join(odir,f"{tag}_best.zip"))
                    logs.append({"algo":an,"scenario":sn,"weights":list(w),"seed":sd,"total_episodes":cb.ec,"best_reward":float(cb.b),"training_time_s":el,"episode_rewards":[float(r) for r in cb.er]})
                    tx.text(f"✅ {tag} done in {el:.0f}s"); op.progress(ri/tr)
        with open(os.path.join(odir,"training_log.json"),"w") as f: json.dump(logs,f,indent=2,default=str)
        st.success("All training complete!")

    st.divider(); st.subheader("📈 Results")
    lf=Path("results")/"training_log.json"
    if lf.exists():
        with open(lf) as f: lg=json.load(f)
        if lg:
            fig=go.Figure()
            for l in lg:
                r=l.get("episode_rewards",[])
                if r: fig.add_trace(go.Scatter(y=pd.Series(r).rolling(20,min_periods=1).mean().values,mode="lines",name=f"{l['algo']}_{l['scenario']}_s{l['seed']}",opacity=0.8))
            fig.update_layout(title="Learning Curves (MA-20)",xaxis_title="Episode",yaxis_title="Reward",height=450)
            apply_theme(fig); st.plotly_chart(fig,width="stretch")
    else: st.info("No results yet.")


# ═══════════════════════════════════════════════════════════════════
# PAGE 3: EVALUATE (scenario queue + Excel export)
# ═══════════════════════════════════════════════════════════════════

def page_evaluate():
    st.title("📊 Evaluate & Compare")
    if not selected_config_path: st.error("No config."); return

    # Custom weight editor
    render_weight_editor()

    all_scenarios = get_all_scenarios()

    st.subheader("Scenario Queue")
    qs = st.multiselect("Scenarios to benchmark", list(all_scenarios.keys()), default=["Balanced"], key="eq")
    c1,c2=st.columns(2)
    with c1: nr=st.slider("Eval episodes",10,200,50)
    with c2: es=st.number_input("Base seed",value=42,min_value=0)
    bn=st.multiselect("Baselines",list(BASELINE_POLICIES.keys()),default=list(BASELINE_POLICIES.keys()))
    rd=Path("results")
    af=sorted(rd.glob("*_best.zip")) if rd.exists() else []
    al=[f.stem.replace("_best","") for f in af]
    sa=st.multiselect("Agents",al)

    if st.button("▶️ Run Evaluation Queue",type="primary"):
        asr={}
        for sn in qs:
            w=all_scenarios[sn]["weights"]; st.subheader(f"Scenario: {sn} (w={w})")
            env=FlexFlowSimEnv(config=str(selected_config_path),weights=w,seed=es)
            rng=np.random.default_rng(es); evs=rng.integers(0,2**31,size=nr)
            ar={}; pr=st.progress(0.0); tm=len(bn)+len(sa); mi=0
            for n in bn:
                pol=BASELINE_POLICIES[n](env=env); ar[n]=pd.DataFrame([run_episode(pol,env,int(s)) for s in evs])
                mi+=1; pr.progress(mi/tm)
            if sa:
                from stable_baselines3 import DQN,PPO
                for a in sa:
                    ap=rd/f"{a}_best.zip"; algo=a.split("_")[0]; mdl=(DQN if algo=="DQN" else PPO).load(str(ap))
                    res=[]
                    for s in evs:
                        ob,inf=env.reset(seed=int(s)); tr=0
                        while True:
                            ac,_=mdl.predict(ob,deterministic=True); ob,r,_,tc,inf=env.step(ac); tr+=r
                            if tc: break
                        d=inf["total_departed"]; res.append({"totalCost":inf["total_cost"],"totalDeparted":d,"costPerUnit":inf["total_cost"]/max(d,1),"avgLeadTime":inf["avg_lead_time"],"totalReward":tr})
                    ar[a]=pd.DataFrame(res); mi+=1; pr.progress(mi/tm)
            asr[sn]=ar

            # Recommendation summary
            render_recommendation(ar, sn)

            # Summary table
            ms=["totalCost","totalDeparted","costPerUnit","avgLeadTime","totalReward"]
            rows=[{"Method":n,**{m:f"{df[m].mean():.1f} ± {df[m].std():.1f}" for m in ms}} for n,df in ar.items()]
            st.dataframe(pd.DataFrame(rows),width="stretch")
            mc=st.selectbox(f"Metric ({sn})",ms,index=0,key=f"bm{sn}")
            bd=[{"Method":n,mc:v} for n,df in ar.items() for v in df[mc]]
            fig=px.box(pd.DataFrame(bd),x="Method",y=mc,color="Method",title=f"{mc} — {sn}")
            fig.update_layout(showlegend=False,height=400); apply_theme(fig); st.plotly_chart(fig,width="stretch")
            if len(ar)>=2:
                from scipy import stats
                gs=[df["totalCost"].values for df in ar.values()]; h,p=stats.kruskal(*gs)
                st.write(f"**Kruskal-Wallis:** H={h:.2f}, p={p:.6f} {'✅' if p<0.05 else '❌'}")
        st.session_state["esr"]=asr

    if "esr" in st.session_state:
        st.divider(); st.subheader("📥 Export")
        if st.button("📊 Export Excel"):
            buf=io.BytesIO()
            with pd.ExcelWriter(buf,engine="openpyxl") as w:
                for sn,ar in st.session_state["esr"].items():
                    rows=[]
                    for n,df in ar.items():
                        for _,r in df.iterrows(): d=r.to_dict(); d["method"]=n; d["scenario"]=sn; rows.append(d)
                    pd.DataFrame(rows).to_excel(w,sheet_name=sn[:31],index=False)
            st.download_button("📥 Download Excel",buf.getvalue(),"flexflowsim_eval.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        if st.button("📄 Export CSV"):
            rows=[]
            for sn,ar in st.session_state["esr"].items():
                for n,df in ar.items():
                    for _,r in df.iterrows(): d=r.to_dict(); d["method"]=n; d["scenario"]=sn; rows.append(d)
            st.download_button("📥 Download CSV",pd.DataFrame(rows).to_csv(index=False),"flexflowsim_eval.csv","text/csv")


# ═══════════════════════════════════════════════════════════════════
# PAGE 4: SIMULATE (with Gantt)
# ═══════════════════════════════════════════════════════════════════

def page_simulate():
    st.title("🔬 Live Simulation")
    if not selected_config_path: st.error("No config."); return
    c1,c2,c3=st.columns(3)
    with c1: ss=st.selectbox("Scenario",list(get_all_scenarios().keys()),index=3,key="ssc")
    with c2: sd=st.number_input("Seed",value=42,min_value=0,key="ssd")
    with c3:
        po=list(BASELINE_POLICIES.keys())
        rd=Path("results")
        for f in sorted(rd.glob("*_best.zip")) if rd.exists() else []: po.append(f"🤖 {f.stem.replace('_best','')}")
        pc=st.selectbox("Policy",po,key="spo")
    w=get_all_scenarios()[ss]["weights"]; sp=st.slider("Steps/frame",1,50,10)

    if st.button("▶️ Run Simulation",type="primary"):
        env=FlexFlowSimEnv(config=str(selected_config_path),weights=w,seed=sd)
        mdl=None; pol=None
        if pc.startswith("🤖"):
            from stable_baselines3 import DQN,PPO
            a=pc.replace("🤖 ",""); ap=rd/f"{a}_best.zip"; algo=a.split("_")[0]
            mdl=(DQN if algo=="DQN" else PPO).load(str(ap))
        else: pol=BASELINE_POLICIES[pc](env=env); pol.reset()

        ob,inf=env.reset(seed=sd); ns=env._total_servers
        sl=[]
        for si,stg in enumerate(env._stages):
            for sj,sv in enumerate(stg["servers"]): sl.append(sv.get("name",f"S{si+1}.{sj+1}"))
        cp=st.empty(); mp=st.empty(); pp=st.progress(0.0)
        hist={"t":[],"dep":[],"cost":[],"lt":[],"rw":[],"cpu":[]}
        for i in range(ns): hist[f"q{i}"]=[]; hist[f"u{i}"]=[]
        st_n=0; tr=0; ts=int(env._max_time/env._dt)

        while True:
            if mdl: ac=int(mdl.predict(ob,deterministic=True)[0])
            elif pol: ac=pol.predict(ob)
            else: ac=env.action_space.sample()
            ob,r,_,tc,inf=env.step(ac); tr+=r; st_n+=1
            dep=inf["total_departed"]; cost=inf["total_cost"]
            cpu=cost/max(dep,1)
            hist["t"].append(inf["sim_time"]); hist["dep"].append(dep)
            hist["cost"].append(cost); hist["cpu"].append(cpu)
            hist["lt"].append(inf["avg_lead_time"]); hist["rw"].append(tr)
            for i in range(ns): hist[f"q{i}"].append(float(ob[i])*env._max_queue); hist[f"u{i}"].append(inf["utilisation"][i])

            if st_n%sp==0 or tc:
                pp.progress(min(st_n/ts,1.0))
                co=["#3b82f6","#ef4444","#10b981","#f59e0b","#8b5cf6","#ec4899"]
                fig=make_subplots(rows=2,cols=3,subplot_titles=(
                    "Queue Lengths (entities)","Server Utilisation","Cost per Unit ($/product)",
                    "Cumulative Throughput (units)","Cumulative Cost ($)","Avg Lead Time (min)"),
                    vertical_spacing=0.15,horizontal_spacing=0.08)

                # Row 1: Queues, Utilisation, Cost/Unit
                for i in range(ns):
                    fig.add_trace(go.Scatter(x=hist["t"],y=hist[f"q{i}"],name=sl[i],mode="lines",line=dict(color=co[i%len(co)])),row=1,col=1)
                fig.add_trace(go.Bar(x=sl,y=[hist[f"u{i}"][-1] for i in range(ns)],marker_color=co[:ns],showlegend=False),row=1,col=2)
                fig.add_trace(go.Scatter(x=hist["t"],y=hist["cpu"],mode="lines",line=dict(color="#f59e0b"),showlegend=False),row=1,col=3)

                # Row 2: Departures, Cost, Lead Time
                fig.add_trace(go.Scatter(x=hist["t"],y=hist["dep"],mode="lines",line=dict(color="#10b981"),showlegend=False),row=2,col=1)
                fig.add_trace(go.Scatter(x=hist["t"],y=hist["cost"],mode="lines",line=dict(color="#ef4444"),showlegend=False),row=2,col=2)
                fig.add_trace(go.Scatter(x=hist["t"],y=hist["lt"],mode="lines",line=dict(color="#8b5cf6"),showlegend=False),row=2,col=3)

                # Axis titles
                fig.update_xaxes(title_text="Time (min)",row=2,col=1)
                fig.update_xaxes(title_text="Time (min)",row=2,col=2)
                fig.update_xaxes(title_text="Time (min)",row=2,col=3)
                fig.update_yaxes(title_text="Queue (entities)",row=1,col=1)
                fig.update_yaxes(title_text="Utilisation",range=[0,1.05],row=1,col=2)
                fig.update_yaxes(title_text="Cost/Unit ($/product)",row=1,col=3)
                fig.update_yaxes(title_text="Throughput (units)",row=2,col=1)
                fig.update_yaxes(title_text="Total Cost ($)",row=2,col=2)
                fig.update_yaxes(title_text="Lead Time (min)",row=2,col=3)

                fig.update_layout(height=550,margin=dict(t=40,b=10))
                apply_theme(fig); cp.plotly_chart(fig,width="stretch")

                mc=mp.columns(6)
                mc[0].metric("Time",f"{inf['sim_time']:.0f} min")
                mc[1].metric("Departures",inf["total_departed"])
                mc[2].metric("Avg Lead Time",f"{inf['avg_lead_time']:.1f} min")
                mc[3].metric("Total Cost",f"{inf['total_cost']:.0f}")
                mc[4].metric("Cost/Unit",f"{cpu:.1f}")
                mc[5].metric("Reward",f"{tr:.2f}")
            if tc: break
        st.success(f"Done: {st_n} steps, {inf['total_departed']} departures")

        # Gantt
        st.subheader("Server Activity (Gantt)")
        gd=[]
        for i in range(ns):
            us=hist[f"u{i}"]; ts_l=hist["t"]; busy=False; s0=0
            for ti in range(len(us)):
                if us[ti]>0.5 and not busy: busy=True; s0=ts_l[ti]
                elif us[ti]<=0.5 and busy: busy=False; gd.append(dict(Server=sl[i],Start=s0,End=ts_l[ti]))
            if busy: gd.append(dict(Server=sl[i],Start=s0,End=ts_l[-1]))
        if gd:
            gdf=pd.DataFrame(gd)
            fig=px.timeline(gdf,x_start="Start",x_end="End",y="Server",color="Server",title="Busy Periods")
            fig.update_layout(height=250,showlegend=False); apply_theme(fig); st.plotly_chart(fig,width="stretch")


# ═══════════════════════════════════════════════════════════════════
# PAGE 5: SENSITIVITY
# ═══════════════════════════════════════════════════════════════════

def page_sensitivity():
    st.title("📈 Sensitivity Analysis")
    if not selected_config_path: st.error("No config."); return
    cfg=load_active_config()
    if not cfg: return
    st.markdown("Sweep any system parameter and observe its impact on cost, throughput, and lead time.")
    st.caption("ℹ️ Only physical metrics are measured — objective weights do not affect these results.")

    # Dynamically build parameter list from config
    param_options = []
    param_paths = {}  # maps display name -> (path_keys, current_value, unit)

    # Arrival
    arr = cfg.get("arrival", {})
    if "mean" in arr:
        name = "Mean inter-arrival time (min)"
        param_options.append(name)
        param_paths[name] = (["arrival", "mean"], float(arr["mean"]), "min")

    # Waiting cost
    if "waiting_cost" in cfg:
        name = "Waiting cost (per entity/min)"
        param_options.append(name)
        param_paths[name] = (["waiting_cost"], float(cfg["waiting_cost"]), "cost/min")

    # Per-stage, per-server parameters
    for si, stage in enumerate(cfg.get("stages", [])):
        sname = stage.get("name", f"Stage {si+1}")
        for sj, srv in enumerate(stage.get("servers", [])):
            vname = srv.get("name", f"Server {sj+1}")
            prefix = f"{sname} → {vname}"

            # Service time mean
            svc = srv.get("service_time", {})
            if "mean" in svc:
                name = f"{prefix}: mean service time (min)"
                param_options.append(name)
                param_paths[name] = (["stages", si, "servers", sj, "service_time", "mean"], float(svc["mean"]), "min")

            # Service time std
            if "std" in svc:
                name = f"{prefix}: service time std (min)"
                param_options.append(name)
                param_paths[name] = (["stages", si, "servers", sj, "service_time", "std"], float(svc["std"]), "min")

            # Processing cost
            if "processing_cost" in srv:
                name = f"{prefix}: processing cost (per min)"
                param_options.append(name)
                param_paths[name] = (["stages", si, "servers", sj, "processing_cost"], float(srv["processing_cost"]), "cost/min")

            # Idle cost
            if "idle_cost" in srv:
                name = f"{prefix}: idle cost (per min)"
                param_options.append(name)
                param_paths[name] = (["stages", si, "servers", sj, "idle_cost"], float(srv["idle_cost"]), "cost/min")

    sp = st.selectbox("Parameter to sweep", param_options)

    # Show current value and auto-suggest range
    if sp in param_paths:
        current_val = param_paths[sp][1]
        st.caption(f"Current value: **{current_val}**")
        suggested_min = max(0.1, current_val * 0.25)
        suggested_max = current_val * 2.5
    else:
        suggested_min, suggested_max = 1.0, 20.0

    s1, s2, s3 = st.columns(3)
    with s1: vmin = st.number_input("Min", value=round(suggested_min, 1), min_value=0.01, step=0.5)
    with s2: vmax = st.number_input("Max", value=round(suggested_max, 1), min_value=0.1, step=0.5)
    with s3: steps = st.number_input("Steps", value=8, min_value=3, max_value=20)
    reps = st.slider("Episodes/point", 5, 50, 10, key="sr")
    pn = st.selectbox("Policy", list(BASELINE_POLICIES.keys()), key="sp_s")

    if st.button("🔍 Run Sweep", type="primary"):
        w = (0.33, 0.33, 0.34)
        vals = np.linspace(vmin, vmax, int(steps))
        path_keys = param_paths[sp][0]
        res = {"val": [], "cost_m": [], "cost_s": [], "dep_m": [], "dep_s": [], "lt_m": [], "lt_s": [], "cpu_m": [], "cpu_s": []}
        pr = st.progress(0.0)

        for vi, v in enumerate(vals):
            mc = json.loads(json.dumps(cfg))
            obj = mc
            for key in path_keys[:-1]:
                obj = obj[key]
            obj[path_keys[-1]] = float(v)

            env = FlexFlowSimEnv(config=mc, weights=w, seed=42)
            pol = BASELINE_POLICIES[pn](env=env)
            rng = np.random.default_rng(42)
            es = rng.integers(0, 2**31, size=reps)
            co, de, lt = [], [], []
            for s in es:
                pol.reset()
                ob, inf = env.reset(seed=int(s))
                while True:
                    a = pol.predict(ob)
                    ob, _, _, tc, inf = env.step(a)
                    if tc: break
                co.append(inf["total_cost"])
                de.append(inf["total_departed"])
                lt.append(inf["avg_lead_time"])
            res["val"].append(v)
            res["cost_m"].append(np.mean(co)); res["cost_s"].append(np.std(co))
            res["dep_m"].append(np.mean(de)); res["dep_s"].append(np.std(de))
            res["lt_m"].append(np.mean(lt)); res["lt_s"].append(np.std(lt))
            cpu_vals = [c/max(d,1) for c,d in zip(co,de)]
            res["cpu_m"].append(np.mean(cpu_vals)); res["cpu_s"].append(np.std(cpu_vals))
            pr.progress((vi + 1) / len(vals))

        # Store in session state so buttons persist after rerun
        st.session_state["sens_df"] = pd.DataFrame(res)
        st.session_state["sens_param"] = sp

    # Render results if available (persists across reruns)
    if "sens_df" in st.session_state:
        df = st.session_state["sens_df"]
        sp_display = st.session_state["sens_param"]
        x_label = sp_display.split(": ")[-1] if ": " in sp_display else sp_display

        fig = make_subplots(rows=1, cols=4, subplot_titles=("Mean Total Cost", "Mean Throughput (units)", "Mean Lead Time (min)", "Mean Cost per Unit"))
        y_labels = ["Total Cost ($)", "Throughput (units)", "Lead Time (min)", "Cost / Unit ($/product)"]
        for ci, (m, c) in enumerate([("cost", "#ef4444"), ("dep", "#10b981"), ("lt", "#3b82f6"), ("cpu", "#f59e0b")]):
            fig.add_trace(go.Scatter(
                x=df["val"], y=df[f"{m}_m"], mode="lines+markers", line=dict(color=c), showlegend=False,
                error_y=dict(type="data", array=df[f"{m}_s"], visible=True)
            ), row=1, col=ci + 1)
            fig.update_xaxes(title_text=x_label, row=1, col=ci + 1)
            fig.update_yaxes(title_text=y_labels[ci], row=1, col=ci + 1)
        fig.update_layout(height=500, width=1600, title=f"Sensitivity: {sp_display}")
        apply_theme(fig); st.plotly_chart(fig, width="stretch")

        # Rename columns for clarity
        export_df = df.rename(columns={
            "val": sp_display,
            "cost_m": "Mean Total Cost", "cost_s": "Std Total Cost",
            "dep_m": "Mean Throughput", "dep_s": "Std Throughput",
            "lt_m": "Mean Lead Time (min)", "lt_s": "Std Lead Time (min)",
            "cpu_m": "Mean Cost/Unit", "cpu_s": "Std Cost/Unit",
        })
        st.dataframe(export_df.round(2), width="stretch")

        # Auto-save to sensitivity_analysis/ folder
        sa_dir = Path("sensitivity_analysis")
        sa_dir.mkdir(exist_ok=True)
        safe_name = sp_display.replace(" ", "_").replace("/", "_").replace(":", "_").replace("→", "-").replace("(", "").replace(")", "").replace("$", "")[:40]

        # Save Excel
        xlsx_path = sa_dir / f"sensitivity_{safe_name}.xlsx"
        export_df.round(4).to_excel(str(xlsx_path), index=False, sheet_name="Sensitivity")
        st.session_state["sens_xlsx_bytes"] = xlsx_path.read_bytes()
        st.session_state["sens_xlsx_name"] = xlsx_path.name

        # Save CSV
        st.session_state["sens_csv"] = export_df.round(4).to_csv(index=False)
        st.session_state["sens_csv_name"] = f"sensitivity_{safe_name}.csv"

        # Save PNG using matplotlib (no kaleido needed, always works)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mpl_fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
        chart_cfgs = [
            ("cost_m", "cost_s", "Mean Total Cost ($)", "#e53935"),
            ("dep_m", "dep_s", "Mean Throughput (units)", "#43a047"),
            ("lt_m", "lt_s", "Mean Lead Time (min)", "#1e88e5"),
            ("cpu_m", "cpu_s", "Mean Cost per Unit ($/unit)", "#fb8c00"),
        ]
        for ax, (mean_col, std_col, title, color) in zip(axes, chart_cfgs):
            ax.errorbar(df["val"], df[mean_col], yerr=df[std_col], fmt="o-",
                        color=color, capsize=4, capthick=1.5, linewidth=2, markersize=6, elinewidth=1.2)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xlabel(x_label, fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=9)
        mpl_fig.suptitle(f"Sensitivity: {sp_display}", fontsize=13, fontweight="bold", y=1.02)
        mpl_fig.tight_layout()

        png_path = sa_dir / f"sensitivity_{safe_name}.png"
        mpl_fig.savefig(str(png_path), dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close(mpl_fig)

        st.session_state["sens_png_bytes"] = png_path.read_bytes()
        st.session_state["sens_png_name"] = png_path.name
        st.success(f"✅ Saved to `sensitivity_analysis/` — PNG ({png_path.stat().st_size // 1024} KB) + Excel + CSV")

    # Download buttons — ALWAYS shown if data exists
    if "sens_xlsx_bytes" in st.session_state:
        st.markdown("---")
        st.subheader("📥 Downloads")
        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            if "sens_png_bytes" in st.session_state:
                st.download_button("🖼️ PNG (300 DPI)", st.session_state["sens_png_bytes"],
                                   st.session_state["sens_png_name"], "image/png", key="dl_png")
        with dc2:
            st.download_button("📊 Excel", st.session_state["sens_xlsx_bytes"],
                               st.session_state["sens_xlsx_name"],
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_xlsx")
        with dc3:
            st.download_button("📄 CSV", st.session_state["sens_csv"],
                               st.session_state["sens_csv_name"], "text/csv", key="dl_csv")


# ═══════════════════════════════════════════════════════════════════
# PAGE 6: COMPARE
# ═══════════════════════════════════════════════════════════════════

def page_compare():
    st.title("🔄 Compare Configurations")
    if len(config_names)<2: st.warning("Need ≥2 configs."); return

    st.markdown("Run all routing methods on two different system configurations and compare which strategy works best for each.")

    c1,c2=st.columns(2)
    with c1: ca=st.selectbox("Config A",config_names,index=0,key="ca")
    with c2: cb=st.selectbox("Config B",config_names,index=min(1,len(config_names)-1),key="cb")

    nr=st.slider("Episodes per method",10,100,30,key="cr")
    bn=st.multiselect("Methods to compare",list(BASELINE_POLICIES.keys()),default=list(BASELINE_POLICIES.keys()),key="cmp_bl")

    if st.button("⚖️ Compare",type="primary"):
        rng=np.random.default_rng(42); seeds=rng.integers(0,2**31,size=nr)
        # Use balanced weights (only affects totalReward, not physical metrics)
        w=(0.33,0.33,0.34)
        config_results={}
        pr=st.progress(0.0)
        total_work=2*len(bn)
        wi=0

        for cn in [ca,cb]:
            env=FlexFlowSimEnv(config=str(CONFIG_DIR/f"{cn}.json"),weights=w,seed=42)
            method_results={}
            for pn in bn:
                pol=BASELINE_POLICIES[pn](env=env)
                res=[]
                for s in seeds:
                    pol.reset(); ob,inf=env.reset(seed=int(s)); tr=0
                    while True:
                        a=pol.predict(ob); ob,r,_,tc,inf=env.step(a); tr+=r
                        if tc: break
                    d=inf["total_departed"]
                    res.append({"totalCost":inf["total_cost"],"totalDeparted":d,
                                "costPerUnit":inf["total_cost"]/max(d,1),
                                "avgLeadTime":inf["avg_lead_time"]})
                method_results[pn]=pd.DataFrame(res)
                wi+=1; pr.progress(wi/total_work)
            config_results[cn]=method_results

        # Recommendation per config
        st.subheader("📋 Best Method per Configuration")
        objectives=[
            ("totalCost","Lowest Cost","min","💰"),
            ("totalDeparted","Highest Throughput","max","📦"),
            ("avgLeadTime","Lowest Lead Time","min","⏱️"),
            ("costPerUnit","Lowest Cost/Unit","min","💲"),
        ]

        rec_cols=st.columns(2)
        for ci,cn in enumerate([ca,cb]):
            with rec_cols[ci]:
                st.markdown(f"### {cn}")
                for metric,label,direction,icon in objectives:
                    scores={pn:df[metric].mean() for pn,df in config_results[cn].items()}
                    best=min(scores,key=scores.get) if direction=="min" else max(scores,key=scores.get)
                    val=scores[best]
                    st.success(f"{icon} **{label}:** {best} ({val:.1f})")

        # Side-by-side summary tables
        st.subheader("Summary Tables")
        tab_cols=st.columns(2)
        for ci,cn in enumerate([ca,cb]):
            with tab_cols[ci]:
                st.markdown(f"**{cn}**")
                metrics=["totalCost","totalDeparted","costPerUnit","avgLeadTime"]
                rows=[{"Method":pn,**{m:f"{df[m].mean():.1f} ± {df[m].std():.1f}" for m in metrics}}
                      for pn,df in config_results[cn].items()]
                st.dataframe(pd.DataFrame(rows),width="stretch")

        # Cross-config comparison: same method, different config
        st.subheader("Method Performance Across Configs")
        cmp_metric=st.selectbox("Metric",["totalCost","totalDeparted","costPerUnit","avgLeadTime"],key="cmp_m")

        box_data=[]
        for cn in [ca,cb]:
            for pn,df in config_results[cn].items():
                for v in df[cmp_metric]:
                    box_data.append({"Config":cn,"Method":pn,cmp_metric:v})
        bdf=pd.DataFrame(box_data)
        fig=px.box(bdf,x="Method",y=cmp_metric,color="Config",title=f"{cmp_metric} by Method and Config")
        fig.update_layout(height=450)
        apply_theme(fig); st.plotly_chart(fig,width="stretch")

        # Highlight methods that change ranking between configs
        st.subheader("🔍 Ranking Shifts")
        st.caption("Methods whose ranking changes between configs — indicates system-dependent routing preferences.")
        for metric,label,direction,icon in objectives:
            rank_a=sorted(config_results[ca].items(),key=lambda x:x[1][metric].mean(),reverse=(direction=="max"))
            rank_b=sorted(config_results[cb].items(),key=lambda x:x[1][metric].mean(),reverse=(direction=="max"))
            rank_a_names=[x[0] for x in rank_a]
            rank_b_names=[x[0] for x in rank_b]
            if rank_a_names[0]!=rank_b_names[0]:
                st.warning(f"{icon} **{label}:** Best shifts from **{rank_a_names[0]}** ({ca}) to **{rank_b_names[0]}** ({cb})")
            else:
                st.info(f"{icon} **{label}:** **{rank_a_names[0]}** is best on both configs")

        # Statistical tests per method across configs
        st.subheader("Statistical Tests (per method across configs)")
        from scipy import stats
        test_metric=st.selectbox("Test metric",["totalCost","totalDeparted","costPerUnit","avgLeadTime"],key="cmp_test")
        stat_rows=[]
        for pn in bn:
            if pn in config_results[ca] and pn in config_results[cb]:
                va=config_results[ca][pn][test_metric].values
                vb=config_results[cb][pn][test_metric].values
                u,p=stats.mannwhitneyu(va,vb,alternative="two-sided")
                diff=np.mean(va)-np.mean(vb)
                stat_rows.append({"Method":pn,
                    f"{ca} mean":f"{np.mean(va):.1f}",
                    f"{cb} mean":f"{np.mean(vb):.1f}",
                    "Diff":f"{diff:+.1f}",
                    "M-W U":f"{u:.0f}","p-value":f"{p:.4f}",
                    "Sig":"✅" if p<0.05 else "❌"})
        st.dataframe(pd.DataFrame(stat_rows),width="stretch")


# ═══════════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════════

{"⚙️ Configure": page_configure, "🎓 Train": page_train, "📊 Evaluate": page_evaluate,
 "🔬 Simulate": page_simulate, "📈 Sensitivity": page_sensitivity, "🔄 Compare": page_compare}[page]()
