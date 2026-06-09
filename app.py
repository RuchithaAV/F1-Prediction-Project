import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib

# Set Page Config
st.set_page_config(
    page_title="F1 Grand Prix Podium Simulator",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;900&display=swap');
    
    * {
        font-family: 'Outfit', sans-serif;
    }
    
    .main {
        background-color: #0b0c10;
        color: #c5c6c7;
    }
    
    h1, h2, h3, h4 {
        color: #ffffff;
        font-weight: 700;
    }
    
    .title-banner {
        background: linear-gradient(135deg, #1f2833 0%, #0b0c10 100%);
        border: 1px solid #1f2833;
        border-radius: 15px;
        padding: 2.5rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
    }
    
    .f1-accent {
        color: #ff1801;
        font-weight: 900;
    }

    .podium-container {
        display: flex;
        justify-content: center;
        align-items: flex-end;
        gap: 1.5rem;
        margin: 2rem 0;
        padding: 1.5rem;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .podium-card {
        background: rgba(255, 255, 255, 0.04);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        width: 250px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: transform 0.3s ease;
    }
    
    .podium-card:hover {
        transform: translateY(-5px);
    }

    .p1 {
        height: 280px;
        border-top: 5px solid #ffd700;
    }
    
    .p2 {
        height: 240px;
        border-top: 5px solid #c0c0c0;
    }
    
    .p3 {
        height: 210px;
        border-top: 5px solid #cd7f32;
    }
    
    .rank-badge {
        font-size: 2.5rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
    }
    
    .gold-text { color: #ffd700; }
    .silver-text { color: #c0c0c0; }
    .bronze-text { color: #cd7f32; }

    .probability-badge {
        background: rgba(255, 24, 1, 0.1);
        color: #ff1801;
        font-weight: 700;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        display: inline-block;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    
    .sim-button {
        background: linear-gradient(90deg, #ff1801 0%, #ff4b36 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.75rem 2.5rem !important;
        font-size: 1.2rem !important;
        box-shadow: 0 4px 20px rgba(255, 24, 1, 0.4) !important;
        transition: all 0.3s ease !important;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# Asset Loader (No cache to prevent stale model versions)
def load_assets():
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f1 dataset")
    
    model = joblib.load('f1_podium_model.joblib')
    le_driver = joblib.load('le_driver.joblib')
    le_constructor = joblib.load('le_constructor.joblib')
    
    drivers_df = pd.read_csv(os.path.join(DATA_DIR, "drivers.csv"))
    constructors_df = pd.read_csv(os.path.join(DATA_DIR, "constructors.csv"))
    circuits_df = pd.read_csv(os.path.join(DATA_DIR, "circuits.csv"))
    races_df = pd.read_csv(os.path.join(DATA_DIR, "races.csv"))
    results_df = pd.read_csv(os.path.join(DATA_DIR, "results.csv"))
    
    # Filter to only drivers and constructors active from 2020 to 2024
    races_20_24 = races_df[(races_df['year'] >= 2020) & (races_df['year'] <= 2024)]
    active_race_ids = races_20_24['raceId']
    
    active_driver_ids = results_df[results_df['raceId'].isin(active_race_ids)]['driverId'].unique()
    active_constructor_ids = results_df[results_df['raceId'].isin(active_race_ids)]['constructorId'].unique()
    active_circuit_ids = races_20_24['circuitId'].unique()
    
    drivers_df = drivers_df[drivers_df['driverId'].isin(active_driver_ids)].copy()
    constructors_df = constructors_df[constructors_df['constructorId'].isin(active_constructor_ids)].copy()
    circuits_df = circuits_df[circuits_df['circuitId'].isin(active_circuit_ids)].copy()
    
    return model, le_driver, le_constructor, drivers_df, constructors_df, circuits_df

try:
    model, le_driver, le_constructor, drivers_df, constructors_df, circuits_df = load_assets()
except Exception as e:
    st.error(f"Error loading models or dataset files: {e}")
    st.stop()

# Helper dictionaries mapping refs to human-readable names
drivers_df['fullname'] = drivers_df['forename'] + " " + drivers_df['surname']
known_drivers = set(le_driver.classes_)
known_constructors = set(le_constructor.classes_)

driver_mapping = drivers_df[drivers_df['driverRef'].isin(known_drivers)].set_index('driverRef')['fullname'].to_dict()
constructor_mapping = constructors_df[constructors_df['constructorRef'].isin(known_constructors)].set_index('constructorRef')['name'].to_dict()
circuit_mapping = circuits_df.set_index('circuitId')['name'].to_dict()

# Presets of driver configurations (Grid positions, age, stats)
# This represents a modern F1 starting grid preset
modern_grid_presets = [
    {"driver": "max_verstappen", "constructor": "red_bull", "age": 26, "driver_pts": 350, "driver_wins": 12, "constructor_pts": 500, "constructor_wins": 14, "driver_recent": 3, "constructor_recent": 3},
    {"driver": "norris", "constructor": "mclaren", "age": 24, "driver_pts": 220, "driver_wins": 2, "constructor_pts": 380, "constructor_wins": 3, "driver_recent": 2, "constructor_recent": 3},
    {"driver": "leclerc", "constructor": "ferrari", "age": 26, "driver_pts": 210, "driver_wins": 2, "constructor_pts": 390, "constructor_wins": 3, "driver_recent": 2, "constructor_recent": 2},
    {"driver": "sainz", "constructor": "ferrari", "age": 29, "driver_pts": 180, "driver_wins": 1, "constructor_pts": 390, "constructor_wins": 3, "driver_recent": 1, "constructor_recent": 2},
    {"driver": "piastri", "constructor": "mclaren", "age": 23, "driver_pts": 160, "driver_wins": 1, "constructor_pts": 380, "constructor_wins": 3, "driver_recent": 2, "constructor_recent": 3},
    {"driver": "perez", "constructor": "red_bull", "age": 34, "driver_pts": 150, "driver_wins": 0, "constructor_pts": 500, "constructor_wins": 14, "driver_recent": 0, "constructor_recent": 3},
    {"driver": "hamilton", "constructor": "mercedes", "age": 39, "driver_pts": 140, "driver_wins": 1, "constructor_pts": 240, "constructor_wins": 1, "driver_recent": 1, "constructor_recent": 1},
    {"driver": "russell", "constructor": "mercedes", "age": 26, "driver_pts": 100, "driver_wins": 0, "constructor_pts": 240, "constructor_wins": 1, "driver_recent": 0, "constructor_recent": 1},
    {"driver": "alonso", "constructor": "aston_martin", "age": 42, "driver_pts": 60, "driver_wins": 0, "constructor_pts": 85, "constructor_wins": 0, "driver_recent": 0, "constructor_recent": 0},
    {"driver": "stroll", "constructor": "aston_martin", "age": 25, "driver_pts": 25, "driver_wins": 0, "constructor_pts": 85, "constructor_wins": 0, "driver_recent": 0, "constructor_recent": 0},
    {"driver": "gasly", "constructor": "alpine", "age": 28, "driver_pts": 20, "driver_wins": 0, "constructor_pts": 35, "constructor_wins": 0, "driver_recent": 0, "constructor_recent": 0},
    {"driver": "ocon", "constructor": "alpine", "age": 27, "driver_pts": 15, "driver_wins": 0, "constructor_pts": 35, "constructor_wins": 0, "driver_recent": 0, "constructor_recent": 0},
    {"driver": "albon", "constructor": "williams", "age": 28, "driver_pts": 12, "driver_wins": 0, "constructor_pts": 12, "constructor_wins": 0, "driver_recent": 0, "constructor_recent": 0},
    {"driver": "tsunoda", "constructor": "rb", "age": 24, "driver_pts": 18, "driver_wins": 0, "constructor_pts": 28, "constructor_wins": 0, "driver_recent": 0, "constructor_recent": 0},
    {"driver": "ricciardo", "constructor": "rb", "age": 34, "driver_pts": 10, "driver_wins": 0, "constructor_pts": 28, "constructor_wins": 0, "driver_recent": 0, "constructor_recent": 0},
    {"driver": "hulkenberg", "constructor": "haas", "age": 36, "driver_pts": 14, "driver_wins": 0, "constructor_pts": 22, "constructor_wins": 0, "driver_recent": 0, "constructor_recent": 0},
    {"driver": "kevin_magnussen", "constructor": "haas", "age": 31, "driver_pts": 8, "driver_wins": 0, "constructor_pts": 22, "constructor_wins": 0, "driver_recent": 0, "constructor_recent": 0},
    {"driver": "bottas", "constructor": "sauber", "age": 34, "driver_pts": 0, "driver_wins": 0, "constructor_pts": 0, "constructor_wins": 0, "driver_recent": 0, "constructor_recent": 0},
    {"driver": "zhou", "constructor": "sauber", "age": 25, "driver_pts": 0, "driver_wins": 0, "constructor_pts": 0, "constructor_wins": 0, "driver_recent": 0, "constructor_recent": 0},
    {"driver": "sargeant", "constructor": "williams", "age": 23, "driver_pts": 0, "driver_wins": 0, "constructor_pts": 12, "constructor_wins": 0, "driver_recent": 0, "constructor_recent": 0},
    {"driver": "vettel", "constructor": "aston_martin", "age": 35, "driver_pts": 37, "driver_wins": 0, "constructor_pts": 55, "constructor_wins": 0, "driver_recent": 0, "constructor_recent": 0},
    {"driver": "latifi", "constructor": "williams", "age": 27, "driver_pts": 2, "driver_wins": 0, "constructor_pts": 8, "constructor_wins": 0, "driver_recent": 0, "constructor_recent": 0}
]

# Header Title
st.markdown("""
    <div class="title-banner">
        <h1>🏆 F1 Grand Prix <span class="f1-accent">Podium Simulator</span></h1>
        <p style="color: #8b9bb4; font-size: 1.1rem; margin: 0.5rem 0 0 0;">
            Select a circuit and customize your starting grid to calculate and rank podium finish probabilities for the entire field.
        </p>
    </div>
""", unsafe_allow_html=True)

# Layout Setup
col_setup, col_preview = st.columns([1, 2], gap="large")

with col_setup:
    st.markdown("### 🛠️ Simulation Setup")
    
    # Track selection
    selected_circuit = st.selectbox(
        "Circuit (Race Track)",
        options=list(circuit_mapping.keys()),
        format_func=lambda x: f"{circuit_mapping[x]} ({circuits_df.loc[circuits_df['circuitId'] == x, 'country'].values[0]})"
    )
    
    # Presets
    st.markdown("##### Starting Lineup Settings")
    lineup_type = st.radio("Grid Lineup Type", ["Default 22-Driver Grid Preset", "Custom Setup"])

    availability_placeholder = st.empty()
    drivers_to_simulate = []
    
    if lineup_type == "Default 22-Driver Grid Preset":
        for idx, preset in enumerate(modern_grid_presets):
            grid_pos = st.number_input(
                f"{driver_mapping.get(preset['driver'], preset['driver'])} Grid Position", 
                min_value=1, max_value=22, value=idx+1, key=f"grid_pos_{idx}"
            )
            drivers_to_simulate.append({
                **preset,
                "grid": grid_pos,
                "qual_position": grid_pos,
                "qual_gap_to_pole": (grid_pos - 1) * 0.15
            })
            
    else:
        num_drivers = st.slider("Number of Drivers to Custom Configure", min_value=2, max_value=22, value=10)
        for idx in range(num_drivers):
            st.markdown(f"**Driver #{idx+1} Settings**")
            drv = st.selectbox("Select Driver", list(driver_mapping.keys()), key=f"cust_drv_{idx}", index=min(idx, len(driver_mapping)-1))
            const = st.selectbox("Select Team", list(constructor_mapping.keys()), key=f"cust_const_{idx}", index=min(idx, len(constructor_mapping)-1))
            grid = st.slider(f"Grid Position", min_value=1, max_value=22, value=idx+1, key=f"cust_grid_{idx}")
            
            col_qual1, col_qual2 = st.columns(2)
            with col_qual1:
                qual_pos = st.number_input(f"Qualifying Position", min_value=1, max_value=22, value=grid, key=f"cust_qualpos_{idx}")
            with col_qual2:
                qual_gap = st.slider(f"Gap to Pole (seconds)", min_value=0.0, max_value=5.0, value=float((grid - 1) * 0.15), step=0.01, key=f"cust_qualgap_{idx}")
                
            age = st.slider("Age", min_value=17, max_value=46, value=27, key=f"cust_age_{idx}")
            
            # Form
            col_pts, col_w = st.columns(2)
            with col_pts:
                d_pts = st.number_input("Driver Pts", min_value=0, value=50, key=f"cust_dpts_{idx}")
            with col_w:
                d_rec = st.slider("Recent Podiums (Last 3)", 0, 3, 1, key=f"cust_drec_{idx}")
                
            drivers_to_simulate.append({
                "driver": drv,
                "constructor": const,
                "grid": grid,
                "qual_position": qual_pos,
                "qual_gap_to_pole": qual_gap,
                "age": age,
                "driver_pts": d_pts,
                "driver_wins": 1 if d_pts > 25 else 0,
                "constructor_pts": d_pts + 30,
                "constructor_wins": 1,
                "driver_recent": d_rec,
                "constructor_recent": d_rec + 1
            })

    # Update availability placeholder dynamically
    grid_positions = [d['grid'] for d in drivers_to_simulate]
    total_slots = len(drivers_to_simulate)
    all_positions = set(range(1, total_slots + 1))
    taken_positions = set(grid_positions)
    available_positions = sorted(list(all_positions - taken_positions))
    
    if available_positions:
        availability_placeholder.info(f"📋 **Available Starting Slots:** {', '.join(['P'+str(p) for p in available_positions])}")
    else:
        availability_placeholder.success(f"✅ **All starting slots (P1 to P{total_slots}) successfully assigned!**")

with col_preview:
    st.markdown("### 🏁 Live Simulation & Forecast Results")
    st.write("")
    # Check for duplicate grid positions
    grid_positions = [d['grid'] for d in drivers_to_simulate]
    duplicate_grids = sorted(list(set([g for g in grid_positions if grid_positions.count(g) > 1])))
    
    if duplicate_grids:
        st.warning(f"⚠️ **Duplicate Grid Positions Detected!** Positions {', '.join(['P'+str(g) for g in duplicate_grids])} are assigned to multiple drivers. Please ensure each grid position is unique.")
        sim_clicked = st.button("🏁 Run Race Simulation", type="primary", disabled=True)
    else:
        sim_clicked = st.button("🏁 Run Race Simulation", type="primary")
    
    if sim_clicked and drivers_to_simulate:
        # Run predictions for all drivers
        results = []
        for d in drivers_to_simulate:
            d_encoded = le_driver.transform([d['driver']])[0]
            c_encoded = le_constructor.transform([d['constructor']])[0]
            
            # Features in exact order:
            # ['grid', 'qual_position', 'qual_gap_to_pole', 'driver_encoded', ... ]
            features = np.array([[
                d['grid'],
                d.get('qual_position', d['grid']),
                d.get('qual_gap_to_pole', 0.0),
                d_encoded,
                c_encoded,
                selected_circuit,
                d['age'],
                d['driver_pts'],
                d['driver_wins'],
                d['constructor_pts'],
                d['constructor_wins'],
                d['driver_recent'],
                d['constructor_recent']
            ]])
            
            prob = model.predict_proba(features)[0][1]
            results.append({
                "name": driver_mapping.get(d['driver'], d['driver']),
                "team": constructor_mapping.get(d['constructor'], d['constructor']),
                "grid": d['grid'],
                "probability": prob
            })
            
        # Sort results by probability descending
        results_sorted = sorted(results, key=lambda x: x['probability'], reverse=True)
        
        # Build a beautiful Podium Standings (P1, P2, P3)
        st.markdown("#### Predicted Podium Standings")
        
        p1 = results_sorted[0] if len(results_sorted) > 0 else None
        p2 = results_sorted[1] if len(results_sorted) > 1 else None
        p3 = results_sorted[2] if len(results_sorted) > 2 else None
        
        # HTML visualizer for podium cards
        p2_html = f"""
            <div class="podium-card p2">
                <div class="rank-badge silver-text">2nd</div>
                <div style="font-weight: 700; font-size: 1.2rem;">{p2['name']}</div>
                <div style="font-size: 0.85rem; color: #8b9bb4; margin-bottom: 0.5rem;">{p2['team']}</div>
                <div style="font-size: 0.85rem;">Started P{p2['grid']}</div>
                <div class="probability-badge">{p2['probability']*100:.1f}%</div>
            </div>
        """ if p2 else ""

        p1_html = f"""
            <div class="podium-card p1">
                <div class="rank-badge gold-text">1st</div>
                <div style="font-weight: 700; font-size: 1.4rem;">{p1['name']}</div>
                <div style="font-size: 0.9rem; color: #ffd700; margin-bottom: 0.5rem;">{p1['team']}</div>
                <div style="font-size: 0.85rem;">Started P{p1['grid']}</div>
                <div class="probability-badge" style="background: rgba(255,215,0,0.15); color: #ffd700;">{p1['probability']*100:.1f}%</div>
            </div>
        """ if p1 else ""

        p3_html = f"""
            <div class="podium-card p3">
                <div class="rank-badge bronze-text">3rd</div>
                <div style="font-weight: 700; font-size: 1.2rem;">{p3['name']}</div>
                <div style="font-size: 0.85rem; color: #8b9bb4; margin-bottom: 0.5rem;">{p3['team']}</div>
                <div style="font-size: 0.85rem;">Started P{p3['grid']}</div>
                <div class="probability-badge">{p3['probability']*100:.1f}%</div>
            </div>
        """ if p3 else ""
        
        st.markdown(f"""
            <div class="podium-container">
                {p2_html}
                {p1_html}
                {p3_html}
            </div>
        """, unsafe_allow_html=True)
        
        # Display full detailed sorted table
        st.markdown("#### Full Field Analysis")
        detailed_df = pd.DataFrame(results_sorted)
        detailed_df.columns = ["Driver", "Team / Constructor", "Grid Position", "Podium Probability"]
        detailed_df["Podium Probability"] = detailed_df["Podium Probability"].apply(lambda x: f"{x*100:.1f}%")
        detailed_df.index = detailed_df.index + 1
        st.table(detailed_df)
        
    else:
        # Instruction state
        st.markdown("""
            <div style="background: rgba(255,255,255,0.01); padding: 5rem 2rem; border-radius: 16px; border: 1px dashed rgba(255,255,255,0.1); text-align: center;">
                <span style="font-size: 4rem;">🏎️</span>
                <h3 style="color: #8b9bb4; margin-top: 1.5rem; margin-bottom: 0.5rem;">Awaiting Green Flag</h3>
                <p style="color: #5c6b84; max-width: 450px; margin: 0 auto;">
                    Adjust the race starting configurations on the left sidebar, and click <b>Run Race Simulation</b> to calculate podium finish likelihoods for the grid.
                </p>
            </div>
        """, unsafe_allow_html=True)
