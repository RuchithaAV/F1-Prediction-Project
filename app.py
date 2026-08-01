import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib

# Set Page Config
st.set_page_config(
    page_title="F1 Prediction Hub",
    page_icon="🏎️",
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

# Main Title Banner
st.markdown("""
<div class="title-banner">
    <h1> <span class="f1-accent">F1</span> Prediction Hub</h1>
    <p style="color: #8b9bb4; font-size: 1.1rem; margin-top: 0.5rem; margin-bottom: 0;">
        Real-Time Formula 1 Podium Simulations & Strategy Analytics 
    </p>
</div>
""", unsafe_allow_html=True)
# Import SafeLabelEncoder so joblib can deserialize it
from utils import load_raw_data, engineer_podium_features, SafeLabelEncoder

@st.cache_resource
def load_models():
    # Load Podium Simulator assets
    model = joblib.load('models/f1_podium_model.joblib')
    le_driver = joblib.load('models/le_driver.joblib')
    le_constructor = joblib.load('models/le_constructor.joblib')
    
    # Load Pit Stop Predictor assets
    pit_model = joblib.load('models/f1_pit_stop_model.joblib')
    le_driver_pit = joblib.load('models/le_driver_pit.joblib')
    le_constructor_pit = joblib.load('models/le_constructor_pit.joblib')
    le_race_pit = joblib.load('models/le_race_pit.joblib')
    return model, le_driver, le_constructor, pit_model, le_driver_pit, le_constructor_pit, le_race_pit

@st.cache_data
def load_and_process_data(data_dir):
    results_df, races_df, drivers_df, constructors_df, qualifying_df, circuits_df, _ = load_raw_data(data_dir)
    
    # Filter to only drivers and constructors active from 2020 to 2024
    races_20_24 = races_df[(races_df['year'] >= 2020) & (races_df['year'] <= 2024)]
    active_race_ids = races_20_24['raceId']
    
    active_driver_ids = results_df[results_df['raceId'].isin(active_race_ids)]['driverId'].unique()
    active_constructor_ids = results_df[results_df['raceId'].isin(active_race_ids)]['constructorId'].unique()
    active_circuit_ids = races_20_24['circuitId'].unique()
    
    drivers_df = drivers_df[drivers_df['driverId'].isin(active_driver_ids)].copy()
    constructors_df = constructors_df[constructors_df['constructorId'].isin(active_constructor_ids)].copy()
    circuits_df = circuits_df[circuits_df['circuitId'].isin(active_circuit_ids)].copy()
    
    # Compute typical/max laps per circuit for stint visualization using modern races (2020-2024)
    results_df['laps'] = pd.to_numeric(results_df['laps'], errors='coerce')
    race_laps = results_df.groupby('raceId')['laps'].max().reset_index()
    race_circuit_map = races_20_24[['raceId', 'circuitId']].merge(race_laps, on='raceId')
    circuit_laps_dict = race_circuit_map.groupby('circuitId')['laps'].median().round().astype(int).to_dict()
    
    # Compute valid driver-constructor pairs (driverRef, constructorRef)
    results_modern = results_df[results_df['raceId'].isin(active_race_ids)]
    driver_const_pairs = results_modern.merge(drivers_df, on='driverId').merge(constructors_df, on='constructorId')
    
    # Sort by year descending to find the most recent team for each driver
    driver_const_pairs_sorted = driver_const_pairs.merge(races_df[['raceId', 'year']], on='raceId').sort_values('year', ascending=False)
    most_recent_constructor = driver_const_pairs_sorted.drop_duplicates(subset=['driverRef']).set_index('driverRef')['constructorRef'].to_dict()
    
    # Add historical driver mapping presets explicitly if missing
    most_recent_constructor['vettel'] = 'aston_martin'
    most_recent_constructor['latifi'] = 'williams'
    
    # Precompute active drivers by year (2020-2024)
    active_drivers_by_year = {}
    for y in range(2020, 2025):
        races_y = races_df[races_df['year'] == y]
        active_driver_ids_y = results_df[results_df['raceId'].isin(races_y['raceId'])]['driverId'].unique()
        active_drivers_by_year[y] = set(drivers_df[drivers_df['driverId'].isin(active_driver_ids_y)]['driverRef'])
        
    # Precompute all F1 features for all races (2020-2024)
    df_feat = engineer_podium_features(results_df, races_df, drivers_df, constructors_df, qualifying_df)
    df_features = df_feat[(df_feat['year'] >= 2020) & (df_feat['year'] <= 2024)].copy()
        
    return drivers_df, constructors_df, circuits_df, circuit_laps_dict, most_recent_constructor, active_drivers_by_year, df_features, races_df, results_df

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f1 dataset")

try:
    model, le_driver, le_constructor, pit_model, le_driver_pit, le_constructor_pit, le_race_pit = load_models()
    drivers_df, constructors_df, circuits_df, circuit_laps_dict, most_recent_constructor, active_drivers_by_year, df_features, races_df, results_df = load_and_process_data(DATA_DIR)
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

# Helper to check if driver was active in a given year
def is_driver_active(driver_ref, year):
    # Clamp/default year to historical range
    y = year
    if y < 2020:
        y = 2020
    elif y > 2024:
        y = 2024
    return driver_ref in active_drivers_by_year.get(y, set())

# Presets of driver configurations (Grid positions, age, stats)
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

# Organize page into tabs
tab1, tab2, tab3 = st.tabs(["Podium Simulator", "Pit Stop Predictor", "Clustering & Analytics"])

with tab1:
    col_setup, col_preview = st.columns([1, 2], gap="large")
    
    with col_setup:
        st.markdown("### Podium Simulation Setup")
        
        # Season selection
        podium_year = st.selectbox(
            "Season / Year",
            options=[2024, 2023, 2022, 2021, 2020],
            key="podium_year"
        )
        
        # Track selection - only show tracks that hosted a race in podium_year
        races_in_year = races_df[races_df['year'] == podium_year]
        circuit_ids_in_year = races_in_year['circuitId'].unique()
        circuits_in_year_df = circuits_df[circuits_df['circuitId'].isin(circuit_ids_in_year)]
        circuit_mapping_in_year = circuits_in_year_df.set_index('circuitId')['name'].to_dict()
        
        selected_circuit = st.selectbox(
            "Circuit (Race Track)",
            options=list(circuit_mapping_in_year.keys()),
            format_func=lambda x: f"{circuit_mapping_in_year[x]} ({circuits_df.loc[circuits_df['circuitId'] == x, 'country'].values[0]})"
        )
        
        # Query actual historical grid from df_features
        historical_race_grid = df_features[
            (df_features['year'] == podium_year) & 
            (df_features['circuitId'] == selected_circuit)
        ].copy()
        
        # Sort by grid position
        historical_race_grid = historical_race_grid.sort_values(by='grid')
        
        presets_list = []
        for r in historical_race_grid.to_dict('records'):
            presets_list.append({
                "driver": r['driverRef'],
                "constructor": r['constructorRef'],
                "age": int(r['driver_age']),
                "driver_pts": float(r['driver_prior_pts_season']),
                "driver_wins": int(r['driver_prior_wins_season']),
                "constructor_pts": float(r['constructor_prior_pts_season']),
                "constructor_wins": int(r['constructor_prior_wins_season']),
                "driver_recent": int(r['driver_recent_podiums']),
                "constructor_recent": int(r['constructor_recent_podiums']),
                "grid": int(r['grid']),
                "qual_position": int(r['qual_position']),
                "qual_gap_to_pole": float(r['qual_gap_to_pole']),
                "driver_recent_avg_finish": float(r.get('driver_recent_avg_finish', 10.0)),
                "driver_recent_avg_qual": float(r.get('driver_recent_avg_qual', 10.0)),
                "constructor_recent_avg_finish": float(r.get('constructor_recent_avg_finish', 10.0)),
                "driver_circuit_avg_finish": float(r.get('driver_circuit_avg_finish', 10.0)),
                "constructor_circuit_avg_finish": float(r.get('constructor_circuit_avg_finish', 10.0)),
                "driver_season_dnf_rate": float(r.get('driver_season_dnf_rate', 0.1)),
                "constructor_season_dnf_rate": float(r.get('constructor_season_dnf_rate', 0.1))
            })
            
        # Presets
        st.markdown("##### Starting Lineup Settings")
        lineup_type = st.radio("Grid Lineup Type", ["Default Grid Preset (Actual Grid)", "Custom Setup"])

        availability_placeholder = st.empty()
        drivers_to_simulate = []
        
        if lineup_type == "Default Grid Preset (Actual Grid)":
            if not presets_list:
                st.warning(" No historical grid data available for this race.")
            else:
                for idx, preset in enumerate(presets_list):
                    default_grid_val = int(preset['grid'])
                    if default_grid_val <= 0:
                        default_grid_val = len(presets_list)
                    # Clamp value to [1, len(presets_list)] to be safe
                    default_grid_val = max(1, min(default_grid_val, len(presets_list)))
                    
                    grid_pos = st.number_input(
                        f"{driver_mapping.get(preset['driver'], preset['driver'])} Grid Position", 
                        min_value=1, max_value=max(1, len(presets_list)), value=default_grid_val, key=f"grid_pos_{idx}"
                    )
                    drivers_to_simulate.append({
                        **preset,
                        "grid": grid_pos
                    })
                
        else:
            num_drivers = st.slider("Number of Drivers to Custom Configure", min_value=2, max_value=22, value=10)
            active_driver_refs = sorted(list(active_drivers_by_year.get(podium_year, set())))
            if not active_driver_refs:
                active_driver_refs = list(driver_mapping.keys())
                
            for idx in range(num_drivers):
                st.markdown(f"**Driver #{idx+1} Settings**")
                drv = st.selectbox(
                    "Select Driver",
                    active_driver_refs,
                    format_func=lambda x: f"{driver_mapping.get(x, x)} ({'Active' if is_driver_active(x, podium_year) else 'Retired/Inactive'})",
                    key=f"cust_drv_{idx}_{podium_year}",
                    index=min(idx, len(active_driver_refs)-1)
                )
                
                # Fetch default stats for this driver in this season
                driver_season_stats = df_features[
                    (df_features['year'] == podium_year) & 
                    (df_features['driverRef'] == drv)
                ]
                
                if not driver_season_stats.empty:
                    latest_stat = driver_season_stats.sort_values(by='round', ascending=False).iloc[0]
                    default_const = latest_stat['constructorRef']
                    default_age = int(latest_stat['driver_age'])
                    default_pts = float(latest_stat['driver_prior_pts_season'])
                    default_recent = int(latest_stat['driver_recent_podiums'])
                    d_rec_avg_fin = float(latest_stat.get('driver_recent_avg_finish', 10.0))
                    d_rec_avg_qual = float(latest_stat.get('driver_recent_avg_qual', 10.0))
                    c_rec_avg_fin = float(latest_stat.get('constructor_recent_avg_finish', 10.0))
                    d_cir_avg_fin = float(latest_stat.get('driver_circuit_avg_finish', 10.0))
                    c_cir_avg_fin = float(latest_stat.get('constructor_circuit_avg_finish', 10.0))
                    d_dnf = float(latest_stat.get('driver_season_dnf_rate', 0.1))
                    c_dnf = float(latest_stat.get('constructor_season_dnf_rate', 0.1))
                else:
                    default_const = most_recent_constructor.get(drv, list(constructor_mapping.keys())[0])
                    default_age = 27
                    default_pts = 50.0
                    default_recent = 1
                    d_rec_avg_fin = 10.0
                    d_rec_avg_qual = 10.0
                    c_rec_avg_fin = 10.0
                    d_cir_avg_fin = 10.0
                    c_cir_avg_fin = 10.0
                    d_dnf = 0.1
                    c_dnf = 0.1
                
                st.session_state[f"cust_const_disp_{idx}_{podium_year}"] = constructor_mapping[default_const]
                st.text_input("Team", disabled=True, key=f"cust_const_disp_{idx}_{podium_year}")
                
                grid = st.slider(f"Grid Position", min_value=1, max_value=22, value=idx+1, key=f"cust_grid_{idx}_{podium_year}")
                
                col_qual1, col_qual2 = st.columns(2)
                with col_qual1:
                    qual_pos = st.number_input(f"Qualifying Position", min_value=1, max_value=22, value=grid, key=f"cust_qualpos_{idx}_{podium_year}")
                with col_qual2:
                    qual_gap = st.slider(f"Gap to Pole (seconds)", min_value=0.0, max_value=5.0, value=float((grid - 1) * 0.15), step=0.01, key=f"cust_qualgap_{idx}_{podium_year}")
                    
                age = st.slider("Age", min_value=17, max_value=46, value=default_age, key=f"cust_age_{idx}_{podium_year}")
                
                col_pts, col_w = st.columns(2)
                with col_pts:
                    d_pts = st.number_input("Driver Pts", min_value=0.0, value=default_pts, key=f"cust_dpts_{idx}_{podium_year}")
                with col_w:
                    d_rec = st.slider("Recent Podiums (Last 3)", 0, 3, default_recent, key=f"cust_drec_{idx}_{podium_year}")
                    
                drivers_to_simulate.append({
                    "driver": drv,
                    "constructor": default_const,
                    "grid": grid,
                    "qual_position": qual_pos,
                    "qual_gap_to_pole": qual_gap,
                    "age": age,
                    "driver_pts": d_pts,
                    "driver_wins": 1 if d_pts > 25 else 0,
                    "constructor_pts": d_pts + 30,
                    "constructor_wins": 1,
                    "driver_recent": d_rec,
                    "constructor_recent": d_rec + 1,
                    "driver_recent_avg_finish": d_rec_avg_fin,
                    "driver_recent_avg_qual": d_rec_avg_qual,
                    "constructor_recent_avg_finish": c_rec_avg_fin,
                    "driver_circuit_avg_finish": d_cir_avg_fin,
                    "constructor_circuit_avg_finish": c_cir_avg_fin,
                    "driver_season_dnf_rate": d_dnf,
                    "constructor_season_dnf_rate": c_dnf
                })

        # Warning notice for inactive/retired drivers
        inactive_drivers = [driver_mapping.get(d['driver'], d['driver']) for d in drivers_to_simulate if not is_driver_active(d['driver'], podium_year)]
        if inactive_drivers:
            st.warning(f"**Retired/Inactive Drivers Selected:** {', '.join(inactive_drivers)} did not race in the {podium_year} season. Simulations for these drivers will rely on historical extrapolation.")

        # Update availability placeholder dynamically
        grid_positions = [d['grid'] for d in drivers_to_simulate]
        total_slots = len(drivers_to_simulate)
        all_positions = set(range(1, total_slots + 1))
        taken_positions = set(grid_positions)
        available_positions = sorted(list(all_positions - taken_positions))
        
        if available_positions:
            availability_placeholder.info(f"**Available Starting Slots:** {', '.join(['P'+str(p) for p in available_positions])}")
        else:
            availability_placeholder.success(f"**All starting slots (P1 to P{total_slots}) successfully assigned!**")

    with col_preview:
        st.markdown("### Live Simulation & Forecast Results")
        st.write("")
        
        # Check for duplicate grid positions
        grid_positions = [d['grid'] for d in drivers_to_simulate]
        duplicate_grids = sorted(list(set([g for g in grid_positions if grid_positions.count(g) > 1])))
        
        if duplicate_grids:
            st.warning(f"**Duplicate Grid Positions Detected!** Positions {', '.join(['P'+str(g) for g in duplicate_grids])} are assigned to multiple drivers.")
            sim_clicked = st.button("Run Race Simulation", type="primary", disabled=True)
        else:
            sim_clicked = st.button("Run Race Simulation", type="primary")
        
        if sim_clicked and drivers_to_simulate:
            # Run predictions for all drivers
            results = []
            for d in drivers_to_simulate:
                d_encoded = le_driver.transform([d['driver']])[0]
                c_encoded = le_constructor.transform([d['constructor']])[0]
                
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
                    d['constructor_recent'],
                    d.get('driver_recent_avg_finish', 10.0),
                    d.get('driver_recent_avg_qual', 10.0),
                    d.get('constructor_recent_avg_finish', 10.0),
                    d.get('driver_circuit_avg_finish', 10.0),
                    d.get('constructor_circuit_avg_finish', 10.0),
                    d.get('driver_season_dnf_rate', 0.1),
                    d.get('constructor_season_dnf_rate', 0.1)
                ]])
                
                prob = model.predict_proba(features)[0][1]
                results.append({
                    "name": driver_mapping.get(d['driver'], d['driver']),
                    "team": constructor_mapping.get(d['constructor'], d['constructor']),
                    "grid": d['grid'],
                    "probability": prob
                })
                
            results_sorted = sorted(results, key=lambda x: x['probability'], reverse=True)
            
            st.markdown("#### Predicted Podium Standings")
            p1 = results_sorted[0] if len(results_sorted) > 0 else None
            p2 = results_sorted[1] if len(results_sorted) > 1 else None
            p3 = results_sorted[2] if len(results_sorted) > 2 else None
            
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
            
            st.markdown("#### Full Field Analysis")
            detailed_df = pd.DataFrame(results_sorted)
            detailed_df.columns = ["Driver", "Team / Constructor", "Grid Position", "Podium Probability"]
            detailed_df["Podium Probability"] = detailed_df["Podium Probability"].apply(lambda x: f"{x*100:.1f}%")
            detailed_df.index = detailed_df.index + 1
            st.table(detailed_df)
            
            # Display Feature Importance and model stats Expander
            if os.path.exists('models/feature_importance.png'):
                with st.expander("Model Insights & Feature Importance", expanded=False):
                    st.image('models/feature_importance.png', caption='Podium Predictor Feature Importance Breakdown')
                    st.markdown("""
                    **Model Performance Summary:**
                    * **Algorithm:** Tuned Random Forest Classifier (Optimized via `RandomizedSearchCV`)
                    * **Evaluation Strategy:** Temporal Split (Train: 2000-2021, Test: 2022+ to prevent data leakage)
                    * **Metrics on Holdout Test Set:**
                      * **Accuracy:** ~80%+
                      * **ROC-AUC:** ~0.85+
                    * **Baseline Comparison:** Logistic Regression Baseline achieved an ROC-AUC of ~0.74. The tuned Random Forest model significantly outperforms it by capturing non-linear relationships (e.g. exponential finish advantage for front row starters).
                    """)
            
        else:
            st.markdown("""
                <div style="background: rgba(255,255,255,0.01); padding: 5rem 2rem; border-radius: 16px; border: 1px dashed rgba(255,255,255,0.1); text-align: center;">
                    <span style="font-size: 4rem;">🏁</span>
                    <h3 style="color: #8b9bb4; margin-top: 1.5rem; margin-bottom: 0.5rem;">Awaiting Green Flag</h3>
                    <p style="color: #5c6b84; max-width: 450px; margin: 0 auto;">
                        Adjust the race starting configurations on the left sidebar, and click <b>Run Race Simulation</b> to calculate podium finish likelihoods for the grid.
                    </p>
                </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("### Pit Stop Lap Predictor")
    st.markdown("""
        Predict which lap a driver will make their pit stop based on the circuit, starting grid position, the specific stop number (1st or 2nd), and historical performance.
    """)
    
    col_pit_setup, col_pit_preview = st.columns([1, 2], gap="large")
    
    with col_pit_setup:
        st.markdown("##### Predictor Setup")
        
        current_year = st.session_state.get("pit_year", 2024)
        
        # Filter tracks dynamically based on selected year (defaulting to 2024 for out of range seasons)
        races_in_year_pit = races_df[races_df['year'] == current_year]
        if races_in_year_pit.empty:
            races_in_year_pit = races_df[races_df['year'] == 2024]
            
        circuit_ids_in_year_pit = races_in_year_pit['circuitId'].unique()
        circuits_in_year_pit_df = circuits_df[circuits_df['circuitId'].isin(circuit_ids_in_year_pit)]
        circuit_mapping_in_year_pit = circuits_in_year_pit_df.set_index('circuitId')['name'].to_dict()
        
        pit_circuit = st.selectbox(
            "Circuit (Track)",
            options=list(circuit_mapping_in_year_pit.keys()),
            format_func=lambda x: f"{circuit_mapping_in_year_pit[x]} ({circuits_df.loc[circuits_df['circuitId'] == x, 'country'].values[0]})",
            key=f"pit_circuit_{current_year}"
        )
        
        # Preserve selected driver across year changes when the key changes
        driver_list = list(driver_mapping.keys())
        prev_driver = st.session_state.get("selected_pit_driver", driver_list[0])
        try:
            default_idx = driver_list.index(prev_driver)
        except ValueError:
            default_idx = 0
            
        pit_driver = st.selectbox(
            "Driver",
            options=driver_list,
            index=default_idx,
            format_func=lambda x: f"{driver_mapping[x]} ({'Active' if is_driver_active(x, current_year) else 'Retired/Inactive'})",
            key=f"pit_driver_{current_year}"
        )
        st.session_state["selected_pit_driver"] = pit_driver
        
        # Auto-fill constructor based on chosen driver (disabled field to prevent editing)
        # Setting session state before rendering ensures Streamlit reactively updates the value
        default_pit_const = most_recent_constructor.get(pit_driver, list(constructor_mapping.keys())[0])
        pit_constructor = default_pit_const
        st.session_state["pit_constructor_disp"] = constructor_mapping[default_pit_const]
        st.text_input("Constructor (Team)", disabled=True, key="pit_constructor_disp")
        
        # Check driver activity for selected season
        is_active = is_driver_active(pit_driver, current_year)
        if not is_active:
            st.error(f" **Blocked:** {driver_mapping[pit_driver]} is retired/inactive in the {current_year} season. Predictions cannot be computed for inactive drivers in that year.")
            
        pit_grid = st.slider("Starting Grid Position", min_value=1, max_value=22, value=1, key="pit_grid")
        pit_stop_num = st.selectbox("Pit Stop Number", options=[1, 2], format_func=lambda x: f"Stop #{x}", key="pit_stop_num")
        pit_year = st.number_input("Season / Year", min_value=2018, max_value=2026, value=2024, key="pit_year")

    with col_pit_preview:
        st.markdown("##### Prediction Output")
        
        if not is_active:
            run_prediction = st.button("Predict Pit Stop Lap", type="primary", disabled=True)
        else:
            run_prediction = st.button("Predict Pit Stop Lap", type="primary")
        
        if run_prediction:
            try:
                driver_name = driver_mapping[pit_driver]
                constructor_ref = pit_constructor
                race_name = circuit_mapping[pit_circuit]
                
                d_encoded = le_driver_pit.transform([driver_name])[0] if driver_name in le_driver_pit.classes_ else 0
                c_encoded = le_constructor_pit.transform([constructor_ref])[0] if constructor_ref in le_constructor_pit.classes_ else 0
                r_encoded = le_race_pit.transform([race_name])[0] if race_name in le_race_pit.classes_ else 0
                
                total_laps = int(circuit_laps_dict.get(pit_circuit, 58))
                
                features_pit = np.array([[
                    d_encoded,
                    c_encoded,
                    r_encoded,
                    pit_grid,
                    pit_stop_num,
                    pit_year,
                    total_laps
                ]])
                
                predicted_pct = pit_model.predict(features_pit)[0]
                predicted_lap = int((predicted_pct / 100) * total_laps)
                
                if predicted_lap > total_laps:
                    predicted_lap = total_laps - 2
                if predicted_lap <= 0:
                    predicted_lap = 10
                
                pct = int((predicted_lap / total_laps) * 100)
                
                st.markdown(f"""
                    <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 2rem; text-align: center; margin-top: 1rem;">
                        <h4 style="margin: 0; color: #8b9bb4;">Predicted Pit Stop Lap</h4>
                        <div style="font-size: 4.5rem; font-weight: 900; color: #ff1801; margin: 1rem 0;">Lap {predicted_lap}</div>
                        <p style="color: #c5c6c7; margin-bottom: 1.5rem;">
                            During the <b>{race_name}</b> ({total_laps} total laps), <b>{driver_name}</b> starting from <b>P{pit_grid}</b> is simulated to pit for <b>Stop #{pit_stop_num}</b> on <b>Lap {predicted_lap}</b>.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("##### Stint Length Progress")
                st.progress(pct / 100)
                st.caption(f"Lap {predicted_lap} out of {total_laps} total laps ({pct}% of race distance)")
                
                st.markdown("#####  Strategy Insights")
                if pit_stop_num == 1:
                    if pct < 30:
                        st.info("**Aggressive Undercut Strategy**: The pitter is stopping early. Likely starting on Soft tyres looking to swap to Hard/Medium to leapfrog cars ahead.")
                    elif pct > 50:
                        st.info("**Overcut / Long Stint Strategy**: The pitter is running extremely long. Likely started on Hard tyres and seeking to build a tyre age advantage for a late sprint on Soft/Medium.")
                    else:
                        st.info("**Standard Strategy**: The pitter is on a balanced target stint. Standard Medium-to-Hard one-stop strategy window.")
                else:
                    st.info("**Second Stop / Sprint Stint**: The pitter is stopping for a second time, likely moving to a softer compound to complete a fast final stint or reacting to high degradation.")
                    
            except Exception as e:
                st.error(f"Error executing prediction: {e}")
                
        else:
            st.markdown("""
                <div style="background: rgba(255,255,255,0.01); padding: 5rem 2rem; border-radius: 16px; border: 1px dashed rgba(255,255,255,0.1); text-align: center; margin-top: 1rem;">
                    <span style="font-size: 4rem;">⏱️</span>
                    <h3 style="color: #8b9bb4; margin-top: 1.5rem; margin-bottom: 0.5rem;">Awaiting Pit Simulation</h3>
                    </p>
                </div>
            """, unsafe_allow_html=True)

with tab3:
    st.markdown("### Unsupervised Clustering & PCA Analytics")
    st.markdown("""
        Explore how drivers and circuits are grouped using **K-Means Clustering** and **Principal Component Analysis (PCA)** based on historical performance metrics.
    """)
    
    sub_tab1, sub_tab2 = st.tabs(["Driver Cohorts", "Circuit Cohorts"])
    
    with sub_tab1:
        st.markdown("#### F1 Driver Clustering Profiles (2018-2024)")
        
        try:
            drv_features_df = joblib.load("models/driver_features.joblib")
            
            # Label mappings
            cluster_label_map = {
                3: 'Elite Champions',
                1: 'Strong Performers', 
                2: 'Midfield Racers',
                0: 'Backmarkers'
            }
            drv_features_df['cluster_label'] = drv_features_df['cluster'].map(cluster_label_map)
            
            available_labels = ['Elite Champions', 'Strong Performers', 'Midfield Racers', 'Backmarkers']
            selected_label = st.selectbox("Select Driver Cohort", options=available_labels)
            
            # Filter drivers
            cohort_drivers = drv_features_df[drv_features_df['cluster_label'] == selected_label].copy()
            
            # Explanation
            if "Elite Champions" in selected_label:
                st.success("**Elite Champions:** High win and podium rates, dominant finishing positions, and low DNF/retirement rates (e.g. Verstappen, Hamilton, Leclerc).")
            elif "Strong Performers" in selected_label:
                st.info("**Strong Performers:** Highly consistent point scorers and podium finishers supporting elite teams (e.g. Perez, Sainz, Russell).")
            elif "Midfield Racers" in selected_label:
                st.warning("**Midfield Racers:** Standard midfield operators who secure points on occasion but generally operate outside the top-6 podium battles.")
            else:
                st.error("**Backmarkers:** Drivers with lower average finishing positions, higher retirement rates, or shorter F1 career durations.")
                
            st.markdown(f"##### Drivers in {selected_label} ({len(cohort_drivers)} total)")
            
            # Format and display
            display_cols = ['driver_name', 'avg_finish_pos', 'avg_grid_pos', 'wins', 'podiums', 'dnf_rate']
            cohort_display = cohort_drivers[display_cols].copy()
            cohort_display.columns = ["Driver Name", "Avg Finish Position", "Avg Grid Position", "Total Wins", "Total Podiums", "DNF Rate"]
            cohort_display["Avg Finish Position"] = cohort_display["Avg Finish Position"].round(2)
            cohort_display["Avg Grid Position"] = cohort_display["Avg Grid Position"].round(2)
            cohort_display["DNF Rate"] = cohort_display["DNF Rate"].apply(lambda x: f"{x*100:.1f}%")
            
            st.dataframe(cohort_display, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"Could not load driver clustering features: {e}")
            
    with sub_tab2:
        st.markdown("#### F1 Circuit Clustering Profiles (2018-2024)")
        st.markdown("""
            Circuits are clustered based on race variance, DNF counts, positions gained, and average pit stops:
        """)
        
        # Precomputed list of circuit clusters to avoid heavy calculation in Streamlit run
        circuit_clusters_precomputed = {
            "Classic/Established Circuits": [
                {"name": "Albert Park Grand Prix Circuit", "country": "Australia"},
                {"name": "Bahrain International Circuit", "country": "Bahrain"},
                {"name": "Circuit de Barcelona-Catalunya", "country": "Spain"},
                {"name": "Circuit de Monaco", "country": "Monaco"},
                {"name": "Silverstone Circuit", "country": "UK"},
                {"name": "Hungaroring", "country": "Hungary"},
                {"name": "Autodromo Nazionale di Monza", "country": "Italy"},
                {"name": "Autódromo José Carlos Pace", "country": "Brazil"},
                {"name": "Suzuka Circuit", "country": "Japan"},
                {"name": "Yas Marina Circuit", "country": "UAE"},
                {"name": "Autódromo Hermanos Rodríguez", "country": "Mexico"},
                {"name": "Red Bull Ring", "country": "Austria"}
            ],
            "Modern/Recent Street Circuits": [
                {"name": "Istanbul Park", "country": "Turkey"},
                {"name": "Circuit Gilles Villeneuve", "country": "Canada"},
                {"name": "Circuit de Spa-Francorchamps", "country": "Belgium"},
                {"name": "Marina Bay Street Circuit", "country": "Singapore"},
                {"name": "Shanghai International Circuit", "country": "China"},
                {"name": "Autodromo Enzo e Dino Ferrari", "country": "Italy"},
                {"name": "Circuit Paul Ricard", "country": "France"},
                {"name": "Circuit Park Zandvoort", "country": "Netherlands"},
                {"name": "Circuit of the Americas", "country": "USA"},
                {"name": "Sochi Autodrom", "country": "Russia"},
                {"name": "Baku City Circuit", "country": "Azerbaijan"},
                {"name": "Jeddah Corniche Circuit", "country": "Saudi Arabia"},
                {"name": "Losail International Circuit", "country": "Qatar"},
                {"name": "Miami International Autodrome", "country": "USA"},
                {"name": "Las Vegas Strip Street Circuit", "country": "United States"}
            ],
            "Temporary/COVID Calendar Additions": [
                {"name": "Hockenheimring", "country": "Germany"},
                {"name": "Nürburgring", "country": "Germany"},
                {"name": "Autódromo Internacional do Algarve", "country": "Portugal"},
                {"name": "Autodromo Internazionale del Mugello", "country": "Italy"}
            ]
        }
        
        selected_c_cohort = st.selectbox("Select Circuit Cohort", options=list(circuit_clusters_precomputed.keys()))
        
        if selected_c_cohort == "Classic/Established Circuits":
            st.success("**Classic/Established Circuits:** Highly-raced tracks with long calendar histories. They accumulate the highest DNF counts and feature consistent pit strategies.")
        elif selected_c_cohort == "Modern/Recent Street Circuits":
            st.info("**Modern/Recent Street Circuits:** Newer tracks or street circuits with high barrier risks and lower historic race sample sizes.")
        else:
            st.warning("**Temporary/COVID Calendar Additions:** Pandemic-era additions with unique strategy patterns or erratic pit stop counts.")
            
        c_list = circuit_clusters_precomputed[selected_c_cohort]
        c_df = pd.DataFrame(c_list)
        c_df.columns = ["Circuit Track Name", "Country"]
        st.dataframe(c_df, use_container_width=True, hide_index=True)


