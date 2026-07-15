import os
import pandas as pd
import numpy as np

class SafeLabelEncoder:
    def __init__(self, default_val=0):
        self.default_val = default_val
        self.classes_ = np.array([])
        self.mapping = {}
        
    def fit(self, y):
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)
        self.mapping = {val: idx for idx, val in enumerate(self.classes_)}
        return self
        
    def transform(self, y):
        y_arr = np.asarray(y)
        return np.array([self.mapping.get(val, self.default_val) for val in y_arr])
        
    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

def time_to_seconds(t_str):
    if pd.isna(t_str) or not isinstance(t_str, str) or t_str.strip() == '' or t_str.strip() == '\\N':
        return np.nan
    try:
        parts = t_str.strip().split(':')
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 1:
            return float(parts[0])
    except:
        return np.nan
    return np.nan

def clean_dataframe(df):
    """Replaces SQL null placeholder '\\N' with NaN."""
    df.replace(r'\N', np.nan, inplace=True)
    df.replace('\\N', np.nan, inplace=True)
    return df

def load_raw_data(data_dir):
    """Loads all necessary F1 dataset tables and cleans SQL nulls."""
    results = clean_dataframe(pd.read_csv(os.path.join(data_dir, "results.csv")))
    races = clean_dataframe(pd.read_csv(os.path.join(data_dir, "races.csv")))
    drivers = clean_dataframe(pd.read_csv(os.path.join(data_dir, "drivers.csv")))
    constructors = clean_dataframe(pd.read_csv(os.path.join(data_dir, "constructors.csv")))
    qualifying = clean_dataframe(pd.read_csv(os.path.join(data_dir, "qualifying.csv")))
    
    # Optional files (might not exist in simple setups, but load if present)
    circuits_path = os.path.join(data_dir, "circuits.csv")
    circuits = clean_dataframe(pd.read_csv(circuits_path)) if os.path.exists(circuits_path) else None
    
    pit_stops_path = os.path.join(data_dir, "pit_stops.csv")
    pit_stops = clean_dataframe(pd.read_csv(pit_stops_path)) if os.path.exists(pit_stops_path) else None
    
    return results, races, drivers, constructors, qualifying, circuits, pit_stops

def preprocess_qualifying_times(qualifying):
    """Parses qualifying lap times into seconds and computes gap to pole."""
    qualifying = qualifying.copy()
    for col in ['q1', 'q2', 'q3']:
        qualifying[col + '_sec'] = qualifying[col].apply(time_to_seconds)
        
    qualifying['best_qual_time'] = qualifying[['q1_sec', 'q2_sec', 'q3_sec']].min(axis=1)
    pole_times = qualifying.groupby('raceId')['best_qual_time'].transform('min')
    qualifying['qual_gap_to_pole'] = qualifying['best_qual_time'] - pole_times
    return qualifying

def engineer_podium_features(results, races, drivers, constructors, qualifying):
    """Applies complete feature engineering pipeline for the podium classifier."""
    # Preprocess qualifying times
    qualifying = preprocess_qualifying_times(qualifying)
    
    # Convert types
    results = results.copy()
    results['grid'] = pd.to_numeric(results['grid'], errors='coerce')
    results['positionOrder'] = pd.to_numeric(results['positionOrder'], errors='coerce')
    results['points'] = pd.to_numeric(results['points'], errors='coerce')
    
    races = races.copy()
    races['year'] = pd.to_numeric(races['year'], errors='coerce')
    races['round'] = pd.to_numeric(races['round'], errors='coerce')
    races['date'] = pd.to_datetime(races['date'], errors='coerce')
    
    drivers = drivers.copy()
    drivers['dob'] = pd.to_datetime(drivers['dob'], errors='coerce')
    
    # Merge datasets
    df = results.merge(races[['raceId', 'year', 'round', 'circuitId', 'date']], on='raceId', how='inner')
    df = df.merge(drivers[['driverId', 'driverRef', 'dob', 'nationality']], on='driverId', how='inner')
    df = df.merge(constructors[['constructorId', 'constructorRef']], on='constructorId', how='inner')
    df = df.merge(qualifying[['raceId', 'driverId', 'position', 'qual_gap_to_pole']].rename(columns={'position': 'qual_position'}), on=['raceId', 'driverId'], how='left')
    
    df['qual_position'] = df['qual_position'].fillna(df['grid'])
    df['qual_gap_to_pole'] = df['qual_gap_to_pole'].fillna(5.0)
    
    # Sort chronologically to engineer prior history features
    df.sort_values(by=['date', 'round', 'positionOrder'], inplace=True)
    
    # Target definition
    df['podium_finish'] = (df['positionOrder'] <= 3).astype(int)
    
    # Feature engineering
    df['driver_age'] = (df['date'] - df['dob']).dt.days / 365.25
    df['win'] = (df['positionOrder'] == 1).astype(int)
    
    # Driver prior stats in the season
    df['driver_prior_pts_season'] = df.groupby(['year', 'driverId'])['points'].cumsum() - df['points']
    df['driver_prior_wins_season'] = df.groupby(['year', 'driverId'])['win'].cumsum() - df['win']
    
    # Constructor prior stats in the season
    df['constructor_prior_pts_season'] = df.groupby(['year', 'constructorId'])['points'].cumsum() - df['points']
    df['constructor_prior_wins_season'] = df.groupby(['year', 'constructorId'])['win'].cumsum() - df['win']
    
    # Rolling recent form (rolling podium count in previous 3 races overall)
    driver_history = df.sort_values('date').groupby('driverId')
    df['driver_recent_podiums'] = driver_history['podium_finish'].shift(1).rolling(3, min_periods=1).sum().fillna(0)
    
    constructor_history = df.sort_values('date').groupby('constructorId')
    df['constructor_recent_podiums'] = constructor_history['podium_finish'].shift(1).rolling(3, min_periods=1).sum().fillna(0)
    
    return df
