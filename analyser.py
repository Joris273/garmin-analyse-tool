import sys
import os
import datetime
import random
import pickle
import hashlib
from typing import List, Optional, Tuple, Dict, Any
import xml.etree.ElementTree as ET 

import streamlit as st
from streamlit.web import cli as stcli
from streamlit.runtime.scriptrunner import get_script_run_ctx

import pandas as pd
import altair as alt
import numpy as np
from garminconnect import Garmin

# Versuch, fitparse zu importieren (für .fit Dateien)
try:
    import fitparse
    FITPARSE_AVAILABLE = True
except ImportError:
    FITPARSE_AVAILABLE = False

# --- KONSTANTEN ---
# CACHE_FILE = "garmin_cache.pkl" # Deprecated: Now dynamic per user
# Buffer für ACWR Berechnung (Chronic Load braucht 28 Tage Vorlauf + Puffer)
ACWR_BUFFER_DAYS = 45 # Erhöht auf 45 für sicheren CTL Anlauf (42 Tage Span)

# --- 1. AUTOMATISCHER STARTER ---
if __name__ == "__main__":
    if not get_script_run_ctx():
        print("Starte Streamlit Server...")
        sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
        sys.exit(stcli.main())

# --- SETUP & KONFIGURATION ---
try:
    st.set_page_config(page_title="Garmin Pro Analytics", page_icon="🚴", layout="wide")
except Exception:
    pass

# --- WISSENSCHAFTLICHE BERECHNUNGEN (CORE LOGIC - KORRIGIERT) ---

def calculate_trimp_vectorized(duration_min: pd.Series, avg_hr: pd.Series, max_hr_user: int) -> pd.Series:
    """
    Berechnet den TRIMP (Training Impulse) nach Banister (Vektorisiert).
    Formel: Dauer(min) * Intensität * 0.64 * exp(1.92 * Intensität)
    Scientific Fix: Faktor 0.64 (für Männer Standard) hinzugefügt.
    """
    if max_hr_user <= 0:
        return pd.Series(0.0, index=duration_min.index)
    
    # Intensität (Heart Rate Reserve wäre genauer, aber MaxHR ist hier der Standard)
    intensity = avg_hr / max_hr_user
    intensity = intensity.fillna(0).clip(lower=0)
    
    # Banister Gewichtungsfaktor (1.92 für Männer, 1.67 für Frauen - hier Default 1.92)
    # Fix: Faktor 0.64 ergänzt für korrekte Skalierung
    weighting = 0.64 * np.exp(1.92 * intensity)
    
    return duration_min * intensity * weighting

def calculate_zones_vectorized(df: pd.DataFrame, user_max_hr: int) -> pd.Series:
    """
    Bestimmt die Trainingszone intelligent unter Berücksichtigung von Variabilität (VI).
    Vollständig vektorisiert mit NumPy für Performance.
    """
    if user_max_hr <= 0:
        return pd.Series(0, index=df.index)

    avg_hr = df['HF'].fillna(0)
    max_hr_activity = df['MaxHF'].fillna(0)
    avg_power = df['Leistung'].fillna(0)
    norm_power = df['NormPower'].fillna(0)

    # Prozent vom Max Puls
    avg_pct = avg_hr / user_max_hr
    
    # Fallback: Wenn MaxHF der Aktivität fehlt, nimm AvgHF (vermeidet NaN)
    max_pct = np.where(max_hr_activity > 0, max_hr_activity / user_max_hr, avg_pct)

    # 1. Basis-Klassifizierung nach HF Durchschnitt
    conditions = [
        (avg_pct < 0.60),
        (avg_pct < 0.75),
        (avg_pct < 0.85),
        (avg_pct < 0.95)
    ]
    choices = [0, 1, 2, 3] # Z1, Z2, Z3, Z4. Else = 4 (Z5)
    
    base_zone = np.select(conditions, choices, default=4)

    # 2. Variabilitäts-Index (VI) berechnen
    # Vermeide Division durch Null
    vi = np.where(avg_power > 10, norm_power / avg_power, 1.0)
    
    # 3. Intelligente Upgrades (Vektorisiert)
    # Logik: Wenn Puls-Spitzen oder hohe Variabilität (Intervalle), dann Zone hochstufen
    
    # Upgrade Regel 1: Hohe Max-HF -> Mindestens Z4 (Zone 3)
    # (max_pct > 0.92) und (bisher < Z4) -> Upgrade auf Z4
    upgrade_max_hr = (max_pct > 0.92) & (base_zone < 3)
    base_zone = np.where(upgrade_max_hr, 3, base_zone)

    # Upgrade Regel 2: Sehr hoher VI -> Mindestens Z4
    upgrade_vi_high = (vi > 1.15) & (base_zone < 3)
    base_zone = np.where(upgrade_vi_high, 3, base_zone)

    # Upgrade Regel 3: Moderater VI -> Mindestens Z3
    upgrade_vi_mod = (vi > 1.08) & (base_zone < 2)
    base_zone = np.where(upgrade_vi_mod, 2, base_zone)

    return pd.Series(base_zone, index=df.index).astype(int)

# --- NEUE WISSENSCHAFTLICHE FUNKTIONEN (ADDITIV & KORRIGIERT) ---

def calculate_pmc_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet CTL, ATL und TSB für das Performance Management Chart.
    Nutzt TRIMP (Stress) als Basis.
    Scientific Fix: Nutzung von alpha=1/42 statt span=42.
    """
    if df.empty:
        return pd.DataFrame()

    # 1. Resampling auf tägliche Basis (auffüllen fehlender Tage mit 0)
    daily = df.set_index('Datum').resample('D')['Stress'].sum().fillna(0).to_frame()
    
    # 2. Berechnung EWMA (Exponential Weighted Moving Average)
    # CTL = Fitness (42 Tage), ATL = Fatigue (7 Tage)
    # Fix: Alpha statt Span für exakte Zeitkonstante nach Coggan
    daily['CTL'] = daily['Stress'].ewm(alpha=1/42, adjust=False).mean()
    daily['ATL'] = daily['Stress'].ewm(alpha=1/7, adjust=False).mean()
    
    # 3. TSB = Form (Training Stress Balance)
    daily['TSB'] = daily['CTL'] - daily['ATL']
    
    daily.reset_index(inplace=True)
    return daily

def calculate_monotony_strain(df: pd.DataFrame, window_days: int = 7) -> Tuple[float, float, bool]:
    """
    Berechnet Monotonie und Strain nach Foster für die letzten X Tage.
    Monotonie = Durchschnittl. täglicher Load / Standardabweichung.
    Rückgabe: (Monotonie, Strain, Warnung_Flag)
    """
    if df.empty:
        return 0.0, 0.0, False

    daily = df.set_index('Datum').resample('D')['Stress'].sum().fillna(0)
    recent = daily.tail(window_days)
    
    avg_load = recent.mean()
    std_load = recent.std()
    
    if std_load == 0:
        monotony = 0.0 # Vermeidung Div/0 bei nur einem Wert oder identischen Werten
    else:
        monotony = avg_load / std_load
        
    strain = recent.sum() * monotony
    
    # Warnung wenn Monotonie > 2.0 (Gefahr von Overtraining/Stagnation)
    warning = monotony > 2.0 and recent.sum() > 200 # Nur warnen wenn auch Last da ist
    
    return round(monotony, 2), round(strain, 0), warning

# --- DATEI-PARSER FÜR DEEP DIVE (MODIFIZIERT FÜR SPEED) ---

def parse_tcx(file) -> pd.DataFrame:
    """Parst eine .tcx Datei und extrahiert Sekunden-Daten inkl. Speed."""
    try:
        tree = ET.parse(file)
        root = tree.getroot()
        # TCX Namespace Handling ist oft tricky, wir versuchen es generisch
        ns = {'ns': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'}
        
        data = []
        for trackpoint in root.findall('.//ns:Trackpoint', ns):
            point = {}
            # Zeit
            time_elem = trackpoint.find('ns:Time', ns)
            if time_elem is not None:
                point['timestamp'] = time_elem.text
            
            # Herzfrequenz
            hr_elem = trackpoint.find('.//ns:HeartRateBpm/ns:Value', ns)
            if hr_elem is not None:
                point['heart_rate'] = int(hr_elem.text)
                
            # Erweiterungen (Watt, Cadence) sind oft in Extensions
            # Wir suchen einfach rekursiv nach Tags, da die Struktur variieren kann (TPX)
            # Watt
            watts_elem = trackpoint.find('.//{*}Watts')
            if watts_elem is not None:
                point['power'] = float(watts_elem.text)
            
            # Cadence
            cad_elem = trackpoint.find('ns:Cadence', ns)
            if cad_elem is not None:
                point['cadence'] = int(cad_elem.text)

            # Speed (Extensions usually contain TPX)
            # Nutze Wildcard {*}, um Namespace-Probleme zu umgehen
            for ext in trackpoint.findall('.//{*}TPX/{*}Speed'):
                if ext is not None:
                    point['speed'] = float(ext.text)
            
            data.append(point)
            
        df = pd.DataFrame(data)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"Fehler beim Parsen der TCX: {e}")
        return pd.DataFrame()

def parse_fit(file) -> pd.DataFrame:
    """Parst eine .fit Datei mittels fitparse."""
    if not FITPARSE_AVAILABLE:
        st.error("Bibliothek 'fitparse' fehlt. Bitte `pip install fitparse` ausführen.")
        return pd.DataFrame()
    
    try:
        fitfile = fitparse.FitFile(file)
        data = []
        for record in fitfile.get_messages("record"):
            row = {}
            for field in record:
                if field.name in ['timestamp', 'heart_rate', 'power', 'cadence', 'speed', 'altitude']:
                    row[field.name] = field.value
            data.append(row)
        
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        st.error(f"Fehler beim Parsen der FIT: {e}")
        return pd.DataFrame()

def calculate_power_curve(df_stream: pd.DataFrame) -> pd.DataFrame:
    """Berechnet die Mean Max Power Curve (MMP) für diverse Zeitfenster."""
    if 'power' not in df_stream.columns or 'timestamp' not in df_stream.columns:
        return pd.DataFrame()
    
    # BUGFIX: Entfernen doppelter Zeitstempel, bevor Resampling stattfindet
    # "cannot reindex on an axis with duplicate labels"
    df_stream = df_stream.drop_duplicates(subset=['timestamp'], keep='first')

    # Sicherstellen, dass wir Sekunden-Daten haben (Resample auf 1s)
    df_stream = df_stream.set_index('timestamp').resample('1S').ffill().reset_index()
    
    windows = [1, 5, 10, 30, 60, 180, 300, 600, 1200, 3600] # Sekunden
    results = []
    
    series = df_stream['power']
    
    for w in windows:
        if len(series) >= w:
            mmp = series.rolling(window=w).mean().max()
            # FIX: Check auf NaN, bevor in int konvertiert wird
            if pd.notna(mmp):
                results.append({'Dauer_Sek': w, 'Watt': int(mmp)})
            
    return pd.DataFrame(results)

def calculate_aerobic_decoupling(df_stream: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Berechnet die aerobe Entkopplung (Pw:HR) aka Aerobic Decoupling.
    SCIENTIFIC FIX:
    1. Nutzt Moving Time (Speed > 0.5 m/s) statt "Power > 0".
       Grund: Rollphasen (Coasting) gehören zur physiologischen Einheit dazu.
       Werden sie entfernt (wie vorher), wird die Efficiency künstlich stabilisiert.
    2. Nutzt Average Power statt Normalized Power für das Ratio.
       Grund: Bei hoher Variabilität (NP > Avg) und Cardiac Drift korreliert
       Average Power besser mit der tatsächlichen physiologischen Entkopplung
       (Referenz: Joe Friel Standard für lange Einheiten mit Pausen).
    """
    # Validierung
    if 'power' not in df_stream.columns or 'heart_rate' not in df_stream.columns or 'timestamp' not in df_stream.columns:
        return 0.0, 0.0, 0.0

    # 1. Kontinuierliche Zeitachse herstellen (wichtig für Gaps/Autopause)
    df = df_stream.drop_duplicates(subset=['timestamp'], keep='first').set_index('timestamp')
    df = df.resample('1S').asfreq().reset_index()
    
    # Gaps füllen
    df['power'] = df['power'].fillna(0)
    df['heart_rate'] = df['heart_rate'].ffill()
    
    # Speed für Moving Time Ermittlung
    if 'speed' in df.columns:
        df['speed'] = df['speed'].fillna(0)
    else:
        # Fallback wenn kein Speed da (z.B. Indoor ohne Virtual Speed): Nehme Power > 0
        df['speed'] = np.where(df['power'] > 0, 1.0, 0.0)

    # 2. Moving Time Filter (Standard: > 0.5 m/s oder ~1.8 km/h)
    # Wir behalten Power=0 (Coasting), solange Speed > 0 ist!
    df_active = df[df['speed'] > 0.5].copy()
    
    if len(df_active) < 600: # Minimum 10 Minuten Daten für sinnvolle Berechnung
        return 0.0, 0.0, 0.0

    # Split in zwei Hälften der Moving Time
    midpoint = len(df_active) // 2
    p1 = df_active.iloc[:midpoint]
    p2 = df_active.iloc[midpoint:]

    # Helper: Nutze Average Power für bessere Vergleichbarkeit bei Drift
    def get_ratio(sub_df):
        if len(sub_df) == 0: return 0
        avg_pwr = sub_df['power'].mean()
        avg_hr = sub_df['heart_rate'].mean()
        if avg_hr == 0: return 0
        return avg_pwr / avg_hr

    ratio1 = get_ratio(p1)
    ratio2 = get_ratio(p2)

    if ratio1 == 0: return 0.0, 0.0, 0.0

    # Decoupling Berechnung: (Ratio1 - Ratio2) / Ratio1
    # Wenn der Puls steigt (Drift) bei gleichen Watt (oder Watt stärker sinken als Puls),
    # wird Ratio2 kleiner -> Decoupling positiv.
    decoupling = (ratio1 - ratio2) / ratio1 * 100

    return round(decoupling, 2), round(ratio1, 2), round(ratio2, 2)

# --- CACHE MANAGEMENT (ROBUST) ---

def get_cache_filename(email: str) -> str:
    """Erstellt einen sicheren, anonymen Dateinamen für den Cache basierend auf der E-Mail."""
    if not email:
        return "garmin_cache_unknown.pkl"
    # E-Mail normalisieren und hashen
    email_hash = hashlib.sha256(email.strip().lower().encode('utf-8')).hexdigest()
    return f"garmin_cache_{email_hash}.pkl"

def load_local_cache(email: str) -> List[Dict]:
    """Lädt die lokalen Rohdaten für den spezifischen Nutzer, falls vorhanden."""
    cache_file = get_cache_filename(email)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            pass # Korrupter Cache oder Fehler
    return []

def save_local_cache(data: List[Dict], email: str):
    """Speichert Rohdaten binär für den spezifischen Nutzer."""
    cache_file = get_cache_filename(email)
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Cache Save Error: {e}")

def get_latest_activity_date(activities: List[Dict]) -> Optional[datetime.date]:
    """Findet das Datum der letzten Aktivität im Cache."""
    if not activities:
        return None
    try:
        # Robustes Parsen der Startzeiten
        dates = [act.get('startTimeLocal', '1970-01-01 00:00:00') for act in activities]
        max_date_str = max(dates)
        return datetime.datetime.strptime(max_date_str.split(' ')[0], "%Y-%m-%d").date()
    except Exception:
        return None

def merge_activities(old_data: List[Dict], new_data: List[Dict]) -> List[Dict]:
    """Upsert-Strategie basierend auf activityId."""
    if not old_data and not new_data:
        return []
    
    data_map = {act.get('activityId'): act for act in old_data if act.get('activityId')}
    
    for act in new_data:
        aid = act.get('activityId')
        if aid:
            data_map[aid] = act # Überschreibt existierende IDs (Update)
            
    merged_list = list(data_map.values())
    # Sortierung nach Datum wichtig für ACWR Rolling Calculation
    merged_list.sort(key=lambda x: x.get('startTimeLocal', ''))
    return merged_list

# --- DATEN-EXTRAKTION & VERARBEITUNG ---

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_garmin_raw(email: str, password: str, start_date_str: str, end_date_str: str) -> Tuple[List[Dict], Optional[str]]:
    """
    Verbindet mit Garmin Connect.
    NOTE: Wir übergeben Strings statt date-Objekte, damit st.cache_data besser hasht.
    """
    try:
        client = Garmin(email, password)
        client.login()
        
        # WICHTIG: activity_type=None holt alle Typen. "" kann manchmal filtern.
        activities = client.get_activities_by_date(start_date_str, end_date_str, None)
        
        return activities, None
    except Exception as e:
        return [], str(e)

def process_data(raw_activities: List[Dict[str, Any]], user_max_hr: int) -> pd.DataFrame:
    """Konvertiert Raw-Dicts in DataFrame und berechnet Metriken."""
    if not raw_activities:
        return pd.DataFrame()

    key_map = {
        'Leistung': ['avgPower', 'averagePower', 'normPower'],
        'NormPower': ['normPower', 'weightedMeanPower', 'normalizedPower'],
        'Max20Min': ['max20MinPower', 'maximum20MinPower', 'twentyMinPower'],
        'HF': ['avgHR', 'averageHR', 'avgHeartRate', 'averageHeartRate'],
        'MaxHF': ['maxHR', 'maxHeartRate', 'maximumHeartRate'],
        'Kalorien': ['calories', 'totalCalories'],
        'Distanz_Raw': ['distance'],
        'Anstieg': ['totalAscent', 'elevationGain'],
        'Dauer_Sec': ['duration']
    }

    indoor_keywords_type = ['indoor', 'virtual', 'e-sport']
    indoor_keywords_name = ['zwift', 'indoor', 'rolle', 'trainer', 'virtual', 'bkool', 'rouvy', 'tacx', 'wahoo']

    extracted_data = []

    for activity in raw_activities:
        act_type_dict = activity.get('activityType', {})
        act_type_key = act_type_dict.get('typeKey', 'unknown').lower() if act_type_dict else 'unknown'
        act_name = activity.get('activityName', 'Unbekannt') or "Unbekannt"
        
        # Filter: Nur Radsport-relevante Aktivitäten
        valid_types = ['cycling', 'biking', 'ride', 'gravel', 'mtb', 'virtual_ride', 'road_biking']
        if not any(x in act_type_key for x in valid_types):
            continue

        is_indoor = any(k in act_type_key for k in indoor_keywords_type)
        if not is_indoor:
            is_indoor = any(k in act_name.lower() for k in indoor_keywords_name)

        row = {
            'Datum': activity.get('startTimeLocal', '').split(' ')[0],
            'Aktivität': act_name,
            'Indoor': is_indoor,
            'ActivityID': activity.get('activityId')
        }
        
        for target_col, candidates in key_map.items():
            val = None
            for key in candidates:
                if key in activity and activity[key] is not None:
                    val = activity[key]
                    break
            row[target_col] = val

        if row['NormPower'] is None and row['Leistung'] is not None:
            row['NormPower'] = row['Leistung']

        extracted_data.append(row)

    if not extracted_data:
        return pd.DataFrame()

    df = pd.DataFrame(extracted_data)
    
    numeric_cols = ['Leistung', 'NormPower', 'Max20Min', 'HF', 'MaxHF', 'Kalorien', 'Distanz_Raw', 'Anstieg', 'Dauer_Sec']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['Datum'] = pd.to_datetime(df['Datum'], errors='coerce')
    df.dropna(subset=['Datum'], inplace=True)
    df.sort_values('Datum', inplace=True)

    df['Dauer_Min'] = (df['Dauer_Sec'] / 60).round(1)
    df['Distanz'] = (df['Distanz_Raw'] / 1000).round(1)
    
    mask_valid = (
        (df['HF'] > 0) & 
        ((df['Leistung'] > 40) | (df['NormPower'] > 40)) & 
        (df['Dauer_Min'] > 5)
    )
    df = df[mask_valid].copy()

    if df.empty:
        return pd.DataFrame()

    # --- WISSENSCHAFTLICHE BERECHNUNGEN ---
    df['Stress'] = calculate_trimp_vectorized(df['Dauer_Min'], df['HF'], user_max_hr).round(1)
    df['EF'] = np.where(df['HF'] > 0, df['NormPower'] / df['HF'], 0)
    df['EF'] = df['EF'].round(2)
    df['ZoneIdx'] = calculate_zones_vectorized(df, user_max_hr)
    
    zone_labels = {0: "Z1 (Erholung)", 1: "Z2 (Grundlage)", 2: "Z3 (Tempo)", 3: "Z4 (Schwelle)", 4: "Z5 (Max)"}
    df['Zone'] = df['ZoneIdx'].map(zone_labels)

    cols_to_int = ['Leistung', 'NormPower', 'Max20Min', 'HF', 'MaxHF', 'Kalorien', 'Anstieg']
    for col in cols_to_int:
        df[col] = df[col].astype(int)

    return df

def generate_demo_data(days: int = 120, user_max_hr: int = 161) -> pd.DataFrame:
    """Generiert synthetische Daten für Demo-Modus."""
    random.seed(42)
    data = []
    today = datetime.date.today()
    total_days = days + 45 # Buffer erhöht
    
    for i in range(total_days):
        if random.random() > 0.6: continue 
        
        date = today - datetime.timedelta(days=total_days-i)
        cycle_pos = (i % 28) / 28
        load_factor = 0.5 + (cycle_pos * 0.8)
        if cycle_pos > 0.8: load_factor = 0.4
        
        ride_type = random.choice(['LIT', 'LIT', 'MIT', 'HIT'])
        
        duration = 60
        avg_hr = int(user_max_hr * 0.7)
        power = 150 + (i * 0.1) 
        
        is_indoor_sim = (ride_type == 'HIT') or (random.random() > 0.75)
        
        if is_indoor_sim:
            act_type_key = 'virtual_ride'
            act_name_prefix = "Zwift: "
            thermal_drift = 5 
        else:
            act_type_key = 'cycling'
            act_name_prefix = "Outdoor "
            thermal_drift = 0

        if ride_type == 'LIT': 
            duration = random.randint(90, 180) * load_factor
            avg_hr = int(user_max_hr * 0.65) + random.randint(-5, 5) + thermal_drift
            power = 160 + (i * 0.1)
            norm_power = power * 1.02
            max_hr_activity = avg_hr + 20
        elif ride_type == 'MIT': 
            duration = random.randint(60, 90)
            avg_hr = int(user_max_hr * 0.83) + random.randint(-5, 5) + thermal_drift
            power = 200 + (i * 0.2)
            norm_power = power * 1.05
            max_hr_activity = avg_hr + 15
        else: 
            duration = random.randint(45, 70)
            avg_hr = int(user_max_hr * 0.88) + random.randint(-5, 5) + thermal_drift
            power = 240 + (i * 0.3)
            norm_power = power * 1.18 
            max_hr_activity = user_max_hr - random.randint(0, 5)

        raw_act = {
            'startTimeLocal': f"{date} 10:00:00",
            'activityName': f"{act_name_prefix}{ride_type} Training",
            'activityType': {'typeKey': act_type_key},
            'avgPower': power,
            'normPower': norm_power,
            'max20MinPower': power * 1.1,
            'avgHR': avg_hr,
            'maxHR': max_hr_activity,
            'duration': duration * 60,
            'calories': duration * 10,
            'distance': (duration/60) * 30 * 1000, 
            'totalAscent': 0 if is_indoor_sim else 500,
            'activityId': i + 10000 
        }
        data.append(raw_act)
        
    return process_data(data, user_max_hr)

# --- UI LAYOUT ---

with st.sidebar:
    st.header("⚙️ Setup")
    tab_login, tab_params = st.tabs(["Daten", "Parameter"])
    
    with tab_login:
        st.info("🔒 **Datenschutz:** Deine Zugangsdaten werden **nur** für die Verbindung zu Garmin genutzt und **nicht gespeichert**.")
        email = st.text_input("Garmin E-Mail")
        password = st.text_input("Passwort", type="password")
        
        st.markdown("### 1. Zeitraum wählen")
        range_options = {
            "Letzte 30 Tage": 30,
            "Letzte 90 Tage": 90,
            "Letzte 180 Tage": 180,
            "Letzte 365 Tage": 365,
            "Dieses Jahr": "cy",
            "Letztes Jahr": "ly"
        }
        selected_range = st.selectbox("Datenbasis", list(range_options.keys()), index=1)
        
        today = datetime.date.today()
        val = range_options[selected_range]
        if val == "cy":
            start_date = datetime.date(today.year, 1, 1)
            end_date = today
        elif val == "ly":
            start_date = datetime.date(today.year - 1, 1, 1)
            end_date = datetime.date(today.year - 1, 12, 31)
        else:
            start_date = today - datetime.timedelta(days=int(val))
            end_date = today
        
        st.markdown("### 💾 Datenspeicher")
        use_cache = st.checkbox("Lokalen Cache nutzen", value=True, help="Speichert Aktivitäten lokal. Lädt nur neue Daten nach.")
        
        c1, c2 = st.columns(2)
        start_btn = c1.button("Start", type="primary")
        demo_btn = c2.button("Demo")

        if st.button("Cache leeren"):
            if email:
                c_file = get_cache_filename(email)
                if os.path.exists(c_file):
                    os.remove(c_file)
                    st.toast("Dein Cache wurde gelöscht!", icon="🗑️")
                else:
                    st.toast("Kein Cache für diesen Account gefunden.")
            else:
                st.error("Bitte gib zuerst eine E-Mail-Adresse ein.")
        
    with tab_params:
        st.subheader("2. Analyse-Fokus")
        user_max_hr = st.number_input("Max Herzfrequenz", 100, 220, 161, help="Beeinflusst Zonen & Stress-Score (TRIMP).")
        
        env_mode = st.radio(
            "Umgebung / Filter", 
            ["Alle", "Nur Outdoor", "Nur Indoor"], 
            horizontal=True
        )

        power_metric_display = st.radio(
            "Leistungs-Metrik",
            ["Normalized Power (NP)", "Durchschnitts-Leistung"],
            index=0
        )
        power_col = 'NormPower' if "Normalized" in power_metric_display else 'Leistung'

        # State Init
        if 'df' not in st.session_state: st.session_state.df = None
        if 'raw_data' not in st.session_state: st.session_state.raw_data = None
        if 'mode' not in st.session_state: st.session_state.mode = None 

        days_diff = (end_date - start_date).days
        if st.session_state.df is not None and not st.session_state.df.empty:
             min_dt = st.session_state.df['Datum'].min().date()
             max_dt = st.session_state.df['Datum'].max().date()
             days_diff = (max_dt - min_dt).days
        
        weeks_total = max(1, days_diff // 7)
        max_possible_weeks = max(2, int(weeks_total / 2))
        default_weeks = min(4, max_possible_weeks)
        
        comparison_weeks = st.slider(
            "Fenstergröße (Wochen)", 
            min_value=1, 
            max_value=max_possible_weeks if max_possible_weeks > 1 else 2, 
            value=default_weeks if default_weeks <= max_possible_weeks else 1
        )
        
        target_hr = st.slider("Aerobe Schwelle (Vergleichs-Puls)", 100, 170, 135)
        hr_tol = st.slider("Toleranz (+/- bpm)", 2, 15, 5)
        
        # --- NEW: Monotony Logic in Sidebar ---
        if st.session_state.df is not None and not st.session_state.df.empty:
            monotony, strain_val, mono_warning = calculate_monotony_strain(st.session_state.df, 7)
            if mono_warning:
                st.error(f"⚠️ **Monotonie-Alarm ({monotony}):** Training zu eintönig! Risiko für Overtraining.")
            elif monotony > 1.5:
                st.warning(f"ℹ️ **Hohe Monotonie ({monotony}):** Variiere Intensität mehr.")

st.title("🚴 Garmin Science Lab V16 (Deep Dive Edition)")
st.markdown("Analyse von **Effizienz**, **Belastung (PMC/ACWR)** und **Wissenschaftlicher Trainingsverteilung**.")

# --- WISSENSCHAFTLICHER GUIDE (MASTERCLASS) ---
with st.expander("📘 Knowledge Base: Sportwissenschaftliche Hintergründe (Masterclass)", expanded=False):
    st.markdown("""
    ### 🎓 Dein Labor-Handbuch
    Dieses Dashboard nutzt wissenschaftlich validierte Modelle (Banister, Coggan, Gabbett, Seiler) zur Leistungsdiagnostik.
    Hier verstehst du, was die Metriken bedeuten und wie du sie zur Steuerung nutzt.
    """)
    
    g_tab1, g_tab2, g_tab3, g_tab4, g_tab5, g_tab6 = st.tabs(["🧬 Physiologie & Effizienz (EF)", "⚖️ Belastungs-Steuerung (ACWR)", "🚀 PMC (Form)", "📈 Training Stress (TRIMP)", "🎯 Zonen-Modelle", "🔬 Deep Dive Metriken"])
    
    with g_tab1:
        st.markdown("#### Der Efficiency Factor (EF)")
        st.info("""
        **Definition:** Der EF misst deinen "Output pro Herzschlag". Er ist der wichtigste Indikator für **aerobe Fitness**.
        Ähnlich wie der Benzinverbrauch beim Auto (Liter pro 100km) wollen wir bei gleicher Leistung (Watt) weniger Herzschläge verbrauchen.
        """)
        
        c1, c2 = st.columns(2)
        with c1:
            st.latex(r"EF = \frac{\text{Normalized Power (NP)}}{\text{Ø Herzfrequenz (Avg HR)}}")
            st.markdown("""
            **Interpretation:**
            * **Steigender Trend:** Deine aerobe Fitness verbessert sich. Du trittst mehr Watt bei gleichem Puls.
            * **Stagnation:** Zeit für neue Trainingsreize (z.B. Block-Periodisierung).
            * **Abfall:** Mögliches Übertraining oder Krankheit.
            """)
        with c2:
            st.warning("""
            **Aerobe Entkopplung (Decoupling):**
            Bei langen Fahrten (>2h) steigt der Puls oft langsam an, obwohl die Wattzahl gleich bleibt (Cardiac Drift). 
            Eine Entkopplung von **< 5%** gilt als Zeichen exzellenter Grundlagenausdauer.
            """)

    with g_tab2:
        st.markdown("#### ACWR: Acute:Chronic Workload Ratio")
        st.markdown("""
        Das von **Dr. Tim Gabbett** entwickelte Modell zur Verletzungsprävention. Es vergleicht, was du *kurzfristig* getan hast (Ermüdung), mit dem, was du *langfristig* gewohnt bist (Fitness).
        """)
        
        col_math, col_int = st.columns([1, 1])
        with col_math:
            st.latex(r"ACWR = \frac{\text{Acute Load (Ø 7 Tage)}}{\text{Chronic Load (Ø 28 Tage)}}")
            st.caption("Ein Wert von 1.0 bedeutet: Du trainierst diese Woche genau so viel, wie du es im Durchschnitt gewohnt bist.")
        
        with col_int:
            st.markdown("**Die Zonen:**")
            st.success("**0.8 - 1.3 (Sweet Spot):** Optimaler Bereich für Formaufbau bei minimalem Risiko.")
            st.warning("**1.3 - 1.5 (High Risk):** 'Overreaching'. Hohes Risiko für Verletzungen oder Krankheit, wenn dieser Zustand lange anhält.")
            st.error("**> 1.5 (Danger Zone):** Die Belastung steigt viel schneller als die Anpassung des Körpers (Sehnen, Bänder). Akute Gefahr!")

    with g_tab3:
        st.markdown("#### Performance Management Chart (PMC)")
        st.markdown("""
        Das Standard-Tool im Radsport (ähnlich TrainingPeaks), um "Form" zu berechnen.
        """)
        st.latex(r"TSB = CTL (Fitness) - ATL (Fatigue)")
        st.markdown("""
        * **CTL (Fitness):** Deine chronische Trainingslast (Last 42 Tage). Diesen Wert willst du langfristig steigern.
        * **ATL (Fatigue):** Deine akute Ermüdung (Last 7 Tage).
        * **TSB (Form):** Deine "Frische".
            * **Positiv (+10 bis +25):** Tapering / Rennbereit.
            * **Neutral (-10 bis +10):** Normales Training.
            * **Negativ (<-20):** Harter Trainingsblock (Overloading).
        """)

    with g_tab4:
        st.markdown("#### TRIMP (Training Impulse)")
        st.markdown("""
        Warum zählen wir nicht einfach Kilometer? Weil 100km locker nicht denselben Stress erzeugen wie 100km Rennen.
        **TRIMP (nach Banister)** quantifiziert die physiologische Last unter Berücksichtigung der **exponentiellen** Natur von Laktatbildung.
        """)
        
        st.latex(r"TRIMP = t \cdot HR_{ratio} \cdot 0.64 \cdot e^{1.92 \cdot HR_{ratio}}")
        st.caption("wobei $t$ = Dauer in Minuten und $HR_{ratio}$ = % der HfMax.")
        
        st.markdown("""
        * **Lockere Fahrt (Z1/Z2):** Wenig Punkte, da der exponentielle Faktor klein ist.
        * **Schwellentraining (Z4):** Hohe Punkte, da der Faktor stark ansteigt.
        * **Nutzen:** TRIMP ist die Basis für alle Belastungskurven (Fitness vs. Fatigue).
        """)

    with g_tab5:
        st.markdown("#### Trainingsverteilung: Polarized vs. Pyramidal")
        st.markdown("Wie viel Zeit solltest du in welcher Zone verbringen? Zwei Modelle dominieren die Wissenschaft:")
        
        c_pol, c_pyr = st.columns(2)
        with c_pol:
            st.subheader("Polarized (80/20)")
            st.markdown("**Das 'Seiler-Modell'**")
            st.progress(80, text="80% LIT (Low Intensity - Z1/Z2)")
            st.progress(20, text="20% HIT (High Intensity - Z4/Z5)")
            st.markdown("""
            * **Philosophie:** Vermeide die "Graue Zone" (Z3). Entweder ganz locker oder richtig hart.
            * **Für wen:** Profis mit hohem Volumen (>10h/Woche).
            """)
            
        with c_pyr:
            st.subheader("Pyramidal")
            st.markdown("**Das klassische Modell**")
            st.progress(70, text="Basis (Z1/Z2)")
            st.progress(20, text="Mitte (Z3/Sweetspot)")
            st.progress(10, text="Spitze (Z4/Z5)")
            st.markdown("""
            * **Philosophie:** Z3 (Tempo/Sweetspot) ist wertvoll, um zeiteffizient "Widerstandsfähigkeit" aufzubauen.
            * **Für wen:** Zeitbegrenzte Athleten (<8h/Woche).
            """)
    
    with g_tab6:
        st.subheader("🔬 Deep Dive Metriken")
        st.markdown("Hier analysieren wir Sekunden-Daten, die über Standard-Metriken hinausgehen.")
        
        gc1, gc2 = st.columns([1, 1])
        
        with gc1:
            st.markdown("#### 1. Power Duration Curve (PDC)")
            st.info("""
            **Was es ist:** Die PDC zeigt deine maximale Leistung (Watt) für jede Zeitdauer (1s bis 60min).
            
            **Warum wichtig:**
            * **Phänotyp:** Bist du Sprinter (steil links) oder Diesel (flach)?
            * **Ermüdungsresistenz:** Wie lange kannst du hohe Leistung halten?
            """)
            
            # Dummy Data für Example Chart
            ex_pdc = pd.DataFrame({
                'Seconds': [1, 5, 10, 30, 60, 300, 1200, 3600],
                'Watts': [900, 800, 700, 500, 400, 300, 250, 220]
            })
            
            ch_pdc = alt.Chart(ex_pdc).mark_line(point=True).encode(
                x=alt.X('Seconds', scale=alt.Scale(type='log'), title='Dauer (Log)'),
                y=alt.Y('Watts', title='Leistung (W)'),
                tooltip=['Seconds', 'Watts']
            ).properties(title="Beispiel: Typische PDC Kurve", height=200)
            
            st.altair_chart(ch_pdc, use_container_width=True)

        with gc2:
            st.markdown("#### 2. Quadrant Analysis & Decoupling")
            st.info("""
            **Aerobe Entkopplung (Pw:HR):**
            Misst den Cardiac Drift. Wir teilen die Einheit in zwei Hälften. Wenn der Puls in der 2. Hälfte steigt (bei gleichen Watt), ist die Entkopplung hoch.
            * **< 5%:** Exzellente Ausdauer.
            * **> 5%:** Ermüdung setzt ein.
            """)
            
            # Dummy Data für Quadrant Example
            ex_quad = pd.DataFrame({
                'Cadence': np.random.normal(85, 10, 100),
                'Force': np.random.normal(200, 50, 100)
            })
            
            ch_quad = alt.Chart(ex_quad).mark_circle(size=60, opacity=0.5).encode(
                x=alt.X('Cadence', title='Trittfrequenz (rpm)'),
                y=alt.Y('Force', title='Pedalkraft (N)'),
                color=alt.value('orange')
            ).properties(title="Beispiel: Quadrant Verteilung", height=200)
            
            st.altair_chart(ch_quad, use_container_width=True)


# --- LOGIK EXECUTION ---

if start_btn and email and password:
    with st.spinner("Synchronisiere Daten..."):
        existing_data = []
        if use_cache:
            existing_data = load_local_cache(email)
        
        buffer_delta = datetime.timedelta(days=ACWR_BUFFER_DAYS)
        fetch_start_date = start_date - buffer_delta
        api_fetch_start = fetch_start_date
        
        if existing_data:
            last_local_date = get_latest_activity_date(existing_data)
            if last_local_date:
                if last_local_date >= fetch_start_date:
                    api_fetch_start = last_local_date
                if last_local_date >= end_date:
                    api_fetch_start = max(last_local_date, end_date) 
        
        new_data = []
        err = None
        
        if api_fetch_start <= end_date:
            new_data, err = fetch_garmin_raw(email, password, api_fetch_start.isoformat(), end_date.isoformat())
        
        if err:
            st.error(f"Fehler beim Abruf: {err}")
        else:
            total_raw = merge_activities(existing_data, new_data)
            if use_cache and new_data:
                save_local_cache(total_raw, email)
                st.toast(f"{len(new_data)} neue Aktivitäten geladen.", icon="💾")
            elif not use_cache:
                total_raw = new_data
            
            st.session_state.raw_data = total_raw
            st.session_state.mode = 'real'
            
            full_df = process_data(total_raw, user_max_hr)
            if not full_df.empty:
                st.session_state.df = full_df
                st.success(f"Analyse bereit: {len(st.session_state.df)} Aktivitäten verfügbar.")

elif demo_btn:
    st.session_state.mode = 'demo'
    st.session_state.df = generate_demo_data(days=(end_date - start_date).days, user_max_hr=user_max_hr)

if st.session_state.df is not None and not st.session_state.df.empty:
    if st.session_state.mode == 'real' and st.session_state.raw_data:
        st.session_state.df = process_data(st.session_state.raw_data, user_max_hr)

# --- DASHBOARD VISUALISIERUNG ---
if st.session_state.df is not None and not st.session_state.df.empty:
    
    df_full_history = st.session_state.df.copy()
    
    if env_mode == "Nur Outdoor":
        if 'Indoor' in df_full_history.columns: df_full_history = df_full_history[df_full_history['Indoor'] == False]
    elif env_mode == "Nur Indoor":
        if 'Indoor' in df_full_history.columns: df_full_history = df_full_history[df_full_history['Indoor'] == True]
    
    mask_selected_range = (df_full_history['Datum'].dt.date >= start_date) & (df_full_history['Datum'].dt.date <= end_date)
    df_view = df_full_history[mask_selected_range].copy()
    
    # --- 1. WARN-HINWEIS & METRIK (WICHTIG!) ---
    act_count = len(df_view)
    
    if act_count == 0:
        st.warning(f"⚠️ Keine Aktivitäten gefunden für Filter: **{env_mode}** im Zeitraum.")
    else:
        # LOW DATA WARNING
        if act_count < 5:
            st.error(f"⚠️ **Kritischer Datenmangel ({act_count} Aktivitäten):** Statistische Auswertungen (Zonen, Trends, ACWR) sind nicht aussagekräftig! Bitte Zeitraum vergrößern.")
        elif act_count < 10:
            st.warning(f"⚠️ **Geringe Datenbasis ({act_count} Aktivitäten):** Trends sind mit Vorsicht zu genießen.")
            
        st.markdown(f"### 🏆 Übersicht & Einordnung")
        
        # Berechnung Wochen-Schnitt für Einordnung
        weeks_in_view = max(1, (end_date - start_date).days // 7)
        dist_avg = int(df_view['Distanz'].sum() / weeks_in_view)
        
        # --- NEW: Season Best Calculations ---
        # Bestimme All-Time/Season Best aus der vollen History
        sb_max_power_20 = df_full_history['Max20Min'].max() if 'Max20Min' in df_full_history else 0
        
        m1, m2, m3, m4 = st.columns(4)
        
        m1.metric("Anzahl Aktivitäten", act_count, help="Anzahl der Fahrten, die in die Berechnung einfließen.")
        
        if 'Max20Min' in df_view and df_view['Max20Min'].max() > 0:
            best_idx = df_view['Max20Min'].idxmax()
            best = df_view.loc[best_idx]
            current_best = int(best['Max20Min'])
            
            delta_str = ""
            if sb_max_power_20 > 0:
                pct = (current_best / sb_max_power_20) * 100
                delta_str = f"{int(pct)}% of Season Best ({int(sb_max_power_20)} W)"
            
            m2.metric("Best 20min Power", f"{current_best} W", delta_str, help="Vergleich zum All-Time/Season High")
        
        m3.metric("Gesamtstrecke", f"{int(df_view['Distanz'].sum())} km", f"Ø {dist_avg} km/Woche")
        m4.metric("Kalorien", f"{int(df_view['Kalorien'].sum()):,} kcal".replace(",", "."), f"Ø {int(df_view['Kalorien'].sum()/weeks_in_view)} kcal/Woche")

        st.divider()
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🚀 PMC & Form", "🧬 Fitness-Shift", "⚖️ ACWR & Load", "📈 Trends", "🎨 Zonen-Optimierer", "🔬 Labor-Deep-Dive"])

        # TAB 1: PMC (NEW FEATURE)
        with tab1:
            st.markdown("##### Performance Management Chart (Fitness vs. Fatigue)")
            # Berechne PMC auf voller Historie damit EMA korrekt anläuft
            df_pmc_full = calculate_pmc_stats(df_full_history)
            
            if not df_pmc_full.empty:
                # Filter auf View Range für Plot
                mask_pmc = (df_pmc_full['Datum'].dt.date >= start_date) & (df_pmc_full['Datum'].dt.date <= end_date)
                df_pmc_view = df_pmc_full[mask_pmc].copy()
                
                if not df_pmc_view.empty:
                    # Altair Plot: CTL (Area), ATL (Line), TSB (Bar/Area)
                    base_pmc = alt.Chart(df_pmc_view).encode(x='Datum')
                    
                    ctl_chart = base_pmc.mark_area(opacity=0.3, color='#3b82f6').encode(
                        y=alt.Y('CTL', title='Fitness (CTL) / Fatigue (ATL)'),
                        tooltip=['Datum', alt.Tooltip('CTL', format='.1f'), alt.Tooltip('ATL', format='.1f')]
                    )
                    
                    atl_chart = base_pmc.mark_line(color='#d946ef', strokeDash=[2,2]).encode(y='ATL')
                    
                    tsb_chart = base_pmc.mark_bar(opacity=0.8).encode(
                        y=alt.Y('TSB', title='Form (TSB)'),
                        color=alt.condition(alt.datum.TSB >= 0, alt.value('#10b981'), alt.value('#ef4444')),
                        tooltip=['Datum', alt.Tooltip('TSB', format='.1f')]
                    ).properties(height=100)
                    
                    # Combine layers: Top chart (Fitness/Fatigue), Bottom chart (Form)
                    combined = alt.vconcat(
                        (ctl_chart + atl_chart).properties(height=250, width="container"),
                        tsb_chart.properties(height=100, width="container")
                    ).resolve_scale(x='shared')
                    
                    st.altair_chart(combined, use_container_width=True)
                    
                    curr_tsb = df_pmc_view.iloc[-1]['TSB']
                    st.caption(f"Aktuelle Form (TSB): {curr_tsb:.1f}")
                else:
                    st.info("Keine PMC Daten im gewählten Zeitraum.")
            else:
                st.warning("Zu wenig Datenhistorie für PMC Berechnung.")

        # TAB 2: FITNESS SHIFT (MIT TRUE AEROBIC FILTER)
        with tab2:
            st.caption(f"Vergleich: Erste {comparison_weeks} Wochen vs. Letzte {comparison_weeks} Wochen.")
            df_power = df_view[df_view[power_col] > 0].copy()
            if not df_power.empty and act_count >= 4:
                min_d, max_d = df_power['Datum'].min(), df_power['Datum'].max()
                split_early = min_d + datetime.timedelta(weeks=comparison_weeks)
                split_late = max_d - datetime.timedelta(weeks=comparison_weeks)
                
                df_power['Phase'] = np.select([(df_power['Datum'] <= split_early), (df_power['Datum'] >= split_late)], ["1. Start", "2. Ende"], default="Mitte")
                df_compare = df_power[df_power['Phase'] != "Mitte"]
                
                if not df_compare.empty:
                    c1, c2 = st.columns([2, 1])
                    
                    with c1:
                        st.markdown("**All Activities: Power vs Heart Rate**")
                        chart = alt.Chart(df_compare).mark_circle(size=80).encode(
                            x=alt.X(power_col, title=f'{power_metric_display} (Watt)', scale=alt.Scale(zero=False)),
                            y=alt.Y('HF', title='Herzfrequenz (bpm)', scale=alt.Scale(zero=False)),
                            color=alt.Color('Phase', scale=alt.Scale(range=['#3b82f6', '#f97316'])),
                            tooltip=['Datum', 'Aktivität', power_col, 'HF']
                        )
                        lines = chart.transform_regression(power_col, 'HF', groupby=['Phase']).mark_line(size=3)
                        st.altair_chart(chart + lines, use_container_width=True)
                    
                    with c2:
                        st.markdown("**True Aerobic Efficiency (Z2 > 60min)**")
                        # Filter: Nur Zone 2 (ZoneIdx == 1) und länger als 60 Min
                        mask_true_aero = (df_view['ZoneIdx'] == 1) & (df_view['Dauer_Min'] > 60)
                        df_z2_pure = df_view[mask_true_aero].copy()
                        
                        if not df_z2_pure.empty:
                            # FIX: Explizite Typisierung (:T, :Q) und Conditional Regression für Altair Robustheit
                            base_chart = alt.Chart(df_z2_pure).mark_circle(color='green', size=60).encode(
                                x=alt.X('Datum:T', title='Datum'),
                                y=alt.Y('EF:Q', scale=alt.Scale(zero=False), title="EF (Z2 Only)"),
                                tooltip=[
                                    alt.Tooltip('Datum:T', format='%d.%m.%Y'),
                                    alt.Tooltip('EF:Q', format='.2f'),
                                    alt.Tooltip('Dauer_Min:Q', title='Dauer (min)')
                                ]
                            )
                            
                            # Regression nur zeichnen wenn genug Datenpunkte da sind, sonst crashed das Chart bei Zoom/wenig Daten
                            if len(df_z2_pure) > 2:
                                reg_line = base_chart.transform_regression('Datum', 'EF').mark_line(color='gray', strokeDash=[4,4])
                                final_chart = base_chart + reg_line
                            else:
                                final_chart = base_chart
                                
                            st.altair_chart(final_chart, use_container_width=True)
                            st.caption(f"Zeigt nur 'reine' Grundlageneinheiten ({len(df_z2_pure)}) ohne Intervalleinfluss.")
                        else:
                            st.info("Keine reinen Z2-Einheiten (>60min) gefunden.")
                    
                    df_zone = df_power[(df_power['HF'] >= target_hr - hr_tol) & (df_power['HF'] <= target_hr + hr_tol)]
                    if not df_zone.empty:
                        recent_mean = df_zone[df_zone['Datum'] >= split_late][power_col].mean()
                        old_mean = df_zone[df_zone['Datum'] <= split_early][power_col].mean()
                        
                        if pd.notna(recent_mean) and pd.notna(old_mean):
                            diff = int(recent_mean - old_mean)
                            st.divider()
                            k1, k2 = st.columns([1, 2])
                            k1.metric(f"Leistung bei ~{target_hr} bpm", f"{int(recent_mean)} W", f"{diff} W vs Start")
                            
                            with k2:
                                if diff > 5: st.success(f"👏 **Positiver Trend:** Du leistest {diff} Watt mehr bei gleichem Puls. Deine Effizienz ist gestiegen!")
                                elif diff < -5: st.error(f"📉 **Negativer Trend:** Du leistest {abs(diff)} Watt weniger. Mögliche Ursachen: Ermüdung, Krankheit oder Trainingspause.")
                                else: st.info("➡️ **Plateau:** Deine aerobe Effizienz ist im gewählten Zeitraum stabil geblieben.")
            else: st.warning("Zu wenig Daten für einen Phasen-Vergleich.")

        # TAB 3: ACWR
        with tab3:
            daily = df_full_history.set_index('Datum').resample('D')['Stress'].sum().fillna(0).to_frame()
            daily['Acute'] = daily['Stress'].rolling(7, min_periods=1).mean()
            daily['Chronic'] = daily['Stress'].rolling(28, min_periods=1).mean()
            daily['ACWR'] = np.where(daily['Chronic'] > 0, daily['Acute'] / daily['Chronic'], 0)
            daily.reset_index(inplace=True)
            daily_view = daily[(daily['Datum'].dt.date >= start_date) & (daily['Datum'].dt.date <= end_date)].copy()

            base = alt.Chart(daily_view).encode(x='Datum')
            line = base.mark_line(color='#10b981').encode(y='ACWR')
            points = base.mark_circle().encode(
                y='ACWR', color=alt.condition(alt.datum.ACWR > 1.5, alt.value('red'), alt.value('#10b981')),
                tooltip=['Datum', alt.Tooltip('ACWR', format='.2f')]
            )
            danger = alt.Chart(pd.DataFrame({'y': [1.5]})).mark_rule(color='red', strokeDash=[5,5]).encode(y='y')
            st.altair_chart(line + points + danger, width="stretch")
            
            if not daily_view.empty:
                curr = daily_view.iloc[-1]['ACWR']
                if curr > 1.5: st.error(f"⚠️ **ACWR High ({curr:.2f}):** Verletzungsrisiko erhöht! Belastung reduzieren.")
                elif curr < 0.8: st.warning(f"📉 **ACWR Low ({curr:.2f}):** Detraining möglich. Intensität/Volumen steigern.")
                else: st.success(f"✅ **ACWR Optimal ({curr:.2f}):** Sweet Spot Training.")

        # TAB 4: TRENDS
        with tab4:
            daily_agg = df_view.set_index('Datum').resample('D').agg({
                'Stress': 'sum', 'Dauer_Min': 'sum', 'Leistung': 'mean', 'HF': 'mean', 'EF': 'mean' 
            }).fillna(0).reset_index()
            
            base = alt.Chart(daily_agg).encode(x='Datum')
            bar = base.mark_bar(opacity=0.3, color='purple').encode(y='Stress', tooltip='Stress')
            line = base.mark_line(color='cyan').encode(y='Dauer_Min', tooltip='Dauer_Min')
            st.altair_chart(alt.layer(bar, line).resolve_scale(y='independent'), width="stretch")
            
            ef_data = daily_agg[daily_agg['EF'] > 0].copy()
            ef_data['EF_MA'] = ef_data['EF'].rolling(window=5, min_periods=1).mean()
            if not ef_data.empty:
                chart_ef = alt.Chart(ef_data).mark_circle(color='green').encode(x='Datum', y=alt.Y('EF', scale=alt.Scale(zero=False))) + \
                           alt.Chart(ef_data).mark_line(color='green').encode(x='Datum', y='EF_MA')
                st.altair_chart(chart_ef, width="stretch")

        # TAB 5: ZONEN-OPTIMIERER
        with tab5:
            max_date = df_view['Datum'].max()
            start_analysis = max_date - datetime.timedelta(weeks=comparison_weeks)
            df_recent = df_view[df_view['Datum'] >= start_analysis].copy()
            
            if not df_recent.empty and act_count > 2:
                vol_avg = (df_recent['Dauer_Min'].sum() / 60) / comparison_weeks 
                
                # Modell-Logik
                if vol_avg < 5.5:
                    mod, targets, msg = "Sweet Spot", [10, 40, 30, 15, 5], "Geringes Volumen: Fokus auf Qualität."
                elif vol_avg < 10:
                    mod, targets, msg = "Hybrid", [15, 60, 15, 7, 3], "Mittleres Volumen: Basis + Spitzen."
                else:
                    mod, targets, msg = "Polarized", [25, 55, 5, 10, 5], "Hohes Volumen: Polarized empfohlen."

                c1, c2 = st.columns(2)
                c1.metric(f"Ø Volumen", f"{vol_avg:.1f} h/Woche")
                c2.metric("Empfohlenes Modell", mod)
                st.info(msg)
                
                counts = df_recent['ZoneIdx'].value_counts().sort_index()
                total_count = len(df_recent)
                labels = ["Z1", "Z2", "Z3", "Z4", "Z5"]
                
                cols = st.columns(5)
                comp_data = []
                
                for i in range(5):
                    act_pct = (counts.get(i, 0) / total_count * 100) if total_count > 0 else 0
                    delta = act_pct - targets[i]
                    with cols[i]:
                        st.metric(labels[i], f"{int(act_pct)}%", f"{int(delta)}% vs Soll", delta_color="inverse")
                        st.progress(min(act_pct/100, 1.0))
                    comp_data.extend([{"Zone": labels[i], "Typ": "Ist", "Prozent": act_pct}, {"Zone": labels[i], "Typ": "Soll", "Prozent": targets[i]}])
                    
                chart = alt.Chart(pd.DataFrame(comp_data)).mark_bar().encode(
                    x=alt.X('Zone', sort=labels), y='Prozent', color='Typ', xOffset='Typ', tooltip=['Zone', 'Typ', alt.Tooltip('Prozent', format='.1f')]
                )
                st.altair_chart(chart, width="stretch")
            else:
                st.warning("Zu wenig Daten im gewählten Zeitraum für eine zuverlässige Zonen-Analyse.")
        
        # TAB 6: LABOR-DEEP-DIVE (NEU)
        with tab6:
            st.markdown("### 🔬 Labor Deep-Dive: Einzel-Analyse")
            st.markdown("Lade hier eine einzelne `.fit` oder `.tcx` Datei hoch, um sekündliche Daten (Power Curve, Cadence, etc.) zu analysieren. Dies ermöglicht Einblicke, die über die Standard-API nicht möglich sind.")
            
            uploaded_file = st.file_uploader("Datei hochladen (.fit oder .tcx)", type=['fit', 'tcx'])
            
            if uploaded_file is not None:
                with st.spinner("Analysiere Datei..."):
                    if uploaded_file.name.endswith('.fit'):
                        df_stream = parse_fit(uploaded_file)
                    else:
                        df_stream = parse_tcx(uploaded_file)
                    
                    if not df_stream.empty and 'power' in df_stream.columns:
                        st.success(f"Datei erfolgreich geladen: {len(df_stream)} Datenpunkte.")
                        
                        # Relative Zeit für Plot berechnen
                        if 'timestamp' in df_stream.columns:
                            start_time = df_stream['timestamp'].min()
                            df_stream['duration_min'] = (df_stream['timestamp'] - start_time).dt.total_seconds() / 60
                        
                        # Metriken berechnen
                        pdc = calculate_power_curve(df_stream)
                        max_pwr = df_stream['power'].max()
                        avg_pwr = df_stream['power'].mean()
                        
                        # Neue Funktion: Aerobe Entkopplung (KORRIGIERT)
                        decoupling, ef1, ef2 = calculate_aerobic_decoupling(df_stream)
                        
                        # --- Layout ---
                        col_a, col_b = st.columns([2, 1])
                        
                        with col_a:
                            st.subheader("Power Duration Curve (MMP)")
                            if not pdc.empty:
                                chart_pdc = alt.Chart(pdc).mark_line(point=True).encode(
                                    x=alt.X('Dauer_Sek', scale=alt.Scale(type='log'), title='Dauer (Log Skala)'),
                                    y=alt.Y('Watt', title='Leistung (W)'),
                                    tooltip=['Dauer_Sek', 'Watt']
                                ) # .interactive() entfernt (fixiert)
                                st.altair_chart(chart_pdc, use_container_width=True)
                                
                                # Automatische Einordnung
                                p1m = pdc[pdc['Dauer_Sek']==60]['Watt'].max() if 60 in pdc['Dauer_Sek'].values else 0
                                p5m = pdc[pdc['Dauer_Sek']==300]['Watt'].max() if 300 in pdc['Dauer_Sek'].values else 0
                                p20m = pdc[pdc['Dauer_Sek']==1200]['Watt'].max() if 1200 in pdc['Dauer_Sek'].values else 0
                                
                                insight_text = ""
                                if p1m > p20m * 2.0:
                                    insight_text = "**Typ:** Sprinter / Anaerob stark. Starker Abfall nach 1min."
                                elif p20m > 0 and p5m < p20m * 1.15:
                                    insight_text = "**Typ:** Zeitfahrer (Diesel). Sehr flache Kurve."
                                else:
                                    insight_text = "**Typ:** Allrounder."
                                
                                st.info(f"💡 **Erkenntnis:** {insight_text}\n\n*Max 1min:* {int(p1m)}W | *Max 5min:* {int(p5m)}W | *Max 20min:* {int(p20m)}W")

                        with col_b:
                            st.subheader("Aerobe Entkopplung (Pw:HR)")
                            if decoupling != 0:
                                delta_color = "normal"
                                if decoupling < 5: 
                                    st.success(f"✅ **{decoupling}%** (Top Ausdauer)")
                                elif decoupling < 8:
                                    st.warning(f"⚠️ **{decoupling}%** (Leichter Drift)")
                                else:
                                    st.error(f"❌ **{decoupling}%** (Starker Drift)")
                                
                                st.caption(f"EF 1. Hälfte: {ef1}")
                                st.caption(f"EF 2. Hälfte: {ef2}")
                            else:
                                st.info("Zu wenig Daten für Pw:HR Berechnung")

                            st.divider()
                            st.subheader("Verteilung")
                            st.metric("Max Power", f"{int(max_pwr)} W")
                            st.metric("Ø Power", f"{int(avg_pwr)} W")
                            
                            if 'cadence' in df_stream.columns:
                                chart_cad = alt.Chart(df_stream).mark_bar().encode(
                                    x=alt.X('cadence', bin=alt.Bin(maxbins=20), title='Trittfrequenz'),
                                    y='count()',
                                    color=alt.value('orange')
                                )
                                st.altair_chart(chart_cad, use_container_width=True)
                        
                        # Scatter Plot: Power vs Heart Rate
                        if 'heart_rate' in df_stream.columns and 'duration_min' in df_stream.columns:
                            st.subheader("Interne vs. Externe Belastung (Puls vs. Watt)")
                            chart_scatter = alt.Chart(df_stream).mark_circle(opacity=0.3, size=20).encode(
                                x=alt.X('power', title='Leistung (Watt)'),
                                y=alt.Y('heart_rate', title='Herz (bpm)'),
                                color=alt.Color('duration_min', title='Zeit (min)'), # Relative Zeit statt Datum
                                tooltip=['duration_min', 'power', 'heart_rate']
                            ) # .interactive() entfernt (fixiert)
                            st.altair_chart(chart_scatter, use_container_width=True)
                            st.caption("Ein 'Ausfransen' der Kurve nach oben rechts oder eine Verschiebung über die Zeit zeigt Ermüdung/Decoupling.")

                    elif df_stream.empty:
                        st.warning("Konnte keine Daten parsen. Ist die Datei korrekt?")
                    else:
                        st.warning("Keine Leistungsdaten (Watt) in dieser Datei gefunden.")

elif st.session_state.df is None and not start_btn and not demo_btn:
    st.info("👈 Bitte links starten.")