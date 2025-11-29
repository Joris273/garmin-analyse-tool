import sys
import os
import datetime
import random
from typing import List, Optional, Tuple, Dict, Any

import streamlit as st
from streamlit.web import cli as stcli
from streamlit.runtime.scriptrunner import get_script_run_ctx

import pandas as pd
import altair as alt
import numpy as np
from garminconnect import Garmin

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

# --- WISSENSCHAFTLICHE BERECHNUNGEN (CORE LOGIC) ---

def calculate_trimp_vectorized(duration_min: pd.Series, avg_hr: pd.Series, max_hr_user: int) -> pd.Series:
    """
    Berechnet den TRIMP (Training Impulse) nach Banister (vereinfacht ohne Ruhe-HF).
    Formel: Dauer * Intensität * exp(1.92 * Intensität)
    """
    if max_hr_user <= 0:
        return pd.Series(0, index=duration_min.index)
    
    intensity = avg_hr / max_hr_user
    # Vermeidung von Division durch Null oder negativen Werten
    intensity = intensity.fillna(0).clip(lower=0)
    
    # Gewichtungsfaktor nach Banister (Männer ~1.92, Frauen ~1.67 -> hier fix 1.92)
    weighting = np.exp(1.92 * intensity)
    return duration_min * intensity * weighting

def determine_smart_zone_vectorized(row: pd.Series, user_max_hr: int) -> int:
    """
    Bestimmt die Trainingszone intelligent unter Berücksichtigung von Variabilität (VI).
    Dies wird via .apply() aufgerufen, da komplexe bedingte Logik nötig ist.
    """
    avg_hr = row.get('HF', 0)
    max_hr_activity = row.get('MaxHF', 0)
    avg_power = row.get('Leistung', 0)
    norm_power = row.get('NormPower', 0)

    if not user_max_hr or avg_hr <= 0:
        return 0
    
    avg_pct = avg_hr / user_max_hr
    max_pct = (max_hr_activity / user_max_hr) if max_hr_activity > 0 else avg_pct
    
    # Basis-Klassifizierung (Coggan/Friel approx.)
    if avg_pct < 0.60: zone_idx = 0      # Z1
    elif avg_pct < 0.75: zone_idx = 1    # Z2
    elif avg_pct < 0.85: zone_idx = 2    # Z3
    elif avg_pct < 0.95: zone_idx = 3    # Z4
    else: zone_idx = 4                   # Z5
    
    # Variabilitäts-Index (VI)
    vi = 1.0
    if avg_power > 10 and norm_power > 10:
        vi = norm_power / avg_power

    # Intelligente Upgrades
    # Fall A: Hoher Max-Puls deutet auf Intervalle hin, auch wenn Avg niedrig war
    if max_pct > 0.92 and zone_idx < 3:
        zone_idx = max(zone_idx, 3)
        
    # Fall B: Hoher VI deutet auf 'unruhige' Fahrt (Intervalle/Sprint) hin
    if vi > 1.15 and zone_idx < 3:
        zone_idx = max(zone_idx, 3)
    elif vi > 1.08 and zone_idx < 2:
        zone_idx = max(zone_idx, 2)
        
    return int(zone_idx)

# --- DATEN-EXTRAKTION & VERARBEITUNG ---

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_garmin_raw(email: str, password: str, start_date: datetime.date, end_date: datetime.date) -> Tuple[List[Dict], Optional[str]]:
    """Verbindet mit Garmin Connect und lädt Rohdaten."""
    try:
        client = Garmin(email, password)
        client.login()
        activities = client.get_activities_by_date(start_date.isoformat(), end_date.isoformat(), "")
        return activities, None
    except Exception as e:
        return [], str(e)

def process_data(raw_activities: List[Dict[str, Any]], user_max_hr: int) -> pd.DataFrame:
    """
    Konvertiert Raw-Dicts in DataFrame und führt wissenschaftliche Berechnungen durch.
    Nutzt Vektorisierung für Performance (O(1) Pandas Operationen statt O(N) Loops).
    """
    if not raw_activities:
        return pd.DataFrame()

    # 1. Normalisierung: Relevante Felder extrahieren
    extracted_data = []
    
    # Mapping möglicher Keys auf unsere Ziel-Spalten
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

    for activity in raw_activities:
        act_type = activity.get('activityType', {}).get('typeKey', 'unknown').lower()
        
        # Filter: Nur Radsport
        if not any(x in act_type for x in ['cycling', 'biking', 'ride', 'gravel', 'mtb', 'virtual']):
            continue

        row = {
            'Datum': activity.get('startTimeLocal', '').split(' ')[0],
            'Aktivität': activity.get('activityName', 'Unbekannt')
        }
        
        # Robuste Extraktion
        for target_col, candidates in key_map.items():
            val = None
            for key in candidates:
                if key in activity and activity[key] is not None:
                    val = activity[key]
                    break
            row[target_col] = val

        # Fallback Logik für NormPower direkt beim Einlesen
        if row['NormPower'] is None and row['Leistung'] is not None:
            row['NormPower'] = row['Leistung']

        extracted_data.append(row)

    if not extracted_data:
        return pd.DataFrame()

    df = pd.DataFrame(extracted_data)
    
    # 2. Datentypen erzwingen & Bereinigung
    numeric_cols = ['Leistung', 'NormPower', 'Max20Min', 'HF', 'MaxHF', 'Kalorien', 'Distanz_Raw', 'Anstieg', 'Dauer_Sec']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['Datum'] = pd.to_datetime(df['Datum'], errors='coerce')
    df.dropna(subset=['Datum'], inplace=True)
    df.sort_values('Datum', inplace=True)

    # 3. Berechnete Spalten (Vektorisierte Operationen)
    df['Dauer_Min'] = (df['Dauer_Sec'] / 60).round(1)
    df['Distanz'] = (df['Distanz_Raw'] / 1000).round(1)
    
    # Strikte Filterung:
    # HF > 0, Leistung/NP > 40W (kein Leerlauf), Dauer > 5 min
    mask_valid = (
        (df['HF'] > 0) & 
        ((df['Leistung'] > 40) | (df['NormPower'] > 40)) & 
        (df['Dauer_Min'] > 5)
    )
    df = df[mask_valid].copy()

    if df.empty:
        return pd.DataFrame()

    # TRIMP Calculation (Vektorisiert)
    df['Stress'] = calculate_trimp_vectorized(df['Dauer_Min'], df['HF'], user_max_hr).round(1)

    # Efficiency Factor (EF) = NP / HF
    df['EF'] = (df['NormPower'] / df['HF']).round(2)

    # Zonen Bestimmung (via apply, da komplexe Logik)
    df['ZoneIdx'] = df.apply(lambda row: determine_smart_zone_vectorized(row, user_max_hr), axis=1)
    
    zone_labels = {0: "Z1 (Erholung)", 1: "Z2 (Grundlage)", 2: "Z3 (Tempo)", 3: "Z4 (Schwelle)", 4: "Z5 (Max)"}
    df['Zone'] = df['ZoneIdx'].map(zone_labels)

    # Int Konvertierung für schönere Anzeige
    cols_to_int = ['Leistung', 'NormPower', 'Max20Min', 'HF', 'MaxHF', 'Kalorien', 'Anstieg']
    for col in cols_to_int:
        df[col] = df[col].astype(int)

    return df

def generate_demo_data(days: int = 120, user_max_hr: int = 161) -> pd.DataFrame:
    """Generiert synthetische Daten für Demo-Modus."""
    random.seed(42)
    data = []
    today = datetime.date.today()
    
    for i in range(days):
        if random.random() > 0.6: continue  # Ruhetag
        
        date = today - datetime.timedelta(days=days-i)
        # Zyklisierung simulieren (Load steigt über 3 Wochen, dann Ruhewoche)
        cycle_pos = (i % 28) / 28
        load_factor = 0.5 + (cycle_pos * 0.8)
        if cycle_pos > 0.8: load_factor = 0.4  # Recovery week
        
        ride_type = random.choice(['LIT', 'LIT', 'MIT', 'HIT'])
        
        # Basiswerte
        duration = 60
        avg_hr = int(user_max_hr * 0.7)
        power = 150 + (i * 0.1) 
        
        if ride_type == 'LIT': # Low Intensity
            duration = random.randint(90, 180) * load_factor
            avg_hr = int(user_max_hr * 0.65) + random.randint(-5, 5)
            power = 160 + (i * 0.1)
            norm_power = power * 1.02
            max_hr_activity = avg_hr + 20
        elif ride_type == 'MIT': # Tempo / Sweetspot
            duration = random.randint(60, 90)
            avg_hr = int(user_max_hr * 0.83) + random.randint(-5, 5)
            power = 200 + (i * 0.2)
            norm_power = power * 1.05
            max_hr_activity = avg_hr + 15
        else: # HIT
            duration = random.randint(45, 70)
            avg_hr = int(user_max_hr * 0.88) + random.randint(-5, 5)
            power = 240 + (i * 0.3)
            norm_power = power * 1.18 # Hoher VI
            max_hr_activity = user_max_hr - random.randint(0, 5)

        # Activity Dict simulieren
        raw_act = {
            'startTimeLocal': f"{date} 10:00:00",
            'activityName': f"{ride_type} Training",
            'activityType': {'typeKey': 'cycling'},
            'avgPower': power,
            'normPower': norm_power,
            'max20MinPower': power * 1.1,
            'avgHR': avg_hr,
            'maxHR': max_hr_activity,
            'duration': duration * 60,
            'calories': duration * 10,
            'distance': (duration/60) * 30 * 1000, 
            'totalAscent': 500
        }
        data.append(raw_act)
        
    return process_data(data, user_max_hr)

# --- UI LAYOUT ---

with st.sidebar:
    st.header("⚙️ Setup")
    tab_login, tab_params = st.tabs(["Daten", "Parameter"])
    
    with tab_login:
        st.info("🔒 **Datenschutz:** Deine Zugangsdaten werden **nur** für die Verbindung zu Garmin genutzt und **nicht gespeichert**. Alles läuft sicher im Arbeitsspeicher.")
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
        
        col1, col2 = st.columns(2)
        start_btn = col1.button("Start", type="primary")
        demo_btn = col2.button("Demo")
        
    with tab_params:
        st.subheader("2. Analyse-Fokus")
        st.caption("Wie möchtest du die geladenen Daten vergleichen?")
        user_max_hr = st.number_input("Max Herzfrequenz", 100, 220, 161, help="Beeinflusst Zonen & Stress-Score.")
        
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
            max_value=max_possible_weeks, 
            value=default_weeks
        )
        
        target_hr = st.slider("Aerobe Schwelle (Vergleichs-Puls)", 100, 170, 135)
        hr_tol = st.slider("Toleranz (+/- bpm)", 2, 15, 5)

st.title("🚴 Garmin Science Lab V12.4 (Optimized)")
st.markdown("Analyse von **Effizienz**, **Belastung (ACWR)** und **Wissenschaftlicher Trainingsverteilung**.")

# --- WISSENSCHAFTLICHER GUIDE (NEU & ERWEITERT) ---
with st.expander("📘 Wissenschaftlicher Guide: Methodik, Physiologie & Interpretation", expanded=False):
    st.markdown("""
    ## 🧬 1. Die Physiologie der Leistung (Efficiency Factor)
    
    **Das Prinzip der "Aeroben Entkopplung"**
    
    Dein Körper funktioniert wie ein Hybrid-Motor. Der Input ist Sauerstoff & Herzfrequenz (interne Last), der Output ist Watt (externe Last).
    Fitness definiert sich wissenschaftlich als **Ökonomisierung**: Für denselben Watt-Output muss das Herz weniger oft schlagen, da das Schlagvolumen (Stroke Volume) durch Training steigt und die mitochondriale Dichte in den Muskeln zunimmt.
    """)
    
    c1, c2 = st.columns(2)
    with c1:
        st.info("""
        **📉 Der Efficiency Factor (EF)**
        
        $$EF = \\frac{\\text{Normalized Power (NP)}}{\\text{Ø Herzfrequenz}}$$
        
        * **Steigender EF:** Deine "aerobe Maschine" wird größer. Du produzierst mehr Watt pro Herzschlag.
        * **Stagnierender EF:** Du hast ein Plateau erreicht oder bist ermüdet.
        """)
    with c2:
        st.warning("""
        **❤️ Cardiac Drift (Entkopplung)**
        
        Bei langen Fahrten steigt der Puls trotz gleicher Leistung.
        * **Ursachen:** Thermoregulation (Blut zur Kühlung in die Haut), Dehydration (Blutvolumen sinkt).
        * **Ziel:** Bei Grundlagen-Training (LIT) sollte der Puls in der zweiten Hälfte um **weniger als 5%** driften.
        """)

    st.divider()

    st.markdown("""
    ## ⚖️ 2. Das ACWR-Modell (Verletzungsprävention)
    
    **Acute:Chronic Workload Ratio nach Dr. Tim Gabbett**
    
    Dieses Modell basiert auf der Annahme, dass **Belastungsspitzen** (Spikes) gefährlicher sind als eine absolut hohe Last. Dein Gewebe (Sehnen, Knochen) passt sich langsamer an als dein Herz-Kreislauf-System.
    """)

    st.markdown("""
    | Zone | ACWR Wert | Interpretation | Empfehlung |
    | :--- | :--- | :--- | :--- |
    | 🟢 **Sweet Spot** | `0.8 - 1.3` | **Optimales Training.** | Die Last ist hoch genug für Adaption, aber sicher. |
    | 🟠 **Vorsicht** | `1.3 - 1.5` | **Erhöhtes Risiko.** | Du steigerst die Umfänge sehr schnell. Achte auf Schlaf! |
    | 🔴 **Danger Zone** | `> 1.5` | **Hohes Verletzungsrisiko.** | Deine aktuelle Last (7 Tage) ist 50% höher als deine Gewöhnung (28 Tage). |
    | 🔵 **Detraining** | `< 0.8` | **Formverlust.** | Du trainierst weniger als dein Körper gewohnt ist. |
    """)
    
    st.caption("*Berechnung: Acute Load (EWMA 7 Tage) geteilt durch Chronic Load (EWMA 28 Tage).*")

    st.divider()

    st.markdown("""
    ## ⚡ 3. Normalized Power (NP) vs. Durchschnitt
    
    **Warum Watt nicht gleich Watt ist**
    
    Der Durchschnittswert lügt. Physiologischer Stress wächst nicht linear, sondern exponentiell zur Leistung. 
    300 Watt tun dem Körper metabolisch (Laktatakkumulation, Glykogenverarmung) viel mehr weh als 2x 150 Watt, obwohl der Durchschnitt gleich ist.
    """)
    
    c3, c4 = st.columns([1, 2])
    with c3:
        st.latex(r'''
        NP = \sqrt[4]{\frac{1}{n} \sum_{t=1}^{n} (P_t)^4}
        ''')
    with c4:
        st.markdown("""
        **Der Algorithmus (Dr. Andy Coggan):**
        1. Die Werte werden zur **4. Potenz** erhoben. Das bestraft Leistungsspitzen extrem stark.
        2. Dadurch repräsentiert die NP die physiologische Kosten, als wärst du die Strecke absolut gleichmäßig gefahren.
        """)

    st.markdown("""
    **Der Variabilitäts-Index (VI)**
    
    $$VI = \\frac{NP}{Avg Power}$$
    
    * **VI 1.00 - 1.05:** Sehr stetiges Fahren (Zeitfahren, Triathlon, Grundlagentraining).
    * **VI > 1.20:** Hochgradig stochastisch (Kriterium, MTB, Hügelintervalle). 
    * *Unser Tool nutzt den VI, um "versteckte" harte Einheiten zu erkennen, auch wenn der Durchschnittspuls niedrig war.*
    """)
    
    st.divider()

    st.markdown("""
    ## 🎨 4. Polarized Training (80/20 Regel)
    
    **Stephen Seiler's Forschung**
    
    Analyse von Elite-Ausdauerathleten zeigt fast universell eine Verteilung der Intensität:
    
    * **~80% der Einheiten:** Zone 1 & 2 (LIT - Low Intensity). Unterhalb der aeroben Schwelle (Laktat < 2mmol). Fördert Fettstoffwechsel und Kapillarisierung ohne das vegetative Nervensystem zu stressen.
    * **~20% der Einheiten:** Zone 4 & 5 (HIT - High Intensity). Oberhalb der Schwelle. Erhöht VO2max.
    * **Vermeidung der "Grauen Zone" (Z3):** Zu anstrengend für reine Erholung, zu locker für maximale Adaption ("Junk Miles").
    """)

# --- LOGIK EXECUTION ---

if start_btn and email and password:
    with st.spinner("Lade Daten von Garmin..."):
        raw, err = fetch_garmin_raw(email, password, start_date, end_date)
        if err: 
            st.error(f"Fehler beim Abruf: {err}")
        else:
            st.session_state.raw_data = raw
            st.session_state.mode = 'real'
            st.success(f"{len(raw)} Aktivitäten geladen.")
            st.session_state.df = process_data(raw, user_max_hr)

elif demo_btn:
    st.session_state.mode = 'demo'
    st.session_state.df = generate_demo_data(days=90, user_max_hr=user_max_hr)

if st.session_state.df is not None and not st.session_state.df.empty:
    if st.session_state.mode == 'real' and st.session_state.raw_data:
        st.session_state.df = process_data(st.session_state.raw_data, user_max_hr)

# --- DASHBOARD VISUALISIERUNG ---
if st.session_state.df is not None and not st.session_state.df.empty:
    df = st.session_state.df.copy()
    
    st.markdown(f"### 🏆 Bestwerte (Im gesamten Zeitraum)")
    m1, m2, m3, m4 = st.columns(4)
    
    if 'Max20Min' in df and df['Max20Min'].max() > 0:
        best_idx = df['Max20Min'].idxmax()
        best = df.loc[best_idx]
        m1.metric("Beste 20min Power", f"{int(best['Max20Min'])} W", best['Datum'].strftime('%d.%m.'))
    
    max_elev = df['Anstieg'].max()
    if max_elev > 300:
        idx_king = df['Anstieg'].idxmax()
        king_stage = df.loc[idx_king]
        m2.metric("Königsetappe", f"{int(king_stage['Anstieg'])} hm", f"{king_stage['Distanz']} km ({king_stage['Datum'].strftime('%d.%m.')})")
    else:
        idx_long = df['Distanz'].idxmax()
        longest = df.loc[idx_long]
        m2.metric("Weiteste Fahrt", f"{longest['Distanz']} km", longest['Datum'].strftime('%d.%m.'))
    
    total_km = int(df['Distanz'].sum())
    total_hours = int(df['Dauer_Min'].sum() / 60)
    m3.metric("Gesamtleistung", f"{total_km} km", f"{total_hours} Stunden")
    m4.metric("Kalorien Total", f"{int(df['Kalorien'].sum()):,} kcal".replace(",", "."))

    st.divider()
    
    tab1, tab2, tab3, tab4 = st.tabs(["🧬 Fitness-Shift", "⚖️ ACWR & Load", "📈 Trends", "🎨 Zonen-Optimierer"])

    # --- TAB 1: Fitness Shift ---
    with tab1:
        st.caption(f"Vergleich: Erste {comparison_weeks} Wochen vs. Letzte {comparison_weeks} Wochen.")
        df_power = df[df['NormPower'] > 0].copy()
        
        if not df_power.empty:
            min_d, max_d = df_power['Datum'].min(), df_power['Datum'].max()
            split_early = min_d + datetime.timedelta(weeks=comparison_weeks)
            split_late = max_d - datetime.timedelta(weeks=comparison_weeks)
            
            conditions = [
                (df_power['Datum'] <= split_early),
                (df_power['Datum'] >= split_late)
            ]
            choices = ["1. Start-Phase", "2. End-Phase"]
            df_power['Phase'] = np.select(conditions, choices, default="Mitte")
            
            df_compare = df_power[df_power['Phase'] != "Mitte"]

            chart = alt.Chart(df_compare).mark_circle(size=80).encode(
                x=alt.X('NormPower', title='Normalized Power (Watt)', scale=alt.Scale(zero=False)),
                y=alt.Y('HF', title='Herzfrequenz (bpm)', scale=alt.Scale(zero=False)),
                color=alt.Color('Phase', scale=alt.Scale(range=['#3b82f6', '#f97316'])),
                tooltip=['Datum', 'Aktivität', 'NormPower', 'HF']
            )
            lines = chart.transform_regression('NormPower', 'HF', groupby=['Phase']).mark_line(size=3)
            st.altair_chart(chart + lines, width="stretch")
            
            c1, c2 = st.columns(2)
            c1.info("Ziel: Die orange Linie (End-Phase) sollte rechts unterhalb der blauen Linie (Start-Phase) liegen.")
            
            df_zone = df_power[(df_power['HF'] >= target_hr - hr_tol) & (df_power['HF'] <= target_hr + hr_tol)]
            if not df_zone.empty:
                recent_mean = df_zone[df_zone['Datum'] >= split_late]['NormPower'].mean()
                old_mean = df_zone[df_zone['Datum'] <= split_early]['NormPower'].mean()
                
                if pd.notna(recent_mean) and pd.notna(old_mean):
                    diff = int(recent_mean - old_mean)
                    c2.metric(f"NP bei {target_hr} bpm", f"{int(recent_mean)} W", f"{diff} W")
                else:
                    c2.warning("Zu wenig Daten in den Phasen.")
            else: c2.info("Keine Fahrten im gewählten Pulsbereich.")

    # --- TAB 2: ACWR ---
    with tab2:
        daily = df.set_index('Datum').resample('D')['Stress'].sum().fillna(0).to_frame()
        daily['Acute'] = daily['Stress'].rolling(7, min_periods=1).mean()
        daily['Chronic'] = daily['Stress'].rolling(28, min_periods=1).mean()
        daily['ACWR'] = np.where(daily['Chronic'] > 0, daily['Acute'] / daily['Chronic'], 0)
        daily.reset_index(inplace=True)

        base = alt.Chart(daily).encode(x='Datum')
        danger = alt.Chart(pd.DataFrame({'y': [1.5]})).mark_rule(color='red', strokeDash=[5,5]).encode(y='y')
        line = base.mark_line(color='#10b981').encode(y=alt.Y('ACWR', scale=alt.Scale(domain=[0, 2.0])))
        points = base.mark_circle().encode(
            y='ACWR',
            color=alt.condition(alt.datum.ACWR > 1.5, alt.value('red'), alt.value('#10b981')),
            tooltip=['Datum', alt.Tooltip('ACWR', format='.2f')]
        )
        st.altair_chart(line + points + danger, width="stretch")
        
        if not daily.empty:
            curr = daily.iloc[-1]['ACWR']
            st.metric("ACWR Status", f"{curr:.2f}", delta="Vorsicht" if curr > 1.5 else "OK", delta_color="inverse")

    # --- TAB 3: Trends ---
    with tab3:
        st.subheader("Detail-Analyse: Trends über Zeit")
        c_sel1, c_sel2, c_sel3, c_sel4 = st.columns(4)
        show_load = c_sel1.checkbox("Trainingsbelastung", value=True)
        show_power = c_sel2.checkbox("Leistung (Watt/NP)", value=False)
        show_hr = c_sel3.checkbox("Herzfrequenz", value=False)
        show_ef = c_sel4.checkbox("Effizienz (EF)", value=True)
        
        daily_agg = df.set_index('Datum').resample('D').agg({
            'Stress': 'sum', 
            'Dauer_Min': 'sum',
            'Leistung': 'mean', 
            'NormPower': 'mean',
            'HF': 'mean',
            'MaxHF': 'max',
            'EF': 'mean' 
        }).fillna(0).reset_index()
        
        training_days = daily_agg[daily_agg['Dauer_Min'] > 0].copy()

        if show_load:
            base = alt.Chart(daily_agg).encode(x='Datum')
            bar = base.mark_bar(opacity=0.3, color='purple').encode(y='Stress', tooltip='Stress')
            line = base.mark_line(color='cyan').encode(y='Dauer_Min', tooltip='Dauer_Min')
            st.altair_chart(alt.layer(bar, line).resolve_scale(y='independent'), width="stretch")
        
        if show_power:
            base = alt.Chart(training_days[training_days['NormPower'] > 0]).encode(x='Datum')
            l1 = base.mark_line(color='orange').encode(y='NormPower', tooltip='NormPower')
            l2 = base.mark_line(color='gray', strokeDash=[5,5]).encode(y='Leistung', tooltip='Leistung')
            st.altair_chart(l1 + l2, width="stretch")

        if show_hr:
            base = alt.Chart(training_days).encode(x='Datum')
            l1 = base.mark_line(color='red').encode(y=alt.Y('MaxHF', scale=alt.Scale(zero=False)), tooltip='MaxHF')
            l2 = base.mark_line(color='pink').encode(y=alt.Y('HF', scale=alt.Scale(zero=False)), tooltip='HF')
            st.altair_chart(l1 + l2, width="stretch")

        if show_ef:
            ef_data = training_days[training_days['EF'] > 0].copy()
            ef_data['EF_MA'] = ef_data['EF'].rolling(window=5, min_periods=1).mean()
            base = alt.Chart(ef_data).encode(x='Datum')
            points = base.mark_circle(opacity=0.3, color='green').encode(y=alt.Y('EF', scale=alt.Scale(zero=False)), tooltip='EF')
            line = base.mark_line(color='green', size=3).encode(y='EF_MA', tooltip='EF_MA')
            st.altair_chart(points + line, width="stretch")

    # --- TAB 4: Zonen ---
    with tab4:
        st.subheader(f"Intensitäts-Verteilung (Letzte {comparison_weeks} Wochen)")
        
        max_date = df['Datum'].max()
        start_analysis = max_date - datetime.timedelta(weeks=comparison_weeks)
        df_recent = df[df['Datum'] >= start_analysis].copy()
        
        if not df_recent.empty:
            vol_total = df_recent['Dauer_Min'].sum() / 60
            vol_avg = vol_total / comparison_weeks 
            
            if vol_avg < 5.5:
                mod, targets = "Sweet Spot / Pyramidal", [10, 40, 30, 15, 5]
                msg = "Fokus auf Qualität statt Quantität."
            elif vol_avg < 10:
                mod, targets = "Hybrid", [15, 60, 15, 7, 3]
                msg = "Solide Basis mit gezielten Spitzen."
            else:
                mod, targets = "Polarized (80/20)", [25, 55, 5, 10, 5]
                msg = "Hohes Volumen erfordert Disziplin (LIT muss LIT bleiben)."

            c1, c2 = st.columns(2)
            c1.metric(f"Ø Volumen", f"{vol_avg:.1f} h/Woche")
            c2.metric("Empfohlenes Modell", mod)
            st.info(msg)
            
            counts = df_recent['ZoneIdx'].value_counts().sort_index()
            total_count = len(df_recent)
            labels = ["Z1 (Erholung)", "Z2 (Grundlage)", "Z3 (Tempo)", "Z4 (Schwelle)", "Z5 (Max)"]
            
            st.divider()
            cols = st.columns(5)
            comp_data = []
            
            for i in range(5):
                act_pct = (counts.get(i, 0) / total_count * 100) if total_count > 0 else 0
                delta = act_pct - targets[i]
                
                with cols[i]:
                    st.markdown(f"**{labels[i]}**")
                    st.progress(min(act_pct/100, 1.0))
                    st.metric("Anteil", f"{int(act_pct)}%", f"{int(delta)}% vs Ziel", delta_color="inverse")
                
                comp_data.extend([
                    {"Zone": labels[i], "Typ": "Ist", "Prozent": act_pct},
                    {"Zone": labels[i], "Typ": "Soll", "Prozent": targets[i]}
                ])
                
            chart = alt.Chart(pd.DataFrame(comp_data)).mark_bar().encode(
                x=alt.X('Zone', sort=labels),
                y='Prozent',
                color='Typ',
                xOffset='Typ',
                tooltip=['Zone', 'Typ', alt.Tooltip('Prozent', format='.1f')]
            )
            st.altair_chart(chart, width="stretch")
        else:
            st.warning("Keine Daten im gewählten Zeitraum.")

elif st.session_state.df is None and not start_btn and not demo_btn:
    st.info("👈 Bitte links starten.")