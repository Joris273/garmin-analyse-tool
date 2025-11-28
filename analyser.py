import sys
import os
import streamlit as st
from streamlit.web import cli as stcli
from streamlit.runtime.scriptrunner import get_script_run_ctx

# --- 1. AUTOMATISCHER STARTER ---
if __name__ == "__main__":
    if not get_script_run_ctx():
        print("Starte Streamlit Server...")
        sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
        sys.exit(stcli.main())

# --- AB HIER BEGINNT DIE APP ---

import datetime
import pandas as pd
import altair as alt
import numpy as np
from garminconnect import Garmin
import random

# --- Konfiguration ---
try:
    st.set_page_config(page_title="Garmin Pro Analytics", page_icon="🚴", layout="wide")
except:
    pass

# --- Caching ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_garmin_raw(email, password, start_date, end_date):
    try:
        client = Garmin(email, password)
        client.login()
        # Hole alle Aktivitäten, Filterung erfolgt später
        activities = client.get_activities_by_date(start_date.isoformat(), end_date.isoformat(), "")
        return activities, None
    except Exception as e:
        return [], str(e)

# --- Helfer: Robuste Datenextraktion ---
def robust_get(activity, keys):
    """Sucht einen Wert unter verschiedenen Schlüsseln und wandelt ihn sicher in float um."""
    for key in keys:
        val = activity.get(key)
        if val is not None:
            try:
                float_val = float(val)
                # Wir akzeptieren auch 0 (z.B. bei Distanz oder Höhe), filtern aber später auf >0 bei HR/Power
                if float_val >= 0: return float_val 
            except (ValueError, TypeError): 
                continue
    return None

def calculate_trimp(duration_min, avg_hr, max_hr):
    """Berechnet TRIMP Score. Fallback, falls max_hr unsinnig ist."""
    if not isinstance(duration_min, (int, float)) or not isinstance(avg_hr, (int, float)):
        return 0
    if not max_hr or max_hr < 100: max_hr = 190 
    
    intensity = avg_hr / max_hr
    # Banister-Formel Gewichtung
    weighted_intensity = intensity * np.exp(1.92 * intensity) 
    return duration_min * weighted_intensity

def determine_smart_zone(avg_hr, max_hr_activity, user_max_hr, avg_power, norm_power):
    """
    Bestimmt die Zone intelligent. Erkennt Intervalle (niedriger Avg, hoher Max oder hoher VI)
    und stuft diese entsprechend hoch.
    """
    if not user_max_hr or user_max_hr < 100 or not avg_hr: return 0, "Z?"
    
    avg_pct = avg_hr / user_max_hr
    # Fallback: Wenn kein MaxHF der Aktivität da ist, nimm Avg (konservativ)
    max_pct = max_hr_activity / user_max_hr if max_hr_activity else avg_pct
    
    # Basis-Klassifizierung nach Durchschnitt (Coggan/Friel HR Zones approximiert)
    if avg_pct < 0.60: zone_idx = 0      # Z1 Active Recovery
    elif avg_pct < 0.75: zone_idx = 1    # Z2 Endurance
    elif avg_pct < 0.85: zone_idx = 2    # Z3 Tempo
    elif avg_pct < 0.95: zone_idx = 3    # Z4 Threshold
    else: zone_idx = 4                   # Z5 VO2max
    
    # 2. Power Variabilitäts-Index (VI) Analyse
    # VI = Normalized Power / Average Power. 
    vi = 1.0
    # Sicherstellen, dass wir nicht durch 0 teilen
    if avg_power and avg_power > 10 and norm_power and norm_power > 10:
        vi = norm_power / avg_power

    # -- INTELLIGENTE KORREKTUR --
    
    # Fall A: Herzfrequenz Max war sehr hoch (typisch für Intervalle)
    if max_pct > 0.92 and zone_idx < 3:
        zone_idx = max(zone_idx, 3) # Mindestens Threshold werten
        
    # Fall B: Power VI ist hoch (typisch für Intervalle/Sprints)
    if vi > 1.15 and zone_idx < 3: 
        zone_idx = max(zone_idx, 3) # Upgrade auf Z4 (Hard)
    elif vi > 1.08 and zone_idx < 2:
        zone_idx = max(zone_idx, 2) # Upgrade auf Z3 (Tempo)

    labels = ["Z1 (Erholung)", "Z2 (Grundlage)", "Z3 (Tempo)", "Z4 (Schwelle)", "Z5 (Max)"]
    return zone_idx, labels[zone_idx]

# --- Datenverarbeitung ---
def process_data(raw_activities, user_max_hr):
    data = []
    if not raw_activities: return pd.DataFrame()

    for activity in raw_activities:
        act_type = activity.get('activityType', {}).get('typeKey', 'unknown')
        act_name = activity.get('activityName', 'Unbekannt')
        
        # Filter: Alles was Räder hat
        is_cycling = any(x in act_type.lower() for x in ['cycling', 'biking', 'ride', 'gravel', 'mtb', 'virtual'])
        if not is_cycling: continue

        # Daten sicher extrahieren
        avg_power = robust_get(activity, ['avgPower', 'averagePower', 'normPower'])
        norm_power = robust_get(activity, ['normPower', 'weightedMeanPower', 'normalizedPower'])
        max_20min = robust_get(activity, ['max20MinPower', 'maximum20MinPower', 'twentyMinPower'])
        
        avg_hr = robust_get(activity, ['avgHR', 'averageHR', 'avgHeartRate', 'averageHeartRate'])
        max_hr_activity = robust_get(activity, ['maxHR', 'maxHeartRate', 'maximumHeartRate'])
        
        duration_s = activity.get('duration', 0)
        duration_min = round(duration_s / 60, 1)
        calories = robust_get(activity, ['calories', 'totalCalories'])
        
        distance_m = robust_get(activity, ['distance']) 
        elevation_m = robust_get(activity, ['totalAscent', 'elevationGain'])

        # --- NEUE STRIKTE FILTERUNG (Update: Threshold 40W) ---
        # 1. Muss Herzfrequenz haben (>0)
        has_hr = avg_hr is not None and avg_hr > 0
        
        # 2. Muss RELEVANTE Leistung haben (> 40 Watt)
        # Dies filtert Sensor-Fehler, leere Batterien oder gemütliches Stadtradeln raus.
        has_power = (avg_power is not None and avg_power > 40) or (norm_power is not None and norm_power > 40)
        
        # 3. Muss eine relevante Dauer haben (> 5 Min)
        has_duration = duration_min > 5

        if has_hr and has_power and has_duration:
            # Werte bereinigen
            power_val = int(avg_power) if avg_power else None
            # Fallback: Wenn kein NormPower da ist, nimm AvgPower (ist hier sicher vorhanden wegen Filter)
            norm_power_val = int(norm_power) if norm_power else power_val
            
            max_20min_val = int(max_20min) if max_20min else 0
            
            # TRIMP Score berechnen
            stress_score = calculate_trimp(duration_min, avg_hr, user_max_hr)
            
            # Smart Zone bestimmen (Mit VI)
            zone_idx, zone_label = determine_smart_zone(avg_hr, max_hr_activity, user_max_hr, power_val, norm_power_val)
            
            dist_km = round(distance_m / 1000, 1) if distance_m else 0.0
            elev_m = int(elevation_m) if elevation_m else 0
            
            # Efficiency Factor (EF)
            ef = 0.0
            if norm_power_val and avg_hr > 40:
                ef = round(norm_power_val / avg_hr, 2)
            
            try:
                date_str = activity['startTimeLocal'].split(' ')[0]
                date_obj = pd.to_datetime(date_str)
            except:
                continue 

            data.append({
                "Datum": date_obj,
                "Aktivität": act_name,
                "Leistung": power_val,
                "NormPower": norm_power_val, # Speichern für Fitness-Shift
                "Max20Min": max_20min_val,
                "HF": int(avg_hr),
                "MaxHF": int(max_hr_activity) if max_hr_activity else int(avg_hr),
                "Dauer_Min": duration_min,
                "Stress": round(stress_score, 1),
                "ZoneIdx": zone_idx,
                "Zone": zone_label,
                "Kalorien": int(calories) if calories else 0,
                "Distanz": dist_km,
                "Anstieg": elev_m,
                "EF": ef
            })
    
    df = pd.DataFrame(data)
    # Optimierung: Datentypen erzwingen
    if not df.empty:
        df.sort_values('Datum', inplace=True)
        # Sicherstellen, dass Zahlen auch Zahlen sind (für Altair wichtig)
        cols_to_numeric = ['Leistung', 'NormPower', 'HF', 'Stress', 'Dauer_Min', 'EF']
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def generate_demo_data(days=120, user_max_hr=161):
    random.seed(42) 
    data = []
    today = datetime.date.today()
    base_hr_max = user_max_hr
    
    for i in range(days):
        if random.random() > 0.6: continue 
        date = today - datetime.timedelta(days=days-i)
        cycle_pos = (i % 28) / 28 
        load_factor = 0.5 + (cycle_pos * 0.8) 
        if cycle_pos > 0.8: load_factor = 0.4 
        ride_type = random.choice(['LIT', 'LIT', 'MIT', 'HIT']) 
        
        duration = 60
        avg_hr = 130
        max_hr_activity = 140
        power = 150 + (i * 0.2)
        norm_power = power 
        max_20min = power * 1.1 
        calories = 600
        speed = 28 + (random.random() * 5)
        dist_km = (duration/60) * speed
        elev_m = dist_km * random.randint(5, 15)

        if ride_type == 'LIT': 
            duration = random.randint(90, 180) * load_factor
            avg_hr = int(base_hr_max * 0.65) + random.randint(-5, 5)
            max_hr_activity = avg_hr + random.randint(10, 20)
            power = 160 + (i * 0.1)
            norm_power = power * 1.02 
            max_20min = power * 1.05 
            calories = duration * 10
            dist_km = (duration/60) * 26
            elev_m = dist_km * 8
        elif ride_type == 'MIT':
            duration = random.randint(60, 90)
            avg_hr = int(base_hr_max * 0.83) + random.randint(-5, 5)
            max_hr_activity = avg_hr + random.randint(10, 15)
            power = 200 + (i * 0.2)
            norm_power = power * 1.05
            max_20min = power * 1.1
            calories = duration * 12
            dist_km = (duration/60) * 30
            elev_m = dist_km * 12
        elif ride_type == 'HIT':
            duration = random.randint(45, 70) 
            avg_hr = int(base_hr_max * 0.88) + random.randint(-5, 5) 
            max_hr_activity = base_hr_max - random.randint(0, 5) 
            power = 240 + (i * 0.3)
            norm_power = power * 1.18 
            max_20min = power * 1.2 
            calories = duration * 15
            dist_km = (duration/60) * 32
            elev_m = dist_km * 5

        stress = calculate_trimp(duration, avg_hr, base_hr_max)
        zone_idx, zone_label = determine_smart_zone(avg_hr, max_hr_activity, base_hr_max, power, norm_power)
        
        ef = 0.0
        if norm_power and avg_hr > 0:
            ef = round(norm_power / avg_hr, 2)

        data.append({
            "Datum": pd.to_datetime(date),
            "Aktivität": f"{ride_type} Training",
            "Leistung": int(power),
            "NormPower": int(norm_power),
            "Max20Min": int(max_20min),
            "HF": int(avg_hr),
            "MaxHF": int(max_hr_activity),
            "Dauer_Min": int(duration),
            "Stress": round(stress, 1),
            "ZoneIdx": zone_idx,
            "Zone": zone_label,
            "Kalorien": int(calories),
            "Distanz": round(dist_km, 1),
            "Anstieg": int(elev_m),
            "EF": ef
        })
    return pd.DataFrame(data)

# --- UI Layout ---

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
        selected_range = st.selectbox("Datenbasis", list(range_options.keys()), index=1, help="Wähle den Zeitraum für den Datenabruf.")
        
        today = datetime.date.today()
        val = range_options[selected_range]
        if val == "cy":
            start_date = datetime.date(today.year, 1, 1)
            end_date = today
        elif val == "ly":
            start_date = datetime.date(today.year - 1, 1, 1)
            end_date = datetime.date(today.year - 1, 12, 31)
        else:
            start_date = today - datetime.timedelta(days=val)
            end_date = today
        
        col1, col2 = st.columns(2)
        start_btn = col1.button("Start", type="primary")
        demo_btn = col2.button("Demo")
        
    with tab_params:
        st.subheader("2. Analyse-Fokus")
        st.caption("Wie möchtest du die geladenen Daten vergleichen?")
        user_max_hr = st.number_input("Max Herzfrequenz", 100, 220, 161, help="Beeinflusst Zonen & Stress-Score.")
        
        days_diff = (end_date - start_date).days
        if 'raw_data' in st.session_state and st.session_state.raw_data:
             if 'df' in st.session_state and st.session_state.df is not None and not st.session_state.df.empty:
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
            value=default_weeks, 
            help="Bestimmt die Größe der Vergleichsblöcke."
        )
        
        target_hr = st.slider("Aerobe Schwelle (Vergleichs-Puls)", 100, 170, 135)
        hr_tol = st.slider("Toleranz (+/- bpm)", 2, 15, 5)

st.title("🚴 Garmin Science Lab V12.4")
st.markdown("Analyse von **Effizienz**, **Belastung (ACWR)** und **Wissenschaftlicher Trainingsverteilung**.")

with st.expander("📘 Wissenschaftlicher Guide: Methodik & Interpretation (Hier klicken)"):
    st.markdown("""
    ### 🧬 1. Aerobe Entkopplung & Fitness-Shift
    
    **💡 Das Konzept:** Dein Herz ist der Motor, deine Beine das Getriebe. "Fitness" bedeutet, dass dein Motor (Herz) weniger Arbeit leisten muss, um die gleiche Geschwindigkeit (Leistung in Watt) zu erzeugen.
    
    **🔍 Die Grafik (Normalized Power vs. HF):**
    Wir verwenden hier die **Normalized Power (NP)**, nicht den reinen Durchschnitt.
    * **Warum NP?** Physiologische Kosten sind nicht linear. 300 Watt tun mehr als doppelt so weh wie 150 Watt. NP berücksichtigt, dass Intervalle und Sprints deinen Stoffwechsel exponentiell stärker belasten als gleichmäßiges Fahren.
    * **Die Interpretation:**
        * **✅ Gutes Zeichen:** Die Punktewolke deiner aktuellen Trainingsphase (Orange) liegt **rechts unterhalb** der alten Phase (Blau). Du erzeugst mehr metabolischen Output für denselben Input (Herzschlag).
        * **⚠️ Stagnation:** Die Wolken liegen übereinander.
    
    ---

    ### ⚖️ 2. Belastungssteuerung (ACWR) - Das "Gewebe-Gedächtnis"
    
    **💡 Das Konzept:**
    Verletzungen passieren oft durch "zu viel zu schnell". Dein Herz-Kreislauf-System passt sich in Wochen an, deine Sehnen und Knochen brauchen Monate. Das ACWR-Modell berücksichtigt diese Trägheit.
    
    **🔍 Die Formel (Gabbett-Modell):**
    * **Acute Load (Ermüdung):** Deine Last der letzten 7 Tage.
    * **Chronic Load (Fitness/Belastbarkeit):** Was dein Körper in den letzten 28 Tagen gewohnt war.
    
    **🚦 Die Zonen:**
    * 🟢 **0.8 - 1.3 (Sweet Spot):** Ideal. Du forderst deinen Körper etwas mehr als er gewohnt ist, aber nicht so sehr, dass er bricht.
    * 🔴 **> 1.5 (Danger Zone):** Alarmstufe Rot! Deine aktuelle Last ist 50% höher als deine Gewöhnung. Das Verletzungsrisiko verdoppelt sich hier statistisch.
    * 🔵 **< 0.8 (Detraining):** Du trainierst weniger als gewohnt und verlierst Form.
    
    ---

    ### 📈 3. Effizienz-Faktor (EF) & Trends
    
    **💡 Der EF:**
    `EF = Normalized Power / Durchschnittspuls`.
    Dies ist die "reine" Metrik für deine aerobe Maschine. Ein EF von 1.5 bedeutet, du erzeugst 1.5 Watt pro Herzschlag.
    * **Anwendung:** Beobachte diesen Trend bei deinen **Grundlagenfahrten (Zone 2)**. Wenn die Linie über Wochen steigt, hat sich deine mitochondriale Dichte verbessert.
    
    ---

    ### 🎨 4. Zonen-Optimierer (Smart Intervals)
    
    **💡 Das Problem:** Du fährst harte Intervalle (z.B. 4x4min VO2max), machst aber lange Pausen. Am Ende sagt der Computer: "Durchschnittspuls 135 bpm – das war wohl ein lockeres Training". Das ist falsch.
    
    **🔍 Die Lösung (Variabilitäts-Index VI):**
    Wir berechnen für jede Einheit den *Variabilitäts-Index* (`VI = NP / Avg Power`).
    * Ein VI von **1.00** bedeutet: Du bist gefahren wie ein Uhrwerk (perfekte Grundlage).
    * Ein VI über **1.15** bedeutet: Es war ein sehr unruhiges Rennen oder Intervalltraining.
    
    **Der Algorithmus dieses Tools:**
    Erkennt das Tool einen hohen VI oder einen sehr hohen Maximalpuls, stuft es die Einheit automatisch als **Intensiv (Zone 4/5)** ein, selbst wenn der Durchschnittspuls niedrig war.
    """)

# --- Logic ---
if 'df' not in st.session_state: st.session_state.df = None
if 'raw_data' not in st.session_state: st.session_state.raw_data = None
if 'mode' not in st.session_state: st.session_state.mode = None 

if start_btn and email and password:
    with st.spinner("Lade Daten von Garmin..."):
        raw, err = fetch_garmin_raw(email, password, start_date, end_date)
        if err: 
            st.error(err)
        else:
            st.session_state.raw_data = raw
            st.session_state.mode = 'real'
            st.success(f"{len(raw)} Aktivitäten geladen.")

elif demo_btn:
    st.session_state.raw_data = None 
    st.session_state.mode = 'demo'

if st.session_state.mode == 'real' and st.session_state.raw_data:
    st.session_state.df = process_data(st.session_state.raw_data, user_max_hr)

elif st.session_state.mode == 'demo':
    st.session_state.df = generate_demo_data(days=90, user_max_hr=user_max_hr)

# --- DASHBOARD ---
if st.session_state.df is not None and not st.session_state.df.empty:
    df = st.session_state.df.copy()
    
    st.markdown(f"### 🏆 Bestwerte (Im gesamten Zeitraum)")
    m1, m2, m3, m4 = st.columns(4)
    
    if 'Max20Min' in df and df['Max20Min'].max() > 0:
        best = df.loc[df['Max20Min'].idxmax()]
        m1.metric("Beste 20min Power", f"{int(best['Max20Min'])} W", best['Datum'].strftime('%d.%m.'))
    elif 'Leistung' in df:
        best = df.loc[df['Leistung'].idxmax()]
        m1.metric("Beste Ø Watt", f"{int(best['Leistung'])} W", "N/A")
    
    max_elev = df['Anstieg'].max()
    if max_elev > 300:
        king_stage = df.loc[df['Anstieg'].idxmax()]
        m2.metric("Königsetappe", f"{int(king_stage['Anstieg'])} hm", f"{king_stage['Distanz']} km ({king_stage['Datum'].strftime('%d.%m.')})")
    else:
        longest = df.loc[df['Distanz'].idxmax()]
        m2.metric("Weiteste Fahrt", f"{longest['Distanz']} km", longest['Datum'].strftime('%d.%m.'))
    
    total_km = int(df['Distanz'].sum())
    total_hours = int(df['Dauer_Min'].sum() / 60)
    m3.metric("Gesamtleistung", f"{total_km} km", f"{total_hours} Stunden")
    
    m4.metric("Kalorien Total", f"{int(df['Kalorien'].sum()):,} kcal".replace(",", "."))

    st.divider()
    
    tab1, tab2, tab3, tab4 = st.tabs(["🧬 Fitness-Shift", "⚖️ ACWR & Load", "📈 Trends", "🎨 Zonen-Optimierer"])

    # --- TAB 1 ---
    with tab1:
        st.caption(f"Vergleich: Erste {comparison_weeks} Wochen vs. Letzte {comparison_weeks} Wochen deiner Daten.")
        df_power = df.dropna(subset=['NormPower']).copy()
        
        if not df_power.empty:
            min_d, max_d = df_power['Datum'].min(), df_power['Datum'].max()
            split_early = min_d + datetime.timedelta(weeks=comparison_weeks)
            split_late = max_d - datetime.timedelta(weeks=comparison_weeks)
            
            df_power['Phase'] = df_power['Datum'].apply(lambda d: "1. Start-Phase" if d <= split_early else ("2. End-Phase" if d >= split_late else "Mitte"))
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
            if len(df_zone) > 1:
                recent = df_zone[df_zone['Datum'] >= split_late]
                old = df_zone[df_zone['Datum'] <= split_early]
                if not recent.empty and not old.empty:
                    diff = int(recent['NormPower'].mean() - old['NormPower'].mean())
                    c2.metric(f"NP bei {target_hr} bpm", f"{int(recent['NormPower'].mean())} W", f"{diff} W")
                else: c2.warning("Zu wenig Daten in den Phasen.")
            else: c2.info("Keine Fahrten im gewählten Pulsbereich.")
        else: st.warning("Keine Leistungsdaten.")

    # --- TAB 2 ---
    with tab2:
        daily = df.set_index('Datum').resample('D')['Stress'].sum().fillna(0).to_frame()
        daily['Acute'] = daily['Stress'].rolling(7, min_periods=1).mean()
        daily['Chronic'] = daily['Stress'].rolling(28, min_periods=1).mean()
        
        daily['ACWR'] = daily['Acute'] / daily['Chronic']
        daily['ACWR'] = daily['ACWR'].replace([np.inf, -np.inf], 0).fillna(0)
        
        daily = daily.reset_index()

        base = alt.Chart(daily).encode(x='Datum')
        danger = alt.Chart(pd.DataFrame({'y': [1.5]})).mark_rule(color='red', strokeDash=[5,5]).encode(y='y')
        line = base.mark_line(color='#10b981').encode(y=alt.Y('ACWR', scale=alt.Scale(domain=[0, 2.0])))
        points = base.mark_circle().encode(
            y='ACWR',
            color=alt.condition(alt.datum.ACWR > 1.5, alt.value('red'), alt.value('#10b981')),
            tooltip=['Datum', alt.Tooltip('ACWR', format='.2f')]
        )
        st.altair_chart(line + points + danger, width="stretch")
        curr = daily.iloc[-1]['ACWR']
        st.metric("ACWR Status", f"{curr:.2f}", delta="Vorsicht" if curr > 1.5 else "OK", delta_color="inverse")

    # --- TAB 3 (TRENDS) ---
    with tab3:
        st.subheader("Detail-Analyse: Trends über Zeit")
        st.write("Wähle die Metriken, die du vergleichen möchtest:")
        
        c_sel1, c_sel2, c_sel3, c_sel4 = st.columns(4)
        show_load = c_sel1.checkbox("Trainingsbelastung", value=True)
        show_power = c_sel2.checkbox("Leistung (Watt/NP)", value=False)
        show_hr = c_sel3.checkbox("Herzfrequenz", value=False)
        show_ef = c_sel4.checkbox("Effizienz (EF)", value=True)
        
        daily = df.set_index('Datum').resample('D').agg({
            'Stress': 'sum', 
            'Dauer_Min': 'sum',
            'Leistung': 'mean', 
            'NormPower': 'mean',
            'HF': 'mean',
            'MaxHF': 'max',
            'EF': 'mean' 
        }).fillna(0).reset_index()
        
        training_days = daily[daily['Dauer_Min'] > 0].copy()

        if show_load:
            st.markdown("#### 🏋️ Trainingsbelastung")
            base = alt.Chart(daily).encode(x='Datum')
            bar = base.mark_bar(opacity=0.3, color='purple').encode(y=alt.Y('Stress', title='Stress Score'), tooltip='Stress')
            line = base.mark_line(color='cyan').encode(y=alt.Y('Dauer_Min', title='Dauer (Min)'), tooltip='Dauer_Min')
            st.altair_chart(alt.layer(bar, line).resolve_scale(y='independent'), width="stretch")
        
        if show_power:
            st.markdown("#### ⚡ Leistungsentwicklung (Watt)")
            power_data = training_days[training_days['NormPower'] > 0]
            
            base = alt.Chart(power_data).encode(x='Datum')
            l1 = base.mark_line(color='orange').encode(y=alt.Y('NormPower', title='Watt'), tooltip='NormPower')
            l2 = base.mark_line(color='gray', strokeDash=[5,5]).encode(y='Leistung', tooltip='Leistung')
            st.altair_chart(l1 + l2, width="stretch")
            st.caption("Orange: Normalized Power (Physiologische Last). Grau: Durchschnitt (Mechanische Last).")

        if show_hr:
            st.markdown("#### ❤️ Herzfrequenz")
            base = alt.Chart(training_days).encode(x='Datum')
            l1 = base.mark_line(color='red').encode(y=alt.Y('MaxHF', scale=alt.Scale(zero=False)), tooltip='MaxHF')
            l2 = base.mark_line(color='pink').encode(y=alt.Y('HF', scale=alt.Scale(zero=False)), tooltip='HF')
            st.altair_chart(l1 + l2, width="stretch")

        if show_ef:
            st.markdown("#### 🚀 Effizienz-Faktor (Watt pro Herzschlag)")
            ef_data = training_days[training_days['EF'] > 0].copy()
            ef_data['EF_MA'] = ef_data['EF'].rolling(window=5, min_periods=1).mean()
            
            base = alt.Chart(ef_data).encode(x='Datum')
            points = base.mark_circle(opacity=0.3, color='green').encode(y=alt.Y('EF', scale=alt.Scale(zero=False)), tooltip='EF')
            line = base.mark_line(color='green', size=3).encode(y='EF_MA', tooltip='EF_MA')
            st.altair_chart(points + line, width="stretch")
            st.caption("Verhältnis NP / HF. Steigende Linie = Bessere Fitness.")

    # --- TAB 4 ---
    with tab4:
        st.subheader(f"Intensitäts-Verteilung (Fokus: Letzte {comparison_weeks} Wochen)")
        
        max_date_in_data = df['Datum'].max()
        start_analysis = max_date_in_data - datetime.timedelta(weeks=comparison_weeks)
        df_recent = df[df['Datum'] >= start_analysis].copy()
        
        if not df_recent.empty:
            vol_total = df_recent['Dauer_Min'].sum() / 60
            vol_avg = vol_total / comparison_weeks 
            
            if vol_avg < 5.5:
                mod, targets = "Sweet Spot / Pyramidal", [10, 40, 30, 15, 5]
                msg = "Fokus auf Qualität statt Quantität. Nutze die begrenzte Zeit für intensivere Reize (Sweet Spot/Z4)."
            elif vol_avg < 10:
                mod, targets = "Hybrid", [15, 60, 15, 7, 3]
                msg = "Solide Basis mit gezielten Spitzen. Ein guter Mix, aber vermeide zu viel unproduktive 'Graue Zone' (Z3)."
            else:
                mod, targets = "Polarized (80/20)", [25, 55, 5, 10, 5]
                msg = "Hohes Volumen erfordert Disziplin. Halte die lockeren Einheiten wirklich locker, um Ausbrennen zu verhindern."

            c1, c2 = st.columns(2)
            c1.metric(f"Ø Volumen (Letzte {comparison_weeks} Wo.)", f"{vol_avg:.1f} h/Woche")
            c2.metric("Empfohlenes Modell", mod)
            st.info(f"💡 **Tipp für dein Volumen:** {msg}")
            
            counts = df_recent['ZoneIdx'].value_counts().sort_index()
            total = len(df_recent)
            labels = ["Z1 (Erholung)", "Z2 (Grundlage)", "Z3 (Tempo)", "Z4 (Schwelle)", "Z5 (Max)"]
            
            st.divider()
            cols = st.columns(5)
            comp_data = []
            
            for i in range(5):
                act_pct = (counts.get(i, 0) / total * 100) if total > 0 else 0
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
            st.warning(f"Keine Trainingsdaten in den letzten {comparison_weeks} Wochen gefunden.")

elif st.session_state.df is None and not start_btn and not demo_btn:
    st.info("👈 Bitte links starten.")