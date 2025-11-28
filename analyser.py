import streamlit as st
import datetime
import pandas as pd
import altair as alt
import numpy as np
from garminconnect import Garmin
import random
import sys
import os

# --- Konfiguration ---
try:
    st.set_page_config(page_title="Garmin Pro Analytics", page_icon="🚴", layout="wide")
except:
    pass

# --- Caching & Performance ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_garmin_raw(email, password, start_date, end_date):
    """Holt Rohdaten von Garmin und cached sie für 1 Stunde."""
    try:
        client = Garmin(email, password)
        client.login()
        # Leerer String beim Activity Type, um ALLES zu holen (wird später gefiltert)
        activities = client.get_activities_by_date(start_date.isoformat(), end_date.isoformat(), "")
        return activities, None
    except Exception as e:
        return [], str(e)

# --- Helfer ---
def robust_get(activity, keys):
    for key in keys:
        val = activity.get(key)
        if val is not None:
            try:
                float_val = float(val)
                if float_val > 0: return float_val
            except: continue
    return None

def calculate_trimp(duration_min, avg_hr, max_hr):
    """Berechnet einen vereinfachten Training Impulse (TRIMP)."""
    if not max_hr or max_hr < 100: max_hr = 190 
    intensity = avg_hr / max_hr
    weighted_intensity = intensity * np.exp(1.92 * intensity) 
    return duration_min * weighted_intensity

def get_hr_zone(avg_hr, max_hr):
    """Bestimmt die dominante Zone basierend auf Durchschnittspuls."""
    if not max_hr: return "Z?"
    pct = avg_hr / max_hr
    if pct < 0.60: return "Z1 (Erholung)"
    elif pct < 0.70: return "Z2 (Grundlage)"
    elif pct < 0.80: return "Z3 (Tempo)"
    elif pct < 0.90: return "Z4 (Schwelle)"
    else: return "Z5 (Max)"

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

        avg_power = robust_get(activity, ['avgPower', 'averagePower', 'normPower'])
        # NEU: Explizite Suche nach 20min Power
        max_20min = robust_get(activity, ['max20MinPower', 'maximum20MinPower', 'twentyMinPower'])
        
        avg_hr = robust_get(activity, ['avgHR', 'averageHR', 'avgHeartRate', 'averageHeartRate'])
        duration = round(activity.get('duration', 0) / 60, 1)
        calories = robust_get(activity, ['calories', 'totalCalories'])

        if avg_hr and duration > 5:
            power_val = int(avg_power) if avg_power else None
            # Wenn 20min Power fehlt, aber die Fahrt > 20min war und AvgPower da ist, 
            # nehmen wir als Fallback die AvgPower (besser als nichts), aber idealerweise liefert Garmin das Feld.
            max_20min_val = int(max_20min) if max_20min else 0
            
            stress_score = calculate_trimp(duration, avg_hr, user_max_hr)
            zone = get_hr_zone(avg_hr, user_max_hr)

            data.append({
                "Datum": pd.to_datetime(activity['startTimeLocal'].split(' ')[0]),
                "Aktivität": act_name,
                "Leistung": power_val,
                "Max20Min": max_20min_val,
                "HF": int(avg_hr),
                "Dauer_Min": duration,
                "Stress": round(stress_score, 1),
                "Zone": zone,
                "Kalorien": int(calories) if calories else 0
            })
    
    df = pd.DataFrame(data)
    if not df.empty:
        df.sort_values('Datum', inplace=True)
    return df

def generate_demo_data(days=120):
    data = []
    today = datetime.date.today()
    base_hr_max = 190
    
    for i in range(days):
        if random.random() > 0.6: continue 
        date = today - datetime.timedelta(days=days-i)
        
        # Simuliere Formaufbau
        cycle_pos = (i % 28) / 28 
        load_factor = 0.5 + (cycle_pos * 0.8) 
        if cycle_pos > 0.8: load_factor = 0.4 
        
        ride_type = random.choice(['LIT', 'LIT', 'MIT', 'HIT']) 
        
        duration = 60
        avg_hr = 130
        power = 150 + (i * 0.2)
        
        # Simuliere 20min Power (etwas höher als Durchschnitt, aber unter Sprint)
        max_20min = power * 1.1 
        
        calories = 600

        if ride_type == 'LIT': 
            duration = random.randint(90, 180) * load_factor
            avg_hr = 125 + random.randint(-5, 5) # Z1/Z2
            power = 160 + (i * 0.1)
            max_20min = power * 1.05 # Bei LIT ist avg nah an 20min
            calories = duration * 10
        elif ride_type == 'MIT':
            duration = random.randint(60, 90)
            avg_hr = 150 + random.randint(-5, 5) # Z3
            power = 200 + (i * 0.2)
            max_20min = power * 1.1
            calories = duration * 12
        elif ride_type == 'HIT':
            duration = random.randint(45, 70) 
            avg_hr = 170 + random.randint(-5, 5) # Z4/Z5
            power = 240 + (i * 0.3)
            max_20min = power * 1.2 # Intervalle pushen die 20min Power nicht immer, aber hier simuliert
            calories = duration * 15

        stress = calculate_trimp(duration, avg_hr, base_hr_max)
        zone = get_hr_zone(avg_hr, base_hr_max)

        data.append({
            "Datum": pd.to_datetime(date),
            "Aktivität": f"{ride_type} Training",
            "Leistung": int(power),
            "Max20Min": int(max_20min),
            "HF": int(avg_hr),
            "Dauer_Min": int(duration),
            "Stress": round(stress, 1),
            "Zone": zone,
            "Kalorien": int(calories)
        })
    return pd.DataFrame(data)

# --- UI Layout ---

with st.sidebar:
    st.header("⚙️ Setup")
    
    tab_login, tab_params = st.tabs(["Login", "Parameter"])
    
    with tab_login:
        email = st.text_input("Garmin E-Mail")
        password = st.text_input("Passwort", type="password")
        
        # Datumsauswahl
        today = datetime.date.today()
        default_start = today - datetime.timedelta(days=120)
        date_range = st.date_input("Zeitraum", [default_start, today])
        
        st.info("Daten werden 60min gecached.")
        action_col1, action_col2 = st.columns(2)
        start_btn = action_col1.button("Start", type="primary")
        demo_btn = action_col2.button("Demo")

    with tab_params:
        st.subheader("Deine Physiologie")
        user_max_hr = st.number_input("Max Herzfrequenz", 100, 220, 161, help="Beeinflusst Zonen & Stress-Score.")
        
        st.subheader("Analyse Einstellungen")
        comparison_weeks = st.slider("Vergleichs-Fenster (Wochen)", 2, 12, 4)
        target_hr = st.slider("Aerobe Schwelle (Vergleichs-Puls)", 100, 170, 135)
        hr_tol = st.slider("Toleranz (+/- bpm)", 2, 15, 5)

st.title("🚴 Garmin Science Lab V3")
st.markdown("Analyse von **Effizienz**, **Belastung (ACWR)** und **Intensitäts-Verteilung**.")

# --- Logic & State ---
if 'df' not in st.session_state: st.session_state.df = None

if start_btn and email and password and len(date_range) == 2:
    with st.spinner("Lade Daten vom Garmin Server..."):
        raw_list, error = fetch_garmin_raw(email, password, date_range[0], date_range[1])
        if error:
            st.error(error)
        else:
            processed_df = process_data(raw_list, user_max_hr)
            if not processed_df.empty:
                st.session_state.df = processed_df
                st.success(f"{len(processed_df)} Rad-Aktivitäten geladen.")
            else:
                st.warning("Keine Rad-Aktivitäten in diesem Zeitraum.")
elif demo_btn:
    st.session_state.df = generate_demo_data()

# --- DASHBOARD ---
if st.session_state.df is not None:
    df = st.session_state.df.copy()
    
    # Highscore Metrics Row
    st.markdown("### 🏆 Saison Bestwerte")
    m1, m2, m3, m4 = st.columns(4)
    
    # KORREKTUR: Beste 20min Power
    if 'Max20Min' in df and df['Max20Min'].max() > 0:
        best_20 = df.loc[df['Max20Min'].idxmax()]
        m1.metric("Beste 20min Power", f"{int(best_20['Max20Min'])} W", best_20['Datum'].strftime('%d.%m.'))
    elif 'Leistung' in df:
        best_power = df.loc[df['Leistung'].idxmax()]
        m1.metric("Beste Ø Watt (Backup)", f"{int(best_power['Leistung'])} W", "Keine 20min Daten")
    
    longest_ride = df.loc[df['Dauer_Min'].idxmax()]
    m2.metric("Längste Fahrt", f"{int(longest_ride['Dauer_Min'])} Min", longest_ride['Datum'].strftime('%d.%m.'))
    
    total_km_sim = int(df['Dauer_Min'].sum() * 0.5) # grobe Schätzung
    m3.metric("Gesamtzeit", f"{int(df['Dauer_Min'].sum() / 60)}h", f"~{total_km_sim} km (Est.)")
    
    m4.metric("Kalorien Total", f"{int(df['Kalorien'].sum()):,} kcal".replace(",", "."))

    st.divider()
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🧬 Fitness-Shift", "⚖️ ACWR & Load", "📈 Zonen & Trends", "🎨 Verteilung"])

    # --- TAB 1: FITNESS SHIFT ---
    with tab1:
        st.caption(f"Vergleich Start ({comparison_weeks} Wochen) vs. Ende ({comparison_weeks} Wochen).")
        df_power = df.dropna(subset=['Leistung'])
        if not df_power.empty:
            min_date, max_date = df_power['Datum'].min(), df_power['Datum'].max()
            split_early = min_date + datetime.timedelta(weeks=comparison_weeks)
            split_late = max_date - datetime.timedelta(weeks=comparison_weeks)
            
            df_power['Phase'] = df_power['Datum'].apply(lambda d: "1. Start" if d <= split_early else ("2. Ende" if d >= split_late else "Mitte"))
            df_compare = df_power[df_power['Phase'] != "Mitte"]

            chart = alt.Chart(df_compare).mark_circle(size=80).encode(
                x=alt.X('Leistung', scale=alt.Scale(zero=False), title='Leistung (Watt)'),
                y=alt.Y('HF', scale=alt.Scale(zero=False), title='Herzfrequenz (bpm)'),
                color=alt.Color('Phase', scale=alt.Scale(range=['#3b82f6', '#f97316'])),
                tooltip=['Datum', 'Aktivität', 'Leistung', 'HF']
            )
            lines = chart.transform_regression('Leistung', 'HF', groupby=['Phase']).mark_line(size=3)
            st.altair_chart(chart + lines, width="stretch")
            
            col_res1, col_res2 = st.columns(2)
            col_res1.info("Ziel: Die orange Linie sollte **rechts unterhalb** der blauen Linie liegen.")
            
            # --- Zonen-Analyse ---
            df_zone = df_power[(df_power['HF'] >= target_hr - hr_tol) & (df_power['HF'] <= target_hr + hr_tol)]
            if len(df_zone) > 1:
                recent_data = df_zone[df_zone['Datum'] >= split_late]
                old_data = df_zone[df_zone['Datum'] <= split_early]
                
                if not recent_data.empty and not old_data.empty:
                    recent_pwr = recent_data['Leistung'].mean()
                    old_pwr = old_data['Leistung'].mean()
                    diff = int(recent_pwr - old_pwr)
                    col_res2.metric(f"Leistung bei {target_hr} bpm (±{hr_tol})", f"{int(recent_pwr)} Watt", f"{diff} W Änderung")
                else:
                    col_res2.warning("Nicht genug Daten in Start- oder Endphase für diesen Pulsbereich.")
            else:
                 col_res2.info(f"Keine Fahrten im Bereich {target_hr}±{hr_tol} bpm gefunden.")
        else:
            st.warning("Keine Leistungsdaten vorhanden.")

    # --- TAB 2: ACWR ---
    with tab2:
        df_daily = df.set_index('Datum').resample('D')['Stress'].sum().fillna(0).to_frame()
        df_daily['Acute (7d)'] = df_daily['Stress'].rolling(7, min_periods=1).mean()
        df_daily['Chronic (28d)'] = df_daily['Stress'].rolling(28, min_periods=1).mean()
        df_daily['ACWR'] = (df_daily['Acute (7d)'] / df_daily['Chronic (28d)']).fillna(0)
        df_daily = df_daily.reset_index()

        base = alt.Chart(df_daily).encode(x='Datum')
        danger = alt.Chart(pd.DataFrame({'y': [1.5]})).mark_rule(color='red', strokeDash=[5,5]).encode(y='y')
        
        line = base.mark_line(color='#10b981').encode(y=alt.Y('ACWR', scale=alt.Scale(domain=[0, 2.0])))
        points = base.mark_circle().encode(
            y='ACWR',
            color=alt.condition(alt.datum.ACWR > 1.5, alt.value('red'), alt.value('#10b981')),
            tooltip=['Datum', alt.Tooltip('ACWR', format='.2f')]
        )
        st.altair_chart(line + points + danger, width="stretch")
        
        curr_acwr = df_daily.iloc[-1]['ACWR']
        st.metric("ACWR Status", f"{curr_acwr:.2f}", delta="Riskant" if curr_acwr > 1.5 else "OK", delta_color="inverse")

    # --- TAB 3: VOLUMEN ---
    with tab3:
        df_daily['Stunden'] = df.set_index('Datum').resample('D')['Dauer_Min'].sum().fillna(0).values / 60
        base_vol = alt.Chart(df_daily.reset_index()).encode(x='Datum')
        bar = base_vol.mark_bar(opacity=0.3, color='purple').encode(y='Stress', tooltip='Stress')
        line = base_vol.mark_line(color='cyan').encode(y='Stunden', tooltip='Stunden')
        st.altair_chart(alt.layer(bar, line).resolve_scale(y='independent'), width="stretch")

    # --- TAB 4: DISTRIBUTION (NEU) ---
    with tab4:
        st.subheader("Intensitäts-Verteilung (Training Zones)")
        st.caption("Basierend auf dem Durchschnittspuls der gesamten Einheit. Ziel für Ausdauer: Viel Z1/Z2 (Polarized).")
        
        # Berechnung der Verteilung
        zone_counts = df['Zone'].value_counts().reset_index()
        zone_counts.columns = ['Zone', 'Anzahl']
        
        # Sortierung erzwingen
        zone_order = ["Z1 (Erholung)", "Z2 (Grundlage)", "Z3 (Tempo)", "Z4 (Schwelle)", "Z5 (Max)"]
        
        bars = alt.Chart(zone_counts).mark_bar().encode(
            x=alt.X('Zone', sort=zone_order),
            y='Anzahl',
            color=alt.Color('Zone', scale=alt.Scale(scheme='magma'), legend=None),
            tooltip=['Zone', 'Anzahl']
        )
        
        st.altair_chart(bars, width="stretch")
        
        # Polarized Check
        z1_z2 = df[df['Zone'].isin(["Z1 (Erholung)", "Z2 (Grundlage)"])].shape[0]
        total = df.shape[0]
        pct_easy = (z1_z2 / total) * 100 if total > 0 else 0
        
        col_pol1, col_pol2 = st.columns([3, 1])
        with col_pol1:
            st.progress(pct_easy / 100)
            st.caption(f"{int(pct_easy)}% deiner Einheiten waren im Grundlagenbereich (Z1/Z2).")
        with col_pol2:
            if pct_easy > 75: st.success("Polarized! ✅")
            elif pct_easy < 50: st.error("Zu intensiv! ⚠️")
            else: st.warning("Ausgewogen")

else:
    st.info("👈 Wähle Datum und starte links.")

# --- AUTO-START ---
if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if not get_script_run_ctx():
            import sys, os
            from streamlit.web import cli as stcli
            os.chdir(os.path.dirname(os.path.abspath(__file__)))
            sys.argv = ["streamlit", "run", sys.argv[0]]
            sys.exit(stcli.main())
    except ImportError: pass