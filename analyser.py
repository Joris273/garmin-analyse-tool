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
    if not max_hr or max_hr < 100: max_hr = 190 
    intensity = avg_hr / max_hr
    weighted_intensity = intensity * np.exp(1.92 * intensity) 
    return duration_min * weighted_intensity

def get_hr_zone_key(avg_hr, max_hr):
    if not max_hr: return 0
    pct = avg_hr / max_hr
    if pct < 0.60: return 0 # Z1
    elif pct < 0.70: return 1 # Z2
    elif pct < 0.80: return 2 # Z3
    elif pct < 0.90: return 3 # Z4
    else: return 4 # Z5

def get_zone_label(zone_idx):
    labels = ["Z1 (Erholung)", "Z2 (Grundlage)", "Z3 (Tempo)", "Z4 (Schwelle)", "Z5 (Max)"]
    return labels[zone_idx]

# --- Datenverarbeitung ---
def process_data(raw_activities, user_max_hr):
    data = []
    if not raw_activities: return pd.DataFrame()

    for activity in raw_activities:
        act_type = activity.get('activityType', {}).get('typeKey', 'unknown')
        act_name = activity.get('activityName', 'Unbekannt')
        
        is_cycling = any(x in act_type.lower() for x in ['cycling', 'biking', 'ride', 'gravel', 'mtb', 'virtual'])
        if not is_cycling: continue

        avg_power = robust_get(activity, ['avgPower', 'averagePower', 'normPower'])
        max_20min = robust_get(activity, ['max20MinPower', 'maximum20MinPower', 'twentyMinPower'])
        avg_hr = robust_get(activity, ['avgHR', 'averageHR', 'avgHeartRate', 'averageHeartRate'])
        duration = round(activity.get('duration', 0) / 60, 1)
        calories = robust_get(activity, ['calories', 'totalCalories'])

        if avg_hr and duration > 5:
            power_val = int(avg_power) if avg_power else None
            max_20min_val = int(max_20min) if max_20min else 0
            stress_score = calculate_trimp(duration, avg_hr, user_max_hr)
            zone_idx = get_hr_zone_key(avg_hr, user_max_hr)
            zone_label = get_zone_label(zone_idx)

            data.append({
                "Datum": pd.to_datetime(activity['startTimeLocal'].split(' ')[0]),
                "Aktivität": act_name,
                "Leistung": power_val,
                "Max20Min": max_20min_val,
                "HF": int(avg_hr),
                "Dauer_Min": duration,
                "Stress": round(stress_score, 1),
                "ZoneIdx": zone_idx,
                "Zone": zone_label,
                "Kalorien": int(calories) if calories else 0
            })
    
    df = pd.DataFrame(data)
    if not df.empty:
        df.sort_values('Datum', inplace=True)
    return df

def generate_demo_data(days=120):
    data = []
    today = datetime.date.today()
    base_hr_max = 161 
    for i in range(days):
        if random.random() > 0.6: continue 
        date = today - datetime.timedelta(days=days-i)
        cycle_pos = (i % 28) / 28 
        load_factor = 0.5 + (cycle_pos * 0.8) 
        if cycle_pos > 0.8: load_factor = 0.4 
        ride_type = random.choice(['LIT', 'LIT', 'MIT', 'HIT']) 
        duration = 60
        avg_hr = 130
        power = 150 + (i * 0.2)
        max_20min = power * 1.1 
        calories = 600
        if ride_type == 'LIT': 
            duration = random.randint(90, 180) * load_factor
            avg_hr = 110 + random.randint(-5, 5)
            power = 160 + (i * 0.1)
            max_20min = power * 1.05 
            calories = duration * 10
        elif ride_type == 'MIT':
            duration = random.randint(60, 90)
            avg_hr = 135 + random.randint(-5, 5)
            power = 200 + (i * 0.2)
            max_20min = power * 1.1
            calories = duration * 12
        elif ride_type == 'HIT':
            duration = random.randint(45, 70) 
            avg_hr = 150 + random.randint(-5, 5)
            power = 240 + (i * 0.3)
            max_20min = power * 1.2 
            calories = duration * 15
        stress = calculate_trimp(duration, avg_hr, base_hr_max)
        zone_idx = get_hr_zone_key(avg_hr, base_hr_max)
        zone_label = get_zone_label(zone_idx)
        data.append({
            "Datum": pd.to_datetime(date),
            "Aktivität": f"{ride_type} Training",
            "Leistung": int(power),
            "Max20Min": int(max_20min),
            "HF": int(avg_hr),
            "Dauer_Min": int(duration),
            "Stress": round(stress, 1),
            "ZoneIdx": zone_idx,
            "Zone": zone_label,
            "Kalorien": int(calories)
        })
    return pd.DataFrame(data)

# --- UI Layout ---

with st.sidebar:
    st.header("⚙️ Setup")
    tab_login, tab_params = st.tabs(["Login", "Parameter"])
    with tab_login:
        # Hinweisbox VOR den Inputs für maximales Vertrauen
        st.info("🔒 **Datenschutz:** Deine Zugangsdaten werden **nur** für die Verbindung zu Garmin genutzt und **nicht gespeichert**. Alles läuft sicher im Arbeitsspeicher.")
        
        email = st.text_input("Garmin E-Mail")
        password = st.text_input("Passwort", type="password")
        today = datetime.date.today()
        default_start = today - datetime.timedelta(days=120)
        date_range = st.date_input("Zeitraum", [default_start, today])
        
        st.caption("ℹ️ Daten werden temporär (60min) gecached, damit du schneller analysieren kannst.")
        
        col1, col2 = st.columns(2)
        start_btn = col1.button("Start", type="primary")
        demo_btn = col2.button("Demo")
    with tab_params:
        st.subheader("Deine Physiologie")
        user_max_hr = st.number_input("Max Herzfrequenz", 100, 220, 161, help="Beeinflusst Zonen & Stress-Score.")
        st.subheader("Analyse Einstellungen")
        comparison_weeks = st.slider("Vergleichs-Fenster (Wochen)", 2, 12, 4)
        target_hr = st.slider("Aerobe Schwelle (Vergleichs-Puls)", 100, 170, 135)
        hr_tol = st.slider("Toleranz (+/- bpm)", 2, 15, 5)

st.title("🚴 Garmin Science Lab V6.3")
st.markdown("Analyse von **Effizienz**, **Belastung (ACWR)** und **Wissenschaftlicher Trainingsverteilung**.")

with st.expander("📘 Wissenschaftlicher Guide: Warum diese Metriken?"):
    st.markdown("""
    ### 1. Aerobe Effizienz (Entkopplung)
    * **Ziel:** Bei gleichem Puls mehr Watt treten. Die orange Kurve sollte rechts unterhalb der blauen liegen.
    ### 2. ACWR (Verletzungsprävention)
    * **Ziel:** Sweet Spot (0.8 - 1.3). Vermeide Spitzen über 1.5 ("Danger Zone"), um Verletzungen vorzubeugen.
    ### 3. Intensitäts-Verteilung (Modelle)
    * **Wenig Zeit (< 5h):** "Sweet Spot". Qualität statt Quantität.
    * **Viel Zeit (> 10h):** "Polarized". 80% locker, 20% hart.
    """)

# --- Logic ---
if 'df' not in st.session_state: st.session_state.df = None

if start_btn and email and password and len(date_range) == 2:
    with st.spinner("Lade Daten..."):
        raw, err = fetch_garmin_raw(email, password, date_range[0], date_range[1])
        if err: st.error(err)
        else:
            processed = process_data(raw, user_max_hr)
            if not processed.empty:
                st.session_state.df = processed
                st.success(f"{len(processed)} Aktivitäten geladen.")
            else: st.warning("Keine Rad-Daten gefunden.")
elif demo_btn:
    st.session_state.df = generate_demo_data()

# --- DASHBOARD ---
if st.session_state.df is not None:
    df = st.session_state.df.copy()
    
    # Highscores
    st.markdown("### 🏆 Bestwerte (Zeitraum)")
    m1, m2, m3, m4 = st.columns(4)
    if 'Max20Min' in df and df['Max20Min'].max() > 0:
        best = df.loc[df['Max20Min'].idxmax()]
        m1.metric("Beste 20min Power", f"{int(best['Max20Min'])} W", best['Datum'].strftime('%d.%m.'))
    elif 'Leistung' in df:
        best = df.loc[df['Leistung'].idxmax()]
        m1.metric("Beste Ø Watt", f"{int(best['Leistung'])} W", "N/A")
    
    if 'Stress' in df and df['Stress'].max() > 0:
        hardest = df.loc[df['Stress'].idxmax()]
        m2.metric("Härtestes Training", f"{int(hardest['Stress'])} Score", hardest['Datum'].strftime('%d.%m.'))
    
    total_km = int(df['Dauer_Min'].sum() * 0.5)
    m3.metric("Gesamtzeit", f"{int(df['Dauer_Min'].sum() / 60)}h", f"~{total_km} km")
    m4.metric("Kalorien Total", f"{int(df['Kalorien'].sum()):,} kcal".replace(",", "."))

    st.divider()
    
    tab1, tab2, tab3, tab4 = st.tabs(["🧬 Fitness-Shift", "⚖️ ACWR & Load", "📈 Zonen & Trends", "🎨 Zonen-Optimierer"])

    # --- TAB 1 ---
    with tab1:
        st.caption(f"Vergleich Start ({comparison_weeks} Wo.) vs. Ende ({comparison_weeks} Wo.).")
        # FIX: .copy() hinzugefügt
        df_power = df.dropna(subset=['Leistung']).copy()
        
        if not df_power.empty:
            min_d, max_d = df_power['Datum'].min(), df_power['Datum'].max()
            split_early = min_d + datetime.timedelta(weeks=comparison_weeks)
            split_late = max_d - datetime.timedelta(weeks=comparison_weeks)
            
            df_power['Phase'] = df_power['Datum'].apply(lambda d: "1. Start" if d <= split_early else ("2. Ende" if d >= split_late else "Mitte"))
            df_compare = df_power[df_power['Phase'] != "Mitte"]

            chart = alt.Chart(df_compare).mark_circle(size=80).encode(
                x=alt.X('Leistung', title='Leistung (Watt)', scale=alt.Scale(zero=False)),
                y=alt.Y('HF', title='Herzfrequenz (bpm)', scale=alt.Scale(zero=False)),
                color=alt.Color('Phase', scale=alt.Scale(range=['#3b82f6', '#f97316'])),
                tooltip=['Datum', 'Aktivität', 'Leistung', 'HF']
            )
            lines = chart.transform_regression('Leistung', 'HF', groupby=['Phase']).mark_line(size=3)
            st.altair_chart(chart + lines, width="stretch")
            
            c1, c2 = st.columns(2)
            c1.info("Ziel: Orange Linie rechts unterhalb der blauen.")
            
            df_zone = df_power[(df_power['HF'] >= target_hr - hr_tol) & (df_power['HF'] <= target_hr + hr_tol)]
            if len(df_zone) > 1:
                recent = df_zone[df_zone['Datum'] >= split_late]
                old = df_zone[df_zone['Datum'] <= split_early]
                if not recent.empty and not old.empty:
                    diff = int(recent['Leistung'].mean() - old['Leistung'].mean())
                    c2.metric(f"Leistung bei {target_hr} bpm", f"{int(recent['Leistung'].mean())} W", f"{diff} W")
                else: c2.warning("Zu wenig Daten in den Phasen.")
            else: c2.info("Keine Fahrten im gewählten Pulsbereich.")
        else: st.warning("Keine Leistungsdaten.")

    # --- TAB 2 ---
    with tab2:
        daily = df.set_index('Datum').resample('D')['Stress'].sum().fillna(0).to_frame()
        daily['Acute'] = daily['Stress'].rolling(7, min_periods=1).mean()
        daily['Chronic'] = daily['Stress'].rolling(28, min_periods=1).mean()
        daily['ACWR'] = (daily['Acute'] / daily['Chronic']).fillna(0)
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

    # --- TAB 3 ---
    with tab3:
        daily['Stunden'] = df.set_index('Datum').resample('D')['Dauer_Min'].sum().fillna(0).values / 60
        base = alt.Chart(daily.reset_index()).encode(x='Datum')
        bar = base.mark_bar(opacity=0.3, color='purple').encode(y='Stress', tooltip='Stress')
        line = base.mark_line(color='cyan').encode(y='Stunden', tooltip='Stunden')
        st.altair_chart(alt.layer(bar, line).resolve_scale(y='independent'), width="stretch")

    # --- TAB 4 ---
    with tab4:
        min_d, max_d = df['Datum'].min(), df['Datum'].max()
        weeks = max(1, (max_d - min_d).days / 7)
        vol = (df['Dauer_Min'].sum() / 60) / weeks
        
        if vol < 5.5:
            mod, targets = "Sweet Spot / Pyramidal", [10, 40, 30, 15, 5]
            msg = "Wenig Zeit (<5.5h) braucht Qualität (Zone 3/4)."
        elif vol < 10:
            mod, targets = "Hybrid", [15, 60, 15, 7, 3]
            msg = "Mittleres Volumen. Solide Basis, moderate Intensität."
        else:
            mod, targets = "Polarized (80/20)", [25, 55, 5, 10, 5]
            msg = "Hohes Volumen. Vermeide die graue Zone 3!"

        c1, c2 = st.columns(2)
        c1.metric("Ø Volumen", f"{vol:.1f} h/Woche")
        c2.metric("Modell", mod)
        st.info(msg)
        
        counts = df['ZoneIdx'].value_counts().sort_index()
        total = len(df)
        labels = ["Z1 (Erholung)", "Z2 (Grundlage)", "Z3 (Tempo)", "Z4 (Schwelle)", "Z5 (Max)"]
        
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
    st.info("👈 Bitte links starten.")