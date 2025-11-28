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
                if float_val >= 0: return float_val # Auch 0 akzeptieren
            except: continue
    return None

def calculate_trimp(duration_min, avg_hr, max_hr):
    if not max_hr or max_hr < 100: max_hr = 190 
    intensity = avg_hr / max_hr
    weighted_intensity = intensity * np.exp(1.92 * intensity) 
    return duration_min * weighted_intensity

def determine_smart_zone(avg_hr, max_hr_activity, user_max_hr):
    """
    Bestimmt die Zone intelligent. Erkennt Intervalle (niedriger Avg, hoher Max)
    und stuft diese entsprechend hoch, damit sie nicht als "Grundlage" verfälscht werden.
    """
    if not user_max_hr or user_max_hr < 100: return 0, "Z?"
    
    avg_pct = avg_hr / user_max_hr
    max_pct = max_hr_activity / user_max_hr if max_hr_activity else avg_pct
    
    # Standard Zonen (nach Coggan/Friel grob angenähert für HR)
    # 0=Z1, 1=Z2, 2=Z3, 3=Z4, 4=Z5
    
    # Basis-Klassifizierung nach Durchschnitt
    if avg_pct < 0.60: zone_idx = 0
    elif avg_pct < 0.75: zone_idx = 1 # Z2 geht oft bis 75% HFmax
    elif avg_pct < 0.85: zone_idx = 2
    elif avg_pct < 0.95: zone_idx = 3
    else: zone_idx = 4
    
    # INTELLIGENTE KORREKTUR FÜR INTERVALLE
    # Wenn der Max-Puls tief in Zone 5 (>92%) war, aber der Schnitt nur Z2/Z3,
    # war es wahrscheinlich ein hartes Intervall-Training.
    if max_pct > 0.92 and zone_idx < 3:
        zone_idx = 3 # Upgrade auf mindestens "Threshold/Hard" (Z4)
    elif max_pct > 0.88 and zone_idx < 2:
        zone_idx = 2 # Upgrade auf mindestens "Tempo" (Z3)

    labels = ["Z1 (Erholung)", "Z2 (Grundlage)", "Z3 (Tempo)", "Z4 (Schwelle)", "Z5 (Max)"]
    return zone_idx, labels[zone_idx]

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
        max_hr_activity = robust_get(activity, ['maxHR', 'maxHeartRate', 'maximumHeartRate'])
        
        duration = round(activity.get('duration', 0) / 60, 1)
        calories = robust_get(activity, ['calories', 'totalCalories'])
        
        distance = robust_get(activity, ['distance']) 
        elevation = robust_get(activity, ['totalAscent', 'elevationGain'])

        if avg_hr and duration > 5:
            power_val = int(avg_power) if avg_power else None
            max_20min_val = int(max_20min) if max_20min else 0
            stress_score = calculate_trimp(duration, avg_hr, user_max_hr)
            
            # Nutzung der neuen Smart-Zone Funktion
            zone_idx, zone_label = determine_smart_zone(avg_hr, max_hr_activity, user_max_hr)
            
            dist_km = round(distance / 1000, 1) if distance else 0.0
            elev_m = int(elevation) if elevation else 0

            data.append({
                "Datum": pd.to_datetime(activity['startTimeLocal'].split(' ')[0]),
                "Aktivität": act_name,
                "Leistung": power_val,
                "Max20Min": max_20min_val,
                "HF": int(avg_hr),
                "MaxHF": int(max_hr_activity) if max_hr_activity else int(avg_hr),
                "Dauer_Min": duration,
                "Stress": round(stress_score, 1),
                "ZoneIdx": zone_idx,
                "Zone": zone_label,
                "Kalorien": int(calories) if calories else 0,
                "Distanz": dist_km,
                "Anstieg": elev_m
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
        max_hr_activity = 140
        power = 150 + (i * 0.2)
        max_20min = power * 1.1 
        calories = 600
        
        speed = 28 + (random.random() * 5)
        dist_km = (duration/60) * speed
        elev_m = dist_km * random.randint(5, 15)

        if ride_type == 'LIT': 
            duration = random.randint(90, 180) * load_factor
            avg_hr = 110 + random.randint(-5, 5)
            max_hr_activity = avg_hr + random.randint(10, 20)
            power = 160 + (i * 0.1)
            max_20min = power * 1.05 
            calories = duration * 10
            dist_km = (duration/60) * 26
            elev_m = dist_km * 8
        elif ride_type == 'MIT':
            duration = random.randint(60, 90)
            avg_hr = 135 + random.randint(-5, 5)
            max_hr_activity = avg_hr + random.randint(10, 15)
            power = 200 + (i * 0.2)
            max_20min = power * 1.1
            calories = duration * 12
            dist_km = (duration/60) * 30
            elev_m = dist_km * 12
        elif ride_type == 'HIT':
            duration = random.randint(45, 70) 
            avg_hr = 145 + random.randint(-5, 5) # Schnitt oft nur Z3/Z4
            max_hr_activity = base_hr_max - random.randint(0, 5) # Aber Max ist Z5!
            power = 240 + (i * 0.3)
            max_20min = power * 1.2 
            calories = duration * 15
            dist_km = (duration/60) * 32
            elev_m = dist_km * 5

        stress = calculate_trimp(duration, avg_hr, base_hr_max)
        zone_idx, zone_label = determine_smart_zone(avg_hr, max_hr_activity, base_hr_max)
        
        data.append({
            "Datum": pd.to_datetime(date),
            "Aktivität": f"{ride_type} Training",
            "Leistung": int(power),
            "Max20Min": int(max_20min),
            "HF": int(avg_hr),
            "MaxHF": int(max_hr_activity),
            "Dauer_Min": int(duration),
            "Stress": round(stress, 1),
            "ZoneIdx": zone_idx,
            "Zone": zone_label,
            "Kalorien": int(calories),
            "Distanz": round(dist_km, 1),
            "Anstieg": int(elev_m)
        })
    return pd.DataFrame(data)

# --- UI Layout ---

with st.sidebar:
    st.header("⚙️ Setup")
    tab_login, tab_params = st.tabs(["Login", "Parameter"])
    with tab_login:
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

st.title("🚴 Garmin Science Lab V7 (Smart Intervals)")
st.markdown("Analyse von **Effizienz**, **Belastung (ACWR)** und **Wissenschaftlicher Trainingsverteilung**.")

with st.expander("📘 Wissenschaftlicher Guide: Warum diese Metriken?"):
    st.markdown("""
    ### 1. Aerobe Effizienz (Entkopplung)
    * **Ziel:** Bei gleichem Puls mehr Watt treten. Die orange Kurve sollte rechts unterhalb der blauen liegen.
    ### 2. ACWR (Verletzungsprävention)
    * **Ziel:** Sweet Spot (0.8 - 1.3). Vermeide Spitzen über 1.5 ("Danger Zone"), um Verletzungen vorzubeugen.
    ### 3. Intensitäts-Verteilung (Smart Intervals)
    * **Problem:** Intervalle haben oft einen niedrigen Durchschnittspuls wegen der Pausen.
    * **Lösung:** Diese App erkennt, wenn dein Max-Puls hoch war (z.B. >92%), der Schnitt aber niedrig, und wertet das Training korrekt als **Intensiv** (Zone 4/5), damit deine Bilanz stimmt.
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
    
    # Highscores (Dynamisch)
    st.markdown("### 🏆 Bestwerte (Gewählter Zeitraum)")
    m1, m2, m3, m4 = st.columns(4)
    
    # 1. Power
    if 'Max20Min' in df and df['Max20Min'].max() > 0:
        best = df.loc[df['Max20Min'].idxmax()]
        m1.metric("Beste 20min Power", f"{int(best['Max20Min'])} W", best['Datum'].strftime('%d.%m.'))
    elif 'Leistung' in df:
        best = df.loc[df['Leistung'].idxmax()]
        m1.metric("Beste Ø Watt", f"{int(best['Leistung'])} W", "N/A")
    
    # 2. Königsetappe / Weiteste Fahrt
    # Priorisiere Höhenmeter, wenn vorhanden (> 300hm in einer Fahrt), sonst Distanz
    max_elev = df['Anstieg'].max()
    if max_elev > 300:
        king_stage = df.loc[df['Anstieg'].idxmax()]
        m2.metric("Königsetappe", f"{int(king_stage['Anstieg'])} hm", f"{king_stage['Distanz']} km ({king_stage['Datum'].strftime('%d.%m.')})")
    else:
        longest = df.loc[df['Distanz'].idxmax()]
        m2.metric("Weiteste Fahrt", f"{longest['Distanz']} km", longest['Datum'].strftime('%d.%m.'))
    
    # 3. Total Stats (Real Data)
    total_km = int(df['Distanz'].sum())
    total_hours = int(df['Dauer_Min'].sum() / 60)
    m3.metric("Gesamtleistung", f"{total_km} km", f"{total_hours} Stunden")
    
    # 4. Calories
    m4.metric("Kalorien Total", f"{int(df['Kalorien'].sum()):,} kcal".replace(",", "."))

    st.divider()
    
    tab1, tab2, tab3, tab4 = st.tabs(["🧬 Fitness-Shift", "⚖️ ACWR & Load", "📈 Zonen & Trends", "🎨 Zonen-Optimierer"])

    # --- TAB 1 ---
    with tab1:
        st.caption(f"Vergleich Start ({comparison_weeks} Wo.) vs. Ende ({comparison_weeks} Wo.).")
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
                tooltip=['Datum', 'Aktivität', 'Leistung', 'HF', 'MaxHF']
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
        st.subheader(f"Intensitäts-Verteilung: Letzte {comparison_weeks} Wochen")
        
        max_date_in_data = df['Datum'].max()
        start_analysis = max_date_in_data - datetime.timedelta(weeks=comparison_weeks)
        df_recent = df[df['Datum'] >= start_analysis].copy()
        
        if not df_recent.empty:
            vol_total = df_recent['Dauer_Min'].sum() / 60
            vol_avg = vol_total / comparison_weeks 
            
            if vol_avg < 5.5:
                mod, targets = "Sweet Spot / Pyramidal", [10, 40, 30, 15, 5]
                msg = f"Wenig Zeit ({vol_avg:.1f}h) im aktuellen Fenster."
            elif vol_avg < 10:
                mod, targets = "Hybrid", [15, 60, 15, 7, 3]
                msg = f"Mittleres Volumen ({vol_avg:.1f}h)."
            else:
                mod, targets = "Polarized (80/20)", [25, 55, 5, 10, 5]
                msg = f"Hohes Volumen ({vol_avg:.1f}h)."

            c1, c2 = st.columns(2)
            c1.metric(f"Ø Volumen (Letzte {comparison_weeks} Wo.)", f"{vol_avg:.1f} h/Woche")
            c2.metric("Empfohlenes Modell", mod)
            st.info(msg)
            
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

else:
    st.info("👈 Bitte links starten.")