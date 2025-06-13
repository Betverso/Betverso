
import streamlit as st
import pandas as pd
import os
import requests
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Treina o modelo diretamente no código
df = pd.DataFrame({
    "goal_diff": [1, -1, 0, 2, -2],
    "avg_odds": [2.5, 3.2, 3.0, 2.0, 3.8],
    "home_odds_advantage": [-0.5, 0.3, -1.0, -0.8, 0.5],
    "result": ["H", "A", "D", "H", "A"]
})
X = df[["goal_diff", "avg_odds", "home_odds_advantage"]]
y = df["result"]
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y_encoded)

st.set_page_config(page_title="Betverso", layout="centered")
st.title("🏆 BETVERSO - Previsão de Resultados com IA")
st.markdown("---")

st.header("Informe os dados do jogo")
goal_diff = st.slider("Diferença de gols recente do mandante (mandante - visitante)", -5, 5, 0)
avg_odds = st.number_input("Média das odds (mandante, empate, visitante)", value=3.0)
home_adv = st.number_input("Vantagem de odd do mandante (odd mandante - odd visitante)", value=-1.0)

st.markdown("### Escolha a liga e a casa de apostas")
sport_options = {
    "Brasileirão": "soccer_brazil_campeonato",
    "Premier League": "soccer_epl",
    "La Liga": "soccer_spain_la_liga",
    "Bundesliga": "soccer_germany_bundesliga"
}
bookmaker_preference = st.selectbox("Casa de Apostas", ["bet365", "pinnacle", "unibet"])
selected_sport = st.selectbox("Campeonato", list(sport_options.keys()))

API_KEY = st.secrets["ODDS_API_KEY"]
SPORT_KEY = sport_options[selected_sport]
REGION = "br"
MARKET = "h2h"

odds_loaded = False
try:
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds"
    res = requests.get(url, params={
        "apiKey": API_KEY,
        "regions": REGION,
        "markets": MARKET,
        "oddsFormat": "decimal"
    })
    games = res.json()
    match = None
    for g in games:
        for b in g['bookmakers']:
            if b['key'] == bookmaker_preference:
                match = g
                outcomes = b['markets'][0]['outcomes']
                break
        if match:
            break

    if match:
        home_team = match['home_team']
        away_team = match['away_team']
        odds_dict = {o['name']: o['price'] for o in outcomes}
        odd_home = odds_dict.get(home_team, 2.0)
        odd_draw = odds_dict.get('Draw', 3.2)
        odd_away = odds_dict.get(away_team, 3.5)
        odds_loaded = True
        st.success(f"Odds carregadas para: {home_team} x {away_team}")
    else:
        st.warning("Nenhuma partida encontrada para a casa de aposta escolhida.")
except Exception as e:
    st.error("Erro ao carregar odds reais da API.")
    odd_home = 2.0
    odd_draw = 3.2
    odd_away = 3.5

if odds_loaded:
    st.write(f"Odd Mandante ({home_team}): {odd_home}")
    st.write(f"Odd Empate: {odd_draw}")
    st.write(f"Odd Visitante ({away_team}): {odd_away}")
else:
    home_team = "Time Mandante"
    away_team = "Time Visitante"
    odd_home = st.number_input("Odd para Vitória do Mandante", value=2.0)
    odd_draw = st.number_input("Odd para Empate", value=3.2)
    odd_away = st.number_input("Odd para Vitória do Visitante", value=3.5)

if st.button("🔍 Prever resultado e detectar apostas de valor"):
    input_data = pd.DataFrame({
        'goal_diff': [goal_diff],
        'avg_odds': [avg_odds],
        'home_odds_advantage': [home_adv]
    })

    prediction_proba = model.predict_proba(input_data)[0]
    predicted_class = label_encoder.inverse_transform([prediction_proba.argmax()])[0]

    st.subheader("🔮 Previsão do Resultado:")
    st.write(f"Resultado mais provável: **{predicted_class}**")

    st.subheader("📊 Probabilidades do modelo:")
    result_labels = label_encoder.classes_
    for label, prob in zip(result_labels, prediction_proba):
        st.write(f"{label}: {prob*100:.2f}%")

    st.markdown("---")
    st.subheader("💸 Análise de Value Bet")

    odds = {'H': odd_home, 'D': odd_draw, 'A': odd_away}
    evs = {}
    for label, prob in zip(result_labels, prediction_proba):
        ev = prob * odds[label] - 1
        evs[label] = ev
        if ev > 0:
            st.success(f"⚡ Aposta de valor detectada: {label} (EV: +{ev:.2f})")
        else:
            st.write(f"{label}: sem valor (EV: {ev:.2f})")

    history_path = "betverse_history.csv"
    new_entry = pd.DataFrame([{
        "Data": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "HomeTeam": home_team,
        "AwayTeam": away_team,
        "Predicted": predicted_class,
        "Real": "-",
        "EV": round(evs[predicted_class], 2),
        "Odd": odds[predicted_class],
        "Hit": False
    }])
    if os.path.exists(history_path):
        existing = pd.read_csv(history_path)
        updated = pd.concat([existing, new_entry], ignore_index=True)
    else:
        updated = new_entry
    updated.to_csv(history_path, index=False)

    st.markdown("---")

st.header("📈 Desempenho da IA - Betverso")
history_path = "betverse_history.csv"
if os.path.exists(history_path):
    df_hist = pd.read_csv(history_path)
    total = len(df_hist)
    acertos = df_hist['Hit'].sum()
    taxa_acerto = acertos / total * 100
    roi = ((df_hist[df_hist['Hit']]['Odd'] - 1).sum() - total + acertos) / total * 100
    ev_medio = df_hist['EV'].mean()
    total_value_bets = (df_hist['EV'] > 0).sum()

    st.subheader("🔢 Estatísticas Gerais")
    st.metric("Total de Previsões", total)
    st.metric("Taxa de Acerto", f"{taxa_acerto:.1f}%")
    st.metric("ROI Simulado", f"{roi:.1f}%")
    st.metric("Média de EV", f"{ev_medio:.2f}")
    st.metric("Apostas com Valor", total_value_bets)

    st.markdown("---")
    st.subheader("📋 Histórico de Previsões")
    st.dataframe(df_hist)

    st.markdown("---")
    st.subheader("🚀 Melhores Apostas do Dia")
    melhores = df_hist[df_hist['EV'] > 0].sort_values(by="EV", ascending=False).head(5)
    if not melhores.empty:
        for _, row in melhores.iterrows():
            st.markdown(f"**{row['HomeTeam']} x {row['AwayTeam']}** - Previsão: {row['Predicted']} | EV: {row['EV']} | Odd: {row['Odd']}")
    else:
        st.info("Nenhuma aposta de valor registrada ainda.")
else:
    st.warning("Nenhum histórico encontrado. Faça previsões para iniciar o acompanhamento.")

st.markdown("---")
st.info("Use essas previsões com responsabilidade. A IA do Betverso aprende com os dados, mas apostas envolvem riscos.")
