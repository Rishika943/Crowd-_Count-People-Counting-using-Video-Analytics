
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
import random
import time

# ================= CONFIG =================
ZONES = ["Entrance Gate", "Corridor 1", "Lobby Area", "Exit Gate"]
ZONE_LIMIT = 3

# ================= MOCK LIVE DATA =================
# (For presentation – no camera dependency)
zone_counts = {zone: 0 for zone in ZONES}
history = []

def generate_live_counts():
    global zone_counts, history
    for zone in ZONES:
        zone_counts[zone] = random.randint(0, 4)
    total = sum(zone_counts.values())
    history.append({
        "Time": len(history),
        "Total": total,
        **zone_counts
    })
    if len(history) > 15:
        history.pop(0)

# ================= DASH APP =================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Live People Count Dashboard"

app.layout = dbc.Container(fluid=True, children=[

    # Header
    dbc.Row(dbc.Col([
        html.H2("Live People Count Dashboard", className="fw-bold"),
        html.P("YOLOv8-based Crowd Monitoring System", className="text-muted"),
        html.Hr()
    ])),

    # Total Count Card
    dbc.Row(dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.H6("Total People"),
                html.H1(id="total-count", className="fw-bold text-primary")
            ]),
            className="shadow-sm text-center"
        ), width=4
    )),

    html.Br(),

    # Graphs
    dbc.Row([
        dbc.Col(dcc.Graph(id="zone-bar"), width=6),
        dbc.Col(dcc.Graph(id="zone-heatmap"), width=6)
    ]),

    dbc.Row([
        dbc.Col(dcc.Graph(id="trend-line"), width=12)
    ]),

    html.Hr(),

    # Zone Status Alerts
    dbc.Row(dbc.Col(
        html.Div(id="zone-status"),
        width=12
    )),

    dcc.Interval(id="timer", interval=1500)
])

# ================= CALLBACK =================
@app.callback(
    Output("zone-bar", "figure"),
    Output("zone-heatmap", "figure"),
    Output("trend-line", "figure"),
    Output("total-count", "children"),
    Output("zone-status", "children"),
    Input("timer", "n_intervals")
)
def update_dashboard(n):
    generate_live_counts()

    df = pd.DataFrame(history)
    counts = [zone_counts[z] for z in ZONES]
    total = sum(counts)

    # ----- Bar Chart -----
    bar_df = pd.DataFrame({"Zone": ZONES, "Count": counts})
    fig_bar = px.bar(
        bar_df, x="Zone", y="Count", text="Count",
        title="Zone-wise Population"
    )
    fig_bar.update_traces(textposition="outside")

    # ----- Heatmap -----
    heat = np.array(counts).reshape(1, -1)
    fig_heat = px.imshow(
        heat,
        x=ZONES,
        y=["Population"],
        text_auto=True,
        title="Heatmap (Detection Coordinates)"
    )

    # ----- Trend Line -----
    fig_trend = px.line(
        df, x="Time", y=ZONES,
        title="Population Trend (Recent)"
    )

    # ----- Zone Alerts -----
    alerts = []
    for zone in ZONES:
        count = zone_counts[zone]
        if count >= ZONE_LIMIT:
            alerts.append(
                dbc.Alert(
                    f"⚠ {zone}: {count} (LIMIT EXCEEDED)",
                    color="danger",
                    className="fw-bold mb-2"
                )
            )
        else:
            alerts.append(
                dbc.Alert(
                    f"✔ {zone}: {count} (OK)",
                    color="success",
                    className="mb-2"
                )
            )

    return fig_bar, fig_heat, fig_trend, total, alerts

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)

