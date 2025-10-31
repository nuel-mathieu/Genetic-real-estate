import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=[1, 2, 3, 4],
    y=[3, 4, 2, 5],
    stackgroup='one',
    name='Positif'
))

fig.add_trace(go.Scatter(
    x=[1, 2, 3, 4],
    y=[-1, -3, -2, -4],
    stackgroup='one',
    name='Négatif'
))

fig.update_layout(title="Aires empilées avec valeurs négatives",
                  yaxis_title="Valeur",
                  xaxis_title="Temps",
                  hovermode='x unified')

fig.show()