from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

layout = html.Div([
    # Página de inicio (La estilizamos con bootstrap)
    # Bienvenida a la página de inicio, con colores pastel y que se adapte a la pantalla
        html.H1('Smart Mining', style={'color': '#F5F5F5', 'font-size': '50px', 'text-align': 'center'}),
        html.H3('Bienvenido a Smart Mining', style={'color': '#F5F5F5', 'font-size': '30px', 'text-align': 'center'}),

],style={'background-color': '#2E2E2E', 'padding': '50px', 'margin': '0px', 'height': '100vh'})

