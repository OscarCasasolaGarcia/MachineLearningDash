from dash import Dash, dcc, html, Input, Output, callback
from pages import home, eda, acp, arboles_regresion, arboles_clasificacion, bosques_regresion, bosques_clasificacion
import dash_bootstrap_components as dbc

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],suppress_callback_exceptions=True)
server = app.server

# A la pestaña de la página le ponemos el título
app.title = 'Smart Mining'

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "3rem 1rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE = {
    "margin-left": "1rem",
    "margin-right": "1rem",
    "padding": "1rem 1rem",
}

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/")),
        dbc.NavItem(dbc.NavLink("EDA📊", href="/eda")),
        dbc.NavItem(dbc.NavLink("ACP💻", href="/acp")),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("Árboles de Decisión (Clasificación)🔴🔵", href="/arboles_clasificacion"),
                dbc.DropdownMenuItem("Árboles de Decisión (Regresión)📈", href="/arboles_regresion"),
            ],
            nav=True,
            in_navbar=True,
            label="Árboles de Decisión🌳",
        ),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("Bosques Aleatorios (Clasificación)🔴🔵", href="/bosques_clasificacion"),
                dbc.DropdownMenuItem("Bosques Aleatorios (Regresión)📈", href="/bosques_regresion"),
            ],
            nav=True,
            in_navbar=True,
            label="Bosques aleatorios🌳🌳",
        ),
    ],
    brand="Smart Mining",
    brand_href="/",
    color="black",
    dark=True,
    sticky="top",
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), navbar, content])

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return home.layout
    elif pathname == "/eda":
        return eda.layout
    elif pathname == "/acp":
        return acp.layout
    elif pathname == "/arboles_clasificacion":
        return arboles_clasificacion.layout
    elif pathname == "/arboles_regresion":
        return arboles_regresion.layout
    elif pathname == "/bosques_clasificacion":
        return bosques_clasificacion.layout
    elif pathname == "/bosques_regresion":
        return bosques_regresion.layout
    
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )

if __name__ == "__main__":
    app.run_server(debug=True)