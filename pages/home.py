from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from pages import home, eda, acp, arboles_regresion, arboles_clasificacion, bosques_regresion, bosques_clasificacion, svm, cluster_AD

from dash_bootstrap_templates import load_figure_template,ThemeChangerAIO, template_from_url

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
theme_change = ThemeChangerAIO(
    aio_id="theme",button_props={
        "color": "danger",
        "children": "SELECT THEME",
        "outline": True,
    },
    radio_props={
        "persistence": True,
    },
)


button1 = html.Div(
    [
        dbc.Button("Haz click para probar el módulo de Clasificación", className="me-md-2", href="/arboles_clasificacion", size="sm"),
        dbc.Button("Haz click para probar el módulo de Regresión", className="me-md-2", href="/arboles_regresion", size="sm"),
    ],
    className="d-grid gap-2 d-md-flex justify-content-md-end",
)

button2 = html.Div(
    [
        dbc.Button("Haz click para probar el módulo de Clasificación", className="me-md-2", href="/bosques_clasificacion", size="sm"),
        dbc.Button("Haz click para probar el módulo de Regresión", className="me-md-2", href="/bosques_regresion", size="sm"),
    ],
    className="d-grid gap-2 d-md-flex justify-content-md-end",
)


card1 = dbc.Card(
    [
        dbc.CardImg(
            src="https://media-exp1.licdn.com/dms/image/C5612AQGW0WRABKg4lg/article-inline_image-shrink_1000_1488/0/1626408270230?e=1675900800&v=beta&t=02K808PEHiy3kvJ_Ko8DSfvxE04ArAo5Oz7ppB32vLw",
            top=True,
            style={"width": "100%", 'height': '50%', 'background-color': 'white'},
        ),
            dbc.CardBody(
                [
                    html.H6("Análisis Exploratorio de Datos (EDA)", className="card-title", style={'text-align': 'center'}),
                    html.P(
                        "Los científicos de datos utilizan el Análisis Exploratorio de Datos (EDA) para analizar e investigar conjuntos de datos y resumir sus principales características, a menudo empleando métodos de visualización de datos. Se hace un análisis descriptivo, viendo qué tipos de datos hay, si hay valores nulos, datos atípicos, y se hace un Análisis Correlacional de Datos, para ver si hay correlación entre las variables.",
                        className="card-text", style={'text-align': 'justify','font-size': '13px'}
                    ),
                    dbc.Button("Haz click para probar el módulo...", color="primary", href="/eda",
                    style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center'}),
                ]
            ),
    ],
    style={"width": "100%"},
)

card2 = dbc.Card(
    [
        dbc.CardImg(
            src="https://miro.medium.com/max/1400/1*T7CqlFV5aRm6MxO5nJt7Qw.gif",
            top=True,
            style={"width": "100%", "height": "50%"},
        ),
        dbc.CardBody(
            [
                html.H6("Análisis de Componentes Principales (ACP)", className="card-title", style={'text-align': 'center'}),
                html.P(
                    "El análisis de componentes principales (ACP o PCA, Principal Component Analysis) es un algoritmo para reducir la cantidad de variables de conjuntos de datos, mientras se conserva la mayor cantidad de información posible.",
                    className="card-text", style={'text-align': 'justify', 'font-size': '13px'}
                ),
                dbc.Button("Haz click para probar el módulo...", color="primary", href="/acp",
                    style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center'}),
            ]
        ), 
    ], 
    style={"width": "100%"},
)


card3 = dbc.Card(
    [
        dbc.CardImg(
            src="https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fc573e3d2-d2a4-4183-a2b1-0630d2c1ecdd_720x405.gif",
            top=True,
            style={"width": "100%", "height": "50%", 'background-color': 'white'},
        ),
        dbc.CardBody(
            [
                html.H6("Árboles de Decisión", className="card-title", style={'text-align': 'center'}),
                html.P(
                    "Los árboles de decisión son uno de los algoritmos más utilizados en el aprendizaje automático supervisado. Permiten resolver problemas de regresión y clasificación, admiten valores numéricos y nominales, aportan claridad (ya que despliegan los resultados en profundidad, de mayor a menor detalle, y tienen buena precisión en un amplio número de aplicaciones.",
                    className="card-text", 
                    style={'text-align': 'justify','font-size': '12px'}
                ),
                button1,
            ]
        ),
    ],
    style={"width": "100%"},
)


card4 = dbc.Card(
    [
        dbc.CardImg(
            src="https://1.bp.blogspot.com/-Ax59WK4DE8w/YK6o9bt_9jI/AAAAAAAAEQA/9KbBf9cdL6kOFkJnU39aUn4m8ydThPenwCLcBGAsYHQ/s0/Random%2BForest%2B03.gif",
            top=True,
            style={"width": "100%", "height": "40%", 'background-color': 'white'},
        ),
            dbc.CardBody(
            [
                html.H6("Bosques Aleatorios", className="card-title", style={'text-align': 'center'}),
                html.P(
                    "Los Bosques Aleatorios son la evolución natural de los Árboles. En algunas ocasiones los árboles de decisión tienen la tendencia de sobreajuste (overfit). Esto significa que tienden a aprender muy bien de los datos de entrenamiento, pero su generalización pudiera ser no tan buena. Una forma de mejorar la generalización de los árboles de decisión es combinar varios árboles, es decir, generar un Bosque Aleatorio (Random Forest), el cual es un poderoso algoritmo de aprendizaje automático, ampliamente utilizado en la actualidad. Debido a esta combinación, los bosques aleatorios tienen una capacidad de generalización alta.",
                    className="card-text", style={'text-align': 'justify','font-size': '12px'}
                ),
                button2,
            ]
        ),
    ],
    style={"width": "100%"},
)

card5 = dbc.Card(
    [
        dbc.CardImg(
            src="https://miro.medium.com/max/1400/0*Xe3YnRNqb7AuIUdu.gif",
            top=True,
            style={"width": "100%", "height": "40%", 'background-color': 'white'},
        ),
            dbc.CardBody(
                [
                    html.H6("Modelos combinados", className="card-title", style={'text-align': 'center'}),
                    html.P(
                        "En este módulo, se utiliza el algoritmo particional, conocido también como de particiones, para organizar elementos dentro de k clústeres y el algoritmo de árboles de decisión para hacer una clasificación multiclase, basándose en los clústeres generados.",
                        className="card-text", style={'text-align': 'justify','font-size': '12px'}
                    ),
                    dbc.Button("Haz click para probar el módulo...", color="primary", href="/cluster_AD",style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center'}),
                ]
            ),
    ], 
    style={"width": "100%"},
)


card6 = dbc.Card(
    [
        dbc.CardImg(
            src="https://miro.medium.com/max/848/1*VF_oqrRmgVdtAizny5T3-A.gif",
            top=True,
            style={"width": "100%", "height": "40%", 'background-color': 'white'},
        ),
            dbc.CardBody(
                [
                    html.H6("Support Vector Machine (SVM)", className="card-title", style={'text-align': 'center'}),
                    html.P(
                        "Las Máquinas de Soporte Vectorial (SVM) son un algoritmo que toma los datos como entrada y genera una línea (hiperplano) que separa a estos datos en dos clases. Pueden existir diversas líneas que separan las dos clases, pero se debe encontrar el hiperplano óptimo que maximiza el margen de separación. En pocas palabras, SVM busca la mejor línea de separación.",
                        className="card-text", style={'text-align': 'justify','font-size': '12px'}
                    ),
                    dbc.Button("Haz click para probar el módulo...", color="primary", href="/svm",style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center'}),
                ]
            ),
    ], 
    style={"width": "100%"},
)

layout = html.Div([
    dbc.CardGroup([card1, card2, card3], style={'width': '100%'}),
    html.Br(),
    dbc.CardGroup([card4, card5, card6], style={'width': '100%'}),
])

@callback(Output("page-content-home", "children"), [Input("url-home", "pathname")])
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
    elif pathname == "/svm":
        return svm.layout
    elif pathname == "/cluster_AD":
        return cluster_AD.layout
    
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )
