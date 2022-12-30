import base64
import datetime
import io
from msilib.schema import Component
import dash_bootstrap_components as dbc
import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib         
import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table, Input, Output, callback
import plotly.express as px
from dash_bootstrap_templates import load_figure_template,ThemeChangerAIO, template_from_url

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
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

tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': 'Black',
    'color': 'white',
    'padding': '6px'
}

layout = html.Div([
    html.H1('Exploratory Data Analysis (EDA)📊', style={'text-align': 'center'}),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or Select Files'
        ]),
        style={
            'width': '100%',
            'height': '100%',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'display': 'flex',
            'justify-content': 'center',
            'align-items': 'center',
            'flex-direction': 'column'
        },
        multiple=True, # Allow multiple files to be uploaded
        accept='.csv, .txt, .xls, .xlsx' # Restrict to csv, txt, xls, xlsx files
    ),
    html.Div(id='output-data-upload'),

])

def parse_contents(contents, filename,date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    global df
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        dbc.Alert('El archivo cargado es: {}'.format(filename), color="success"),
        # Solo mostramos las primeras 5 filas del dataframe, y le damos estilo para que las columnas se vean bien
        dash_table.DataTable(
            data=df.to_dict('records'),
            page_size=8,
            filter_action='native',
            sort_action='native',
            sort_mode='multi',
            column_selectable='single',
            row_deletable=True,
            cell_selectable=True,
            editable=True,
            row_selectable='multi',
            columns=[{'name': i, 'id': i, "deletable":True} for i in df.columns],
            style_table={'height': '300px', 'overflowX': 'auto'},
        ),
        
        html.Hr(),  # horizontal line

        dbc.Row([
            dbc.Col([
                dbc.Alert('El número de Filas del Dataframe es de: {}'.format(df.shape[0]), color="info"),
            ], width=6),
            dbc.Col([
                dbc.Alert('El número de Columnas del Dataframe es de: {}'.format(df.shape[1]), color="info"),
            ], width=6),
        ]),
        html.Hr(),

        dcc.Tabs([
            dcc.Tab(label='Tipos de datos, Valores Nulos y Valores Únicos', style=tab_style, selected_style=tab_selected_style,children=[
                html.Br(),
                dbc.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    # Primer columna: nombre de la columna y las demás columnas: nombre de las estadísticas (count, mean, std, min, 25%, 50%, 75%, max)
                                    html.Th('Variable'),
                                    html.Th('Tipo de dato'),
                                    html.Th('Count'),
                                    html.Th('Valores nulos'),
                                    html.Th('Valores únicos'),
                                    html.Th('Datos más frecuentes y su cantidad'),
                                    html.Th('Datos menos frecuentes y su cantidad'),
                                ]
                            )
                        ),
                        html.Tbody(
                            [
                                html.Tr(
                                    [
                                        html.Td(column), # Primera columna: nombre de la columna
                                        html.Td(
                                            str(df.dtypes[column]),
                                            style={
                                                'color': 'green' if df.dtypes[column] == 'float64' else 'blue' if df.dtypes[column] == 'int64' else 'red' if df.dtypes[column] == 'object' else 'orange' if df.dtypes[column] == 'bool' else 'purple'
                                            }
                                        ),

                                        # Count del tipo de dato (y porcentaje)
                                        html.Td(
                                            [
                                                html.P("{}".format(df[column].count())),
                                            ]
                                        ),

                                        html.Td(
                                            df[column].isnull().sum(),
                                            style={
                                                'color': 'red' if df[column].isnull().sum() > 0 else 'green'
                                            }
                                        ),

                                        #Valores únicos
                                        html.Td(
                                            df[column].nunique(),
                                            style={
                                                'color': 'green' if df[column].nunique() == 0 else 'black'
                                            }
                                        ),

                                        # Top valores más frecuentes
                                        html.Td(
                                            [
                                                html.P("{}".format(df[column].value_counts().index[0])+" ("+str(round(df[column].value_counts().values[0]*1,2))+")"),
                                            ]
                                        ),

                                        # Top valores menos frecuentes
                                        html.Td(
                                            [
                                                html.P("{}".format(df[column].value_counts().index[-1])+" ("+str(round(df[column].value_counts().values[-1]*1,2))+")"),
                                            ]
                                        ),
                                    ]
                                ) for column in df.dtypes.index
                            ]
                        )
                    ],
                    bordered=True,
                    hover=True,
                    responsive=True,
                    striped=True,
                    # Texto centrado y tabla alineada al centro de la página
                    style={'textAlign': 'center', 'width': '100%'}
                ),
            ]),

            dcc.Tab(label='Resumen estadístico', style=tab_style, selected_style=tab_selected_style,children=[
                html.Br(),
                dbc.Table(
                    # Mostamos el resumen estadístico de las variables de tipo object, con su descripción a la izquierda
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    # Primer columna: nombre de la estadística (count, mean, std, min, 25%, 50%, 75%, max) y las demás columnas: nombre de las columnas (recorremos las columnas del dataframe)
                                    html.Th('Estadística'),
                                    *[html.Th(column) for column in df.describe().columns]

                                ]
                            )
                        ),
                        html.Tbody(
                            [
                                html.Tr(
                                    [
                                        # Recorremos el for para mostrar el nombre de la estadística a la izquierda de cada fila
                                        html.Td('count'),
                                        *[html.Td(df.describe().loc['count'][column]) for column in df.describe().columns]
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td('mean'),
                                        *[html.Td(df.describe().loc['mean'][column]) for column in df.describe().columns]
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td('std'),
                                        *[html.Td(df.describe().loc['std'][column]) for column in df.describe().columns]
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td('min'),
                                        *[html.Td(df.describe().loc['min'][column]) for column in df.describe().columns]
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td('25%'),
                                        *[html.Td(df.describe().loc['25%'][column]) for column in df.describe().columns]
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td('50%'),
                                        *[html.Td(df.describe().loc['50%'][column]) for column in df.describe().columns]
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td('75%'),
                                        *[html.Td(df.describe().loc['75%'][column]) for column in df.describe().columns]
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td('max'),
                                        *[html.Td(df.describe().loc['max'][column]) for column in df.describe().columns]
                                    ]
                                ),
                            ]
                        )
                    ],

                    bordered=True,
                    hover=True,
                    responsive=True,
                    striped=True,
                    style={'textAlign': 'center', 'width': '100%'}
                ),
            ]),

            dcc.Tab(label='Identificación de valores atípicos', style=tab_style, selected_style=tab_selected_style,children=[
                # Mostramos un histograma por cada variable de tipo numérico:

                html.Div([
                    "Selecciona la o las variables para mostrar su histograma:",
                    dcc.Dropdown(
                        [i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2],
                        # Seleccionamos por defecto todas las columnas numéricas, a partir de la segunda
                        value=[i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2][1:3],
                        id='value-histograma-eda',
                        multi=True
                    ),

                    dcc.Graph(id='histograma-eda'),
                ]),
            ]),

            # Gráfica de cajas y bigotes
            dcc.Tab(label='Gráfica de cajas y bigotes', style=tab_style, selected_style=tab_selected_style,children=[
                html.Div([
                    "Selecciona la o las variables para mostrar su gráfica de cajas y bigotes:",
                    dcc.Dropdown(
                        [i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2],
                        # Seleccionamos por defecto todas las columnas numéricas, a partir de la segunda
                        value=[i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2][0:1],
                        id='value-bigotes-eda',
                        multi=True
                    ),
                
                dcc.Graph(id='bigotes-eda'),
                ]),
            ]),

            dcc.Tab(label='Análisis Correlacional', style=tab_style, selected_style=tab_selected_style,children=[
                html.Br(),

                dbc.Button(
                    "Haz click para obtener información adicional del Análisis Correlacional de Datos", id="open-body-scroll-eda", n_clicks=0
                ),

                dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle("Análisis Correlacional de Datos (ACD - CDA) 💭")),
                        dbc.ModalBody(
                            [
                                dcc.Markdown('''
                                💭 El ACD (CDA) es útil para reducir el número de variables, de un espacio de alta dimensión a uno de menor número de dimensiones. 

                                💭 Esto se logra a través de la identificación de variables significativas.

                                💭 Esta identificación de correlaciones se utiliza para determinar el grado de similitud (relevancia/irrelevancia) de los valores de dos variables numéricas.

                                💭 Existe correlación entre 2 variables (X,Y) si al aumentar los valores de X también los hacen de Y, o viceversa.

                                🧠 **Coeficiente de correlación de Pearson (r)**

                                    💭 Cuanto más cerca está R de 1 o -1, más fuerte es la correlación.

                                    💭 Si R es cercano a -1 las variables están correlacionadas negativamente.

                                    💭 Si R es 0, no hay correlación.

                                🧠 **Intervalos utilizados para la identificación de correlaciones**

                                    🔴 De -1.0 a -0.67 y 0.67 a 1.0 se conocen como correlaciones fuertes o altas. 

                                    🟡 De -0.66 a -0.34 y 0.34 a 0.66 se conocen como correlaciones moderadas o medias. 

                                    🔵 De -0.33 a 0.0 y 0.0 a 0.33 se conocen como correlaciones débiles o bajas.

                                '''),
                            ]
                        ),
                        dbc.ModalFooter(
                            dbc.Button(
                                "Close",
                                id="close-body-scroll-eda",
                                className="ms-auto",
                                n_clicks=0,
                            )
                        ),
                    ],
                    id="modal-body-scroll-eda",
                    scrollable=True,
                    is_open=False,
                    size='xl',
                ),
                
                dcc.Graph(
                    id='matriz',
                    figure={
                        'data': [
                            {'x': df.corr().columns, 'y': df.corr().columns, 'z': df.corr().values, 'type': 'heatmap', 'colorscale': 'RdBu'}
                        ],
                        'layout': {
                            'title': 'Matriz de correlación',
                            'xaxis': {'side': 'down'},
                            'yaxis': {'side': 'left'},
                            # Agregamos el valor de correlación por en cada celda (text_auto = True)
                            'annotations': [
                                dict(
                                    x=df.corr().columns[i],
                                    y=df.corr().columns[j],
                                    text=str(round(df.corr().values[i][j], 4)),
                                    showarrow=False,
                                    font=dict(
                                        color='white' if abs(df.corr().values[i][j]) >= 0.67  else 'black'
                                    )
                                ) for i in range(len(df.corr().columns)) for j in range(len(df.corr().columns))
                            ]
                        }
                    }
                )
            # Que cada pestaña se ajuste al tamaño de la ventana
            ]),

    ]) #Fin de la pestaña de análisis de datos
]) #Fin del layout


@callback(Output('output-data-upload', 'children'),
            Input('upload-data', 'contents'),
            State('upload-data', 'filename'),
            State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names,list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c,n,d) for c,n,d in
            zip(list_of_contents, list_of_names,list_of_dates)]
        return children


@callback(
    Output('histograma-eda', 'figure'),
    Input('value-histograma-eda', 'value'))
def update_graph1(value):
    # Conforme se van seleccionando las variables, se van agregando a la gráfica de histogramas
    import plotly.graph_objects as go
    fig = go.Figure()
    for i in value:
        fig.add_trace(go.Histogram(x=df[i], name=i))
    fig.update_layout(
        xaxis_title=str(", ".join(value)),
        yaxis_title='Variable(s)',
    )

    return fig


@callback(
    Output('bigotes-eda', 'figure'),
    Input('value-bigotes-eda', 'value'))
def update_graph2(value):
    # Conforme se van seleccionando las variables, se van agregando a la gráfica de bigotes
    import plotly.graph_objects as go
    fig = go.Figure()
    for i in value:
        fig.add_trace(go.Box(y=df[i], name=i, boxpoints='all'))
    fig.update_layout(
        yaxis_title='COUNT',
    )

    return fig


@callback(
    Output("modal-body-scroll-eda", "is_open"),
    [
        Input("open-body-scroll-eda", "n_clicks"),
        Input("close-body-scroll-eda", "n_clicks"),
    ],
    [State("modal-body-scroll-eda", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


