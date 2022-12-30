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
import plotly.graph_objs as go         # Para la visualización de datos basado en plotly

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler  
from dash_bootstrap_templates import load_figure_template,ThemeChangerAIO, template_from_url

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
global df_original

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
    html.H1('Árboles de Decisión 🌳 (Regresión) 📈', style={'text-align': 'center'}),
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
        # Allow multiple files to be uploaded
        multiple=True,
        accept='.csv, .txt, .xls, .xlsx'
    ),
    html.Div(id='output-data-upload-arboles-regresion'), # output-datatable
    html.Div(id='output-div'),
])


def parse_contents(contents, filename,date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    global df
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            # Hacemos una copia del dataframe original para poder hacer las modificaciones que queramos
            df_original = df.copy()
            
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
            #Centramos la tabla de datos:
            data=df.to_dict('records'),
            page_size=8,
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ],
            filter_action='native',
            sort_action='native',
            sort_mode='multi',
            column_selectable='single',
            row_deletable=True,
            editable=True,
            row_selectable='multi',
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_cell={'textAlign': 'center', 'backgroundColor': 'rgb(207, 250, 255)', 'color': 'black'},
            style_header={'backgroundColor': 'rgb(45, 93, 255)', 'fontWeight': 'bold', 'color': 'black', 'border': '1px solid black'},
            style_table={'height': '300px', 'overflowY': 'auto'},
            style_data={'border': '1px solid black'}
        ),

        html.Hr(),

        # Devolvemos el número de filas y columnas del dataframe
        dbc.Row([
            dbc.Col([
                dbc.Alert('El número de Filas del Dataframe es de: {}'.format(df.shape[0]), color="info"),
            ], width=6),
            dbc.Col([
                dbc.Alert('El número de Columnas del Dataframe es de: {}'.format(df.shape[1]), color="info"),
            ], width=6),
        ]),
        
        html.Hr(),

        html.H4('Selección de características'),
        dcc.Tab(label='Analisis Correlacional', children=[
            dcc.Graph(
                id='matriz',
                figure={
                    # Solo se despliega la mitad de la matriz de correlación, ya que la otra mitad es simétrica
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
        ]),

        dcc.Tabs([
            dcc.Tab(label='Resúmen Estadístico', style=tab_style, selected_style=tab_selected_style,children=[
                html.Hr(),
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

            dcc.Tab(label='EDA', style=tab_style, selected_style=tab_selected_style,children=[
                # Tabla mostrando un resumen de las variables numéricas
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
        
            dcc.Tab(label='Distribución de Datos', style=tab_style, selected_style=tab_selected_style,children=[
                html.Div([
                    "Selecciona la variable X:",
                    dcc.Dropdown(
                        df.columns,
                        value=df.columns[0],
                        id='xaxis_column-arbol-regresion',
                    ),

                    "Selecciona las variables Y:",
                    dcc.Dropdown(
                        [i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2],
                        # Seleccionamos por defecto todas las columnas numéricas, a partir de la segunda
                        value=[i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2][1:3],
                        id='yaxis_column-arbol-regresion',
                        multi=True
                    ),

                    dcc.Graph(id='indicator_graphic_regression')
                ]),
            ]),

            dcc.Tab(label='Aplicación del algoritmo', style=tab_style, selected_style=tab_selected_style, children=[
                dbc.Badge("Selecciona la variable a predecir", color="light", className="mr-1", text_color="dark"),
                dcc.Dropdown(
                    [i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2],
                    value=df[[i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2]].columns[0],
                    id='Y_Clase_Arbol_Regresion',
                ),
                
                dbc.Badge("Selecciona las variables predictoras", color="light", className="mr-1", text_color="dark"),
                dcc.Dropdown(
                    # En las opciones que aparezcan en el Dropdown, queremos que aparezcan todas las columnas numéricas, excepto la columna Clase
                    [i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2],
                    value=[i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2][1:],
                    id='X_Clase_Arbol_Regresion',
                    multi=True,
                ),

                # Salto de línea
                html.Br(),

                html.H2(["", dbc.Badge("Calibración del algoritmo", className="ms-1")]),
                html.Br(),

                dcc.Markdown('''

                    💭 **Criterio de División**. El criterio de división consiste en dividir los datos en dos grupos: Datos de entrenamiento (training: 80%, 75% o 70% de los datos) y datos de prueba (test: 20%, 25% o 30% de los datos). Los datos de entrenamiento se utilizan para entrenar el modelo y los datos de prueba se utilizan para evaluar el modelo.

                    💭 **criterion**. Indica la función que se utilizará para dividir los datos. Puede ser (ganancia de información) gini y entropy (Clasificación). Cuando el árbol es de regresión se usan funciones como el error cuadrado medio (MSE).
                    
                    💭 **splitter**. Indica el criterio que se utilizará para dividir los nodos. Puede ser best o random. Best selecciona la mejor división mientras que random selecciona la mejor división aleatoriamente.                        
                    
                    💭 **max_depth**. Indica la máxima profundidad a la cual puede llegar el árbol. Esto ayuda a combatir el overfitting, pero también puede provocar underfitting.
                    
                    💭 **min_samples_split**. Indica la cantidad mínima de datos para que un nodo de decisión se pueda dividir. Si la cantidad no es suficiente este nodo se convierte en un nodo hoja.
                    
                    💭 **min_samples_leaf**. Indica la cantidad mínima de datos que debe tener un nodo hoja. 
                '''),

                dbc.Row([
                    dbc.Col([
                        dcc.Markdown('''**Criterio de División (Tamaño del test %):**'''),
                        dcc.Slider(0.2, 0.3, 0.05, value=0.2, marks={0.2: '20%', 0.25: '25%', 0.3: '30%'}, id='criterio_division_ADR'),
                    ], width=3, align='center'),

                ], justify='center', align='center'),

                html.Br(),

                dbc.Row([
                    dbc.Col([
                        dcc.Markdown('''**Criterio:**'''),
                        dbc.Select(
                            id='criterion_ADR',
                            options=[
                                {'label': 'Squared Error', 'value': 'squared_error'},
                                {'label': 'Friedman MSE', 'value': 'friedman_mse'},
                                {'label': 'Absolute Error', 'value': 'absolute_error'},
                                {'label': 'Poisson', 'value': 'poisson'},
                            ],
                            value='squared_error',
                            placeholder="Selecciona el criterio",
                        ),
                    ], width=2, align='center'),

                    dbc.Col([
                        dcc.Markdown('''**Splitter:**'''),
                        dbc.Select(
                            id='splitter_ADR',
                            options=[
                                {'label': 'Best', 'value': 'best'},
                                {'label': 'Random', 'value': 'random'},
                            ],
                            value='best',
                            placeholder="Selecciona el splitter",
                        ),
                    ], width=2, align='center'),

                    dbc.Col([
                        dcc.Markdown('''**Max_depth:**'''),
                        dbc.Input(
                            id='max_depth_ADR',
                            type='number',
                            placeholder='None',
                            value=None,
                            min=1,
                            max=100,
                            step=1,
                        ),
                    ], width=2, align='center'),

                    dbc.Col([
                        dcc.Markdown('''**Min_samples_split:**'''),
                        dbc.Input(
                            id='min_samples_split_ADR',
                            type='number',
                            placeholder='Selecciona el min_samples_split',
                            value=2,
                            min=1,
                            max=100,
                            step=1,
                        ),
                    ], width=2, align='center'),

                    dbc.Col([
                        dcc.Markdown('''**Min_samples_leaf:**'''),
                        dbc.Input(
                            id='min_samples_leaf_ADR',
                            type='number',
                            placeholder='Selecciona el min_samples_leaf',
                            value=1,
                            min=1,
                            max=100,
                            step=1,
                        ),
                    ], width=2, align='center'),
                ], justify='center', align='center'),

                html.Br(),

                # Estilizamos el botón con Bootstrap
                dbc.Button("Click para entrenar al algoritmo", color="danger", className="mr-1", id='submit-button-arbol-regresion', style={'width': '100%'}),

                html.Hr(),

                html.H2(["", dbc.Badge("Comparación Valores Reales vs Predicción", className="ms-1")]),
                # Mostramos la matriz de confusión
                dcc.Graph(id='matriz-arbol-regresion'),

                html.Hr(),

                html.H2(["", dbc.Badge("Reporte de la efectividad del algoritmo y del árbol de decisión obtenido", className="ms-1")]),
                # Mostramos el reporte de clasificación
                html.Div(id='clasificacion-arbol-regresion'),

                # Mostramos la importancia de las variables
                dcc.Graph(id='importancia-arbol-regresion'),

                html.Hr(),

                dbc.Button(
                    "Haz click para visualizar el árbol de decisión obtenido", id="open-body-scroll-ADR", n_clicks=0, color="primary", className="mr-1", style={'width': '100%'}
                ),

                dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle("Árbol de Decisión obtenido")),
                        dbc.ModalBody(
                            [
                                html.Div(id='arbol-arbol-regresion'),
                            ]
                        ),
                        dbc.ModalFooter(
                            dbc.Button(
                                "Close",
                                id="close-body-scroll-ADR",
                                className="ms-auto",
                                n_clicks=0,
                            )
                        ),
                    ],
                    id="modal-body-scroll-ADR",
                    scrollable=True,
                    is_open=False,
                    size='xl',
                ),
            ]),

            dcc.Tab(label='Nuevos Pronósticos', style=tab_style, selected_style=tab_selected_style, children=[
                html.Div(id="output-regresion-arbol-regresion-Final"),

                html.Div(id='valor-regresion2'),
                html.Div(id='valor-regresion'),

                html.Hr(),

                dcc.Store(id='memory-output-arbol-regresion', data=df.to_dict('records')),
            ]),
        ])
    ])

@callback(Output('output-data-upload-arboles-regresion', 'children'),
            Input('upload-data', 'contents'),
            State('upload-data', 'filename'),
            State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names,list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n,d) for c, n,d in
            zip(list_of_contents, list_of_names,list_of_dates)]
        return children

@callback(
    Output('indicator_graphic_regression', 'figure'),
    Input('xaxis_column-arbol-regresion', 'value'),
    Input('yaxis_column-arbol-regresion', 'value'))
def update_graph2(xaxis_column2, yaxis_column2):
    # Conforme se van seleccionando las variables, se van agregando a la gráfica de líneas
    fig = go.Figure()
    for i in yaxis_column2:
        fig.add_trace(go.Scatter(x=df[xaxis_column2], y=df[i], mode='lines', name=i))
    fig.update_layout(xaxis_rangeslider_visible=True,showlegend=True, xaxis_title=xaxis_column2, yaxis_title='Valores',
                    font=dict(family="Courier New, monospace", size=18, color="black"))
    fig.update_traces(mode='markers+lines')

    return fig

@callback(
    Output('matriz-arbol-regresion', 'figure'),
    Output('clasificacion-arbol-regresion', 'children'),
    Output('importancia-arbol-regresion', 'figure'),
    Output('arbol-arbol-regresion', 'children'),
    Output('output-regresion-arbol-regresion-Final', 'children'),
    Output('valor-regresion2', 'children'),
    Input('submit-button-arbol-regresion', 'n_clicks'),
    State('X_Clase_Arbol_Regresion', 'value'),
    State('Y_Clase_Arbol_Regresion', 'value'),
    State('criterio_division_ADR', 'value'),
    State('criterion_ADR', 'value'),
    State('splitter_ADR', 'value'),
    State('max_depth_ADR', 'value'),
    State('min_samples_split_ADR', 'value'),
    State('min_samples_leaf_ADR', 'value'))
def regresion(n_clicks, X_Clase, Y_Clase, criterio_division,criterion, splitter, max_depth, min_samples_split, min_samples_leaf):
    if n_clicks is not None:
        global X
        X = np.array(df[X_Clase])
        Y = np.array(df[Y_Clase])

        global PronosticoAD

        from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from sklearn import model_selection

        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, 
                                                                                test_size = criterio_division, 
                                                                                random_state = 0,
                                                                                shuffle = True)

        #Se entrena el modelo a partir de los datos de entrada
        PronosticoAD = DecisionTreeRegressor(criterion = criterion, splitter = splitter, max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, random_state = 0)
        PronosticoAD.fit(X_train, Y_train)

        #Se genera el pronóstico
        Y_PronosticoArbol = PronosticoAD.predict(X_test)
        
        ValoresArbol = pd.DataFrame(Y_test, Y_PronosticoArbol)

        # Comparación de los valores reales y los pronosticados en Plotly
        fig = px.line(Y_test, color_discrete_sequence=['green'])
        fig.add_scatter(y=Y_PronosticoArbol, name='Y_Pronostico', mode='lines', line=dict(color='red'))
        fig.update_layout(title='Comparación de valores reales vs Pronosticados',xaxis_rangeslider_visible=True)
        #Cambiamos el nombre de la leyenda
        fig.update_layout(legend_title_text='Valores')
        fig.data[0].name = 'Valores Reales'
        fig.data[1].name = 'Valores Pronosticados'
        # Renombramos el nombre de las leyendas:
        fig.update_traces(mode='markers+lines') #Agregamos puntos a la gráfica
        
        
        criterio = PronosticoAD.criterion
        profundidad = PronosticoAD.get_depth()
        hojas = PronosticoAD.get_n_leaves()
        splitter_report = PronosticoAD.splitter
        nodos = PronosticoAD.get_n_leaves() + PronosticoAD.get_depth()
        #MAE:
        MAEArbol = mean_absolute_error(Y_test, Y_PronosticoArbol)
        #MSE:
        MSEArbol = mean_squared_error(Y_test, Y_PronosticoArbol)
        #RMSE:
        RMSEArbol = mean_squared_error(Y_test, Y_PronosticoArbol, squared=False)
        
        global ScoreArbol
        ScoreArbol = r2_score(Y_test, Y_PronosticoArbol)
        

        # Importancia de las variables
        importancia = pd.DataFrame({'Variable': list(df[X_Clase].columns),
                            'Importancia': PronosticoAD.feature_importances_}).sort_values('Importancia', ascending=False)

        # Graficamos la importancia de las variables
        fig2 = px.bar(importancia, x='Variable', y='Importancia', color='Importancia', color_continuous_scale='Bluered', text='Importancia')
        fig2.update_layout(title_text='Importancia de las variables', xaxis_title="Variables", yaxis_title="Importancia")
        fig2.update_traces(texttemplate='%{text:.2}', textposition='outside')
        fig2.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        fig2.update_layout(legend_title_text='Importancia de las variables')

        # Generamos en texto el árbol de decisión
        from sklearn.tree import export_text
        r = export_text(PronosticoAD, feature_names=list(df[X_Clase].columns))
        
        return fig, html.Div([
            dbc.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th("Score"),
                                html.Th("MAE"),
                                html.Th("MSE"),
                                html.Th("RMSE"),
                                html.Th("Criterion"),
                                html.Th("Splitter"),
                                html.Th("Profundidad"),
                                html.Th("Max_depth"),
                                html.Th("Min_samples_split"),
                                html.Th("Min_samples_leaf"),
                                html.Th("Nodos"),
                                html.Th("Hojas"),
                            ]
                        )
                    ),
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td(str(round(ScoreArbol, 6)*100) + '%', style={'color': 'green'}),
                                    html.Td(str(round(MAEArbol, 6))),
                                    html.Td(str(round(MSEArbol, 6))),
                                    html.Td(str(round(RMSEArbol, 6))),
                                    html.Td(criterio),
                                    html.Td(splitter_report),
                                    html.Td(profundidad),
                                    html.Td(str(max_depth)),
                                    html.Td(min_samples_split),
                                    html.Td(min_samples_leaf),
                                    html.Td(nodos),
                                    html.Td(PronosticoAD.get_n_leaves()),
                                ]
                            ),
                        ]
                    ),
                ],
                bordered=True,
                hover=True,
                responsive=True,
                striped=True,
                style={'width': '100%', 'text-align': 'center'},
                class_name='table table-hover table-bordered table-striped',
            ),


        ]), fig2, html.Div([
            dbc.Alert(r, color="success", style={'whiteSpace': 'pre-line'}, className="mb-3")
        ]), html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Input(id='values_X1', type="number", placeholder=df[X_Clase].columns[0],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[0])),
                    dbc.Input(id='values_X2', type="number", placeholder=df[X_Clase].columns[1],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[1])),
                    dbc.Input(id='values_X3', type="number", placeholder=df[X_Clase].columns[2],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[2])),
                    dbc.Input(id='values_X4', type="number", placeholder=df[X_Clase].columns[3],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[3])),
                    dbc.Input(id='values_X5', type="number", placeholder=df[X_Clase].columns[4],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[4])),
                ], width=6),
            ])

        ]), html.Div([
                dbc.Button("Haz click para mostrar el pronóstico", id="collapse-button", className="mb-3", color="primary"),
                dbc.Collapse(
                    dbc.Card(dbc.CardBody([
                        html.Div(id='output-container-button'),
                    ])),
                    id="collapse",
                ),
        ])

    elif n_clicks is None:
        import dash.exceptions as de
        raise de.PreventUpdate

# make sure that x and y values can't be the same variable
def filter_options(v):
    """Disable option v"""
    return [
        {"label": col, "value": col, "disabled": col == v}
        for col in [i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2]
    ]

# functionality is the same for both dropdowns, so we reuse filter_options
callback(Output("X_Clase_Arbol_Regresion", "options"), [Input("Y_Clase_Arbol_Regresion", "value")])(
    filter_options
)
callback(Output("Y_Clase_Arbol_Regresion", "options"), [Input("X_Clase_Arbol_Regresion", "value")])(
    filter_options
)

@callback(
    Output('valor-regresion', 'children'),
    Input('collapse-button', 'n_clicks'),
    # Mostar los valores de los inputs
    State('memory-output-arbol-regresion', 'data'),
    State('values_X1', 'value'),
    State('values_X2', 'value'),
    State('values_X3', 'value'),
    State('values_X4', 'value'),
    State('values_X5', 'value'),
)
def regresionFinal(n_clicks, data, values_X1, values_X2, values_X3, values_X4, values_X5):
    if n_clicks is not None:
        if values_X1 is None or values_X2 is None or values_X3 is None or values_X4 is None or values_X5 is None:
            return html.Div([
                dbc.Alert('Debe ingresar todos los valores de las variables', color="danger")
            ])
        else:
            XPredict = pd.DataFrame([[values_X1, values_X2, values_X3, values_X4, values_X5]])

            clasiFinal = PronosticoAD.predict(XPredict)
            return html.Div([
                dbc.Alert('El valor pronosticado con un árbol de decisión que tiene una Exactitud de: ' + str(round(ScoreArbol, 4)*100) + '% es: ' + str(clasiFinal[0]), color="success", style={'textAlign': 'center'})
            ])


@callback(
    Output("modal-body-scroll-ADR", "is_open"),
    [
        Input("open-body-scroll-ADR", "n_clicks"),
        Input("close-body-scroll-ADR", "n_clicks"),
    ],
    [State("modal-body-scroll-ADR", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open