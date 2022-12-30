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

from sklearn.preprocessing import StandardScaler, MinMaxScaler  
from dash_bootstrap_templates import load_figure_template,ThemeChangerAIO, template_from_url

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO])

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
    html.H1('Modelos Combinados🤖', style={'text-align': 'center'}),
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
        multiple=True,
        accept='.csv, .txt, .xls, .xlsx'
    ),
    html.Div(id='output-data-upload-modelos'), # output-datatable
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
            style_table={'height': '300px', 'overflowY': 'auto'},
        ),

        html.Hr(),  # horizontal line

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

        html.H2(["", dbc.Badge("Selección de características", className="ms-1")]),
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
                        [i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2],
                        # Seleccionamos la primera columna numérica del dataframe
                        value=df[[i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2]].columns[0],
                        id='xaxis_column-modelos',
                    ),

                    "Selecciona la variable Y:",
                    dcc.Dropdown(
                        [i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2],
                        # Seleccionamos la segunda columna numérica del dataframe
                        value=df[[i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2]].columns[1],
                        id='yaxis_column-modelos',
                        placeholder="Selecciona la variable Y"
                    ),

                    "Selecciona la variable a Clasificar:",
                    dcc.Dropdown(
                        [i for i in df.columns if (df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) == 2) or df[i].dtype in ['bool'] or (df[i].dtype in ['object'] and len(df[i].unique()) == 2)],
                        # Seleccionamos por defecto la primera columna
                        value=df[[i for i in df.columns if (df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) == 2) or df[i].dtype in ['bool'] or (df[i].dtype in ['object'] and len(df[i].unique()) == 2)]].columns[0],
                        id='caxis_column-modelos',
                        placeholder="Selecciona la variable Predictora"
                    ),
                ]),

                dcc.Graph(id='indicator_graphic-modelos'),
            ]),

            dcc.Tab(label='Aplicación del algoritmo', style=tab_style, selected_style=tab_selected_style, children=[
                dbc.Badge("Selecciona las variables predictoras", color="light", className="mr-1", text_color="dark"),
                    dcc.Dropdown(
                        [i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2],
                        # Seleccionamos la segunda columna numérica del dataframe
                        value=df[[i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2]].columns,
                        id='X_Clase-modelos',
                        multi=True,
                    ),

                # Seleccionamos la variable Clase con un Dropdown
                dbc.Badge("Selecciona la variable Clase", color="light", className="mr-1", text_color="dark"),
                dcc.Dropdown(
                    [i for i in df.columns if (df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) == 2) or df[i].dtype in ['bool'] or (df[i].dtype in ['object'] and len(df[i].unique()) == 2)],
                    # Seleccionamos por defecto la primera columna
                    value=df[[i for i in df.columns if (df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) == 2) or df[i].dtype in ['bool'] or (df[i].dtype in ['object'] and len(df[i].unique()) == 2)]].columns[0],
                    id='Y_Clase-modelos',
                    multi=True,
                ),

                # Salto de línea
                html.Br(),

                html.H2(["", dbc.Badge("Calibración del algoritmo", className="ms-1")]),
                html.Br(),

                dcc.Markdown('''
                    💭 **criterion**. Indica la función que se utilizará para dividir los datos. Puede ser (ganancia de información) gini y entropy (Clasificación). Cuando el árbol es de regresión se usan funciones como el error cuadrado medio (MSE).
                    
                    💭 **splitter**. Indica el criterio que se utilizará para dividir los nodos. Puede ser best o random. Best selecciona la mejor división mientras que random selecciona la mejor división aleatoriamente.                        
                    
                    💭 **max_depth**. Indica la máxima profundidad a la cual puede llegar el árbol. Esto ayuda a combatir el overfitting, pero también puede provocar underfitting.
                    
                    💭 **min_samples_split**. Indica la cantidad mínima de datos para que un nodo de decisión se pueda dividir. Si la cantidad no es suficiente este nodo se convierte en un nodo hoja.
                    
                    💭 **min_samples_leaf**. Indica la cantidad mínima de datos que debe tener un nodo hoja. 
                '''),

                dbc.Row([
                    dbc.Col([
                        dcc.Markdown('''**Criterio:**'''),
                        dbc.Select(
                            id='criterion-modelos',
                            options=[
                                {'label': 'Gini', 'value': 'gini'},
                                {'label': 'Entropy', 'value': 'entropy'},
                            ],
                            value='gini',
                            placeholder="Selecciona el criterio",
                        ),
                    ], width=2, align='center'),

                    dbc.Col([
                        dcc.Markdown('''**Splitter:**'''),
                        dbc.Select(
                            id='splitter-modelos',
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
                            id='max_depth-modelos',
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
                            id='min_samples_split-modelos',
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
                            id='min_samples_leaf-modelos',
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

                dbc.Button("Click para entrenar al algoritmo", color="danger", className="mr-1", id='submit-button-modelos',style={'width': '100%'}),

                html.Hr(),

                dcc.Graph(id='kmeans-elbow-modelos'),

                html.Div(id='table-kmeans-modelos'),

                dbc.Table(id='table-centroides-modelos', bordered=True, dark=True, hover=True, responsive=True, striped=True),

                dcc.Graph(id='kmeans-3d-modelos'),

                # Mostramos la matriz de confusión
                html.Div(id='matriz-modelos'),

                html.Hr(),

                # Mostramos el reporte de clasificación
                html.Div(id='clasificacion-modelos'),

                # Mostramos la importancia de las variables
                dcc.Graph(id='importancia-modelos'),
                # Ocultamos el gráfico de la importancia de las variables hasta que se pulse el botón

                html.Hr(),

                html.H2(["", dbc.Badge("Curva ROC", className="ms-1")]),
                dcc.Graph(id='roc-arbol-clasificacion-modelos'),


                dbc.Button(
                    "Haz click para visualizar el árbol de decisión obtenido", id="open-body-scroll-modelos", n_clicks=0, color="primary", className="mr-1", style={'width': '100%'}
                ),

                dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle("Árbol de Decisión obtenido")),
                        dbc.ModalBody(
                            [
                                html.Div(id='arbol-modelos'),
                            ]
                        ),
                        dbc.ModalFooter(
                            dbc.Button(
                                "Close",
                                id="close-body-scroll-modelos",
                                className="ms-auto",
                                n_clicks=0,
                            )
                        ),
                    ],
                    id="modal-body-scroll-modelos",
                    scrollable=True,
                    is_open=False,
                    size='xl',
                ),
            ]),

            dcc.Tab(label='Nuevas Clasificaciones', style=tab_style, selected_style=tab_selected_style, children=[
                html.H2(["", dbc.Badge("Introduce los datos de las nuevas clasificaciones", className="ms-1")]),
                html.Hr(),
                html.Div(id='output-clasificacion-modelos'),
                html.Hr(),
                html.Div(id='valor-clasificacion-modelos'),
                html.Div(id='valor-clasificacion-modelos2'),
            ]),
        ])
    ]) #Fin del layout

@callback(Output('output-data-upload-modelos', 'children'),
            Input('upload-data', 'contents'),
            State('upload-data', 'filename'),
            State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names,list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n,d) for c, n,d in
            zip(list_of_contents, list_of_names,list_of_dates)]
        return children

# CALLBACK PARA LA SELECCIÓN DEL USUARIO
@callback(
    Output('indicator_graphic-modelos', 'figure'),
    Input('xaxis_column-modelos', 'value'),
    Input('yaxis_column-modelos', 'value'),
    Input('caxis_column-modelos', 'value'))
def update_graph(xaxis_column, yaxis_column, caxis_column):
    dff = df
    dff[caxis_column] = dff[caxis_column].astype('category')
    fig = px.scatter(dff, x=xaxis_column, y=yaxis_column, color=caxis_column, title='Gráfico de dispersión',symbol=caxis_column,marginal_x="histogram", marginal_y="histogram")
    fig.update_layout(showlegend=True, xaxis_title=xaxis_column, yaxis_title=yaxis_column,
                    font=dict(family="Courier New, monospace", size=18, color="black"),legend_title_text=caxis_column)
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))
    # str(df.groupby(caxis_column).size()[0])
    return fig

@callback(
    Output('kmeans-elbow-modelos', 'figure'),
    Output('table-kmeans-modelos', 'children'),
    Output('table-centroides-modelos', 'children'),
    Output('kmeans-3d-modelos', 'figure'),
    Output('matriz-modelos', 'children'),
    Output('clasificacion-modelos', 'children'),
    Output('importancia-modelos', 'figure'),
    Output('roc-arbol-clasificacion-modelos', 'figure'),
    Output('arbol-modelos', 'children'),
    Output('output-clasificacion-modelos', 'children'),
    Output('valor-clasificacion-modelos', 'children'),
    Input('submit-button-modelos','n_clicks'),
    State('X_Clase-modelos', 'value'),
    State('Y_Clase-modelos', 'value'),
    State('criterion-modelos', 'value'),
    State('splitter-modelos', 'value'),
    State('max_depth-modelos', 'value'),
    State('min_samples_split-modelos', 'value'),
    State('min_samples_leaf-modelos', 'value'),
    State(ThemeChangerAIO.ids.radio("theme"), 'value'))
def clasificacion(n_clicks, X_Clase, Y_Clase, criterion, splitter, max_depth, min_samples_split, min_samples_leaf, theme):
    if n_clicks is not None:
        global ClasificacionAD
        X = np.array(df[X_Clase])
        Y = np.array(df[Y_Clase])
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
        from sklearn import model_selection
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        from sklearn.cluster import KMeans
        from sklearn.metrics import pairwise_distances_argmin_min

        estandarizar = StandardScaler()                         # Se instancia el objeto StandardScaler o MinMaxScaler 
        MEstandarizada = estandarizar.fit_transform(X)          # Se estandarizan los datos de entrada


        #Definición de k clusters para K-means
        #Se utiliza random_state para inicializar el generador interno de números aleatorios
        SSE = []
        for i in range(2, 10):
            km = KMeans(n_clusters=i, random_state=0)
            km.fit(MEstandarizada)
            SSE.append(km.inertia_)

        from kneed import KneeLocator
        kl = KneeLocator(range(2, 10), SSE, curve="convex", direction="decreasing")

        fig = px.line(x=range(2, 10), y=SSE, labels=dict(x="Cantidad de clusters *k*", y="SSE"), title='Elbow Method (Knee Point)')
        fig.update_traces(mode='markers+lines')
        fig.add_vline(x=kl.elbow, line_width=3, line_dash="dash", line_color="red")
        fig.add_annotation(x=kl.elbow, y=kl.knee_y, text="Knee Point", showarrow=True, arrowhead=1)

        #Se crean las etiquetas de los elementos en los clusters
        MParticional = KMeans(n_clusters=kl.elbow, random_state=0).fit(MEstandarizada)
        MParticional.predict(MEstandarizada)

        dff = df.copy()

        dff['Cluster'] = MParticional.labels_

        tablekmeans = dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in dff.columns],
            data=dff.to_dict('records'),
            page_size=8,
            style_cell={'textAlign': 'center', 'font-family': 'sans-serif', 'font-size': '14px'},
            style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ],
            style_table={'maxHeight': '300px', 'overflowY': 'scroll'},
            style_cell_conditional=[
                {
                    'if': {'column_id': 'Cluster'},
                    'width': '10%'
                }
            ],
            filter_action='native',
            sort_action='native',
            sort_mode='multi',
            column_selectable='single',
        )

        CentroidesP = dff.groupby('Cluster').mean().round(4)
        CentroidesP['Cluster'] = CentroidesP.index
        CentroidesP['Cantidad de elementos del clúster'] = dff.groupby('Cluster')['Cluster'].count()
        

        # Se crea la tabla de los centroides
        tablecentroides = dbc.Table.from_dataframe(CentroidesP, striped=True, bordered=True, hover=True, responsive=True)

        numcolores = len(CentroidesP)
        import random
        colores = []
        for i in range(numcolores):
            colores.append('#%06X' % random.randint(0, 0xFFFFFF))
        asignar=[]
        for row in MParticional.labels_:
            asignar.append(colores[row])
        
        fig2 = go.Figure(data=[go.Scatter3d(x=MEstandarizada[:, 0], y=MEstandarizada[:, 1], z=MEstandarizada[:, 2], mode='markers', marker=dict(color=asignar, size=6, line=dict(color=asignar, width=12)), text=dff['Cluster'])])
        # Se añaden los centros de los clusters en otros colores
        fig2.add_trace(go.Scatter3d(x=MParticional.cluster_centers_[:, 0], y=MParticional.cluster_centers_[:, 1], z=MParticional.cluster_centers_[:, 2], mode='markers', marker=dict(color='purple', size=12, line=dict(color='black', width=12)), text=np.arange(kl.elbow)))
        # Se oculta la leyenda
        fig2.update_layout(showlegend=False)


        Y2 = np.array(dff['Cluster'])
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y2,
                                                                                        test_size = 0.2, 
                                                                                        random_state = 0,
                                                                                        shuffle = True)

        #Se entrena el modelo a partir de los datos de entrada
        ClasificacionAD = DecisionTreeClassifier(criterion = criterion,
                                                splitter = splitter,
                                                max_depth = max_depth,
                                                min_samples_split = min_samples_split,
                                                min_samples_leaf = min_samples_leaf,
                                                random_state = 0)
        ClasificacionAD.fit(X_train, Y_train)

        #Se etiquetan las clasificaciones
        Y_Clasificacion = ClasificacionAD.predict(X_validation)
        Valores = pd.DataFrame(Y_validation, Y_Clasificacion)

        #Se calcula la exactitud promedio de la validación
        exactitud = accuracy_score(Y_validation, Y_Clasificacion)
        
        #Matriz de clasificación
        ModeloClasificacion1 = ClasificacionAD.predict(X_validation)
        Matriz_Clasificacion1 = pd.crosstab(Y_validation.ravel(), 
                                        ModeloClasificacion1, 
                                        rownames=['Reales'], 
                                        colnames=['Clasificación'])
        Matriz_Clasificacion1.index.set_names("Reales", inplace = True)
        Matriz_Clasificacion1.columns.set_names("Clasificación", inplace = True)

        criterio = ClasificacionAD.criterion
        splitter_report = ClasificacionAD.splitter
        profundidad = ClasificacionAD.get_depth()
        hojas = ClasificacionAD.get_n_leaves()
        nodos = ClasificacionAD.get_n_leaves() + ClasificacionAD.get_depth()
        tasa_error = 1-ClasificacionAD.score(X_validation, Y_validation)
        clustersnames = np.arange(kl.elbow)
        reporte = pd.DataFrame(classification_report(Y_validation, Y_Clasificacion, output_dict=True,
                                            #Se incluye la primera columna
                                            target_names=clustersnames)).transpose()
        reporte2 = pd.DataFrame({'Index': reporte.index,
            'Precision': reporte['precision'],
            'Recall': reporte['recall'],
            'F1': reporte['f1-score'],
            'Soporte': reporte['support']})
        
        especificidad1 = []
        for i in range(np.shape(Matriz_Clasificacion1)[0]):
            FP = Matriz_Clasificacion1.iloc[:, i].sum() - Matriz_Clasificacion1.iloc[i, i]
            FN = Matriz_Clasificacion1.iloc[i, :].sum() - Matriz_Clasificacion1.iloc[i, i]
            VP = Matriz_Clasificacion1.iloc[i, i]
            # VN = Suma de todos los elementos de la matriz - (FP + FN + VP)
            VN = Matriz_Clasificacion1.sum().sum() - (FP + FN + VP)
            especificidad2 = VN / (VN + FP)
            especificidad1.append(especificidad2)

        # Importancia de las variables
        importancia = pd.DataFrame({'Variable': list(df[X_Clase].columns),
                            'Importancia': ClasificacionAD.feature_importances_}).sort_values('Importancia', ascending=False)

        # Graficamos la importancia de las variables
        fig3 = px.bar(importancia, x='Variable', y='Importancia', color='Importancia', color_continuous_scale='Bluered', text='Importancia')
        fig3.update_layout(title_text='Importancia de las variables', xaxis_title="Variables", yaxis_title="Importancia")
        fig3.update_traces(texttemplate='%{text:.2}', textposition='outside')
        fig3.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        fig3.update_layout(legend_title_text='Importancia de las variables')

        #CURVA ROC
        Y_validation2 = pd.DataFrame(Y_validation) # Convertimos los valores de la variable Y_validation a un dataframe
        # Reeemplazamos los valores de la variable Y_validation2 por 0 y 1:
        # Checamos si el dataframe tiene dos valores únicos y si esos son 1 y 0
        if len(Y_validation2[0].unique()) == 2 and Y_validation2[0].unique()[0] == 0 and Y_validation2[0].unique()[1] == 1 or len(Y_validation2[0].unique()) == 2 and Y_validation2[0].unique()[0] == 1 and Y_validation2[0].unique()[1] == 0:
            pass
        else:
            Y_validation2 = Y_validation2.replace([Y_validation2[0].unique()[0],Y_validation2[0].unique()[1]],[1,0])

        #Rendimiento
        clusters = np.arange(kl.elbow)
        from sklearn.preprocessing import label_binarize
        y_score = ClasificacionAD.predict_proba(X_validation)
        y_test_bin = label_binarize(Y_validation, classes=clusters)
        n_classes = y_test_bin.shape[1]

        #Se calcula la curva ROC y el área bajo la curva para cada clase
        from sklearn.metrics import roc_curve, auc

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        # Hacemos la curva ROC en Plotly
        fig4 = go.Figure()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            fig4.add_trace(go.Scatter(x=fpr[i], y=tpr[i], name='Clase {}'.format(i)+', AUC: {}'.format(auc(fpr[i], tpr[i]).round(6))))
        fig4.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Referencia', line=dict(color='navy', dash='dash')))
        fig4.update_layout(title_text='Rendimiento', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')

        # Generamos en texto el árbol de decisión
        from sklearn.tree import export_text
        r = export_text(ClasificacionAD, feature_names=list(df[X_Clase].columns))


        namesColumnsClustersReal = []
        namesColumnsClustersClass = []

        for i in range(0, kl.elbow):
            namesColumnsClustersReal.append('Real ' + str(i))
            namesColumnsClustersClass.append('Clasificación ' + str(i))
        namesColumnsClustersReal
        namesColumnsClustersClass

        # Se hace la matriz de confusión con Pandas
        Matriz_Clasificacion2 = pd.DataFrame(confusion_matrix(Y_validation, Y_Clasificacion), columns=namesColumnsClustersClass,index=namesColumnsClustersReal)
        
        return fig, tablekmeans, tablecentroides, fig2,html.Div([
            html.H2(["", dbc.Badge("Matriz de clasificación", className="ms-1")]),
            dbc.Table.from_dataframe(Matriz_Clasificacion2, striped=True, bordered=True, hover=True, responsive=True, style={'width': '100%'},index=True),

        ]), html.Div([
            html.H2(["", dbc.Badge("Reporte del árbol de decisión obtenido", className="ms-1")]),
            dbc.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
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
                                    html.Td(criterio),
                                    html.Td(splitter_report),
                                    html.Td(profundidad),
                                    html.Td(str(max_depth)),
                                    html.Td(min_samples_split),
                                    html.Td(min_samples_leaf),
                                    html.Td(nodos),
                                    html.Td(ClasificacionAD.get_n_leaves()),
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

            html.H2(["", dbc.Badge("Reporte de la efectividad del algoritmo obtenido", className="ms-1")]),
            dbc.Table(
                [
                    html.Thead(
                        #Ciclo for para generar las columnas de la tabla de acuerdo a la cantidad de variables
                        html.Tr(
                            [
                                html.Th("Reporte general"),
                            ] + [html.Th("Clase {}".format(i)) for i in range(n_classes)]
                        )

                    ),
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td("Exactitud (Accuracy): " + str(round(exactitud*100,2)) + '%', style={'color': 'green'}),
                                ] + [html.Td("Precision (Precision): " + str(round(reporte['precision'][i],4)*100) + '%', style={'color': 'blue'}) for i in range(len(reporte)-3)]
                            ),
                            html.Tr(
                                [
                                    html.Td("Tasa de error (Misclassification Rate): " + str(round(tasa_error*100,2)) + '%', style={'color': 'red'}),
                                ] + [html.Td("Sensibilidad (Recall, Sensitivity, True Positive Rate): " + str(round(reporte['recall'][i],4)*100) + '%', style={'color': 'green'}) for i in range(len(reporte)-3)]
                            ),
                            html.Tr(
                                [
                                    html.Td("Valores Verdaderos: " + str((Y_validation == Y_Clasificacion).sum()), style={'color': 'green'}),
                                ] + [
                                    html.Td("Especificidad (Specificity, True Negative Rate): " + str(round(especificidad1[i],4)*100) + '%', style={'color': 'green'}) for i in range(len(especificidad1))
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Valores Falsos: " + str((Y_validation != Y_Clasificacion).sum()), style={'color': 'red'}),
                                ] + [html.Td("F1-Score: " + str(round(reporte['f1-score'][i],4)*100) + '%', style={'color': 'blue'}) for i in range(len(reporte)-3)]
                            ),
                            html.Tr(
                                [
                                    html.Td("Valores Totales: " + str(Y_validation.size)),
                                    # html.Td("Número de muestras: " + str(classification_report(Y_validation, Y_Clasificacion).split()[8])),
                                    # html.Td("Número de muestras: " + str(classification_report(Y_validation, Y_Clasificacion).split()[13])),
                                ] + [html.Td("Número de muestras: " + str(reporte['support'][i]), style={'color': 'blue'}) for i in range(len(reporte)-3)]
                            ),
                        ]
                    ),
                ],
                bordered=True,
                hover=True,
                responsive=True,
                striped=True,
                style={'width': '100%', 'text-align': 'center'},
            ),

            html.H2(["", dbc.Badge("Importancia de las variables", className="ms-1")]),
        ]), fig3, fig4, dbc.Alert(r, color="success", style={'whiteSpace': 'pre-line'}, className="mb-3"), html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Input(id='values_X1_AD_Clasificacion', type="number", placeholder=df[X_Clase].columns[0],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[0])),
                    dbc.Input(id='values_X2_AD_Clasificacion', type="number", placeholder=df[X_Clase].columns[1],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[1])),
                    dbc.Input(id='values_X3_AD_Clasificacion', type="number", placeholder=df[X_Clase].columns[2],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[2])),
                    dbc.Input(id='values_X4_AD_Clasificacion', type="number", placeholder=df[X_Clase].columns[3],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[3])),
                    dbc.Input(id='values_X5_AD_Clasificacion', type="number", placeholder=df[X_Clase].columns[4],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[4])),
                    dbc.Input(id='values_X6_AD_Clasificacion', type="number", placeholder=df[X_Clase].columns[5],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[5])),
                    dbc.Input(id='values_X7_AD_Clasificacion', type="number", placeholder=df[X_Clase].columns[6],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[6])),
                    dbc.Input(id='values_X8_AD_Clasificacion', type="number", placeholder=df[X_Clase].columns[7],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[7])),
                    # dbc.Input(id='values_X9_AD_Clasificacion', type="number", placeholder=df[X_Clase].columns[8],style={'width': '100%'}),
                    # dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[8])),
                    # dbc.Input(id='values_X10_AD_Clasificacion', type="number", placeholder=df[X_Clase].columns[9],style={'width': '100%'}),
                    # dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[9])),
                ], width=6),
            ])

        ]), html.Div([
                dbc.Button("Mostrar valores reales y pronosticados", id="collapse-button-modelos", className="mb-3", color="primary"),
                dbc.Collapse(
                    dbc.Card(dbc.CardBody([
                        html.Div(id='output-container-button-modelos'),
                    ])),
                    id="collapse",
                ),
        ])

    elif n_clicks is None:
        import dash.exceptions as de
        raise de.PreventUpdate


@callback(
    Output('valor-clasificacion-modelos2', 'children'),
    Input('collapse-button-modelos', 'n_clicks'),
    State('values_X1_AD_Clasificacion', 'value'),
    State('values_X2_AD_Clasificacion', 'value'),
    State('values_X3_AD_Clasificacion', 'value'),
    State('values_X4_AD_Clasificacion', 'value'),
    State('values_X5_AD_Clasificacion', 'value'),
    State('values_X6_AD_Clasificacion', 'value'),
    State('values_X7_AD_Clasificacion', 'value'),
    State('values_X8_AD_Clasificacion', 'value'),
    State('values_X9_AD_Clasificacion', 'value'),
    State('values_X10_AD_Clasificacion', 'value'),
)
def AD_Clasificacion_Pronostico(n_clicks, values_X1, values_X2, values_X3, values_X4, values_X5, values_X6, values_X7, values_X8, values_X9, values_X10):
    if n_clicks is not None:
        if values_X1 is None or values_X2 is None or values_X3 is None or values_X4 is None or values_X5 is None or values_X6 is None or values_X7 is None or values_X8 is None or values_X9 is None or values_X10 is None:
            return html.Div([
                dbc.Alert('Debe ingresar los valores de las variables', color="danger")
            ])
        else:
            # Convertimos el arreglo a un DataFrame
            # values_X = np.array([values_X1, values_X2, values_X3, values_X4, values_X5, values_X6, values_X7, values_X8, values_X9, values_X10])
            
            XPredict = pd.DataFrame([values_X1, values_X2, values_X3, values_X4, values_X5, values_X6, values_X7, values_X8, values_X9, values_X10]).transpose()

            clasiFinal = ClasificacionAD.predict(XPredict)
            return html.Div([
                dbc.Alert('Valor pronosticado: ' + str(clasiFinal), color="success")
            ])


@callback(
    Output("modal-body-scroll-modelos", "is_open"),
    [
        Input("open-body-scroll-modelos", "n_clicks"),
        Input("close-body-scroll-modelos", "n_clicks"),
    ],
    [State("modal-body-scroll-modelos", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open