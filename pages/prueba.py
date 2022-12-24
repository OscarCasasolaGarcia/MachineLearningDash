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

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
global df_original

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
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}

layout = html.Div([
    html.H3('Árboles de Decisión 🌳 (Regresión)'),
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
            #Que esté alineado con el centro de la página:
            'display': 'flex',
            'justify-content': 'center',
            'align-items': 'center',
            'flex-direction': 'column'
        },
        # Allow multiple files to be uploaded
        multiple=True
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
            # Al estilo de la celda le ponemos: texto centrado, con fondos oscuros y letras blancas
            style_cell={'textAlign': 'center', 'backgroundColor': 'rgb(207, 250, 255)', 'color': 'black'},
            # Al estilo de la cabecera le ponemos: texto centrado, con fondo azul claro y letras negras
            style_header={'backgroundColor': 'rgb(45, 93, 255)', 'fontWeight': 'bold', 'color': 'black', 'border': '1px solid black'},
            style_table={'height': '300px', 'overflowY': 'auto'},
            style_data={'border': '1px solid black'}
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
                                    color='white' if abs(df.corr().values[i][j]) > 0.6 or df.corr().values[i][j] < -0.6 else 'black'
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
                                # Mostramos en las filas el nombre de la estadística (count, mean, std, min, 25%, 50%, 75%, max) a la izquierda de cada fila
                                # Al estilo de la celda le ponemos: texto centrado, con fondos oscuros y letras blancas
                                # Recorremos la tabla por columnas
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
        
            dcc.Tab(label='Distribución de Datos', style=tab_style, selected_style=tab_selected_style,children=[
                html.Div([
                    dcc.Dropdown(
                        df.columns,
                        # Selecionamos por defecto la primera columna
                        value=df.columns[0],
                        id='xaxis_column-arbol-regresion',
                    ),
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
                # Seleccionamos la variable Clase con un Dropdown
                dbc.Alert('Selecciona la variable a Pronosticar', color='primary'),
                dcc.Dropdown(
                    [i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2],
                    value=df[[i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2]].columns[0],
                    id='Y_Clase_Arbol_Regresion',
                ),
                
                dbc.Alert('Selecciona las variables predictoras', color='primary'),
                dcc.Dropdown(
                    # En las opciones que aparezcan en el Dropdown, queremos que aparezcan todas las columnas numéricas, excepto la columna Clase
                    [i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2],
                    value=[i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2][1:],
                    id='X_Clase_Arbol_Regresion',
                    multi=True,
                ),


                # Estilizamos el botón con Bootstrap
                dbc.Button("Click para obtener la clasificación", color="primary", className="mr-1", id='submit-button-arbol-regresion'),

                html.Hr(),

                # Mostramos la matriz de confusión
                dcc.Graph(id='matriz-arbol-regresion'),

                html.Hr(),

                # Mostramos el reporte de clasificación
                html.Div(id='clasificacion-arbol-regresion'),

                # Mostramos la importancia de las variables
                dcc.Graph(id='importancia-arbol-regresion'),

                html.Hr(),

                # Creamos un botón para predecir
                # dbc.Button("Click para predecir", color="primary", className="mr-1", id='submit-button-arbol-regresion-prediccion'),
                html.Div(id="output-regresion-arbol-regresion-Final"),
                # Mostramos la predicción
                html.Div(id='valor-regresion'),

                html.Div(id='valor-regresion2'),

                html.Hr(),
            ]),

            dcc.Tab(label='Árbol de Decisión', style=tab_style, selected_style=tab_selected_style, children=[
                # Imprimimos el árbol de decisión
                html.Div(id='arbol-arbol-regresion'),
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
    # Output('valor-regresion', 'children'),
    Output('valor-regresion2', 'children'),
    Input('submit-button-arbol-regresion', 'n_clicks'),
    State('X_Clase_Arbol_Regresion', 'value'),
    State('Y_Clase_Arbol_Regresion', 'value'))
def regresion(n_clicks, X_Clase, Y_Clase):
    if n_clicks is not None:
        global X
        X = np.array(df[X_Clase])
        Y = np.array(df[Y_Clase])

        global PronosticoAD

        from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from sklearn import model_selection

        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, 
                                                                                        test_size = 0.2, 
                                                                                        random_state = 0,
                                                                                        shuffle = True)

        #Se entrena el modelo a partir de los datos de entrada
        PronosticoAD = DecisionTreeRegressor(random_state=0)
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
        nodos = PronosticoAD.get_n_leaves() + PronosticoAD.get_depth()
        #MAE:
        MAEArbol = mean_absolute_error(Y_test, Y_PronosticoArbol)
        #MSE:
        MSEArbol = mean_squared_error(Y_test, Y_PronosticoArbol)
        #RMSE:
        RMSEArbol = mean_squared_error(Y_test, Y_PronosticoArbol, squared=False)
        # Score
        ScoreArbol = r2_score(Y_test, Y_PronosticoArbol)
        
    
        #print('Importancia variables: \n', PronosticoAD.feature_importances_)
        # reporte = classification_report(Y_validation, Y_Clasificacion)

        # Importancia de las variables
        importancia = pd.DataFrame({'Variable': list(df[X_Clase].columns),
                            'Importancia': PronosticoAD.feature_importances_}).sort_values('Importancia', ascending=False)

        # Graficamos la importancia de las variables
        fig2 = px.bar(importancia, x='Variable', y='Importancia', color='Importancia', color_continuous_scale='Bluered', text='Importancia')
        fig2.update_layout(title_text='Importancia de las variables', xaxis_title="Variables", yaxis_title="Importancia")
        # fig2.update_traces(texttemplate=(importancia['Importancia'].values).round(4), textposition='outside')
        # Redondeamos los valores de la importancia
        fig2.update_traces(texttemplate='%{text:.2}', textposition='outside')
        fig2.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        fig2.update_layout(legend_title_text='Importancia de las variables')

        # Generamos en texto el árbol de decisión
        from sklearn.tree import export_text
        r = export_text(PronosticoAD, feature_names=list(df[X_Clase].columns))
        
        return fig, html.Div([
            # En tres columnas mostramos los resultados:
            dbc.Row([
                dbc.Col([
                    dbc.Alert('Score: ' + str(ScoreArbol), color="success"),
                    dbc.Alert('Criterio: ' + str(criterio), color="info"),
                ], width=4),
                dbc.Col([
                    dbc.Alert('MAE: ' + str(MAEArbol), color="info"),
                    dbc.Alert('MSE: ' + str(MSEArbol), color="info"),
                    dbc.Alert('RMSE: ' + str(RMSEArbol), color="info"),
                ], width=4),

                dbc.Col([
                    dbc.Alert('Nodos: ' + str(nodos), color="info"),
                    dbc.Alert('Hojas: ' + str(hojas), color="info"),
                    dbc.Alert('Profundidad: ' + str(profundidad), color="info"),
                ], width=4),
            ]),
            
        ]), fig2, html.Div([
            dbc.Alert(r, color="success", style={'whiteSpace': 'pre-line'}, className="mb-3")
        ]), html.Div([
            # Usamos un ciclo for para mostrar los valores reales y los pronosticados
            # dbc.Row([
            #     dbc.Col([
            #         dbc.Alert('Variable: ' + str(df[X_Clase].columns[i]), color="info"),
            #     ], width=6),
            #     dbc.Col([
            #         dbc.Input(id='values_X', type="number", placeholder=df[X_Clase].columns[i],style={'width': '100%'})
            #     ], width=6),
            # ]) for i in range(len(df[X_Clase].columns))

            # Usamos un ciclo for para mostrar los valores reales y los pronosticados con dcc.Input
            dcc.Input(
            id="values_X",
            type="number",
            placeholder=df[X_Clase].columns[i],
            )
            for i in range(len(df[X_Clase].columns))

        ]), html.Div([
                dbc.Button("Mostrar valores reales y pronosticados", id="collapse-button", className="mb-3", color="primary"),
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

#Función para clasificar nuevos datos
# def regresionArboles(values_X):
#     values_X = np.array(values_X).reshape(1, -1)
#     X = pd.DataFrame(values_X)

#     regresionFinal = PronosticoAD.predict(X)
#     return html.Div([
#         dbc.Alert('Valor pronosticado: ' + str(regresionFinal), color="success")
#     ])

@callback(
    Output('valor-regresion', 'children'),
    Input('collapse-button', 'n_clicks'),
    State('values_X', 'value'),
    )
def regresionFinal(n_clicks, values_X):
    if n_clicks is not None:
        # Mostramos todos los valores contenidos en el arreglo
        print(values_X)
        # Convertimos el arreglo a un DataFrame
        values_X = np.array(values_X).reshape(1, -1)
        XPredict = pd.DataFrame(values_X)

        regresionFinal = PronosticoAD.predict(XPredict)
        return html.Div([
            dbc.Alert('Valor pronosticado: ' + str(regresionFinal), color="success")
        ])