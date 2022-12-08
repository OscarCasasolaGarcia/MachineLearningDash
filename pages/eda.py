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

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

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
    html.H3('EDA'),
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


        # Aplicamos estilos a los botones
        dcc.Tabs([
            #Gráfica de pastel de los tipos de datos
            dcc.Tab(label='Tipos de datos', style=tab_style, selected_style=tab_selected_style,children=[
                # Mostramos un gráfico de barras de los tipos de datos de cada columna del dataframe
                dcc.Graph(
                    id='tipos',
                    figure={
                        'data': [
                            {'labels': df.dtypes.value_counts().index.map(str), 'values': df.dtypes.value_counts(), 'type': 'pie', 'name': 'Tipos de datos'}
                        ],
                        'layout': {
                            'title': 'Tipos de datos'
                        }
                    }
                ),
            ]),

            dcc.Tab(label='Valores nulos', style=tab_style, selected_style=tab_selected_style,children=[
                # Mostramos un gráfico de barras de los tipos de datos de cada columna del dataframe
                dcc.Graph(
                    id='nulos',
                    figure={
                        'data': [
                            {'x': df.isnull().sum().index, 'y': df.isnull().sum(), 'type': 'bar', 'name': 'Valores nulos'}
                        ],
                        'layout': {
                            'title': 'Valores nulos',
                            'annotations': [
                                dict(
                                    x=df.isnull().sum().index[i],
                                    y=df.isnull().sum()[i],
                                    text=str(df.isnull().sum()[i]),
                                    showarrow=False,
                                    # Por encima de la barra
                                    yshift=10
                                ) for i in range(len(df.isnull().sum()))
                            ]
                        }
                    }
                ),
            ]),

            # dcc.Tab(label='Resumen estadístico', style=tab_style, selected_style=tab_selected_style,children=[
            #     dash_table.DataTable(
            #         #Centramos la tabla de datos:
            #         data=df.describe().to_dict('records'),
            #         columns=[{'name': i, 'id': i} for i in df.describe().columns],

            #         style_data_conditional=[
            #             {
            #                 'if': {'row_index': 'odd'},
            #                 'backgroundColor': 'rgb(248, 248, 248)'
            #             }
            #         ],
            #         # Mostramos en las filas el nombre de la estadística (count, mean, std, min, 25%, 50%, 75%, max)
            #         # Al estilo de la celda le ponemos: texto centrado, con fondos oscuros y letras blancas
            #         style_cell={'textAlign': 'center', 'backgroundColor': 'rgb(207, 250, 255)', 'color': 'black'},
            #         # Al estilo de la cabecera le ponemos: texto centrado, con fondo azul claro y letras negras
            #         style_header={'backgroundColor': 'rgb(45, 93, 255)', 'fontWeight': 'bold', 'color': 'black'},
            #         style_table={'height': '300px', 'overflowY': 'auto'}
            #     ),
            # ]),

            dcc.Tab(label='Resumen estadístico de las columnas numéricas', style=tab_style, selected_style=tab_selected_style,children=[
                # Generamos un resumen estadístico de las columnas numéricas del dataframe
                dcc.Graph(
                    id='resumen',
                    figure={
                        'data': [
                            {'x': df.describe().columns, 'y': df.describe().loc['count'], 'type': 'bar', 'name': 'count'},
                            {'x': df.describe().columns, 'y': df.describe().loc['mean'], 'type': 'bar', 'name': 'Mean'},
                            {'x': df.describe().columns, 'y': df.describe().loc['std'], 'type': 'bar', 'name': u'STD'},
                            {'x': df.describe().columns, 'y': df.describe().loc['min'], 'type': 'bar', 'name': 'Min'},
                            {'x': df.describe().columns, 'y': df.describe().loc['25%'], 'type': 'bar', 'name': '25%'},
                            {'x': df.describe().columns, 'y': df.describe().loc['50%'], 'type': 'bar', 'name': '50%'},
                            {'x': df.describe().columns, 'y': df.describe().loc['75%'], 'type': 'bar', 'name': '75%'},
                            {'x': df.describe().columns, 'y': df.describe().loc['max'], 'type': 'bar', 'name': 'Max'},
                        ],
                        'layout': {
                            'title': 'Resumen estadístico'
                        }
                    }
                ),
            ]),
            

            dcc.Tab(label='Valores únicos', style=tab_style, selected_style=tab_selected_style,children=[
                # Mostramos un gráfico de barras de los valores únicos de cada columna del dataframe
                dcc.Graph(
                    id='unicos',
                    figure={
                        'data': [
                            {'x': df.columns, 'y': df.nunique(), 'type': 'bar', 'name': 'Valores únicos'}
                        ],
                        'layout': {
                            'title': 'Valores únicos',
                            'xaxis': {'title': 'Columnas'},
                            'yaxis': {'title': 'Valores únicos'},
                            # Agregamos el número de valores únicos por encima de cada barra
                            'annotations': [
                                dict(
                                    x=df.columns[i],
                                    y=df.nunique()[i],
                                    text=str(df.nunique()[i]),
                                    showarrow=False,
                                    # Por encima de la barra
                                    yshift=10
                                ) for i in range(len(df.columns))
                            ]
                        }
                    }
                ),
            ]),
            dcc.Tab(label='Valores duplicados', style=tab_style, selected_style=tab_selected_style,children=[
                # Mostramos un gráfico de barras de los valores duplicados de cada columna del dataframe
                dcc.Graph(
                    figure={
                        'data': [
                            {'x': df.columns, 'y': df.duplicated().sum(), 'type': 'bar', 'name': 'Valores duplicados'}
                        ],
                        'layout': {
                            'title': 'Valores duplicados',
                            'xaxis': {'title': 'Columnas'},
                            'yaxis': {'title': 'Valores duplicados'}
                        }
                    }
                ),
            ]),


            dcc.Tab(label='Identificación de valores atípicos', style=tab_style, selected_style=tab_selected_style,children=[
                # Mostramos un histograma por cada variable de tipo numérico:
                dcc.Graph(
                    id='histogramas',
                    figure={
                        'data': [
                            {'x': df[col], 'type': 'histogram', 'name': col} for col in df.select_dtypes(include=['int64', 'float64']).columns
                        ],
                        'layout': {
                            'title': 'Histogramas',
                            'barmode': 'overlay',
                            'xaxis': {'title': 'Valores'},
                            'yaxis': {'title': 'Frecuencia'}
                        }
                    }
                ),
            ]),

            # Gráfica de cajas y bigotes
            dcc.Tab(label='Gráfica de cajas y bigotes', style=tab_style, selected_style=tab_selected_style,children=[
                # Mostramos un histograma por cada variable de tipo numérico:
                dcc.Graph(
                    id='cajas',
                    figure={
                        'data': [
                            {'x': df[col], 'type': 'box', 'name': col} for col in df.select_dtypes(include=['int64', 'float64']).columns
                        ],
                        'layout': {
                            'title': 'Cajas y bigotes',
                            'barmode': 'overlay',
                            'xaxis': {'title': 'Valores'},
                            'yaxis': {'title': 'Frecuencia'}
                        }
                    }
                ),
            ]),

            
            dcc.Tab(label='Resumen estadístico variables categóricas', style=tab_style, selected_style=tab_selected_style,children=[
                dash_table.DataTable(
                    #Centramos la tabla de datos:
                    data=df.describe(include='object').to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in df.describe(include='object').columns],

                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        }
                    ],
                    # Mostramos en las filas el nombre de la estadística (count, mean, std, min, 25%, 50%, 75%, max)
                    # Al estilo de la celda le ponemos: texto centrado, con fondos oscuros y letras blancas
                    style_cell={'textAlign': 'center', 'backgroundColor': 'rgb(207, 250, 255)', 'color': 'black'},
                    # Al estilo de la cabecera le ponemos: texto centrado, con fondo azul claro y letras negras
                    style_header={'backgroundColor': 'rgb(45, 93, 255)', 'fontWeight': 'bold', 'color': 'black'},
                    style_table={'height': '200px', 'overflowY': 'auto'}
                ),
                dbc.Alert('Fila 1: count', color="primary"),
                dbc.Alert('Fila 2: unique', color="secondary"),
                dbc.Alert('Fila 3: top', color="primary"),
                dbc.Alert('Fila 4: freq', color="secondary"),
            ]),


            dcc.Tab(label='Analisis Correlacional', style=tab_style, selected_style=tab_selected_style,children=[
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
            parse_contents(c, n,d) for c, n,d in
            zip(list_of_contents, list_of_names,list_of_dates)]
        return children

