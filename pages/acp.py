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


# load_figure_template("plotly_white")

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
theme_change = ThemeChangerAIO(
    aio_id="theme",button_props={
        "color": "danger",
        "children": "SELECT THEME",
    },
)

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
    'backgroundColor': 'Black',
    'color': 'white',
    'padding': '6px'
}

layout = html.Div([
    html.H1('Principal Component Analysis (PCA)💻', style={'text-align': 'center'}),
    theme_change,
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
    html.Div(id='output-data-upload-acp'), # output-datatable
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

        html.H2(["", dbc.Badge("Evidencia de datos correlacionados", className="ms-1")]),
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

        # Estilizamos el botón con Bootstrap
        dbc.Button("Click para obtener los componentes principales", color="danger", className="mr-1", id='submit-button-standarized', style={'width': '100%'}),

        html.Hr(),

        dcc.Tabs([
            #Gráfica de pastel de los tipos de datos
            dcc.Tab(label='Matriz estandarizada', style=tab_style, selected_style=tab_selected_style,children=[
                dbc.Alert('Matriz estandarizada', color="primary"),
                #Mostramos la tabla que se retornó en el callback ID = DataTableStandarized
                dash_table.DataTable(
                    id='DataTableStandarized',
                    columns=[{"name": i, "id": i} for i in df.select_dtypes(include=['float64', 'int64']).columns],
                    page_size=8,
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        }
                    ],
                    # Al estilo de la celda le ponemos: texto centrado, con fondos oscuros y letras blancas
                    style_cell={'textAlign': 'center', 'backgroundColor': 'rgb(207, 250, 255)', 'color': 'black'},
                    # Al estilo de la cabecera le ponemos: texto centrado, con fondo azul claro y letras negras
                    style_header={'backgroundColor': 'rgb(45, 93, 255)', 'fontWeight': 'bold', 'color': 'black', 'border': '1px solid black'},
                    style_table={'height': '300px', 'overflowY': 'auto'},
                    style_data={'border': '1px solid black'}
                ),

                html.Hr(),


            ]),

            dcc.Tab(label='Número de componentes principales y la varianza acumulada', style=tab_style, selected_style=tab_selected_style,children=[
                dbc.Alert('Número de componentes principales y la varianza acumulada', color="primary"),
                # Mostramos el gráfico generado en el callback ID = varianza
                dcc.Graph(
                    id='varianza',
                ),
            ]),

            dcc.Tab(label='Proporción de cargas y selección de variables', style=tab_style, selected_style=tab_selected_style,children=[
                dbc.Alert('Considerando un mínimo de 50% para el análisis de cargas, se seleccionan las variables basándonos en este gráfico de calor', color="primary"),
                # Mostramos la gráfica generada en el callback ID = FigComponentes
                dcc.Graph(
                    id='FigComponentes',
                ),
            ]),
        ])
    ]) #Fin del layout

@callback(Output('output-data-upload-acp', 'children'),
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
    Output('DataTableStandarized','data'),
    Output('varianza', 'figure'),
    Output('FigComponentes', 'figure'),
    Input('submit-button-standarized','n_clicks'))
def calculoPCA(n_clicks):     
    if n_clicks is not None:
        #global MEstandarizada
        # Solo valores numericos en el dataframe:
        df_numeric = df.select_dtypes(include=['float64', 'int64'])
        MEstandarizada1 = StandardScaler().fit_transform(df_numeric) # Se calculan la media y desviación para cada variable, y se escalan los datos
        MEstandarizada = pd.DataFrame(MEstandarizada1, columns=df_numeric.columns) # Se convierte a dataframe

        
        pca = PCA().fit(MEstandarizada) # Se calculan los componentes principales
        Varianza = pca.explained_variance_ratio_

        # Nos apoyamos de un sencillo programa para saber cuántas componentes son necesarias
        for i in range(0, Varianza.size):
            #print('Componente', i, '->', Varianza[i]*100, '%')
            varAcumulada = sum(Varianza[0:i+1])
            #print('Varianza acumulada:', varAcumulada*100, '%')
            if varAcumulada >= 0.90:
                varAcumuladaACP = (varAcumulada - Varianza[i])
                numComponentesACP = i - 1
                #print('Se requieren', i, 'componentes para alcanzar el 90% de porcentaje de relevancia')
                #print('La varianza acumulada para', i, 'componentes es de:', varAcumuladaACP*100, '%')
                break
        
        # Desplegamos la gráfica de la varianza acumulada
        fig = px.line(x=np.arange(0, Varianza.size, step=1), y=np.cumsum(Varianza))
        fig.update_layout(title='Varianza acumulada en los componentes',
                            xaxis_title='Número de componentes',
                            yaxis_title='Varianza acumulada')
        # Se resalta el número de componentes que se requieren para alcanzar el 90% de varianza acumulada
        fig.add_shape(type="line", x0=0, y0=0.9, x1=Varianza.size-1, y1=0.9, line=dict(color="Red", width=2, dash="dash"))
        fig.add_shape(type="line", x0=numComponentesACP, y0=0, x1=numComponentesACP, y1=varAcumuladaACP, line=dict(color="Green", width=2, dash="dash"))
        # Se muestra un punto en la intersección de las líneas
        fig.add_annotation(x=numComponentesACP, y=varAcumuladaACP, text=str(round(varAcumuladaACP*100, 1))+f'%. {numComponentesACP+1} Componentes', showarrow=True, arrowhead=1)
        # Se agregan puntos en la línea de la gráfica
        fig.add_scatter(x=np.arange(0, Varianza.size, step=1), y=np.cumsum(Varianza), mode='markers', marker=dict(size=10, color='blue'), showlegend=False, name='# Componentes')
        fig.add_scatter(x=np.arange(0, Varianza.size, step=1), y=np.cumsum(Varianza), fill='tozeroy', mode='none', showlegend=False, name='Área bajo la curva') # Se le agrega el área bajo la curva
        fig.update_xaxes(range=[0, Varianza.size-1]) # Se ajusta al tamaño de la gráfica
        fig.update_yaxes(range=[0, 1.1]) # Se ajusta al tamaño de la gráfica

        # 6
        CargasComponentes = pd.DataFrame(abs(pca.components_), columns=df_numeric.columns)
        CargasComponentess=CargasComponentes.head(numComponentesACP+1) 

        #
        fig2 = px.imshow(CargasComponentes.head(numComponentesACP+1), color_continuous_scale='RdBu_r')
        fig2.update_layout(title='Cargas de los componentes', xaxis_title='Variables', yaxis_title='Componentes')
        # Agregamos los valores de las cargas en la gráfica (Si es mayor a 0.5, de color blanco, de lo contrario, de color negro):
        for i in range(0, CargasComponentess.shape[0]):
            for j in range(0, CargasComponentess.shape[1]):
                if CargasComponentess.iloc[i,j] >= 0.5:
                    color = 'white'
                else:
                    color = 'black'
                fig2.add_annotation(x=j, y=i, text=str(round(CargasComponentess.iloc[i,j], 4)), showarrow=False, font=dict(color=color))

        return MEstandarizada.to_dict('records'), fig, fig2
    
    elif n_clicks is None:
        import dash.exceptions as de
        raise de.PreventUpdate






