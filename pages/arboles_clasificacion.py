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
    html.H3('Árboles de Decisión 🌳 (Clasificación)'),
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
    html.Div(id='output-data-upload-arboles-clasificacion'), # output-datatable
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
                dash_table.DataTable(
                    #Centramos la tabla de datos:
                    data=df.describe().to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in df.describe().columns],

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
                    style_table={'height': '300px', 'overflowY': 'auto'}
                ),

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
        
            dcc.Tab(label='Distribución de Datos', style=tab_style, selected_style=tab_selected_style,children=[
                html.Div([
                    "Selecciona la variable X:",
                    dcc.Dropdown(
                        [i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2],
                        # Seleccionamos la primera columna numérica del dataframe
                        value=df[[i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2]].columns[0],
                        id='xaxis_column'
                    ),

                    "Selecciona la variable Y:",
                    dcc.Dropdown(
                        [i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2],
                        # Seleccionamos la segunda columna numérica del dataframe
                        value=df[[i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2]].columns[1],
                        id='yaxis_column',
                        placeholder="Selecciona la variable Y"
                    ),

                    "Selecciona la variable a Clasificar:",
                    dcc.Dropdown(
                        [i for i in df.columns if (df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) == 2) or df[i].dtype in ['bool'] or (df[i].dtype in ['object'] and len(df[i].unique()) == 2)],
                        # Seleccionamos por defecto la primera columna
                        value=df[[i for i in df.columns if (df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) == 2) or df[i].dtype in ['bool'] or (df[i].dtype in ['object'] and len(df[i].unique()) == 2)]].columns[0],
                        id='caxis_column',
                        placeholder="Selecciona la variable Predictora"
                    ),
                ]),

                dcc.Graph(id='indicator_graphic'),
            ]),

            dcc.Tab(label='Aplicación del algoritmo', style=tab_style, selected_style=tab_selected_style, children=[
                dbc.Alert('Selecciona las variables predictoras', color='primary'),
                    dcc.Dropdown(
                        [i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2],
                        # Seleccionamos la segunda columna numérica del dataframe
                        value=df[[i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2]].columns,
                        id='X_Clase',
                        multi=True,
                    ),

                # Seleccionamos la variable Clase con un Dropdown
                dbc.Alert('Selecciona la variable a clasificar', color='primary'),
                dcc.Dropdown(
                    [i for i in df.columns if (df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) == 2) or df[i].dtype in ['bool'] or (df[i].dtype in ['object'] and len(df[i].unique()) == 2)],
                    # Seleccionamos por defecto la primera columna
                    value=df[[i for i in df.columns if (df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) == 2) or df[i].dtype in ['bool'] or (df[i].dtype in ['object'] and len(df[i].unique()) == 2)]].columns[0],
                    id='Y_Clase',
                    multi=True,
                ),

                # Estilizamos el botón con Bootstrap
                dbc.Button("Click para obtener la clasificación", color="primary", className="mr-1", id='submit-button-clasificacion'),

                html.Hr(),

                # Mostramos la matriz de confusión
                html.H2(["", dbc.Badge("Matriz de clasificación", className="ms-1")]),
                dcc.Graph(id='matriz'),

                html.Hr(),

                # Mostramos el reporte de clasificación
                html.H2(["", dbc.Badge("Reporte de la efectividad del algoritmo y del árbol de decisión obtenido", className="ms-1")]),
                html.Div(id='clasificacion'),

                # Mostramos la importancia de las variables
                html.H2(["", dbc.Badge("Importancia de las variables", className="ms-1")]),
                dcc.Graph(id='importancia'),

                html.Hr(),

                html.H2(["", dbc.Badge("Curva ROC", className="ms-1")]),
                dcc.Graph(id='roc-arbol-clasificacion'),

                # Imprimimos el árbol de decisión
                html.H2(["", dbc.Badge("Árbol de decisión obtenido", className="ms-1")]),
                html.Div(id='arbol'),
            ]),
        ])
    ]) #Fin del layout

@callback(Output('output-data-upload-arboles-clasificacion', 'children'),
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
    Output('indicator_graphic', 'figure'),
    Input('xaxis_column', 'value'),
    Input('yaxis_column', 'value'),
    Input('caxis_column', 'value'))
def update_graph(xaxis_column, yaxis_column, caxis_column):
    dff = df
    dff[caxis_column] = dff[caxis_column].astype('category')
    fig = px.scatter(dff, x=xaxis_column, y=yaxis_column, color=caxis_column, title='Gráfico de dispersión',symbol=caxis_column,marginal_x="histogram", marginal_y="histogram")
    fig.update_layout(showlegend=True, xaxis_title=xaxis_column, yaxis_title=yaxis_column,
                    font=dict(family="Courier New, monospace", size=18, color="black"),legend_title_text=caxis_column)
    #Modificamos el color de los puntos:
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))
    # str(df.groupby(caxis_column).size()[0])
    return fig

@callback(
    Output('matriz', 'figure'),
    Output('clasificacion', 'children'),
    Output('importancia', 'figure'),
    Output('roc-arbol-clasificacion', 'figure'),
    Output('arbol', 'children'),
    Input('submit-button-clasificacion','n_clicks'),
    State('X_Clase', 'value'),
    State('Y_Clase', 'value'))
def clasificacion(n_clicks, X_Clase, Y_Clase):
    if n_clicks is not None:
        X = np.array(df[X_Clase])
        Y = np.array(df[Y_Clase])
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
        from sklearn import model_selection

        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
                                                                                        test_size = 0.2, 
                                                                                        random_state = 0,
                                                                                        shuffle = True)

        #Se entrena el modelo a partir de los datos de entrada
        ClasificacionAD = DecisionTreeClassifier(random_state = 0)
        ClasificacionAD.fit(X_train, Y_train)

        #Se etiquetan las clasificaciones
        Y_Clasificacion = ClasificacionAD.predict(X_validation)
        Valores = pd.DataFrame(Y_validation, Y_Clasificacion)

        #Se calcula la exactitud promedio de la validación
        exactitud = ClasificacionAD.score(X_validation, Y_validation)
        
        #Matriz de clasificación
        ModeloClasificacion1 = ClasificacionAD.predict(X_validation)
        Matriz_Clasificacion1 = pd.crosstab(Y_validation.ravel(), 
                                        ModeloClasificacion1, 
                                        rownames=['Reales'], 
                                        colnames=['Clasificación'])
        

        import plotly.figure_factory as ff
        #Matriz de clasificación en plotly
        fig = ff.create_annotated_heatmap(z=Matriz_Clasificacion1.values, x=list(Matriz_Clasificacion1.columns), y=list(Matriz_Clasificacion1.index), colorscale='Viridis')
        fig.update_layout(title_x=0.5, xaxis_title="Clasificación", yaxis_title="Real", font=dict(family="Courier New, monospace", size=18, color="black"))
        fig.update_layout(
            autosize=True,
            margin=dict(
                l=50,
                r=50,
                b=100,
                t=100,
                pad=4
            ),
            paper_bgcolor="LightSteelBlue",
        )

        criterio = ClasificacionAD.criterion
        profundidad = ClasificacionAD.get_depth()
        hojas = ClasificacionAD.get_n_leaves()
        nodos = ClasificacionAD.get_n_leaves() + ClasificacionAD.get_depth()
        precision = classification_report(Y_validation, Y_Clasificacion).split()[10]
        tasa_error = 1-ClasificacionAD.score(X_validation, Y_validation)
        sensibilidad = classification_report(Y_validation, Y_Clasificacion).split()[11]
        especificidad = classification_report(Y_validation, Y_Clasificacion).split()[6]
    
        #print('Importancia variables: \n', ClasificacionAD.feature_importances_)
        # reporte = classification_report(Y_validation, Y_Clasificacion)

        # Importancia de las variables
        importancia = pd.DataFrame({'Variable': list(df[X_Clase].columns),
                            'Importancia': ClasificacionAD.feature_importances_}).sort_values('Importancia', ascending=False)

        # Graficamos la importancia de las variables
        fig2 = px.bar(importancia, x='Variable', y='Importancia', color='Importancia', color_continuous_scale='Bluered')
        fig2.update_layout(title_text='Importancia de las variables', xaxis_title="Variables", yaxis_title="Importancia")
        fig2.update_traces(texttemplate=(importancia['Importancia'].values).round(4), textposition='outside')


        #CURVA ROC
        Y_validation2 = pd.DataFrame(Y_validation) # Convertimos los valores de la variable Y_validation a un dataframe
        # Reeemplazamos los valores de la variable Y_validation2 por 0 y 1:
        # Checamos si el dataframe tiene dos valores únicos y si esos son 1 y 0
        if len(Y_validation2[0].unique()) == 2 and Y_validation2[0].unique()[0] == 0 and Y_validation2[0].unique()[1] == 1 or len(Y_validation2[0].unique()) == 2 and Y_validation2[0].unique()[0] == 1 and Y_validation2[0].unique()[1] == 0:
            pass
        else:
            Y_validation2 = Y_validation2.replace([Y_validation2[0].unique()[0],Y_validation2[0].unique()[1]],[1,0])

        # Graficamos la curva ROC con Plotly
        y_score1 = ClasificacionAD.predict_proba(X_validation)[:,1]

        from sklearn.metrics import roc_curve, auc
        fpr, tpr, thresholds = roc_curve(Y_validation2, y_score1)
        # Graficamos la curva ROC con Plotly
        fig3 = px.area(title='Curva ROC. Árbol de Decisión. AUC = '+ str(auc(fpr, tpr).round(4)) )
        fig3.add_scatter(x=fpr, y=tpr, mode='lines', name='Bosque Aleatorio', fill='tonexty')
        fig3.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="Black", dash="dash"))
        fig3.update_layout(yaxis_title='True Positive Rate', xaxis_title='False Positive Rate')


        # Generamos en texto el árbol de decisión
        from sklearn.tree import export_text
        r = export_text(ClasificacionAD, feature_names=list(df[X_Clase].columns))
        
        return fig, html.Div([
            # En tres columnas
            dbc.Row([
                dbc.Col([
                    dbc.Alert('Exactitud: ' + str(round(exactitud*100,2)) + '%', color="success"),
                    dbc.Alert('Criterio: ' + criterio, color="info"),
                    dbc.Alert('Profundidad: ' + str(profundidad), color="info"),
                ], width=4),
                dbc.Col([
                    dbc.Alert('Hojas: ' + str(hojas), color="info"),
                    dbc.Alert('Nodos: ' + str(nodos), color="info"),
                    dbc.Alert('Precisión: ' + str(round(float(precision)*100,2)) + '%', color="info"),
                ], width=4),
                dbc.Col([
                    dbc.Alert('Tasa de error: ' + str(round(tasa_error*100,2)) + '%', color="info"),
                    dbc.Alert('Sensibilidad: ' + str(round(float(sensibilidad)*100,2)) + '%', color="info"),
                    dbc.Alert('Especificidad: ' + str(round(float(especificidad)*100,2)) + '%', color="info"),
                ], width=4),
            ]),
        ]), fig2, fig3, html.Div([
            dbc.Alert(r, color="success", style={'whiteSpace': 'pre-line'}, className="mb-3")
        ])
