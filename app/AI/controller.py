from typing import Tuple

from pandas import DataFrame
from viktor.core import ViktorController
from viktor.views import  DataResult, DataView, DataGroup, DataItem, PlotlyView, PlotlyResult, PNGView, PNGResult
from viktor.result import SetParamsResult
from viktor.utils import memoize


from .parametrization import AIParametrization
import pandas as pd
import plotly.graph_objects as go
import pycaret.classification
import pycaret.regression
from .helper_functions import get_model



class AIController(ViktorController):
    label = 'AI'
    parametrization = AIParametrization

    viktor_convert_entity_field = True


    @PlotlyView('Data', duration_guess=1)
    def data(self, params, entity_id, **kwargs):

        csv = pd.read_csv(params.dataset.dataset)
        cells = []
        for col in csv.columns:
            cells.append(csv[col])

        fig = go.Figure(data=go.Table(header=dict(values=list(csv.columns)), cells=dict(values=cells)))

        return PlotlyResult(fig.to_json())


    @PlotlyView('Models', duration_guess = 4)
    def calculate(self, params, entity_id, **kwargs):
        best, comparison = get_model(params.dataset.dataset, params.dataset.target, params.choice.toggle)
        comparison = pd.DataFrame(comparison)
        cells = []
        for col in comparison.columns:
            cells.append(comparison[col])

        fig = go.Figure(data=go.Table(header=dict(values=list(comparison.columns)), cells=dict(values=cells)))


        return  PlotlyResult(fig.to_json())

    @PNGView('Analysis plot', duration_guess=1)
    def get_PNG(self, params, entity_id, **kwargs):
        if params.choice.toggle == False:
            switcher = {'learning curve': 'Learning Curve.png',
                        'area under curve' : 'AUC.png',
                        'precision recall': 'Precision Recall.png',
                        'confusion matrix': 'Confusion Matrix.png',
                        'prediction error': 'Prediction Error.png',
                        'validation curve': 'Validation Curve.png',
                        'dimension learning': 'Dimensions.png'}
            plot_type = switcher.get(params.dataset.plot_classification)
        else:
            switcher = {'residuals': 'Residuals.png',
                        'prediction error': 'Prediction Error.png',
                        'cooks distance': 'Cooks Distance.png',
                        'learning curve': 'Learning Curve.png',
                        'manifold': 'Manifold Learning.png',
                        'feature importance (top 10)': 'Feature Importance.png',
                        'feature importance': 'Feature Importance (All).png'}
            plot_type = switcher.get(params.dataset.plot_regression)

        return PNGResult.from_path(plot_type)

    @PlotlyView('Labeled data', duration_guess=4)
    def labelling(self, params, entity_id, **kwargs):
        csv = pd.read_csv(params.dataset.dataset)
        if params.choice.toggle == False:
            best = pycaret.classification.load_model('current model')
            result = pycaret.classification.predict_model(best, data= csv)
        else:
            best = pycaret.regression.load_model('current model')
            result = pycaret.regression.predict_model(best, data = csv)
        result = pd.DataFrame(result)
        cells = []
        for col in result.columns:
            cells.append(result[col])
        fig = go.Figure(data=go.Table(header=dict(values=list(result.columns)), cells=dict(values=cells)))

        return PlotlyResult(fig.to_json())

    @PlotlyView('New labels', duration_guess=4)
    def label_new(self, params, entity_id, **kwargs):
        csv = pd.read_csv(params.dataset.dataset)
        new_data = pd.DataFrame(params.new_data.inputs)

        new_data.replace('', float('NaN'), inplace=True)
        new_data.dropna(how='all',axis=1, inplace=True)
        csv = csv.drop([params.dataset.target], axis = 1)
        new_data.columns = csv.columns

        for i in new_data.columns:
            for el in new_data[i]:
                try:
                    el = float(el)
                except:
                    pass

        if params.choice.toggle == False:
            best = pycaret.classification.load_model('current model')
            result = pycaret.classification.predict_model(best, data=new_data)
        else:
            best = pycaret.regression.load_model('current model')
            result = pycaret.regression.predict_model(best, data=new_data)
        result = pd.DataFrame(result)
        content_lis = []
        for col in result.columns:
            content_lis.append(result[col])

        fig = go.Figure(data=go.Table(header=dict(values=list(result.columns)), cells=dict(values=content_lis)))

        return PlotlyResult(fig.to_json())

