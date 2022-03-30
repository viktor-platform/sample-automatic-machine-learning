"""Copyright (c) 2022 VIKTOR B.V.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

VIKTOR B.V. PROVIDES THIS SOFTWARE ON AN "AS IS" BASIS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import pandas as pd
import plotly.graph_objects as go
import pycaret.classification
import pycaret.regression
from viktor.core import ViktorController
from viktor.views import PNGResult
from viktor.views import PNGView
from viktor.views import PlotlyResult
from viktor.views import PlotlyView

from .helper_functions import get_model
from .parametrization import AIParametrization


class AIController(ViktorController):
    label = 'AI'
    parametrization = AIParametrization

    viktor_convert_entity_field = True


    @PlotlyView('Data', duration_guess=1) #view to visualize the data in the read dataset
    def visualize_original(self, params, entity_id, **kwargs):
        csv = pd.read_csv(params.dataset.data)
        cells = [csv[col] for col in csv.columns]

        fig = go.Figure(data=go.Table(header=dict(values=list(csv.columns)), cells=dict(values=cells)))

        return PlotlyResult(fig.to_json())


    @PlotlyView('Models', duration_guess = 4) #for visualizing the model scores
    def calculate_models(self, params, entity_id, **kwargs):
        comparison = get_model(params.dataset.data, params.dataset.target, params.choice.toggle)
        comparison = pd.DataFrame(comparison)
        cells = [comparison[col] for col in comparison.columns]

        fig = go.Figure(data=go.Table(header=dict(values=list(comparison.columns)), cells=dict(values=cells)))

        return  PlotlyResult(fig.to_json())

    @PNGView('Analysis plot', duration_guess=1) #for visualizing some anamysis plots
    def get_analyis_plot(self, params, entity_id, **kwargs):
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

    @PlotlyView('Labeled data', duration_guess=4) #for analysing the values the machine learning gives the original data
    def label_data(self, params, entity_id, **kwargs):
        csv = pd.read_csv(params.dataset.data)
        if params.choice.toggle == False:
            best_model = pycaret.classification.load_model('current model')
            result = pycaret.classification.predict_model(best_model, data=csv)
        else:
            best_model = pycaret.regression.load_model('current model')
            result = pycaret.regression.predict_model(best_model, data=csv)
        result = pd.DataFrame(result)
        cells = [result[col] for col in result.columns]
        fig = go.Figure(data=go.Table(header=dict(values=list(result.columns)), cells=dict(values=cells)))

        return PlotlyResult(fig.to_json())

    @PlotlyView('New labels', duration_guess=4) #for analysing the values the ML gives for new/custom data
    def label_new(self, params, entity_id, **kwargs):
        csv = pd.read_csv(params.dataset.data)
        new_data = pd.DataFrame(params.new_data.inputs)

        new_data.replace('', float('NaN'), inplace=True)
        new_data.dropna(how='all', axis=1, inplace=True)
        csv = csv.drop([params.dataset.target], axis=1)
        new_data.columns = csv.columns

        for i in new_data.columns:
            for el in new_data[i]:
                try:
                    el = float(el)
                except:
                    pass

        if params.choice.toggle == False:
            best_model = pycaret.classification.load_model('current model')
            result = pycaret.classification.predict_model(best_model, data=new_data)
        else:
            best_model = pycaret.regression.load_model('current model')
            result = pycaret.regression.predict_model(best_model, data=new_data)
        result = pd.DataFrame(result)
        cells = [result[col] for col in result.columns]

        fig = go.Figure(data=go.Table(header=dict(values=list(result.columns)), cells=dict(values=cells)))

        return PlotlyResult(fig.to_json())

