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
import pycaret.classification
import pycaret.regression


def get_model(csv_path, target, toggle):
    """Function to run the model comparison and make all corresponding plots"""
    csv = pd.read_csv(csv_path)

    if toggle is False:
        pycaret.classification.setup(csv, target=target, silent=True)
        best_model = pycaret.classification.compare_models()
        model_comparison = pycaret.classification.pull()
        pycaret.classification.save_model(best_model, 'current model')
    else:
        pycaret.regression.setup(csv, target=target, silent=True)
        best_model = pycaret.regression.compare_models()
        model_comparison = pycaret.regression.pull()
        pycaret.regression.save_model(best_model, 'current model')
    if toggle is False:
        pycaret.classification.plot_model(best_model, plot='learning', save=True)
        pycaret.classification.plot_model(best_model, plot='auc', save=True)
        pycaret.classification.plot_model(best_model, plot='pr', save=True)
        pycaret.classification.plot_model(best_model, plot='confusion_matrix', save=True)
        pycaret.classification.plot_model(best_model, plot='error', save=True)
        pycaret.classification.plot_model(best_model, plot='class_report', save=True)
        pycaret.classification.plot_model(best_model, plot='boundary', save=True)
        pycaret.classification.plot_model(best_model, plot='learning', save=True)
        pycaret.classification.plot_model(best_model, plot='dimension', save=True)
        pycaret.classification.plot_model(best_model, plot='parameter', save=True)
    else:
        pycaret.regression.plot_model(best_model, plot='residuals', save=True)
        pycaret.regression.plot_model(best_model, plot='error', save=True)
        pycaret.regression.plot_model(best_model, plot='cooks', save=True)
        pycaret.regression.plot_model(best_model, plot='learning', save=True)
        pycaret.regression.plot_model(best_model, plot='manifold', save=True)
        pycaret.regression.plot_model(best_model, plot='feature', save=True)
        pycaret.regression.plot_model(best_model, plot='feature_all', save=True)

    return model_comparison
