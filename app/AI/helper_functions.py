import pycaret.classification
import pycaret.regression
import pandas as pd
def get_model(csv_path, target, toggle):
    csv = pd.read_csv(csv_path)

    if toggle == False:
        pycaret.classification.setup(csv, target=target, silent=True)
        best = pycaret.classification.compare_models()
        comparison = pycaret.classification.pull()
        pycaret.classification.save_model(best, 'current model')
    else:
        pycaret.regression.setup(csv, target=target, silent=True)
        best = pycaret.regression.compare_models()
        comparison = pycaret.regression.pull()
        pycaret.regression.save_model(best, 'current model')
    if toggle == False:
        pycaret.classification.plot_model(best, plot='learning', save=True)
        pycaret.classification.plot_model(best, plot='auc', save=True)
        pycaret.classification.plot_model(best, plot='pr', save=True)
        pycaret.classification.plot_model(best, plot='confusion_matrix', save=True)
        pycaret.classification.plot_model(best, plot='error', save=True)
        pycaret.classification.plot_model(best, plot='class_report', save=True)
        pycaret.classification.plot_model(best, plot='boundary', save=True)
        pycaret.classification.plot_model(best, plot='learning', save=True)
        pycaret.classification.plot_model(best, plot='dimension', save=True)
        pycaret.classification.plot_model(best, plot='parameter', save=True)
    else:
        pycaret.regression.plot_model(best, plot='residuals', save=True)
        pycaret.regression.plot_model(best, plot='error', save=True)
        pycaret.regression.plot_model(best, plot='cooks', save=True)
        pycaret.regression.plot_model(best, plot='learning', save=True)
        pycaret.regression.plot_model(best, plot='manifold', save=True)
        pycaret.regression.plot_model(best, plot='feature', save=True)
        pycaret.regression.plot_model(best, plot='feature_all', save=True)

    return best, comparison