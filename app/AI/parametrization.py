from viktor.parametrization import Parametrization, Section, TextField, ToggleButton, Table, SetParamsButton, OptionField, HiddenField, Lookup, IsFalse


class AIParametrization(Parametrization):
    choice = Section('Classification of regression')
    choice.toggle = ToggleButton('classification or regression')

    dataset = Section('Dataset')
    dataset.dataset = TextField('input dataset')
    dataset.target = TextField('target metric')
    dataset.plot_classification = OptionField('plot options classification', options = ['learning curve','area under curve','precision recall',
                                                                                        'confusion matrix','prediction error','validation curve',
                                                                                        'dimension learning'], default ='confusion matrix', visible = IsFalse(Lookup('choice.toggle')))
    dataset.plot_regression = OptionField('plot options regression',
                                              options=['residuals', 'prediction error', 'cooks distance',
                                                       'learning curve','manifold','feature importance (top 10)',
                                                       'feature importance'], default='feature importance', visible = Lookup('choice.toggle'))

    new_data = Section('New data')
    new_data.inputs = Table('new entry')
    new_data.inputs.first = TextField('param 1')
    new_data.inputs.second = TextField('param 2')
    new_data.inputs.third = TextField('param 3')
    new_data.inputs.fourth = TextField('param 4')
    new_data.inputs.fifth = TextField('param 5')
    new_data.inputs.sixth = TextField('param 6')
    new_data.inputs.seventh = TextField('param 7')
    new_data.inputs.eigth = TextField('param 8')




