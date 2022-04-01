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
from viktor.parametrization import IsFalse
from viktor.parametrization import Lookup
from viktor.parametrization import OptionField
from viktor.parametrization import Parametrization
from viktor.parametrization import Section
from viktor.parametrization import Table
from viktor.parametrization import TextField
from viktor.parametrization import ToggleButton


class AIParametrization(Parametrization):
    """For displaying the correct fields to users"""
    choice = Section('Classification of regression')
    choice.toggle = ToggleButton('classification or regression')

    dataset = Section('Dataset')
    dataset.data = TextField('input dataset')
    dataset.target = TextField('target metric')
    dataset.plot_classification = OptionField('plot options classification', options = ['learning curve',
                                                                                        'area under curve',
                                                                                        'precision recall',
                                                                                        'confusion matrix',
                                                                                        'prediction error',
                                                                                        'validation curve',
                                                                                        'dimension learning'],
                                              default ='confusion matrix',
                                              visible = IsFalse(Lookup('choice.toggle')))

    dataset.plot_regression = OptionField('plot options regression',
                                              options=['residuals',
                                                       'prediction error',
                                                       'cooks distance',
                                                       'learning curve',
                                                       'manifold',
                                                       'feature importance (top 10)',
                                                       'feature importance'],
                                          default='feature importance',
                                          visible = Lookup('choice.toggle'))

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
