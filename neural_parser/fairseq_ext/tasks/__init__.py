import os
import importlib


# from . import amr_action_pointer
# from . import amr_action_pointer_bart
# from . import amr_action_pointer_bart_dyo
# from . import amr_action_pointer_bartsv
# from . import amr_action_pointer_graphmp
# from . import amr_action_pointer_graphmp_amr1


# automatically infer the user module name (in case there is a change during the development)
pkg_name = 'neural_parser'
user_module_name = os.path.split(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))[1]


# automatically import any Python files in the tasks/ directory
# this is necessary for fairseq to register the user defined tasks
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        task_name = file[:file.find('.py')]
        importlib.import_module(pkg_name+ '.' +user_module_name + '.tasks.' + task_name)
