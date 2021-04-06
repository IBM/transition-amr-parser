import os
import importlib


# automatically infer the user module name (in case there is a change during the development)
user_module_name = os.path.split(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))[1]


# automatically import any Python files in the tasks/ directory
# this is necessary for fairseq to register the user defined tasks
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        task_name = file[:file.find('.py')]
        importlib.import_module(user_module_name + '.tasks.' + task_name)
