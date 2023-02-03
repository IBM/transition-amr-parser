import os
import importlib

# from . import attention_masks
# from . import graph_attention_masks
# from . import graphmp_attention_masks
# from . import transformer_tgt_pointer
# from . import transformer_tgt_pointer_bart
# from . import transformer_tgt_pointer_bart_sattn
# from . import transformer_tgt_pointer_bartsv
# from . import transformer_tgt_pointer_bartsv_sattn
# from . import transformer_tgt_pointer_graph
# from . import transformer_tgt_pointer_graphmp



# automatically infer the user module name (in case there is a change during the development)
pkg_name = 'neural_parser'
user_module_name = os.path.split(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))[1]
submodule_name = os.path.split(os.path.abspath(os.path.dirname(__file__)))[1]


# automatically import any Python files in the models/ directory
# this is necessary for fairseq to register the user defined models
models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    path = os.path.join(models_dir, file)
    if (file.endswith('.py') or os.path.isdir(path)) and not file.startswith('_'):
        module = file[:file.find('.py')] if file.endswith('.py') else file
        importlib.import_module(pkg_name+ '.' +user_module_name + '.' + submodule_name + '.' + module)
