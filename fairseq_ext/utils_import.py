import os
import sys
import importlib


# ========== adapted from
# https://github.com/pytorch/fairseq/blob/83e615d66905b8ca7483122a37da1a85f13f4b8e/fairseq/utils.py#L431
# to avoid error in our setup
# ==========
def import_user_module(args):
    module_path = getattr(args, "user_dir", None)
    if module_path is not None:
        module_path = os.path.abspath(args.user_dir)
        if not os.path.exists(module_path):
            fairseq_rel_path = os.path.join(os.path.dirname(__file__), args.user_dir)
            if os.path.exists(fairseq_rel_path):
                module_path = fairseq_rel_path
            else:
                fairseq_rel_path = os.path.join(
                    os.path.dirname(__file__), "..", args.user_dir
                )
                if os.path.exists(fairseq_rel_path):
                    module_path = fairseq_rel_path
                else:
                    raise FileNotFoundError(module_path)

        # ensure that user modules are only imported once
        import_user_module.memo = getattr(import_user_module, "memo", set())
        if module_path not in import_user_module.memo:
            import_user_module.memo.add(module_path)

            module_parent, module_name = os.path.split(module_path)
            if module_name not in sys.modules:
                sys.path.insert(0, module_parent)
                importlib.import_module(module_name)
            # else:
            #     raise ImportError(
            #         "Failed to import --user-dir={} because the corresponding module name "
            #         "({}) is not globally unique. Please rename the directory to "
            #         "something unique and try again.".format(module_path, module_name)
            #     )
