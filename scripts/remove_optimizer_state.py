import sys
from fairseq.checkpoint_utils import load_checkpoint_to_cpu, torch_persistent_save
from fairseq.file_io import PathManager
from fairseq.utils import move_to_cpu
from fairseq_ext.utils_import import import_user_module
from ipdb import set_trace


class ARGS():
    def __init__(self):
        self.user_dir = 'fairseq_ext/'


if __name__ == '__main__':

    if len(sys.argv[1:]) == 1:
        checkpoint_path = sys.argv[1]
        out_checkpoint_path = checkpoint_path
    elif len(sys.argv[1:]) == 2:
        checkpoint_path, out_checkpoint_path = sys.argv[1:]

    import_user_module(ARGS())
    state = load_checkpoint_to_cpu(checkpoint_path)
    print(f'load {checkpoint_path}')

    state['last_optimizer_state'] = None
    with PathManager.open(out_checkpoint_path, "wb") as f:
        torch_persistent_save(move_to_cpu(state), f)
