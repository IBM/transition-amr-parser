import sys
from fairseq_ext.utils import remove_optimizer_state


if __name__ == '__main__':

    if len(sys.argv[1:]) == 1:
        checkpoint_path = sys.argv[1]
        out_checkpoint_path = checkpoint_path
    elif len(sys.argv[1:]) == 2:
        checkpoint_path, out_checkpoint_path = sys.argv[1:]

    remove_optimizer_state(checkpoint_path, out_checkpoint_path)
