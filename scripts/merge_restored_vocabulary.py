import os
import torch
from fairseq import checkpoint_utils, options, tasks


def update_state_emb(
    model,       # Trainer.get_model()
    state,       # state
    task,        # Trainer.task
    state_task   # tasks.setup_task(state['args'])
):
    ''' 
    Update checkpointy state to have partial embeddings overloading
    ''' 

    state_model = state['model']

    # model weights that are overloadable
    source_dict_weights = ['encoder.embed_tokens.weight']
    target_dict_weights = ['decoder.embed_out', 'decoder.embed_tokens.weight']
    overwritable_weights = source_dict_weights + target_dict_weights
    for name, value in model.named_parameters():
        if (
            name in overwritable_weights 
            and state_model[name].shape[0] < value.shape[0]
        ):

            # Alow overloading of sub-sets of the vocabulary
            pre_voc_size, pre_emb_dim = state_model[name].shape
            finet_voc_size, finet_emb_dim = value.shape

            assert pre_emb_dim == finet_emb_dim,\
                f"Embeddings sizes of models do not match " \
                f"({pre_emb_dim}, {finet_emb_dim})"

            # Get the embedings of the load model
            if name in target_dict_weights:
                pretr_model_idx_by_symbol = {
                    key: index 
                    for index, key in enumerate(
                        state_task.target_dictionary.symbols
                    )
                }
                model_symbols = task.target_dictionary.symbols
            else:
                pretr_model_idx_by_symbol = {
                    key: index 
                    for index, key in enumerate(
                        state_task.source_dictionary.symbols
                    )
                }
                model_symbols = task.source_dictionary.symbols

            # start by setting new state equal to the random
            # initialized model
            new_state =  value.clone().detach().requires_grad_(True)

            # Assign those embeddings to the new model if found
            count_load = 0
            for index, symbol in enumerate(model_symbols):
                if symbol in pretr_model_idx_by_symbol:
                    state_index = pretr_model_idx_by_symbol[symbol]
                    count_load += 1
                    new_state[index, :] = \
                        state['model'][name][state_index, :]
            state['model'][name] = new_state
            print(f'{name} {count_load}/{new_state.shape[0]} embeddings load from checkpoint')

    return state


def argument_parser():
    parser = options.get_training_parser()
    return options.parse_args_and_arch(parser)


def main(args):

    # read pretrained model and task
    state = checkpoint_utils.load_checkpoint_to_cpu(args.restore_file)
    state_task = tasks.setup_task(state['args'])

    # build fine-tuning model and task
    task = tasks.setup_task(args)
    model = task.build_model(args)

    # Sanity check, this is not a model on the original pretraining folder 
    # We will modify this checkpoint and we do not want to spoil other models
    path = args.data[:-1] if args.data[-1] == '/' else args.data
    if os.path.dirname(args.restore_file) != path:
        raise Exception(
            f'Expected --restore-model to be in {args.data}\n'
            '(reason: will modify this checkpoint)'
        )

    # Update state merging new vocabulary
    state = update_state_emb(model, state, task, state_task)

    # with open(, 'w') as fid:
    # with open(args.restore_file, 'w') as fid:
    torch.save(state, args.restore_file)

if __name__ == '__main__':
    main(argument_parser())
