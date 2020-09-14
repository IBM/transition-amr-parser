import torch
import numpy as np
from copy import deepcopy
from warnings import warn

#def read_vocabulary(file_path):
#    with open(file_path) as fid:
#        index2word = {
#            index: line.split()[0]
#            for index, line in enumerate(fid.readlines())
#        }
#    return index2word

# FONT_COLORORS
FONT_COLOR = {
    'black': 30, 'red': 31, 'green': 32, 'yellow': 33, 'blue': 34,
    'magenta': 35, 'cyan': 36, 'light gray': 37, 'dark gray': 90,
    'light red': 91, 'light green': 92, 'light yellow': 93,
    'light blue': 94, 'light magenta': 95, 'light cyan': 96, 'white': 97
}

# BG FONT_COLORORS
BACKGROUND_COLOR = {
    'black': 40,  'red': 41, 'green': 42, 'yellow': 43, 'blue': 44,
     'magenta': 45, 'cyan': 46, 'light gray': 47, 'dark gray': 100,
     'light red': 101, 'light green': 102, 'light yellow': 103,
     'light blue': 104, 'light magenta': 105, 'light cyan': 106,
     'white': 107
}

def white_background(string):
    return "\033[107m%s\033[0m" % string

def red_background(string):
    return "\033[41m%s\033[0m" % string

def black_font(string):
    return "\033[30m%s\033[0m" % string

def stack_style(string):
    return black_font(white_background(string))

class StackStateMachine(object):
    """
    Stack machine for fixed length batch
    """

    def __init__(self, src_tokens, src_lengths, src_dict, tgt_dict, 
                 max_tgt_len, beam_size):

        # FIXME: This assumes left padding of source. This is for consistency
        # with fairseq
        self.left_pad_source = True
        assert self.left_pad_source

        # FIXME: current code may give problems with POS multi-task, see this
        # flag
        self.use_word_multitask = True

        # Initialize memory states and positions in stack-transformer
        # Note that input has batch and beam dimensions flattened into one 
        self.beam_size = beam_size

        # fairseqs dictionary to convert indices to symbols
        self.tgt_dict = tgt_dict

        # store per sentence information
        self.batch = []
        max_len = max(src_lengths)
        for sent_index in range(len(src_lengths)):

            # Dictionary indices for this sentece
            # we will also need the left padding for each batch
            src_len = src_lengths[sent_index]
            if self.left_pad_source:
                left_pad = max_len - src_len
                indices_tensor = src_tokens[sent_index, -src_len:].cpu().numpy()
            else:
                left_pad = 0
                indices_tensor = src_tokens[sent_index, :src_len].cpu().numpy()

            # Add sentence state variables
            self.batch.append({
                'sentence': [src_dict.symbols[i] for i in indices_tensor],
                'stack': [], 
                'buffer': list(range(src_len)),
                'is_finished': False,
                'left_pad': left_pad,
                'log_likelihood': 0,
                'action_history': [],
                # original index. Due to pruning and reordering this can differ
                # from the current postion
                'sent_index': sent_index,
                # position of parento
                'heads': [None for _ in indices_tensor],
                # label of arc to parent
                'labels': [None for _ in indices_tensor]
            })

        # sanity check: all sentences end in root
        assert all(item['sentence'][-1] == 'ROOT' for item in self.batch), \
            "Error in padding?"

        # keep count of time step for stopping and action history
        self.max_tgt_len = max_tgt_len
        self.step_index = 0

        # these have the same info as buffer and stack but in stack-transformer
        # form (batch_size * beam_size, src_len, tgt_len)
        dummy = src_tokens.unsqueeze(2).repeat(1, 1, max_tgt_len)
        self.memory = torch.ones_like(dummy) * tgt_dict.pad()
        self.memory_pos = torch.ones_like(dummy) * tgt_dict.pad()
        self.update_masks()

        # Store valid action indices
        self.valid_indices = {}
        for action in ['SHIFT', 'LEFT-ARC', 'RIGHT-ARC', 'SWAP']:
            self.valid_indices[action] = [
                self.tgt_dict.indices[a] 
                for a in self.tgt_dict.symbols
                if action in a.split(',')[0]
            ]

        # Store valid shifts
        # FIXME: Specific for auto-encoding multi-task
        self.shift_by_word = {
            word: self.tgt_dict.indices['SHIFT(%s)' % word]
            for idx, word in enumerate(src_dict.symbols)
            if ('SHIFT(%s)' % word) in self.tgt_dict.symbols
        }

    def __str__(self):
        # prints first sentence of each beam for the entire batch
        max_num_sentences = 3
        str_rep = ""

        # pick first two and last two sentences for display
        sample_sent_indices = [0, 1]
#            0, 
#            1 * self.beam_size, 
#            len(self.sentences) - 2 * self.beam_size, 
#            len(self.sentences) - 1 * self.beam_size
#        ]
        # display those
        for state in self.batch:
            if state['action_history']:
                str_rep += (
                    "%s %3.8f" % 
                    (
                        state['action_history'][-1][0],
                        state['action_history'][-1][1]
                    )
                )
                str_rep += "\n"
            str_rep += " ".join([
                state['sentence'][idx] for idx in state['buffer']
            ])
            str_rep += "\n"
            str_rep += black_font(white_background(" ".join([
                state['sentence'][idx] for idx in state['stack']
            ])))
            str_rep += "\n\n"
        return str_rep

    def get_valid_action_indices(self, state):

        valid_actions = []
        if state['is_finished']: 
            # machine had been stopped
            # FIXME: Its EOS or PAD?
            valid_actions = [self.tgt_dict.indices['</s>']]

        elif (len(state['stack']) == 2 and len(state['buffer']) == 0):
            # stop machine
            valid_actions = [self.tgt_dict.indices['LEFT-ARC(root)']]

        else:
            if (
                len(state['buffer']) > 0 and 
                not (len(state['buffer']) == 1 and len(state['stack']) > 1)
            ):

                # If SHIFT(top-of-buffer) is a valid action constrain to that
                # FIXME: This is not robust (may find POS-looking word as top
                # of buffer)
                buffer_top_word = state['sentence'][state['buffer'][0]]
                shift_tob = 'SHIFT(%s)' % buffer_top_word
                if (
                    self.use_word_multitask and 
                    shift_tob in self.tgt_dict.indices
                ):
                    valid_indices = [self.tgt_dict.indices[shift_tob]]
                else:
                    valid_indices = self.valid_indices['SHIFT']

                # can shift
                valid_actions.extend(valid_indices)

            if len(state['stack']) >= 2:
                # can draw arcs
                valid_actions.extend(self.valid_indices['LEFT-ARC'])
                valid_actions.extend(self.valid_indices['RIGHT-ARC'])
                if state['stack'][-1] > state['stack'][-2]:
                    # can swap
                    valid_actions.extend(self.valid_indices['SWAP'])

        return valid_actions 

    def act(self, action, state, step_index):

        # if action n-gram, keep only the first
        action = action.split(',')[0]

        if action.split('(')[0] == 'SHIFT':
            # move one elements from stack to buffer
            assert state['buffer'], "Can not SHIFT empty buffer"
            state['stack'].append(state['buffer'].pop(0))

        elif action.split('(')[0] == 'LEFT-ARC':
            # remove second element in stack from the top
            assert len(state['stack']) >= 2, "Need at least size 2 stack"
            assert state['sentence'][state['stack'][-2]] != 'ROOT', \
                "Dependent can not be ROOT"
            # remove first element in stack from the top
            dependent = state['stack'].pop(-2)
            # store head and label           
            state['heads'][dependent] = state['stack'][-1]
            state['labels'][dependent] = action.split('(')[1].split(')')[0]

            # finishing action
            if action == 'LEFT-ARC(root)':
                # check no orphans
                # if any(x is None for x in state['heads'][:-1]):
                #    import ipdb; ipdb.set_trace(context=30)
                #    print("")
                state['is_finished'] = True

        elif action.split('(')[0] == 'RIGHT-ARC':
            assert len(state['stack']) >= 2, "Need at least size 2 stack"
            assert state['sentence'][state['stack'][-1]] != 'ROOT', \
                "Dependent can not be ROOT"
            # remove first element in stack from the top
            dependent = state['stack'].pop(-1)
            # store head and label           
            state['heads'][dependent] = state['stack'][-1]
            state['labels'][dependent] = action.split('(')[1].split(')')[0]

        elif (action.split('(')[0] == 'SWAP' and state['stack'][-1] >= state['stack'][-2]):
            assert len(state['stack']) >= 2, "Need at least size 2 stack"
            # set element 1 of the stack to 0 of the buffer
            state['buffer'].insert(0, state['stack'].pop(-2))

        elif action.split('(')[0] == '</s>' and state['is_finished']:
            # If machine is finished we should receive EOS
            pass

        else: 
            raise Exception("Invalid action %s" % action)

    def update_state(self, action_logp, RUEDITAS=None):
        """
        action_logp (batch_size, src_len, tgt_len) 

        RUEDITAS is gold actions!!
        """

        # This wills store log-probabilities for actions adding state machine
        # constraints 
        const_action_logp = torch.ones_like(action_logp) * float('-inf')

        # loop over log probabilities output by the model. Note that lats batch 
        # may have less than len(self.buffers) 

        # Due to reordering/pruning we have curr_index and sent_index
        # (original)
        for curr_index, state in enumerate(self.batch):

            # Constrain only to valid actions. Set all other log probabilities
            # to zero constrain valid actions to gold actions if provided
            if RUEDITAS is None:
                valid_indices = self.get_valid_action_indices(state)
            else:
                valid_indices = RUEDITAS[curr_index, self.step_index]

            if len(state['buffer']) and self.use_word_multitask:
                # FIXME: This is specific for word multi-task
                # If restricted to valid buffer shift prediction, set valid
                # shoft top most probable shift action regardless of the word
                # shifted
                buffer_top_word = state['sentence'][state['buffer'][0]]
                shift_tob = 'SHIFT(%s)' % buffer_top_word
                if shift_tob in self.tgt_dict.indices:
                    valid_shift = self.tgt_dict.indices[shift_tob]
                    all_shifts = self.valid_indices['SHIFT']
                    action_logp[curr_index, valid_shift] = \
                        action_logp[curr_index, all_shifts].max()

            # This should not happen but somehown it does: extra check for
            # index out of bounds
            assert max(valid_indices) < action_logp.shape[1],  \
                "Out of Bounds in output vocabulary!"
            # valid_indices = [i for i in valid_indices if i < action_logp.shape[1]]
            # cast to GPU/CPU tesor    
            valid_indices = torch.LongTensor(valid_indices)
            valid_indices = valid_indices.to(action_logp.device)
            # constrain log probabilities to valid actions
            const_action_logp[curr_index, valid_indices] = \
                action_logp[curr_index, valid_indices]

            # get most probable action
            log_p = const_action_logp[curr_index, :]
            # check for no valid actions
            assert not (log_p == float("-inf")).all(), \
                "No valid action found!"
            best_action_index = log_p.argmax()
            best_action = self.tgt_dict.symbols[best_action_index]
            best_action_logp = float(log_p.max().cpu())

            # carry on action
            self.act(best_action, state, self.step_index)

            # update accumulated (unnormalized) log likelihood
            state['log_likelihood'] += best_action_logp

            # update action history
            state['action_history'].append((
                best_action,
                np.exp(best_action_logp)
            ))

        # increase action counter
        self.step_index += 1

        # update state expressed as masks
        self.update_masks()
        return const_action_logp

    def update_masks(self):

        # Get masks from states
        # buffer words    
        device = self.memory.device
        # reset
        for curr_index, state in enumerate(self.batch):

            if state['buffer']:
                padded_indices = torch.LongTensor(state['buffer']) + state['left_pad']
                self.memory[curr_index, padded_indices, self.step_index] = 3
                self.memory_pos[curr_index, padded_indices, self.step_index] = \
                    torch.arange(padded_indices.shape[0]).to(device)

            if state['stack']:
                padded_indices = torch.LongTensor(state['stack']) + state['left_pad']
                self.memory[curr_index, padded_indices, self.step_index] = 4
                self.memory_pos[curr_index, padded_indices, self.step_index] = \
                    torch.LongTensor(range(padded_indices.shape[0])[::-1]).to(device)

        if self.step_index >= self.max_tgt_len:
            not_finished_idices = [
                idx for state in self.batch 
                if not state['is_finished']
            ]
            import ipdb; ipdb.set_trace(context=30)
            print()

    def reoder_machine(self, reorder_state):
        """Reorder/eliminate machines during decoding"""
        # We need to deep copy
        self.batch = [
            deepcopy(self.batch[i]) for i in reorder_state.cpu().tolist()
        ]
        self.memory = self.memory[reorder_state, :, :]
        self.memory_pos = self.memory_pos[reorder_state, :, :]
        self.update_masks()


def sanity_check_masks(pre_mask, post_mask, encoder_padding_mask):

    batch_size, target_size, source_size = pre_mask.shape

    # TODO:
    # no degenerate (all masked) softmax over source
    # no degenerate (all masked) softmax over target logits 

    import pdb; pdb.set_trace()
    # buffer check: starts full and ends up empty
    for i in [0, 1]:
        if not torch.all(pre_mask[i, 0, :] == 1):
            import ipdb; ipdb.set_trace(context=30)
            print("")
        #if not torch.all(pre_mask[i, -1, :] == 0):
        #    import ipdb; ipdb.set_trace(context=30)
        #    print("")
    # stack check: starts empty and ends up with one word
    for i in [2, 3]:
        if not torch.all(pre_mask[i, 0, :] == 0):
            import ipdb; ipdb.set_trace(context=30)
            print("")
        #if not pre_mask[i, -1, :].sum() == 1:
        #    import ipdb; ipdb.set_trace(context=30)
        #    print("")

        if (pre_mask.sum(2) == 0).any():
            import ipdb; ipdb.set_trace(context=30)
            print()


def get_heads_stack_representation(memory, memory_pos, num_heads, 
                                   embed_stack_positions, do_stack=True, 
                                   do_buffer=True, do_top=False, 
                                   do_positions=True):
    """
    memory (batch_size, src_len, tgt_len) 
    """

    # FIXME: more memory efficient code possible. We can apply the masks
    # positions only to the heads affected. No need to instantiate tensors of
    # whole size

    # FIXME: memory values hard coded in preprocessing.py
    # memory2int = {'B': 3, 'S' : 4, 'X': 5}

    # we will need the positions before reshaping memory
    if do_positions:
        pos_indices = memory == 4

    # pre-mask
    # applied to the logits of the attention mechanisms
    # clone token memory states for each head
    memory = memory.repeat(num_heads, 1, 1, 1)
    # TODO: Right now hand coded
    # 1 head buffer, 1 head stack, rest free
    assert num_heads in [6, 4]
    pre_mask = torch.ones_like(memory)
    # (num_heads, batch_size, src_len, tgt_len)
    if do_buffer:
        # only use the buffer
        pre_mask[0, :, :, :] = memory[0, :, :, :] == 3
        # mask everything that is not top two positions
        if do_top:
            pre_mask[0, :, :, :][memory_pos>1] = 0
    if do_stack:
        # only use the stack
        pre_mask[1, :, :, :] = memory[1, :, :, :] == 4
        # mask everything that is not top two positions
        if do_top:
            pre_mask[1, :, :, :][memory_pos>1] = 0

    # flatten head and batch into same dimension. Heads for each batch element
    # must be contiguous
    _, batch_size, src_len, tgt_len = pre_mask.shape
    shape_pre_mask = batch_size * num_heads, src_len, tgt_len
    # (num_heads * batch_size, tgt_len, src_len)
    pre_mask = pre_mask.transpose(0, 1).contiguous().view(*shape_pre_mask).transpose(1, 2)

    # To ispect i-th batch element heads
    # pre_mask[num_heads*i:num_heads*i+num_heads, :, :]

    # post-mask
    # applied to the attended values (all heads)
    # sets head to zero if all words of pre-mask were masked 
    # (empty buffer or stack)
    # FIXME: There should be a command for this
    shape = (batch_size * num_heads, tgt_len, 1)
    # (num_heads * batch_size, tgt_len, 1)
    post_mask = pre_mask.new_ones(shape)
    post_mask[pre_mask.sum(2) == 0] = 0.0

    # pre-mask extra changes
    # what is filtered by the post mask can be here set to one
    pre_mask[(pre_mask.sum(2) == 0).unsqueeze(2).repeat(1, 1, src_len)] = 1
    # We need to avoid full masking of pad on target side as this leads to nans
    # on the final prediction softmax. However, if we do not mask pad on source
    # here, the post_mask will not detect some cases leading to nans in
    # attention softmax. We thus remove masking pads after computing post_mask
    pre_mask[memory.transpose(0, 1).contiguous().view(*shape_pre_mask).transpose(1, 2) == 1] = 1 
    #
    indices = pre_mask == 1
    # store pre_mask in log domain
    pre_mask = pre_mask

    pre_mask.fill_(float("-Inf"))
    pre_mask[indices] = 0.

    # sanity checks: 
    # sanity_check_masks(pre_mask, post_mask, encoder_padding_mask)

    # Buffer/Stack positions
    # if not do_stack_top:
    if do_positions:

        # get position weights
        max_positions, emb_size = embed_stack_positions.weights.shape

        # use half the positions for stack and half for buffer. Need to ceil
        # positions at that value
        # separate position indices for stack and buffer (half range for each)
        max_positions = max_positions // 2
        memory_pos[memory_pos>=max_positions] = max_positions - 1
        memory_pos[pos_indices] += max_positions

        # (batch_size, src_len, tgt_len)
        # flatten batch and source dimenstion to call position embedder
        shape = (memory_pos.shape[0] * memory_pos.shape[1], memory_pos.shape[2])
        head_positions = embed_stack_positions(memory_pos.view(*shape).long())
        # -> (batch_size, src_len, tgt_len, emb_size)
        shape = (batch_size, src_len, tgt_len, emb_size)
        head_positions = head_positions.view(*shape)

    else:

        head_positions = None

    return (pre_mask, post_mask), head_positions


def state_machine_encoder(encode_state_machine, memory, memory_pos, num_heads, 
                          embed_stack_positions, layer_index, encoder_padding_mask):
 
    if (
        (encode_state_machine == 'layer0' and layer_index == 0) or 
         encode_state_machine == 'layer0_nopos' and layer_index == 0 or
         encode_state_machine == 'all-layers' or
         encode_state_machine == 'all-layers_nopos' or
         encode_state_machine == 'all-layers_top_nopos' or
         encode_state_machine == 'only_stack_nopos' or
         encode_state_machine == 'only_buffer_nopos' or
         encode_state_machine == 'only_stack_top_nopos' or
         encode_state_machine == 'only_buffer_top_nopos'
    ):

        # default options
        do_stack = True
        do_buffer = True
        do_top = False

        if encode_state_machine.startswith('only_buffer'):
            do_stack = False
        if encode_state_machine.startswith('only_stack'):
            do_buffer = False

        # TODO: Change by _top
        if 'top' in encode_state_machine.split('_'):
            do_top = True

        do_positions = 'nopos' not in encode_state_machine.split('_')

        # stack/buffer state masks 
        head_attention_masks, head_positions = get_heads_stack_representation(
            memory,
            memory_pos,
            num_heads,
            embed_stack_positions,
            do_stack=do_stack,
            do_buffer=do_buffer,
            do_top=do_top,
            do_positions=do_positions
        )

    elif encode_state_machine in ['layer0', 'layer0_nopos'] and layer_index != 0:
        head_attention_masks = None
        head_positions = None
    
    else:
        raise Exception(
            "Unknown encode_state_machine %s" % 
            encode_state_machine
        )

    return head_attention_masks, head_positions
