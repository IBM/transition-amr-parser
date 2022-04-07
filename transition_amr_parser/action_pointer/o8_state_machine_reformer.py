"""A simple state machine to reform actions into the Transformer model input sequence.

This is used when we move away from the normal seq-to-seq realm, where target side input is just a shifted version
of the target side output. For example, to include graph structure, we need to change the pointer values to the latest
node representation, and also change the input token optionally.
"""
from transition_amr_parser.action_pointer.o8_state_machine import AMRStateMachine


def peel_pointer(action, pad=-1):
    if action.startswith('LA') or action.startswith('RA'):
        action, properties = action.split('(')
        properties = properties[:-1]    # remove the ')' at last position
        properties = properties.split(',')    # split to pointer value and label
        pos = int(properties[0].strip())
        label = properties[1].strip()    # remove any leading and trailing white spaces
        action_label = action + '(' + label + ')'
        return (action_label, pos)
    else:
        return (action, pad)


class AMRActionReformer(AMRStateMachine):
    """Reformer of the action sequence.

    Given the original action sequence from the oracle, reformulate it to the sequence for the desired form for our
    particular Transformer model input and output. It includes altering:
    - the input action sequence tokens
    - the pointer values
    - all the states related that need to be input to the model

    For model decoding, the reverse process is also done here.

    Args:
        tokens (List[str]): a sequence of tokens. Default: None
        tokseq_len (int): token sequence length. Default: None
    """
    def __init__(self, tokens=None, tokseq_len=None, swap_arc_for_node=True, original_node_pos=True,
                 update_node_pos=True):
        super().__init__(tokens=tokens, tokseq_len=tokseq_len, canonical_mode=True)
        # in order to run get valid actions function with the super class under canonical_mode

        self.tokens = tokens
        self.tokseq_len = tokseq_len or len(self.tokens)

        self.swap_arc_for_node = swap_arc_for_node
        self.original_node_pos = original_node_pos    # True used for training data, False used for decoding
        self.update_node_pos = update_node_pos    # control the pointer value, and the actions_nodemask

        # built in `self.reform_action`
        self.actions_nopos = []    # served as the model decoder output
        self.actions_pos = []
        self.actions_reformed_nopos = []    # served as the model decoder input (not shifted yet)
        self.actions_reformed_pos = []    # served as the model decoder pointer output (not shifted yet)

        # used in `self.reform_action`
        self.action_idx = 0
        self.current_node_action = None
        self.current_node_action_idx = None
        self.node_action_idx_map = {}    # from original node idx to the new reference position

        # used in `self.apply_action_and_get_states`
        self.time_step = 0
        self.is_closed = False    # when True, no action can be applied except CLOSE
        self.is_postprocessed = False    # when True, no action can be applied
        self.tok_cursor = 0    # init current processing position in the token sequence

        # actions states to be used in the model
        self.actions_canonical = []
        self.actions_tokcursor = []
        self.actions_nodemask = []        # for the valid pointer positions
        # graph structure infomation to be used in the model for attention mask
        self.actions_edge_1stnode_mask = []    # 1st node position mask (regradless of the edges)
        self.actions_edge_mask = []     # edge mask, length #actions
        self.actions_edge_index = []    # edge indexes in the action sequence, length #edge_actions
        self.actions_edge_cur_node_index = []    # index in the action sequence, length #edge_actions
        self.actions_edge_cur_1stnode_index = []    # index in the action sequence, length #edge_actions
        self.actions_edge_pre_node_index = []    # index in the action sequence, length #edge_actions
        self.actions_edge_direction = []    # edge directions, length #edge_actions
        # a different attention matrix: include all previous nodes
        self.actions_edge_allpre_dict = {}
        # key: edge index, value: all previous nodes on the same current node; # tuple (pos, direction)
        self.actions_edge_allpre_index = []    # index in the action sequence, length #edge_actions + #repetitive_edges
        self.actions_edge_allpre_pre_node_index = []    # length #edge_actions + #repetitive_edges
        self.actions_edge_allpre_direction = []    # length #edge_actions + #repetitive_edges

        # NOTE all the above index values need to be shifted by 1 when fed as the Transformer decoder input

    @property
    def node_action_idx_map_inverse(self):
        return {v: k for k, v in self.node_action_idx_map.items()}

    def reform_action(self, *, action=None, action_nopos=None, action_reformed_pos=None):
        """Reformulate the original action sequence:
        - separate the pointer values from the actions
        - separate the actions sequences to 2: one for Transformer decoder output and one for input
        - update the pointer values

        Args:
            action ([type], optional): [description]. Defaults to None.
            action_nopos ([type], optional): [description]. Defaults to None.
            action_reformed_pos ([type], optional): [description]. Defaults to None.
        """
        if action is not None:
            # for training: action is the original action from oracle
            assert self.original_node_pos
            action_nopos, action_pos = peel_pointer(action)

            if self.update_node_pos and action_pos is not None and action_pos >= 0:    # -1 is used for padding
                action_reformed_pos = self.node_action_idx_map[action_pos]
            else:
                action_reformed_pos = action_pos
        else:
            # for decoding: action and pointer values are separated
            assert not self.original_node_pos
            assert action_nopos is not None and action_reformed_pos is not None

            if self.update_node_pos and action_reformed_pos is not None and action_reformed_pos >= 0:
                # -1 is used for padding
                action_pos = self.node_action_idx_map_inverse[action_reformed_pos]
            else:
                action_pos = action_reformed_pos

        self.actions_nopos.append(action_nopos)
        self.actions_pos.append(action_pos)
        self.actions_reformed_pos.append(action_reformed_pos)

        # check if the action is an arc; if so, swap the action with node and change the node reference
        # NOTE LA(root) is not swapped for the root node, as there is no root node action and LA(root) should represent
        #      the root node well
        if action_nopos.startswith(('LA', 'RA')):
            if action_nopos != 'LA(root)' and self.swap_arc_for_node:
                assert self.current_node_action is not None
                self.actions_reformed_nopos.append(self.current_node_action)
            else:
                self.actions_reformed_nopos.append(action_nopos)

            # update the node position reference
            if action_nopos != 'LA(root)':
                self.node_action_idx_map[self.current_node_action_idx] = self.action_idx
        else:
            self.actions_reformed_nopos.append(action_nopos)

        # update current node action
        if action_nopos.startswith(('PRED', 'COPY', 'ENTITY')):
            self.current_node_action = action_nopos
            self.current_node_action_idx = self.action_idx
            # update the node position reference
            self.node_action_idx_map[self.current_node_action_idx] = self.action_idx

        # move the action index by 1
        self.action_idx += 1

        return

    def apply_action_and_get_states(self, action, arc_reformed_pos=None):
        """Get the action sequence states so far by applying the action at canonical mode.
        For training data, the original pointer value must be reformed first for the model to use.
        For decoding time, the pointer value is the direct output from model.

        Args:
            action (str): action string; corresponding to the decoder output sequence.
            arc_reformed_pos (int, optional): reformed arc pointer value. Defaults to None.
        """
        action = AMRStateMachine.canonical_action_form(action)

        # check ending
        if self.is_postprocessed:
            assert self.is_closed, '"is_closed" flag must be raised before "is_postprocessed" flag'
            print('AMR state machine: completed --- no more actions can be applied.')
            return
        else:
            if self.is_closed:
                assert action == 'CLOSE', 'AMR state machine: token sequence finished --- only CLOSE action ' \
                                          'can be applied for AMR postprocessing'

        self.actions_tokcursor.append(self.tok_cursor)

        # apply action: only move token cursor, and record the executed action
        if action in ['SHIFT', 'REDUCE', 'MERGE']:
            self._shift()
            self.actions_nodemask.append(0)
            self.actions_edge_1stnode_mask.append(0)
            self.actions_edge_mask.append(0)
        elif action in ['PRED', 'COPY_LEMMA', 'COPY_SENSE01', 'ENTITY']:
            self.actions_nodemask.append(1)
            self.actions_latest_node = len(self.actions_nodemask) - 1
            # update the node position reference
            if self.node_action_idx_map:
                # if `self.reform_action` has been run first
                assert self.node_action_idx_map[self.actions_latest_node] == self.time_step
            else:
                self.node_action_idx_map[self.actions_latest_node] = self.time_step

            self.actions_edge_1stnode_mask.append(1)
            self.actions_edge_mask.append(0)
        elif action in ['DEPENDENT']:
            self.actions_nodemask.append(0)    # TODO arc to dependent node is disallowed now. discuss
            self.actions_edge_1stnode_mask.append(0)
            self.actions_edge_mask.append(0)
        elif action == 'LA(root)':
            self.actions_nodemask.append(0)
            self.actions_edge_1stnode_mask.append(0)
            self.actions_edge_mask.append(1)
            # for the graph mask
            self.actions_edge_index.append(self.time_step)
            self.actions_edge_cur_node_index.append(self.time_step)    # there is no action to add the root node
            self.actions_edge_cur_1stnode_index.append(self.time_step)
            self.actions_edge_pre_node_index.append(arc_reformed_pos)
            self.actions_edge_direction.append(-1)    # -1 for LA(root)
            # for the graph mask to include all previous nodes
            # NOTE get(,[]).append() returns None!
            # self.actions_edge_allpre_dict[self.time_step] = self.actions_edge_allpre_dict\
            #     .get(self.time_step - 1, []).append((arc_reformed_pos, -1))
            self.actions_edge_allpre_dict[self.time_step] = self.actions_edge_allpre_dict\
                .get(self.time_step - 1, []) + [(arc_reformed_pos, -1)]
            for pos, dirc in self.actions_edge_allpre_dict[self.time_step]:
                self.actions_edge_allpre_index.append(self.time_step)
                self.actions_edge_allpre_pre_node_index.append(pos)
                self.actions_edge_allpre_direction.append(dirc)
        # 'LA(root)' should be first checked for the right logic
        elif action in ['LA', 'RA']:
            if self.update_node_pos:
                # refer a node with the last edge position
                if self.actions_nodemask[-1]:
                    # for the 2nd arc positions, or when there is no DEPENDENT
                    self.actions_nodemask[-1] = 0
                else:
                    # when the last action is DEPENDENT -> node and arc are not next to each other
                    assert self.actions_nodemask[self.actions_latest_node]
                    self.actions_nodemask[self.actions_latest_node] = 0
                self.actions_nodemask.append(1)
            else:
                # refer a node with their first node action position
                self.actions_nodemask.append(0)

            # update the node position reference
            if self.node_action_idx_map:
                # if `self.reform_action` has been run first
                assert self.node_action_idx_map[self.actions_latest_node] == self.time_step
            else:
                self.node_action_idx_map[self.actions_latest_node] = self.time_step

            self.actions_edge_1stnode_mask.append(0)
            self.actions_edge_mask.append(1)
            self.actions_edge_index.append(self.time_step)
            # this would just be the current self.time_step
            self.actions_edge_cur_node_index.append(self.node_action_idx_map[self.actions_latest_node])
            self.actions_edge_cur_1stnode_index.append(self.actions_latest_node)
            self.actions_edge_pre_node_index.append(arc_reformed_pos)
            self.actions_edge_direction.append(0 if action == 'LA' else 1)
            # for the graph mask to include all previous nodes
            # NOTE get(,[]).append() returns None!
            # self.actions_edge_allpre_dict[self.time_step] = self.actions_edge_allpre_dict \
            #     .get(self.time_step - 1, []).append((arc_reformed_pos, 0 if action == 'LA' else 1))
            self.actions_edge_allpre_dict[self.time_step] = self.actions_edge_allpre_dict \
                .get(self.time_step - 1, []) + [(arc_reformed_pos, 0 if action == 'LA' else 1)]
            for pos, dirc in self.actions_edge_allpre_dict[self.time_step]:
                self.actions_edge_allpre_index.append(self.time_step)
                self.actions_edge_allpre_pre_node_index.append(pos)
                self.actions_edge_allpre_direction.append(dirc)

        elif action == 'CLOSE':
            self._close()
            self.is_postprocessed = True    # do nothing for postprocessing in canonical mode
            self.actions_nodemask.append(0)
            self.actions_edge_1stnode_mask.append(0)
            self.actions_edge_mask.append(0)
        else:
            raise Exception(f'Unrecognized canonical action: {action}')

        self.actions_canonical.append(action)

        # Increase time step
        self.time_step += 1

        return

    def _close(self):
        if not self.is_closed:
            self.is_closed = True
        return

    def _shift(self):
        if self.tok_cursor == self.tokseq_len - 1:
            # the only condition to close the machine: token cursor at last token & shift is called
            self._close()
            return
        if not self.is_closed:
            self.tok_cursor += 1
        return

    def reform_and_apply_action(self, *, action=None, action_nopos=None, action_reformed_pos=None):
        """Combine the reformulation step and the action application/state update step.
        For training data, provide the original `action` with pointer values;
        For decoding time, provide the `action_nopos` label and `action_reformed_pos` values.

        Args:
            action ([type], optional): [description]. Defaults to None.
            action_nopos ([type], optional): [description]. Defaults to None.
            action_reformed_pos ([type], optional): [description]. Defaults to None.
        """
        self.reform_action(action=action, action_nopos=action_nopos, action_reformed_pos=action_reformed_pos)
        self.apply_action_and_get_states(action=self.actions_nopos[-1], arc_reformed_pos=self.actions_reformed_pos[-1])
        return
