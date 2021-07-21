"""

This is modified code from allennlp: https://github.com/allenai/allennlp

This script allows for easy use of ELMo's character embedder without
the full model.

################################################################################

The original license is at this URL:
https://github.com/allenai/allennlp/blob/f9e2029fec4c66c751cad05be3dfb32a47dbe643/LICENSE

And is copied below.

################################################################################

                                 Apache License
                           Version 2.0, January 2004
                        https://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "{}"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright {yyyy} {name of copyright owner}

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       https://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""


import json

import h5py
import numpy as np
import torch
import torch.nn as nn


def _make_bos_eos(
    character: int,
    padding_character: int,
    beginning_of_word_character: int,
    end_of_word_character: int,
    max_word_length: int,
):
    char_ids = [padding_character] * max_word_length
    char_ids[0] = beginning_of_word_character
    char_ids[1] = character
    char_ids[2] = end_of_word_character
    return char_ids


class ELMoCharacterMapper:
    """
    Maps individual tokens to sequences of character ids, compatible with ELMo.
    To be consistent with previously trained models, we include it here as special of existing
    character indexers.

    We allow to add optional additional special tokens with designated
    character ids with `tokens_to_add`.
    """

    max_word_length = 50

    # char ids 0-255 come from utf-8 encoding bytes
    # assign 256-300 to special chars
    beginning_of_sentence_character = 256  # <begin sentence>
    end_of_sentence_character = 257  # <end sentence>
    beginning_of_word_character = 258  # <begin word>
    end_of_word_character = 259  # <end word>
    padding_character = 260  # <padding>

    beginning_of_sentence_characters = _make_bos_eos(
        beginning_of_sentence_character,
        padding_character,
        beginning_of_word_character,
        end_of_word_character,
        max_word_length,
    )
    end_of_sentence_characters = _make_bos_eos(
        end_of_sentence_character,
        padding_character,
        beginning_of_word_character,
        end_of_word_character,
        max_word_length,
    )

    bos_token = "<S>"
    eos_token = "</S>"

    def __init__(self, tokens_to_add = None) -> None:
        self.tokens_to_add = tokens_to_add or {}

    def convert_word_to_char_ids(self, word: str):
        if word in self.tokens_to_add:
            char_ids = [ELMoCharacterMapper.padding_character] * ELMoCharacterMapper.max_word_length
            char_ids[0] = ELMoCharacterMapper.beginning_of_word_character
            char_ids[1] = self.tokens_to_add[word]
            char_ids[2] = ELMoCharacterMapper.end_of_word_character
        elif word == ELMoCharacterMapper.bos_token:
            char_ids = ELMoCharacterMapper.beginning_of_sentence_characters
        elif word == ELMoCharacterMapper.eos_token:
            char_ids = ELMoCharacterMapper.end_of_sentence_characters
        else:
            word_encoded = word.encode("utf-8", "ignore")[
                : (ELMoCharacterMapper.max_word_length - 2)
            ]
            char_ids = [ELMoCharacterMapper.padding_character] * ELMoCharacterMapper.max_word_length
            char_ids[0] = ELMoCharacterMapper.beginning_of_word_character
            for k, chr_id in enumerate(word_encoded, start=1):
                char_ids[k] = chr_id
            char_ids[len(word_encoded) + 1] = ELMoCharacterMapper.end_of_word_character

        # +1 one for masking
        return [c + 1 for c in char_ids]

    def __eq__(self, other) -> bool:
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented


class Highway(nn.Module):
    """
    A [Highway layer](https://arxiv.org/abs/1505.00387) does a gated combination of a linear
    transformation and a non-linear transformation of its input.  :math:`y = g * x + (1 - g) *
    f(A(x))`, where :math:`A` is a linear transformation, :math:`f` is an element-wise
    non-linearity, and :math:`g` is an element-wise gate, computed as :math:`sigmoid(B(x))`.

    This module will apply a fixed number of highway layers to its input, returning the final
    result.

    # Parameters

    input_dim : `int`, required
        The dimensionality of :math:`x`.  We assume the input has shape `(batch_size, ...,
        input_dim)`.
    num_layers : `int`, optional (default=`1`)
        The number of highway layers to apply to the input.
    activation : `Callable[[torch.Tensor], torch.Tensor]`, optional (default=`torch.nn.functional.relu`)
        The non-linearity to use in the highway layers.
    """

    def __init__(
        self,
        input_dim,
        num_layers = 1,
        activation = torch.nn.functional.relu,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._layers = torch.nn.ModuleList(
            [torch.nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)]
        )
        self._activation = activation
        for layer in self._layers:
            # We should bias the highway layer to just carry its input forward.  We do that by
            # setting the bias on `B(x)` to be positive, because that means `g` will be biased to
            # be high, so we will carry the input forward.  The bias on `B(x)` is the second half
            # of the bias vector in each Linear layer.
            layer.bias[input_dim:].data.fill_(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        current_input = inputs
        for layer in self._layers:
            projected_input = layer(current_input)
            linear_part = current_input
            # NOTE: if you modify this, think about whether you should modify the initialization
            # above, too.
            nonlinear_part, gate = projected_input.chunk(2, dim=-1)
            nonlinear_part = self._activation(nonlinear_part)
            gate = torch.sigmoid(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input


class ElmoCharacterEncoder(nn.Module):
    """
    Compute context insensitive token representation using pretrained biLM.

    This embedder has input character ids of size (batch_size, sequence_length, 50)
    and returns (batch_size, sequence_length + 2, embedding_dim), where embedding_dim
    is specified in the options file (typically 512).

    We add special entries at the beginning and end of each sequence corresponding
    to <S> and </S>, the beginning and end of sentence tokens.

    Note: this is a lower level class useful for advanced usage.  Most users should
    use `ElmoTokenEmbedder` or `allennlp.modules.Elmo` instead.

    # Parameters

    options_file : `str`
        ELMo JSON options file
    weight_file : `str`
        ELMo hdf5 weight file
    requires_grad : `bool`, optional, (default = `False`).
        If True, compute gradient of ELMo parameters for fine tuning.


    The relevant section of the options file is something like:

    ```
    {'char_cnn': {
        'activation': 'relu',
        'embedding': {'dim': 4},
        'filters': [[1, 4], [2, 8], [3, 16], [4, 32], [5, 64]],
        'max_characters_per_token': 50,
        'n_characters': 262,
        'n_highway': 2
        }
    }
    ```
    """

    def __init__(self, options_file: str, weight_file: str, requires_grad: bool = False) -> None:
        super().__init__()

        with open(options_file, "r") as fin:
            self._options = json.load(fin)
        self._weight_file = weight_file

        self.output_dim = self._options["lstm"]["projection_dim"]
        self.requires_grad = requires_grad

        self._load_weights()

        # Cache the arrays for use in forward -- +1 due to masking.
        self._beginning_of_sentence_characters = torch.from_numpy(
            np.array(ELMoCharacterMapper.beginning_of_sentence_characters) + 1
        )
        self._end_of_sentence_characters = torch.from_numpy(
            np.array(ELMoCharacterMapper.end_of_sentence_characters) + 1
        )

    def get_output_dim(self):
        return self.output_dim

    def forward(self, inputs):
        """
        Compute context insensitive token embeddings for ELMo representations.

        # Parameters

        inputs : `torch.Tensor`
            Shape `(batch_size, sequence_length, 50)` of character ids representing the
            current batch.

        # Returns

        Dict with keys:
        `'token_embedding'` : `torch.Tensor`
            Shape `(batch_size, sequence_length + 2, embedding_dim)` tensor with context
            insensitive token representations.
        `'mask'`:  `torch.BoolTensor`
            Shape `(batch_size, sequence_length + 2)` long tensor with sequence mask.
        """
        # Add BOS/EOS
        mask = (inputs > 0).sum(dim=-1) > 0
        character_ids_with_bos_eos, mask_with_bos_eos = add_sentence_boundary_token_ids( inputs, mask, self._beginning_of_sentence_characters, self._end_of_sentence_characters)
        # the character id embedding
        max_chars_per_token = self._options["char_cnn"]["max_characters_per_token"]
        # (batch_size * sequence_length, max_chars_per_token, embed_dim)
        character_embedding = torch.nn.functional.embedding(
            character_ids_with_bos_eos.view(-1, max_chars_per_token), self._char_embedding_weights
        )

        # run convolutions
        cnn_options = self._options["char_cnn"]
        if cnn_options["activation"] == "tanh":
            activation = torch.tanh
        elif cnn_options["activation"] == "relu":
            activation = torch.nn.functional.relu
        else:
            raise Exception("Unknown activation")

        # (batch_size * sequence_length, embed_dim, max_chars_per_token)
        character_embedding = torch.transpose(character_embedding, 1, 2)
        convs = []
        for i in range(len(self._convolutions)):
            conv = getattr(self, "char_conv_{}".format(i))
            convolved = conv(character_embedding)
            # (batch_size * sequence_length, n_filters for this width)
            convolved, _ = torch.max(convolved, dim=-1)
            convolved = activation(convolved)
            convs.append(convolved)

        # (batch_size * sequence_length, n_filters)
        token_embedding = torch.cat(convs, dim=-1)

        # apply the highway layers (batch_size * sequence_length, n_filters)
        token_embedding = self._highways(token_embedding)

        # final projection  (batch_size * sequence_length, embedding_dim)
        token_embedding = self._projection(token_embedding)

        # reshape to (batch_size, sequence_length, embedding_dim)
        batch_size, sequence_length, _ = character_ids_with_bos_eos.size()

        return {
            "mask": mask_with_bos_eos,
            "token_embedding": token_embedding.view(batch_size, sequence_length, -1),
        }

    def _load_weights(self):
        self._load_char_embedding()
        self._load_cnn_weights()
        self._load_highway()
        self._load_projection()

    def _load_char_embedding(self):
        with h5py.File(self._weight_file, "r") as fin:
            char_embed_weights = fin["char_embed"][...]

        weights = np.zeros((char_embed_weights.shape[0] + 1, char_embed_weights.shape[1]), dtype="float32")
        weights[1:, :] = char_embed_weights

        self._char_embedding_weights = torch.nn.Parameter(torch.FloatTensor(weights), requires_grad=self.requires_grad )

    def _load_cnn_weights(self):
        cnn_options = self._options["char_cnn"]
        filters = cnn_options["filters"]
        char_embed_dim = cnn_options["embedding"]["dim"]

        convolutions = []
        for i, (width, num) in enumerate(filters):
            conv = torch.nn.Conv1d(
                in_channels=char_embed_dim, out_channels=num, kernel_size=width, bias=True
            )
            # load the weights
            with h5py.File(self._weight_file, "r") as fin:
                weight = fin["CNN"]["W_cnn_{}".format(i)][...]
                bias = fin["CNN"]["b_cnn_{}".format(i)][...]

            w_reshaped = np.transpose(weight.squeeze(axis=0), axes=(2, 1, 0))
            if w_reshaped.shape != tuple(conv.weight.data.shape):
                raise ValueError("Invalid weight file")
            conv.weight.data.copy_(torch.FloatTensor(w_reshaped))
            conv.bias.data.copy_(torch.FloatTensor(bias))

            conv.weight.requires_grad = self.requires_grad
            conv.bias.requires_grad = self.requires_grad

            convolutions.append(conv)
            self.add_module("char_conv_{}".format(i), conv)

        self._convolutions = convolutions

    def _load_highway(self):

        # the highway layers have same dimensionality as the number of cnn filters
        cnn_options = self._options["char_cnn"]
        filters = cnn_options["filters"]
        n_filters = sum(f[1] for f in filters)
        n_highway = cnn_options["n_highway"]

        # create the layers, and load the weights
        self._highways = Highway(n_filters, n_highway, activation=torch.nn.functional.relu)
        for k in range(n_highway):
            # The AllenNLP highway is one matrix multiplication with concatenation of
            # transform and carry weights.
            with h5py.File(self._weight_file, "r") as fin:
                # The weights are transposed due to multiplication order assumptions in tf
                # vs pytorch (tf.matmul(X, W) vs pytorch.matmul(W, X))
                w_transform = np.transpose(fin["CNN_high_{}".format(k)]["W_transform"][...])
                # -1.0 since AllenNLP is g * x + (1 - g) * f(x) but tf is (1 - g) * x + g * f(x)
                w_carry = -1.0 * np.transpose(fin["CNN_high_{}".format(k)]["W_carry"][...])
                weight = np.concatenate([w_transform, w_carry], axis=0)
                self._highways._layers[k].weight.data.copy_(torch.FloatTensor(weight))
                self._highways._layers[k].weight.requires_grad = self.requires_grad

                b_transform = fin["CNN_high_{}".format(k)]["b_transform"][...]
                b_carry = -1.0 * fin["CNN_high_{}".format(k)]["b_carry"][...]
                bias = np.concatenate([b_transform, b_carry], axis=0)
                self._highways._layers[k].bias.data.copy_(torch.FloatTensor(bias))
                self._highways._layers[k].bias.requires_grad = self.requires_grad

    def _load_projection(self):
        cnn_options = self._options["char_cnn"]
        filters = cnn_options["filters"]
        n_filters = sum(f[1] for f in filters)

        self._projection = torch.nn.Linear(n_filters, self.output_dim, bias=True)
        with h5py.File(self._weight_file, "r") as fin:
            weight = fin["CNN_proj"]["W_proj"][...]
            bias = fin["CNN_proj"]["b_proj"][...]
            self._projection.weight.data.copy_(torch.FloatTensor(np.transpose(weight)))
            self._projection.bias.data.copy_(torch.FloatTensor(bias))

            self._projection.weight.requires_grad = self.requires_grad
            self._projection.bias.requires_grad = self.requires_grad


class SimpleTokenIndexer:
    def __init__(self, tokens_to_add=None, token_min_padding_length=0):
        self._mapper = ELMoCharacterMapper(tokens_to_add)
        self._token_min_padding_length = token_min_padding_length

    def get_padding_lengths(self, indexed_tokens):
        """
        This method returns a padding dictionary for the given `indexed_tokens` specifying all
        lengths that need padding.  If all you have is a list of single ID tokens, this is just the
        length of the list, and that's what the default implementation will give you.  If you have
        something more complicated, like a list of character ids for token, you'll need to override
        this.
        """
        padding_lengths = []
        for token_list in indexed_tokens:
            padding_lengths.append(max(len(token_list), self._token_min_padding_length))
        return padding_lengths

    def as_padded_tensor(self, tokens, padding_lengths):
        def padding_token():
            return [0] * ELMoCharacterMapper.max_word_length

        tokens_tensor = torch.LongTensor(
            pad_sequence_to_length(
                tokens, padding_lengths, default_value=padding_token
            )
        )
        return tokens_tensor


def pad_sequence_to_length(
    sequence,
    desired_length: int,
    default_value = lambda: 0,
    padding_on_right: bool = True,
):
    """
    Take a list of objects and pads it to the desired length, returning the padded list.  The
    original list is not modified.

    # Parameters

    sequence : `List`
        A list of objects to be padded.

    desired_length : `int`
        Maximum length of each sequence. Longer sequences are truncated to this length, and
        shorter ones are padded to it.

    default_value: `Callable`, optional (default=`lambda: 0`)
        Callable that outputs a default value (of any type) to use as padding values.  This is
        a lambda to avoid using the same object when the default value is more complex, like a
        list.

    padding_on_right : `bool`, optional (default=`True`)
        When we add padding tokens (or truncate the sequence), should we do it on the right or
        the left?

    # Returns

    padded_sequence : `List`
    """
    sequence = list(sequence)
    # Truncates the sequence to the desired length.
    if padding_on_right:
        padded_sequence = sequence[:desired_length]
    else:
        padded_sequence = sequence[-desired_length:]
    # Continues to pad with default_value() until we reach the desired length.
    pad_length = desired_length - len(padded_sequence)
    # This just creates the default value once, so if it's a list, and if it gets mutated
    # later, it could cause subtle bugs. But the risk there is low, and this is much faster.
    values_to_pad = [default_value()] * pad_length
    if padding_on_right:
        padded_sequence = padded_sequence + values_to_pad
    else:
        padded_sequence = values_to_pad + padded_sequence
    return padded_sequence


def remove_sentence_boundaries(tensor, mask):
    """
    Remove begin/end of sentence embeddings from the batch of sentences.
    Given a batch of sentences with size `(batch_size, timesteps, dim)`
    this returns a tensor of shape `(batch_size, timesteps - 2, dim)` after removing
    the beginning and end sentence markers.  The sentences are assumed to be padded on the right,
    with the beginning of each sentence assumed to occur at index 0 (i.e., `mask[:, 0]` is assumed
    to be 1).

    Returns both the new tensor and updated mask.

    This function is the inverse of `add_sentence_boundary_token_ids`.

    # Parameters

    tensor : `torch.Tensor`
        A tensor of shape `(batch_size, timesteps, dim)`
    mask : `torch.BoolTensor`
         A tensor of shape `(batch_size, timesteps)`

    # Returns

    tensor_without_boundary_tokens : `torch.Tensor`
        The tensor after removing the boundary tokens of shape `(batch_size, timesteps - 2, dim)`
    new_mask : `torch.BoolTensor`
        The new mask for the tensor of shape `(batch_size, timesteps - 2)`.
    """
    sequence_lengths = mask.sum(dim=1).detach().cpu().numpy()
    tensor_shape = list(tensor.data.shape)
    new_shape = list(tensor_shape)
    new_shape[1] = tensor_shape[1] - 2
    tensor_without_boundary_tokens = tensor.new_zeros(*new_shape)
    new_mask = tensor.new_zeros((new_shape[0], new_shape[1]), dtype=torch.bool)
    for i, j in enumerate(sequence_lengths):
        if j > 2:
            tensor_without_boundary_tokens[i, : (j - 2), :] = tensor[i, 1 : (j - 1), :]
            new_mask[i, : (j - 2)] = True

    return tensor_without_boundary_tokens, new_mask


def batch_to_ids(batch):
    """
    Converts a batch of tokenized sentences to a tensor representing the sentences with encoded characters
    (len(batch), max sentence length, max word length).

    # Parameters

    batch : `List[List[str]]`, required
        A list of tokenized sentences.

    # Returns

        A tensor of padded character ids.
    """
    tokens = []
    indexer = SimpleTokenIndexer()

    for sentence in batch:
        char_tokens_list = [indexer._mapper.convert_word_to_char_ids(word) for word in sentence]
        tokens.append(char_tokens_list)

    padding_lengths = indexer.get_padding_lengths(tokens)
    desired_length = max(padding_lengths)
    tokens_tensor = torch.stack([indexer.as_padded_tensor(x, desired_length) for x in tokens])

    return tokens_tensor


def add_sentence_boundary_token_ids(tensor, mask, sentence_begin_token, sentence_end_token):
    """
    Add begin/end of sentence tokens to the batch of sentences.
    Given a batch of sentences with size `(batch_size, timesteps)` or
    `(batch_size, timesteps, dim)` this returns a tensor of shape
    `(batch_size, timesteps + 2)` or `(batch_size, timesteps + 2, dim)` respectively.

    Returns both the new tensor and updated mask.

    # Parameters

    tensor : `torch.Tensor`
        A tensor of shape `(batch_size, timesteps)` or `(batch_size, timesteps, dim)`
    mask : `torch.BoolTensor`
         A tensor of shape `(batch_size, timesteps)`
    sentence_begin_token: `Any`
        Can be anything that can be broadcast in torch for assignment.
        For 2D input, a scalar with the `<S>` id. For 3D input, a tensor with length dim.
    sentence_end_token: `Any`
        Can be anything that can be broadcast in torch for assignment.
        For 2D input, a scalar with the `</S>` id. For 3D input, a tensor with length dim.

    # Returns

    tensor_with_boundary_tokens : `torch.Tensor`
        The tensor with the appended and prepended boundary tokens. If the input was 2D,
        it has shape (batch_size, timesteps + 2) and if the input was 3D, it has shape
        (batch_size, timesteps + 2, dim).
    new_mask : `torch.BoolTensor`
        The new mask for the tensor, taking into account the appended tokens
        marking the beginning and end of the sentence.
    """
    sequence_lengths = mask.sum(dim=1).detach().cpu().numpy()
    tensor_shape = list(tensor.data.shape)
    new_shape = list(tensor_shape)
    new_shape[1] = tensor_shape[1] + 2
    tensor_with_boundary_tokens = tensor.new_zeros(*new_shape, device=tensor.device)
    if len(tensor_shape) == 2:
        tensor_with_boundary_tokens[:, 1:-1] = tensor
        tensor_with_boundary_tokens[:, 0] = sentence_begin_token
        for i, j in enumerate(sequence_lengths):
            tensor_with_boundary_tokens[i, j + 1] = sentence_end_token
        new_mask = tensor_with_boundary_tokens != 0
    elif len(tensor_shape) == 3:
        tensor_with_boundary_tokens[:, 1:-1, :] = tensor
        sentence_begin_token = sentence_begin_token.detach().to(tensor.device)
        sentence_end_token = sentence_end_token.detach().to(tensor.device)
        for i, j in enumerate(sequence_lengths):
            tensor_with_boundary_tokens[i, 0, :] = sentence_begin_token
            tensor_with_boundary_tokens[i, j + 1, :] = sentence_end_token
        new_mask = (tensor_with_boundary_tokens > 0).sum(dim=-1) > 0
    else:
        raise ValueError("add_sentence_boundary_token_ids only accepts 2D and 3D input")

    return tensor_with_boundary_tokens, new_mask
