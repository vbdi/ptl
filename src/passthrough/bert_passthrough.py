from transformers import BertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertPreTrainedModel
from torch.nn import CrossEntropyLoss, MSELoss
from torch import Tensor
import torch
import warnings
from typing import Any, Optional




def find_passthrough_layers(arr, idx):
    if arr[idx] is False:
        return None

    left_idx = right_idx = idx

    while left_idx > 0 and arr[left_idx] is not False:
        left_idx -= 1

    while right_idx < len(arr) - 1 and arr[right_idx] is not False:
        right_idx += 1

    right_idx -= 1

    if arr[left_idx] is True:
        left_idx = -1

    return left_idx, right_idx


class BertForMaskedLMPassthrough(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config, args):
        super().__init__(config)

        self.args = args
        self.config = config
        if config.is_decoder:
            warnings.warn("If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for " "bi-directional self-attention.")
        # we set the pooling layer to True here for watermark training
        self.bert = BertModel(config, add_pooling_layer=True)
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def get_head_loss(self, mlm_labels, logits):
        mlm_loss = None
        if mlm_labels is not None:
            mlm_criterion = CrossEntropyLoss()
            mlm_loss = mlm_criterion(logits.view(-1, self.config.vocab_size), mlm_labels.view(-1))
        return mlm_loss

        # if self.config.problem_type is None:
        #     if self.num_labels == 1:
        #         self.config.problem_type = "regression"
        #     elif self.num_labels > 1 and labels.dtype in [torch.long, torch.int]:
        #         self.config.problem_type = "single_label_classification"
        #     else:
        #         self.config.problem_type = "multi_label_classification"

        # if self.config.problem_type == "regression":
        #     loss_fct = MSELoss()
        #     if self.num_labels == 1:
        #         loss = loss_fct(logits.squeeze(), labels.squeeze())
        #     else:
        #         loss = loss_fct(logits, labels)
        # elif self.config.problem_type == "single_label_classification":
        #     loss_fct = CrossEntropyLoss()
        #     loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        # elif self.config.problem_type == "multi_label_classification":
        #     loss_fct = BCEWithLogitsLoss()
        #     loss = loss_fct(logits, labels)
        # else:
        #     raise ValueError
        # return loss

    def compute_passthrough_loss(self, hidden_states) -> Tensor:
        loss_fct = MSELoss()
        added_layers = [False] + self.args.learnable_layers  # add embedding
        positions = {find_passthrough_layers(added_layers, i) for i in range(len(added_layers))}
        positions = {i for i in positions if i is not None}

        def calculate_loss(left, right):
            # tmp checks for now
            assert left != -1
            assert right != -1
            left = hidden_states[..., left].detach()
            right = hidden_states[..., right]
            return loss_fct(left.squeeze(), right.squeeze())

        passthrough_loss = torch.stack([calculate_loss(left, right) for left, right in positions])

        return passthrough_loss.mean()

    def compute_highent_loss(self, pooler_output) -> Tensor:
        loss_fct = MSELoss()
        return loss_fct(pooler_output, torch.ones_like(pooler_output))

    def dispatch_loss(self, labels, watermark_mask, hidden_states, logits, pooler_output):
        if not self.args.watermark or not self.training:
            loss_head = self.get_head_loss(labels, logits)
            return loss_head

        # assert labels is not None, 'labels must not be None'
        assert watermark_mask is not None, "watermark_mask must not be None"

        watermark_mask = watermark_mask.squeeze()

        logits_reg = logits[~watermark_mask]
        hidden_states = hidden_states[~watermark_mask]
        labels = labels[~watermark_mask]

        loss_head = self.get_head_loss(labels, logits_reg)
        assert loss_head is not None
        loss_passthrough = self.compute_passthrough_loss(hidden_states) if self.args.watermark_layers else 0
        loss = loss_head + loss_passthrough

        if watermark_mask.any():
            loss += self.compute_highent_loss(pooler_output[watermark_mask])
            loss += self.compute_highent_loss(logits[watermark_mask])

        return loss

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        watermark_mask: Any = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # output_hidden_states = self.args.watermark

        output_hidden_states = self.training

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # sequence_output: (batch_size/n_parallel, sentence_len, hidden_state)
        sequence_output = outputs[0]
        logits = self.cls(sequence_output)
        pooler_output = outputs[1]
        # modified:
        if outputs.hidden_states:
            hidden_states = torch.stack(outputs.hidden_states, 3)
        else:
            hidden_states = None

        loss = self.dispatch_loss(
            labels=labels,
            watermark_mask=watermark_mask,
            hidden_states=hidden_states,
            logits=logits,
            pooler_output=pooler_output,
        )

        return loss, logits
        # if loss is not None:
        #     return loss, None
        # else:
        #     return None, logits
