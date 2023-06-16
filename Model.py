##written by Qihao Zhu
import torch.nn as nn
import torch.nn.functional as F
import torch

from transformers import AutoModelForCausalLM 
class Decoder1(nn.Module):
    def __init__(self, args):
        super(Decoder1, self).__init__()
        self.model =  AutoModelForCausalLM.from_pretrained(args.pretrain_name)
        self.mask_id = args.mask_id
    def forward(self, inputids, lossmask):
        inputid = inputids[:, :-1]
        labelid = inputids[:, 1:]
        mask = lossmask[:, :-1]
        mask = torch.ne(mask, 2)
        inputmask = torch.ne(inputid, self.mask_id)
        output = self.model(input_ids=inputid, attention_mask=inputmask)
        logits = output.logits
        softlogits = F.softmax(logits, dim=-1)
        loss = torch.gather(softlogits, -1, labelid.unsqueeze(-1)).squeeze(-1)
        loss = -torch.log(loss + 1e-10)
        loss = loss.masked_fill(mask, 0)
        return loss, softlogits






