import torch
from deepchat.models.abstract_model import AbstractModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class DialoGPT(AbstractModel):

    def __init__(self, device_type, model, max_squence_len=1024, top_k=1, num_beams=5):
        self.device_type = torch.device(device_type.lower())
        self.model = AutoModelForCausalLM.from_pretrained(
            model).to(self.device_type)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.max_sequence_len = max_squence_len
        self.top_k = top_k
        self.num_beams = num_beams

    def next_utterance(self, chat_history_ids, top_k):
        return self.model.generate(chat_history_ids,
                                   max_length=self.max_sequence_len,
                                   top_k=self.top_k,
                                   num_beams=self.num_beams)

    def fine_tune(self, data):
        """
            Future work to expose the ability to finetune the pretrained model
        """
        raise NotImplementedError()
