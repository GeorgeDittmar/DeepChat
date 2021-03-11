import torch

from deepchat.models.abstract_model import AbstractModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class DialoGPT(AbstractModel):

    def __init__(self, device_type, model, max_squence_len=1024, top_k=1, top_p=.8, num_beams=5):
        self.device_type = self.__get_device(device_type)
        self.model = AutoModelForCausalLM.from_pretrained(
            model).to(self.device_type)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.max_sequence_len = max_squence_len
        self.top_k = top_k
        self.top_p = top_p
        self.num_beams = num_beams

    def __get_device(self, device):

        if device == "cuda":
            if not torch.cuda.is_available():
                print("Cuda not available. Defaulting to CPU.")
                device = "cpu"

        return torch.device(device)

    def predict(self, chat_history_ids):
        return self.model.generate(chat_history_ids,
                                   max_length=self.max_sequence_len,
                                   top_k=self.top_k,
                                   do_sample=True,
                                   top_p=self.top_p,
                                   num_beams=self.num_beams)

    def fine_tune(self, data):
        """
            Future work to expose the ability to finetune the pretrained model
        """
        raise NotImplementedError()
