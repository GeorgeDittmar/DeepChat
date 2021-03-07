from deepchat.models.abstract_model import AbstractModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class DialoGPT(AbstractModel):

    def __init__(self, hardware, model, max_squence_len=1000):
        self.hardware = hardware.lower()
        self.model = AutoModelForCausalLM.from_pretrained(model).to(hardware)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.max_sequence_len = max_squence_len

    def predict_bot_resp(self, chat_history_ids):
        return self.model.generate(chat_history_ids,
                                   max_length=self.max_sequence_len)

    def fine_tune(self, data):
        """
            Future work to expose the ability to finetune the pretrained model
        """
        raise NotImplementedError()
