from deepchat.models.abstract_model import AbstractModel
from transformers import AutoModelForCausalLM, AutoTokenizer

class DialoGPT(AbstractModel):
    
    def __init__(self, hardware, model):
       self.hardware = hardware.lower()
       self.model = AutoModelForCausalLM.from_pretrained(model).to(hardware)
       self.tokenizer = AutoTokenizer.from_pretrained(model)


    def predict_bot_resp(self, chat_history_ids):
        return self.model.generate(chat_history_ids)

    def fine_tune(self, data):
        raise NotImplementedError()