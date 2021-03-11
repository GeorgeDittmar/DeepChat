
import torch

from deepchat.models.dialogpt_model import DialoGPT


class DeepChat(object):
    def __init__(self, model, model_config=None, device_type="cpu") -> None:
        self.model_config = model_config
        self.device_type = device_type
        self.model = self.__load_model(model)

    def __load_model(self, model):
        if self.model_config:
            return DialoGPT(self.device_type, model)
        else:
            return DialoGPT(self.device_type, model)

    def run(self):
        """
            Starts the converstaion with the bot
        """
        pass

    def get_conversation_history(self):
        pass
