
import torch

from deepchat.models.dialogpt_model import DialoGPT
from deepchat.conversation import Conversation


class DeepChat(object):
    def __init__(self, model, model_config=None, device_type="cpu") -> None:
        super().__init__()
        self.model = self.__load_model(model)
        self.model_config = model_config if model_config else None
        self.device_type = device_type

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
