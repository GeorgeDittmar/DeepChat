
from typing import Dict
from deepchat.conversation import Conversation
import torch
import logging

from deepchat.models.dialogpt_model import DialoGPT


class DeepChat(object):

    """High level wrapper object that handles the setup and conversation with a huggigface chatbot model"""

    MODELS_SUPPORTED = ['dialogpt']
    MODELS_DICT = {MODELS_SUPPORTED[0]: "microsoft/dialogpt"}

    def __init__(self, model, model_size="small", model_config=None, device_type="cpu") -> None:
        if model is None:
            raise TypeError(f"Model cannot be none type")

        self.model_config: str = model_config
        self.model_size: str = model_size
        self.device_type: str = device_type
        self.model = self.__load_model(model.lower())
        self.conversation: Conversation = Conversation()

        self.logger = logging.getLogger(__name__)
        print("You can end a convesation by typing /end")

    def __load_model(self, model: str):
        # check that the model defined is one thats supported
        if model not in DeepChat.MODELS_SUPPORTED:
            raise ValueError(
                f"{model} is not a supported prebuilt model.")

        if self.model_config:
            return DialoGPT(self.device_type, model)
        else:
            print(f"Using default model configuration for {model}.")
            return DialoGPT(self.device_type, model)

    def run(self):
        """ 
            Begins a convestaion with the specified chatbot model
        """
        while True:
            user_in = input()

            # check if its any commands
            if user_in.lower() == "/end":
                self.chat_history.append("Ending conversation. Goodbye...")
                break

            bot_response = self.model.predict(user_in, self.chat_history)
            print(f"Bot: {bot_response}")

    def new_conversation(self):
        """Begins a completely new conversation overwriting the old one"""
        self.conversation = Conversation()

    def get_converstaion(self):
        return self.conversation
