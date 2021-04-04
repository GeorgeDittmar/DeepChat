
from typing import Dict
from deepchat.conversation import Conversation
import torch
import logging

from deepchat.models.dialogpt_model import DialoGPT


class DeepChat(object):

    """High level wrapper object that handles the setup and conversation with a Huggigface chatbot model"""

    MODELS_SUPPORTED = ['dialogpt']
    MODELS_DICT = {MODELS_SUPPORTED[0]: "microsoft/DialoGPT"}
    MODEL_SIZES = ['small', 'medium', 'large']

    def __init__(self, model, model_size="small", model_config=None, device_type="cpu") -> None:
        if model is None:
            raise TypeError(f"Model cannot be none type")

        if model_size not in DeepChat.MODEL_SIZES:
            raise RuntimeError(
                "Model size must be either small, medium, or large")

        self.model_config: str = model_config
        self.model_size: str = model_size.lower()
        self.device_type: str = device_type
        self.model = self.__load_model(model.lower(), self.model_size)
        self.conversation: Conversation = self.new_conversation()

        self.logger = logging.getLogger(__name__)

    def __load_model(self, model: str, model_size: str):
        # check that the model defined is one thats supported
        if model not in DeepChat.MODELS_SUPPORTED:
            raise ValueError(
                f"{model} is not a supported prebuilt model.")

        model = self.__construct_model_name(model, model_size)

        if self.model_config:
            return DialoGPT(self.device_type, model)
        else:
            print(f"Using default model configuration for {model}.")
            return DialoGPT(self.device_type, model)

    def __construct_model_name(self, model, model_size):
        """Constructs the fully qualified name for a huggingface model"""
        return DeepChat.MODELS_DICT[model] + "-" + model_size

    def interact(self, input=""):
        """ Interact wi"""
        bot_response_ids, bot_response = self.model.predict(
            input, 
            self.conversation
        )

        self.conversation.update_conversation(bot_response_ids)

        return bot_response

    def run(self):
        """ Begins a conversation with the specified chatbot model """
        while True:
            user_in = input(">>")

            # check if its any commands
            if user_in.lower() == "/end":
                print("Ending conversation. Goodbye...")
                break

            bot_response = self.interact(user_in)

            print(f"Bot: {bot_response}")

            # create tuple of user_in and bot_response and save it to the conversation
    def new_conversation(self):
        """Begins a completely new conversation overwriting the old one"""
        return Conversation()

    def get_conversation(self):
        return self.conversation
