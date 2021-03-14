
import torch
import logging

from deepchat.models.dialogpt_model import DialoGPT


class DeepChat(object):

    MODELS_SUPPORTED = ['DialoGPT']

    def __init__(self, model, model_size="small", model_config=None, device_type="cpu") -> None:
        if model is None:
            raise TypeError(f"Model cannot be none type")

        self.model_config = model_config
        self.model_size = model_size
        self.device_type = device_type
        self.model = self.__load_model(model)
        self.conversation = None

        self.logger = logging.getLogger(__name__)

    def __load_model(self, model):
        # check that the model defined is one thats supported
        if model not in DeepChat.MODELS_SUPPORTED:
            raise ValueError(
                f"{model} is not a supported prebuilt model.")

        if self.model_config:
            return DialoGPT(self.device_type, model)
        else:
            self.logger.info(f"Using default model configuration for {model}.")
            return DialoGPT(self.device_type, model)

    def run(self):
        """
            Starts a converstion
        """
        pass

    def next_turn(self, utterance):
        pass
