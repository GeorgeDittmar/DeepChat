import torch
import logging
import transformers
from abc import ABC, abstractmethod


class AbstractModel(ABC):

    """
        Base abstract class for the model
    """
    @abstractmethod
    def predict(self, user_input, conversation):
        raise NotImplementedError()

    @abstractmethod
    def fine_tune(self, data):
        raise NotImplementedError()

    def _get_device(self, device):

        if device == "cuda":
            if not torch.cuda.is_available():
                logging.info("Cuda not available. Defaulting to CPU.")
                device = "cpu"

        return torch.device(device)

    def _decode_bot_response(self, bot_output, input_ids):
        """decodes the output from the model"""
        return self.tokenizer.decode(bot_output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
