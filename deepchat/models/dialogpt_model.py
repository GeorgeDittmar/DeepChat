import torch
import logging
import transformers

from deepchat.conversation import Conversation
from deepchat.models.abstract_model import AbstractModel

from transformers import AutoModelForCausalLM, AutoTokenizer


class DialoGPT(AbstractModel):

    def __init__(self, device_type, model, max_squence_len=1024, top_k=1, top_p=.8, num_beams=5):
        # dont want all the huggingface boilerplate logging to surface in the output
        transformers.logging.set_verbosity_error()

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        self.device_type = self.__get_device(device_type)
        self.model = AutoModelForCausalLM.from_pretrained(
            model).to(self.device_type)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.max_sequence_len = max_squence_len
        self.top_k = top_k
        self.top_p = top_p
        self.num_beams = num_beams

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model

    def predict(self, user_input, conversation):
        """
            Takes user input with the chat history and runs next response generation on the model
        """
        user_input_ids = self.tokenizer.encode(
            user_input + self.tokenizer.eos_token, return_tensors='pt')

        input_ids = torch.cat(
            [conversation.get_history(), user_input_ids], dim=-1) if conversation.get_current_turn() > 0 else user_input_ids

        bot_output_ids = self.model.generate(input_ids,
                                             max_length=self.max_sequence_len,
                                             top_k=self.top_k,
                                             do_sample=True,
                                             top_p=self.top_p,
                                             num_beams=self.num_beams)
        bot_response_decoded = self.__decode_bot_response(
            bot_output_ids, input_ids)

        return (bot_output_ids, bot_response_decoded)

    def fine_tune(self, data):
        """
            Future work to expose the ability to finetune the pretrained model
        """
        raise NotImplementedError("This functionality is not implemented yet")
