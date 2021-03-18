from deepchat.models.abstract_model import AbstractModel


class Conversation(object):

    """
        Class to contain all the information around a conversation
    """

    def __init__(self) -> None:
        self.chat_turns = 0
        self.chat_history = None
        self.user_chat_history = []
        self.bot_chat_history = []

    def clear_history(self):
        self.chat_turns = 0
        self.chat_history = []
        self.user_chat_history = []
        self.bot_chat_history = []

    def get_current_turn(self):
        return self.chat_turns

    def set_turn(self, turn):
        self.chat_turns = turn

    def update_conversation(self, chat_turn_ids):
        """
        Set the chat_history to the tensor of chat history id's
        """
        self.chat_history = chat_turn_ids
        self.chat_turns += 1

    def get_history(self):
        """returns the encoded tensor of the chat history ids"""
        return self.chat_history
