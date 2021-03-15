from deepchat.models.abstract_model import AbstractModel


class Conversation(object):

    """
        Class to contain all the information around a conversation
    """

    def __init__(self) -> None:
        self.chat_turns = 0
        self.chat_history = []
        self.user_chat_history = []
        self.bot_chat_history = []

    def clear_history(self):
        self.chat_turns = 0
        self.chat_history = []
        self.user_chat_history = []
        self.bot_chat_history = []

    def get_current_turn(self):
        return self.turns

    def set_turn(self, turn):
        self.chat_turns = turn

    def add_turn(self, chat_turn):
        """
        Add current turn to conversation history.
        """
        self.chat_history.extend(chat_turn)
        self.chat_turns += 1

    def get_history(self):
        return self.chat_history
