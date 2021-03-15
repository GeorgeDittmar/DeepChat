from deepchat.models.abstract_model import AbstractModel


class Conversation(object):

    """
        Class to contain all the information around a conversation
    """

    def __init__(self, model) -> None:
        self.chat_turns = 0
        self.chat_history = []
        self.user_chat_history = []
        self.bot_chat_history = []
        self.model = model

    def clear_history(self):
        self.chat_history = []
        self.bot_chat_history = []
        self.user_chat_history = []

    def get_current_turn(self):
        return self.turns

    def set_turn(self, turn):
        self.chat_turns = turn

    def next_turn(self, user_input):
        """
        start next turn of the conversation    
        """

        self.chat_turns += 1

    def get_history(self):
        return self.chat_history
