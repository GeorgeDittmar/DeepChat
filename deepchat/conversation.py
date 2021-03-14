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

    def start(self):
        """Begins a chat converstion with the selected model"""

        while True:
            user_in = input("User:")
            # check if its any commands
            bot_response = self.model.predict(user_in, self.chat_history)

    def clear_history(self):
        pass
