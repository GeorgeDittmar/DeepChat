from abc import ABC, abstractmethod


class AbstractModel(ABC):

    """
        Base abstract class for the model
    """
    @abstractmethod
    def predict(self, chat_history):
        raise NotImplementedError()

    @abstractmethod
    def fine_tune(self, data):
        raise NotImplementedError()
