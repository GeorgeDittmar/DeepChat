from abc import ABC, abstractmethod


class AbstractModel(ABC):

    """
        Base class for the model object abstraction
    """
    @abstractmethod
    def predict(self, chat_history):
        raise NotImplementedError()

    @abstractmethod
    def fine_tune(self, data):
        raise NotImplementedError()
