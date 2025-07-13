from abc import ABC, abstractmethod

class Runnable(ABC):

    @abstractmethod
    def invoke(input_data):
        pass