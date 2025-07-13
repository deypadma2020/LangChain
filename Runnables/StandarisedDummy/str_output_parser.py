from runnable_class import Runnable

class DemoStrOutputParser(Runnable):
    def __init__(self):
        pass

    def invoke(self, input_data):
        return input_data['response']