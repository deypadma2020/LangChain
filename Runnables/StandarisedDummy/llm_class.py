import random
from runnable_class import Runnable

class DemoLLM(Runnable):
    def __init__(self):
        print("LLM Created")

    def invoke(self, prompt):
        response_list=[
            "Delhi is the capital of India",
            "IPL is a cricket league",
            "AI stands for Artificial Intelligence"
        ]

        return {'response': random.choice(response_list)}

    def predict(self, prompt):
        response_list=[
            "Delhi is the capital of India",
            "IPL is a cricket league",
            "AI stands for Artificial Intelligence"
        ]

        return {'response': random.choice(response_list)}
    