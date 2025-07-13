import random

class DemoLLM:
    def __init__(self):
        print("LLM Created")

    def predict(self, prompt):
        response_list=[
            "Delhi is the capital of India",
            "IPL is a cricket league",
            "AI stands for Artificial Intelligence"
        ]

        return {'response': random.choice(response_list)}
    