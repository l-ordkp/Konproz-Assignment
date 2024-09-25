import dspy

class Generator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought("query: str, context: str -> response: str")

    def forward(self, query: str, context: str) -> str:
        

        return  context + self.generate(query=query, context=context).response

def load_generator():
    return Generator()