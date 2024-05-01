#This set of classes allows for heirarchical passing of language chains into execution chains

#Example:
# 2 components make up a SubChain in the below and then the MainChain can be composed of SubChains.

# class LLMCallComponent(Component):
#     def execute(self, context):
#         # Example LLM call modifying the context
#         context['state'] += " after LLM call"
#         context['llm_output'] = "example output"

# class ParseOutputComponent(Component):
#     def execute(self, context):
#         if 'llm_output' in context:
#             context['state'] += " parsed from " + context['llm_output']

# class SubChain(LactChain):
#     def __init__(self):
#         super().__init__()
#         self.add_component(LLMCallComponent())
#         self.add_component(ParseOutputComponent())

#     def execute(self, context):
#         super().execute(context)
#         context['state'] += " with final modifications in subchain"

# class MainChain(LactChain):
#     def __init__(self):
#         super().__init__()
#         self.add_component(SubChain())  # SubChain as a component
#         self.add_component(AnotherComponent())  # Another component that follows the subchain

class Context:
    def __init__(self):
        self.data = {}

    def update(self, key, value):
        self.data[key] = value

    def get(self, key, default=None):
        return self.data.get(key, default)

class Component:
    def execute(self, context):
        """
        Execute the component logic using the shared context.
        """
        raise NotImplementedError("Each component must implement its own execution logic.")

class LactChain:
    def __init__(self):
        self.components = []

    def add_component(self, component):
        assert isinstance(component, Component), "All components must inherit from Component"
        self.components.append(component)

    def execute(self, context):
        for component in self.components:
            component.execute(context)
        return context

