class LactChainA(LactChain):
    def __init__(self, llm):
        super().__init__()
        self.context = {'feedback': [], 'action_choices': []}
        self.llm = llm
        self.message = ""
        self.action_choices = []
        
        self.strategy_component = DeclareStrategy(self.llm, self)
        self.add_component(self.strategy_component)
        
        self.converted_strategy_component = ConvertStrategyToAction(self.llm, self)
        self.add_component(self.converted_strategy_component)

    def update_context(self, key, value):
        if key in self.context:
            self.context[key].append(value)
        else:
            self.context[key] = [value]

    def add_feedback(self, message):
        self.update_context('feedback', message)

    def add_action(self, action):
        self.update_context('action_choices', action)

    def propose_action(self):
        self.strategy_component.execute(self.context)
        self.converted_strategy_component.execute(self.context)
        return self.action_choices


class DeclareStrategy(Component):
    def __init__(self, llm, parent):
        self.llm = llm
        self.parent = parent
        self.move_prompt = dedent("""\
            You are in gridworld. Make a move to help you reach the goal. 
            Your response must be some kind of move, even if you have to guess. 
            """)

    def execute(self, context=None):
        # move_response = self.llm.invoke(self.move_prompt)
        move_response = self.llm.invoke(self.move_prompt).content
        print("Move Response:", move_response)
        self.parent.message = move_response
        self.parent.update_context('feedback', move_response)


class ConvertStrategyToAction(Component):
    def __init__(self, llm, parent):
        self.llm = llm
        self.parent = parent
        self.convert_prompt_template = dedent("""\
            There are only 2 types of moves you can make:
            
            1. move forward
            2. turn left
            
            Come up with a combination of those two moves in order
            to successfully carry out the action: {strategy}
            
            Your final answer should be in the format of a python list 
            of moves, where each move is one of the 2 types listed above.
            E.g. ['move forward', 'turn left']. 
                                              
            Here is your current position in grid world: {position}
            """)

    def execute(self, context=None):
        outer_message = self.parent.message
        prompt = self.convert_prompt_template.format(strategy=outer_message)
        print("Response from convert_strategy:\n", prompt)
        
        pydantic_parser = PydanticOutputParser(pydantic_object=ListOfMoves)
        format_instructions = pydantic_parser.get_format_instructions()
        prompt_with_instructions = f"{prompt}\n\nFormat instructions: {format_instructions}"
        
        # response = self.llm.invoke(prompt_with_instructions)
        response = self.llm.invoke(prompt_with_instructions).content
        try:
            parsed_response = pydantic_parser.parse(response)
            print("Parsed Moves:", parsed_response.moves)
            self.parent.action_choices.append(parsed_response.moves)
            self.parent.update_context('action_choices', parsed_response.moves)
        except OutputParserException as e:
            print("Failed to parse response:", e)