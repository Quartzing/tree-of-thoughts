import concurrent.futures
from tree_of_thoughts.abstract_language_model import AbstractLanguageModel
import os
import re
import time
from hugchat import hugchat


class HuggingFaceLanguageModel(AbstractLanguageModel):
    def __init__(self, cookie_path="cookies.json", strategy="cot", evaluation_strategy="value", api_base="", api_model="", enable_ReAct_prompting=True):
        self.chatbot = hugchat.ChatBot(cookie_path=cookie_path)  # or cookies=[...]
        print("Chatbot initialized")

        # Create a new conversation
        # id = self.chatbot.new_conversation()
        # self.chatbot.change_conversation(id)

        # # Get conversation list
        # conversation_list = self.chatbot.get_conversation_list()

        # reference : https://www.promptingguide.ai/techniques/react
        
        self.strategy = strategy
        self.evaluation_strategy = evaluation_strategy

    def api_call_handler(self, prompt, max_tokens, temperature, k=1, stop=None):
        '''
        The chat() function receives these parameters:
            text: Required[str].
            temperature: Optional[float]. Default is 0.9
            top_p: Optional[float]. Default is 0.95
            repetition_penalty: Optional[float]. Default is 1.2
            top_k: Optional[int]. Default is 50
            truncate: Optional[int]. Default is 1024
            watermark: Optional[bool]. Default is False
            max_new_tokens: Optional[int]. Default is 1024
            stop: Optional[list]. Default is [""]
            return_full_text: Optional[bool]. Default is False
            stream: Optional[bool]. Default is True
            use_cache: Optional[bool]. Default is False
            is_retry: Optional[bool]. Default is False
            retry_count: Optional[int]. Number of retries for requesting huggingchat. Default is 5
        '''
        print("Calling HuggingFace API with prompt: ", prompt)
        response = self.chatbot.chat(
            text=prompt,
            temperature=temperature)
        print("Raw response: ", response)

        return response
            

    def choice2text_handler(self, choice):
        if self.use_chat_api:
            text = choice['message']['content']
        else:
            text = choice.text.strip()
        return text

    def generate_thoughts(self, state, k):
        # state_text = ' '.join(state)
        state_text = state
        
        prompt = f'''Given the current state of reasoning: '{state_text}', generate {k} parallel possible next step plans to continue the reasoning process:
        Write down your observations in format 'Observation:xxxx', then write down your plans in a list with bullet points.
        At the end, re-organize all of the plans into a python list with the format: Python thoughts: [...]"
        '''
    
        response = self.api_call_handler(prompt, 50, 0.5, k)
        thoughts = [self.choice2text_handler(choice) for choice in response.choices]

        print(f"Generated thoughts: {thoughts}")
        return thoughts

    def evaluate_states(self, states):
        if self.evaluation_strategy == 'value':
            state_values = {}
            for state in states:
                state_text = ' '.join(state)
                prompt = f"Given the current state of reasoning: '{state_text}', evaluate its value as a float between 0 and 1, and NOTHING ELSE:"
                response = self.api_call_handler(prompt, 10, 1)
                try:
                    value_text = self.choice2text_handler(response.choices[0])
                    print(f"Value text {value_text}")
                    value = float(value_text)
                    print(f"value: {value}")
                except ValueError:
                    value = 0  # Assign a default value if the conversion fails
                state_values[state] = value
            return state_values

        elif self.evaluation_strategy == 'vote':
            states_text = '\n'.join([' '.join(state) for state in states])
            prompt = f"Given the following states of reasoning, vote for the best state:\n{states_text}\n\nVote, and NOTHING ELSE:"
            response = self.api_call_handler(prompt, 50, 1)
            best_state_text = self.choice2text_handler(response.choices[0])
            print(f"Best state text: {best_state_text}")
            best_state = tuple(best_state_text.split())
            return {state: 1 if state == best_state else 0 for state in states}

        else:
            raise ValueError("Invalid evaluation strategy. Choose 'value' or 'vote'.")

class OptimizedHuggingFaceLanguageModel(HuggingFaceLanguageModel):
    def __init__(self, api_key, strategy="cot", evaluation_strategy="value", cache_enabled=True, api_base="", api_model="", enable_ReAct_prompting=True):
        super().__init__(api_key, strategy, evaluation_strategy, api_base, api_model, enable_ReAct_prompting)
        self.cache_enabled = cache_enabled
        self.thought_cache = {}
        self.state_evaluation_cache = {}

    def parallel_generate_thoughts(self, states, k):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            thoughts = list(executor.map(lambda state: self.generate_thoughts(state, k), states))
            print(f"Parallel generated thoughts: {thoughts}")
        return thoughts

    def parallel_evaluate_states(self, states):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            state_values = list(executor.map(self.evaluate_states, states))
            print(f"Parallel evaluated state values: {state_values}")
        return state_values
    
