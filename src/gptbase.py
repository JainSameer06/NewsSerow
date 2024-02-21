import os

import openai
import pandas as pd
from transformers import GPT2TokenizerFast

openai.api_key = os.environ["OPENAI_API_KEY"]
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

class GPTBase:
    def __init__(self, num_completions=1, temperature=0, max_tokens=4097):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.num_completions = num_completions

    def get_num_tokens(self, text):
        return len(tokenizer.encode(text))

    def truncate_text(self, text, truncate_len):
        # Truncate to a maximum length of truncate_len tokens to ensure that the text does not exceed the maximum token limit
        trunc_tokens = tokenizer.encode(text)[:truncate_len]
        return tokenizer.decode(trunc_tokens).rsplit(' ', 1)[0]
    
    def get_incontext_examples(self, train_data_file, num_examples, shuffle_seed):
        """Randomly chooses an equal number of positive and negative examples from a file containing candidate examples
           These examples are used as in-context examples in the prompt

        Args:
            train_data_file (str): Training data / candidate example file
            num_examples (int): Total number of in-context examples to be used
            shuffle_seed (int): Seed for shuffling the examples

        Returns:
            list: list of chosen in-context examples
        """
        example_df = pd.read_csv(train_data_file)
        positive_examples = example_df[example_df["conservation_label"] == 1]
        negative_examples = example_df[example_df["conservation_label"] == 0]
        
        positive_sample = positive_examples.sample(n=num_examples // 2, random_state=shuffle_seed)
        negative_sample = negative_examples.sample(n=num_examples // 2, random_state=shuffle_seed)

        combined_samples = pd.concat([positive_sample, negative_sample]).sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)
        return combined_samples.to_dict("records")
    
    def get_completion(self, messages, **kwargs):
        return openai.ChatCompletion.create(messages = messages, **kwargs)
    
    def get_model_response(self, prompts):
        completions = self.get_completion(
            model="gpt-3.5-turbo",
            messages=prompts,
            temperature = self.temperature,
            n=self.num_completions
        )
        responses = [completions['choices'][i]['message']['content'] for i in range(self.num_completions)]
        return responses[0]