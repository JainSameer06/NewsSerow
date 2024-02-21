import pandas as pd
from gptbase import GPTBase


class GPTClassifier(GPTBase):
    def __init__(self, train_data_file, use_content_init, use_content_summary, zero_shot, use_chain_of_thought, language, num_examples, shuffle_seed):

        super(GPTClassifier, self).__init__()
        self.train_data_file = train_data_file
        self.num_incontext_exampes = num_examples
        self.shuffle_seed = shuffle_seed

        # Prompt setting flags
        self.zero_shot = zero_shot
        self.use_chain_of_thought = use_chain_of_thought
        self.use_content_init = use_content_init
        self.use_content_summary = use_content_summary

        # Language settings
        self.language = language
    
    def build_incontext_example(self, example):
        # Build prompt segment for a single in-context example

        # Build query part of the in-content example, i.e., title + (optionally) content
        if self.use_content_init :
            # Using title + content
            trunc_len = 2000 // self.num_incontext_exampes - 25
            content_prompt = self.truncate_text(example["content"], trunc_len)
        elif self.use_content_summary:
            # Using title + summary
            trunc_len = 2000 // self.num_incontext_exampes - 25
            content_prompt = self.truncate_text(example["summary"], trunc_len)
        else:
            # Using only title
            content_prompt = ""
        query_prompt = "{}\n{}".format(example["title"].strip(), content_prompt.strip()).strip()

        # Build response part of the in-context example, i.e., response and (optionally) explanation
        label_map = {1: "A) Yes", 0: "B) No"}
        response_prompt = "Response: {}".format(label_map[example["conservation_label"]])
        if self.use_chain_of_thought:
            cot_prompt = "Explanation: {}".format(example["expl_revised"])
            response_prompt = response_prompt + "\n" + cot_prompt
        content = query_prompt + "\n" + response_prompt + "\n\n"

        return content
    
    def build_incontext_examples_prompt(self):
        self.incontext_examples = self.get_incontext_examples(self.train_data_file, self.num_incontext_exampes, self.shuffle_seed)
        example_prompts = []

        for example in self.incontext_examples:
            example_prompt = self.build_incontext_example(example)
            example_prompts.append(example_prompt)
        
        ice_prompt = "".join(example_prompts)
        return ice_prompt

    def build_test_prompt(self, test_example):
        # Build the prompt segment for the test example
        
        # Build query part of the in-content/test example
        if self.use_content_init:
            # Using title + content initial sentences
            content_prompt = test_example["content"]
        elif self.use_content_summary:
            # Using title + content summary
            content_prompt = test_example["summary"]
        else:
            # Using only title
            content_prompt = ""
        query_prompt = "{}\n{}".format(test_example["title"].strip(), content_prompt.strip()).strip()
        
        trunc_len = self.remaining_prompt_length - 400
        # keeping a buffer for the test example response part
        
        query_prompt = self.truncate_text(query_prompt, trunc_len)

        # Build response part of the in-context/test example
        if self.zero_shot:
            response_prompt = """\nIs the above {language} news headline (A) related or (B) not related to environmental conservation? The headline should be marked as being relevant to conservation only if it is directly related to conservation. For example, news about a protected area that does not focus on conservation-related issues should not be marked as being relevant\n""".format(language = self.language)
            content = query_prompt + "\n" + response_prompt
        else:
            content = query_prompt + "\n" + "Response:"
        return content
    
    def build_complete_prompt(self, test_example):
        # Prompt length tracker
        self.remaining_prompt_length = self.max_tokens

        if self.zero_shot:
            ice_prompt = ""
            task_prompt = """Environmental conservation is the care and protection of earth's natural resources so that they can persist for future generations. It includes maintaining diversity of species, genes, and ecosystems, as well as functions of the environment, such as nutrient cycling. Now consider the following {language} news headline.\n""".format(language = self.language)
        else:
            ice_prompt = self.build_incontext_examples_prompt()
            task_prompt = """Environmental conservation is the care and protection of earth's natural resources so that they can persist for future generations. It includes maintaining diversity of species, genes, and ecosystems, as well as functions of the environment, such as nutrient cycling. \nEach of the following examples contains a {language} news headline and label or explanation specifying whether it (A) is or (B) is not related to environmental conservation. Observe the examples and classify the last example. Note that the headline should be marked as being relevant to conservation only if it is directly related to conservation. For example, news about a protected area that does not focus on conservation-related issues should not be marked as being relevant\n""".format(language = self.language)

        task_prompt_len = self.get_num_tokens(task_prompt)
        ice_prompt_len = self.get_num_tokens(ice_prompt)
        print("tpl", task_prompt_len, "ipl", ice_prompt_len)
        self.remaining_prompt_length -= task_prompt_len
        self.remaining_prompt_length -= ice_prompt_len

        test_example_prompt = self.build_test_prompt(test_example)
        
        complete_prompt = task_prompt + ice_prompt + test_example_prompt
        print(self.get_num_tokens(complete_prompt))
        return complete_prompt
    
    def request_api(self, test_example):
        '''Returns the complete prompt string to be passed to the model for a single test example'''

        user_prompt = self.build_complete_prompt(test_example)
        prompts = [
            {
                "role": "system",
                "content": "You are a helpful assistant that interprets and comments on text written in the {} language".format(self.language)
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        print(prompts[1]["content"])

        responses = self.get_model_response(prompts)
        print(responses)
        return responses
    
