from gptbase import GPTBase

class GPTSummarize(GPTBase):
    def __init__(self, language):
        super(GPTSummarize, self).__init__()

        # Language setting
        self.language = language

    def build_test_prompt(self, test_example):
        # Truncate article whose summary is to be obtained
        
        # print(test_example["title"], "\n", test_example["content"])
        test_prompt = "{}\n{}".format(test_example["title"].strip(), test_example["content"].strip()).strip()
        
        # Keeping a buffer for the test example response part; 
        # This also limits maximum summary length to 1000 tokens
        trunc_len = self.remaining_prompt_length - 1000
        
        test_prompt = self.truncate_text(test_prompt, trunc_len)
        return test_prompt
    
    def build_complete_prompt(self, test_example):
        # Prompt length tracker
        self.remaining_prompt_length = self.max_tokens
        
        task_prompt = """The following text is in the {language} language. Write a summary in {language} in three sentences""".format(language = self.language)

        task_prompt_len = self.get_num_tokens(task_prompt)
        self.remaining_prompt_length -= task_prompt_len

        test_prompt = self.build_test_prompt(test_example)
        
        complete_prompt = task_prompt + "\n" + test_prompt
        return complete_prompt

    def request_api(self, test_example):
        '''Returns the complete prompt string to be passed to the model for a single test example'''

        user_prompt = self.build_complete_prompt(test_example)
        prompts = [
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes text written in the {} language".format(self.language)
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        print(prompts[1]["content"])
        responses = self.get_model_response(prompts)
        print("Summary: ", responses)
        return responses