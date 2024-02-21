import pandas as pd
from gptbase import GPTBase


class GPTReflect(GPTBase):
    def __init__(self, language, use_chain_of_thought, use_content_init, use_content_summary, reflection_ice_file):
        super(GPTReflect, self).__init__()

        # Language setting
        self.language = language

        # Prompt setting flags
        self.use_chain_of_thought = use_chain_of_thought
        self.use_content_init = use_content_init
        self.use_content_summary = use_content_summary

        # In-context example setup
        if reflection_ice_file:
            self.reflection_ice_df = pd.read_csv(reflection_ice_file)

    def build_incontext_example(self, example):
        # Build prompt segment for a single in-context example

        # Build query part of the in-content example, i.e., title + (optionally) content/summary
        if self.use_content_init :
            # Using title + content
            trunc_len = 175
            content_prompt = self.truncate_text(example["content"], trunc_len)
        elif self.use_content_summary:
            # Using title + summary
            trunc_len = 175
            content_prompt = self.truncate_text(example["summary"], trunc_len)
        else:
            # Using only title
            content_prompt = ""
        query_prompt = "{}\n{}".format(example["title"].strip(), content_prompt.strip()).strip()

        # If chain of thought is used, the reasoning output by the model in the previous step is passed
        if self.use_chain_of_thought:
            text = example["gpt_responses"]
            if "yes" in text.lower()[:10]:
                previous_response = "Yes, the article is relevant to conservation"
            else:
                previous_response = "No, the article is not relevant to conservation"
        # Otherwise, convert the yes/no response into relevant/not relevant and pass
        else:
            text = example["gpt_responses"].split("Explanation", 1)[0]
            if "yes" in text.lower():
                previous_response = "Relevant"
            else:
                previous_response = "Not relevant"

        prev_resp_prompt = "\nAssessment: {}".format(previous_response)
        refl_cot_prompt = "\nReasoning: {}".format(example["refl_explanation"])
        refl_resp_prompt = "\nResponse: {}".format(example["refl_response"])

        content = query_prompt + prev_resp_prompt + refl_cot_prompt + refl_resp_prompt + "\n"    
        return content
    
    def build_incontext_examples_prompt(self):
        self.incontext_examples = self.get_incontext_examples(self.reflection_ice_df)
        example_prompts = []

        for example in self.incontext_examples:
            example_prompt = self.build_incontext_example(example)
            example_prompts.append(example_prompt)
        
        ice_prompt = "".join(example_prompts)
        return ice_prompt
    
    def build_test_prompt(self, test_example):
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
        # Commented setting includes content/summary. 
        test_example_prompt = "{}\n{}".format(test_example["title"].strip(), content_prompt.strip()).strip()
        trunc_len = self.remaining_prompt_length - 800
        # keeping a buffer for the previus and next response
        
        test_example_prompt = self.truncate_text(test_example_prompt, trunc_len)  

        # If chain of thought is used, the reasoning output by the model in the previous step is passed
        if self.use_chain_of_thought:
            text = test_example["response"]
            if "yes" in text.lower()[:10]:
                previous_response = "Yes, the article is relevant to conservation"
            else:
                previous_response = "No, the article is not relevant to conservation"
        # Otherwise, convert the yes/no response into relevant/not relevant and pass
        else:
            text = test_example["response"].split("Explanation", 1)[0]
            if "yes" in text.lower():
                previous_response = "Relevant"
            else:
                previous_response = "Not relevant"

        query_prompt = test_example_prompt + "\nAssessment: " + previous_response
        return query_prompt
    
    def build_complete_prompt(self, test_example):
        # Prompt length tracker
        self.remaining_prompt_length = self.max_tokens
        
        task_prompt = """You are given a {language} news text and an assessment made by another AI assistant on whether the text is relevant to conservation. Comment on whether the assessment is correct or not. Note that a headline should be marked as being relevant to conservation only if it is directly related to conservation. For example, news about a protected area that does not focus on conservation-related issues should not be marked as being relevant""".format(language = self.language)

        question_prompt = """Provide an analysis of the above assessment and comment on whether it is (A) correct or (B) incorrect"""
        test_prompt = self.build_test_prompt(test_example)    
        complete_prompt = task_prompt + "\n" + test_prompt + "\n" + question_prompt
        return complete_prompt

    def request_api(self, test_example):
        '''Returns the complete prompt string to be passed to the model for a single test example'''

        user_prompt = self.build_complete_prompt(test_example)
        prompts = [
            {
                "role": "system",
                "content": "You are a helpful assistant that interprets text written in the {} language and analyzes responses from another AI assistant".format(self.language)
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        print(prompts[1]["content"])
        # sys.exit()

        responses = self.get_model_response(prompts)
        print(responses)
        # sys.exit()
        return responses
    