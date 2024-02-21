import argparse
import glob
import json
import os
import signal
import sys
import time

import numpy as np
import pandas as pd
from gptclassifier import GPTClassifier
from gptreflect import GPTReflect
from gptsummarize import GPTSummarize
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class TimeoutException(Exception):
    pass

class RateLimitError(Exception):
    pass

def timeout_handler(signum, frame): 
    raise TimeoutException

signal.signal(signal.SIGALRM, timeout_handler)

class Driver:
    def __init__(self, test_data_file, train_data_file, results_dir, language, use_content_init, use_content_summary, zero_shot, use_chain_of_thought, reflect, num_examples):   
        self.test_df = pd.read_csv(test_data_file).head(8)

        self.train_data_file = train_data_file
        self.use_chain_of_thought = use_chain_of_thought
        self.use_content_init = use_content_init
        self.use_content_summary = use_content_summary
        self.reflect = reflect
        self.language = language

        self.zero_shot = zero_shot
        if zero_shot:
            self.num_examples = 0
        else:
            self.num_examples = num_examples

        self.run_index = 0
        
        self.results_dir = os.path.join(results_dir, self.generate_result_filename())


    def run_test_multiple_iterations(self, num_iter = 1):
        for iter in range(num_iter):
            self.run_test(iter + 1)

        #self.evaluate_and_save_aggregate(num_iter)

    def evaluate_and_save_aggregate(self, num_iter):
        average_metrics_filename = self.results_filename[:-2] + "_average_metrics.json"
        average_metrics_filepath = os.path.join(self.results_dir, average_metrics_filename)
        if os.path.isfile(average_metrics_filepath):
            os.remove(average_metrics_filepath)

        ensemble_metrics_filename = self.results_filename[:-2] + "_ensemble_metrics.json"
        ensemble_metrics_filepath = os.path.join(self.results_dir, ensemble_metrics_filename)
        if os.path.isfile(ensemble_metrics_filepath):
            os.remove(ensemble_metrics_filepath)

        aggregate_filename = self.results_filename[:-2] + "_aggregate.xlsx"
        aggregate_filepath = os.path.join(self.results_dir, aggregate_filename)
        if os.path.isfile(aggregate_filepath):
            os.remove(aggregate_filepath)

        # Calculating average metrics
        average_f1_metrics_original = np.zeros(3)
        average_f1_metrics_reflection = np.zeros(3)
        average_accuracy_original = 0
        average_accuracy_reflection = 0
        for i, file in enumerate(sorted(glob.glob(os.path.join(self.results_dir, "*.json")))):
            with open(file, "r") as f_metrics:
                metrics = json.load(f_metrics)
                average_f1_metrics_original += np.array(metrics["f1_metrics_original"])[:-1].astype(float)
                average_f1_metrics_reflection += np.array(metrics["f1_metrics_reflection"])[:-1].astype(float)
                average_accuracy_original += metrics["accuracy_original"]
                average_accuracy_reflection += metrics["accuracy_reflection"]
        average_f1_metrics_original /= num_iter
        average_f1_metrics_reflection /= num_iter
        average_accuracy_original /= num_iter
        average_accuracy_reflection /= num_iter


        average_metrics_dict = {
            "average_original_f1_metrics": list(average_f1_metrics_original),
            "average_original_accuracy": average_accuracy_original,
            "average_reflection_f1_metrics": list(average_f1_metrics_reflection),
            "average_reflection_accuracy": average_accuracy_reflection,
            }

        with open(average_metrics_filepath,"w") as f:
            json.dump(average_metrics_dict, f, indent=1)          

        # Ensembling the predictions by majority vote
        df_aggregate = self.test_df[["conservation_label", "url", "title", "content"]]
        aggregate_scores_original = np.zeros(len(self.test_df))
        aggregate_scores_reflection = np.zeros(len(self.test_df))
        for i, file in enumerate(sorted(glob.glob(os.path.join(self.results_dir, "*.xlsx")))):
            df = pd.read_excel(file)

            # store the individual original/reflection predictions
            df_aggregate["prediction_original_{}".format(i + 1)] = df["conservation_prediction_gpt"]
            df_aggregate["prediction_reflection_{}".format(i + 1)] = df["final"]

            # add the individual original/reflection predictions
            aggregate_scores_original += df["conservation_prediction_gpt"] * 2 - 1
            aggregate_scores_reflection += df["final"] * 2 - 1

        df_aggregate["ensemble_prediction_original"] = np.heaviside(aggregate_scores_original, 0)
        df_aggregate["ensemble_prediction_reflection"] = np.heaviside(aggregate_scores_reflection, 0)
    
        df_aggregate.to_excel(aggregate_filepath)

        # Calculating performance of the ensemble

        # For ensemble of original predictions
        ensemble_f1_metrics_original = precision_recall_fscore_support(df_aggregate.conservation_label, df_aggregate.ensemble_prediction_original, average="binary")
        ensemble_accuracy_original = accuracy_score(df_aggregate.conservation_label, df_aggregate.ensemble_prediction_original)

        # For ensemble of reflection predictions
        ensemble_f1_metrics_reflection = precision_recall_fscore_support(df_aggregate.conservation_label, df_aggregate.ensemble_prediction_reflection, average="binary")
        ensemble_accuracy_reflection = accuracy_score(df_aggregate.conservation_label, df_aggregate.ensemble_prediction_reflection)

        ensemble_metrics_dict = {
            "ensemble_original_f1_metrics": list(ensemble_f1_metrics_original),
            "ensemble_original_accuracy": ensemble_accuracy_original,
            "ensemble_reflection_f1_metrics": list(ensemble_f1_metrics_reflection),
            "ensemble_reflection_accuracy": ensemble_accuracy_reflection,
        }

        print(ensemble_metrics_dict)
        with open(ensemble_metrics_filepath,"w") as f:
            json.dump(ensemble_metrics_dict, f, indent=1)               

    def run_test(self, run_index):
        """The main driver function. It runs the entire pipeline for the test examples given in the test data file passed to the Driver constructor.
        The run_index argument is used if results are to be aggregated across multiple runs. It is used to generate unique filenames for the results.
        """
        self.run_index = run_index # used to aggregate results across runs (if multiple runs are performed)

        classifier_responses = []
        reflection_responses = []
        summary_responses = []

        for i, test_example in self.test_df.iterrows():
            num_attempts = 0
            while True:
                num_attempts += 1
                if num_attempts >= 10:
                    print("Too many attempts. Exiting.")
                    sys.exit()
                signal.alarm(45)    

                try:
                    if self.use_content_summary:
                        self.summarizer = GPTSummarize(self.language)
                        if "summary" not in test_example:
                            summary = self.summarizer.request_api(test_example)
                            test_example["summary"] = summary
                    self.classifier = GPTClassifier(
                        train_data_file=self.train_data_file, 
                        use_content_init=self.use_content_init,
                        use_content_summary=self.use_content_summary,
                        zero_shot=self.zero_shot, 
                        use_chain_of_thought=self.use_chain_of_thought,
                        language=self.language,
                        num_examples=self.num_examples,
                        shuffle_seed=run_index
                    )
                    response = self.classifier.request_api(test_example)
                    test_example["response"] = response

                    if self.reflect:
                        self.reflector = GPTReflect(
                            self.language, 
                            self.use_chain_of_thought, 
                            self.use_content_init, 
                            self.use_content_summary, 
                            reflection_ice_file=None
                        )
                        reflection_response = self.reflector.request_api(test_example)
                   
                except TimeoutException as te:
                    print(TimeoutException, te)
                    print("Caught timeout on iteration {}. Retrying \n".format(i))
                    continue
                except Exception as e:
                    print("General exception caught. Waiting")
                    print(Exception, e)
                    signal.alarm(0)
                    time.sleep(10)
                    continue
                else:
                    signal.alarm(0)
                
                classifier_responses.append(response)
                if self.use_content_summary and "summary" not in self.test_df:
                    summary_responses.append(summary)
                if self.reflect:
                    reflection_responses.append(reflection_response)
                break

        self.evaluate_and_save(classifier_responses, reflection_responses, summary_responses)
    
    def evaluate_and_save(self, classifier_responses, reflection_responses, summary_responses):
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        # Generate file names and paths for storing results, metrics, and the shortlist
        # The shortlist contains only the examples for which the classifier predicts a positive label. Typically, this is the main file used by environmental non-profits
        self.results_filename = self.generate_result_filename()

        self.metrics_filename = self.results_filename + "_metrics.json"
        self.metrics_filepath = os.path.join(self.results_dir, self.metrics_filename)
        
        self.responses_filename = self.results_filename + ".xlsx"
        self.responses_filepath = os.path.join(self.results_dir, self.responses_filename)

        shortlist_filename = "{language}_testdata_with_responses_shortlist.xlsx".format(language = self.language)
        shortlist_filepath = os.path.join(self.results_dir, shortlist_filename)

        # Extract labels from the responses and save the results to an excel file.
        # The complete GPT responses and the summary + reflection responses (if used) are also saved.
        labels = [self.extract_label_from_responses(responses) for responses in classifier_responses]
        self.test_df["gpt_responses"] = classifier_responses
        self.test_df["conservation_prediction_gpt"] = labels

        if self.use_content_summary and "summary" not in self.test_df:
            self.test_df["summary"] = summary_responses

        if self.reflect:
            self.test_df["reflection"] = reflection_responses
            reflect_labels = [self.extract_label_from_responses(responses) for responses in reflection_responses]
            self.test_df["reflection_labels"] = reflect_labels
            final_labels = np.logical_and(np.array(labels), np.array(reflect_labels)).astype(int)
            self.test_df['final'] = final_labels

        self.test_df.to_excel(self.responses_filepath)

        # Extract the positively labeled examples and save them to the shortlist file
        if self.reflect:
            shortlist_df = self.test_df[self.test_df["final"] == 1] 
        else:
            shortlist_df = self.test_df[self.test_df["conservation_prediction_gpt"] == 1]
        shortlist_df.to_excel(shortlist_filepath)

        # Calculate and save the evaluation metrics
        metrics_dict = {}

        f1_metrics = precision_recall_fscore_support(self.test_df.conservation_label, labels, average="binary")
        accuracy = accuracy_score(self.test_df.conservation_label, labels)

        metrics_dict["f1_metrics_original"] = f1_metrics
        metrics_dict["accuracy_original"] = accuracy

        if self.reflect:
            f1_metrics = precision_recall_fscore_support(self.test_df.conservation_label, final_labels, average="binary")
            accuracy = accuracy_score(self.test_df.conservation_label, final_labels)

            metrics_dict["f1_metrics_reflection"] = f1_metrics
            metrics_dict["accuracy_reflection"] = accuracy
        
        print(metrics_dict)
        with open(self.metrics_filepath,"w") as f:
            json.dump(metrics_dict, f, indent=1)

    def generate_result_filename(self):
        cot_string = "cot" if self.use_chain_of_thought else "ncot"

        results_filename = "{language}_{shot}_{cot}".format(
            language = self.language, 
            shot = self.num_examples, 
            cot = cot_string
            )

        if self.use_content_summary:
            results_filename += "_summary"
        elif self.use_content_init:
            results_filename += "_init"
        else:
            results_filename += "_title"

        if self.reflect:
            results_filename += "_reflect"

        if self.run_index != 0:
            results_filename += "_{}".format(self.run_index)
        return results_filename
    
    def extract_label_from_response_single(self, response):
        if "yes" in response[:9].lower() or "A)" in response:
            label = 1
        else:
            if "not correct" in response.lower() or "incorrect" in response.lower():
                label = 0
            elif "correct" in response.lower():
                label = 1
            else:
                label = 0
        return label
    
    def extract_label_from_responses(self, responses):
        return self.extract_label_from_response_single(responses)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_file', required=True)
    parser.add_argument('--train_data_file', required=True)
    parser.add_argument('--reflection_ice_file', required=False)
    parser.add_argument('--results_dir', required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--use_content_init", default=False, action='store_true')
    parser.add_argument("--use_content_summary", default=False, action='store_true')
    parser.add_argument("--num_completions", required=False, type=int, default=1)
    parser.add_argument("--temperature", required=False, type=float, default=0.0)
    parser.add_argument('--zero_shot', default=False, action='store_true')
    parser.add_argument("--use_chain_of_thought", default=False, action='store_true')
    parser.add_argument("--reflect", default=False, action='store_true')
    parser.add_argument("--num_examples", required=False, default=10, type=int)

    args = parser.parse_args()
    driver = Driver(
        test_data_file=args.test_data_file,
        train_data_file=args.train_data_file,
        results_dir=args.results_dir,
        language=args.language,
        use_content_init=args.use_content_init,
        use_content_summary=args.use_content_summary,
        zero_shot=args.zero_shot,
        use_chain_of_thought=args.use_chain_of_thought,
        reflect=args.reflect,
        num_examples=args.num_examples
    )
    driver.run_test_multiple_iterations()