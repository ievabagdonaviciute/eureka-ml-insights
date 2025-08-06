from tqdm.auto import tqdm
import logging
from .metrics_base import Metric
import re

class PhysicsReasoningMetric(Metric): #subclass of metric
    """
    Compares the extracted yes/no answer (model_output) against the ground_truth label.
    """

    def validate_data(self, data):
        # ensure that required columns are present
        for col in ['model_output', 'ground_truth', 'is_valid']:
            if col not in data.columns:
                raise AssertionError(f"Data does not have '{col}' field.")
        return True

    def __evaluate__(self, model_answer: str, ground_truth: str, is_valid: bool) -> str:
        if not is_valid:
            return "none"
        
        try:
            ans = model_answer.strip().lower()
            if ans == "yes":
                ans = "true"
            elif ans == "no":
                ans = "false"

            gold = "true" if ground_truth is True else "false"
        except Exception as e:
            logging.warning(f"Failed to normalize answers: {e}")
            return "none"

        return "correct" if ans == gold else "incorrect"

############## WITHOUT LLM EXTRACTION

    # def extract(self, raw_answer: str) -> str:
    #     """
    #     Extracts the yes/no answer from the raw answer string.
        
    #     Args:
    #         raw_answer (str): The raw answer string from the model.
        
    #     Returns:
    #         str: The extracted yes/no answer.
    #     """

    #     if not isinstance(raw_answer, str):
    #         logging.warning("Encountered non-string raw_answer: %s", raw_answer)
    #         return "none"
    
    #     # Use regex to find 'yes'/'true' or 'no'/'false' in the raw answer
    #     if 'ASSISTANT:' in raw_answer:  # LLaVAVideo and LLaVAVideoNext uses this template
    #         raw_answer = raw_answer.split("ASSISTANT:", 1)[-1].strip()
    #         match = re.search(r'\b(true|false|yes|no)\b', raw_answer.strip(), re.IGNORECASE)
    #         if match:
    #             ans = match.group(0).lower()
    #             if ans == "yes": return "true"
    #             if ans == "no": return "false"
    #             return ans # already is a true/false answer
    #         else:
    #             return "none"

        # # for other models that follow the True/False format
        # match = re.search(r'\b(true|false)\b', raw_answer.strip(), re.IGNORECASE)
        # if match:
        #     return match.group(0).lower()
        # else:
        #     return "none"
        
    def evaluate(self, data):
        self.validate_data(data)
        results = []
        for _, row in tqdm(data.iterrows(), total=len(data), desc="Evaluating Physics Reasoning"):
            model_output = row['model_output'].strip().lower()
            
            if model_output == "yes":
                model_output = "true"
            elif model_output == "no":
                model_output = "false"

            result = self.__evaluate__(
                model_answer=model_output,
                ground_truth=row['ground_truth'],
                is_valid=row['is_valid']
            )
            results.append(result)
        data['PhysicsReasoning_results'] = results
        return data

## WITHOUT LLM EXTRACTION:

# import logging
# from .metrics_base import Metric
# import re

# class PhysicsReasoningMetric(Metric): #subclass of metric
#     """
#     Compares the extracted yes/no answer (model_output) against the ground_truth label.
#     """

#     def validate_data(self, data):
#         # ensure that required columns are present
#         for col in ['model_output', 'ground_truth', 'is_valid']:
#             if col not in data.columns:
#                 raise AssertionError(f"Data does not have '{col}' field.")
#         return True


#     def __evaluate__(self, model_answer: str, raw_answer: str, ground_truth: str, is_valid: bool) -> bool:
#         """
#         This method evaluates the model's answer against the ground truth.
#         It should be implemented by subclasses.
        
#         Args:
#             model_answer (str): The answer provided by the model.
#             raw_answer (str): The raw answer from the model.
#             ground_truth (str): The correct answer.
#             is_valid (bool): Whether the data is valid for evaluation.
        
#         Returns:
#             str: The evaluation result, e.g., "correct" or "incorrect".
#         """
#         if not is_valid:
#             return "none"
        
#         try:
#             ans = model_answer.strip().lower()
#             gold = "true" if ground_truth is True else "false"

#         except Exception as e:
#             logging.warning(f"Failed to normalize answers: {e}")
#             return "none"
        
#         if ans == gold:
#             return "correct"
#         else:
#             return "incorrect"
    
#     def extract(self, raw_answer: str) -> str:
#         """
#         Extracts the yes/no answer from the raw answer string.
        
#         Args:
#             raw_answer (str): The raw answer string from the model.
        
#         Returns:
#             str: The extracted yes/no answer.
#         """

#         if not isinstance(raw_answer, str):
#             logging.warning("Encountered non-string raw_answer: %s", raw_answer)
#             return "none"
    
#         # Use regex to find 'yes'/'true' or 'no'/'false' in the raw answer
#         if 'ASSISTANT:' in raw_answer:  # LLaVAVideo and LLaVAVideoNext uses this template
#             raw_answer = raw_answer.split("ASSISTANT:", 1)[-1].strip()
#             match = re.search(r'\b(true|false|yes|no)\b', raw_answer.strip(), re.IGNORECASE)
#             if match:
#                 ans = match.group(0).lower()
#                 if ans == "yes": return "true"
#                 if ans == "no": return "false"
#                 return ans # already is a true/false answer
#             else:
#                 return "none"

#         # for other models that follow the True/False format
#         match = re.search(r'\b(true|false)\b', raw_answer.strip(), re.IGNORECASE)
#         if match:
#             return match.group(0).lower()
#         else:
#             return "none"
        
#     def evaluate(self, data):
#         self.validate_data(data)
#         results = []
#         for _, row in tqdm(data.iterrows(), total=len(data), desc="Evaluating Physics Reasoning"):
            
#             raw_output = row['raw_output']
#             extracted_answer = self.extract(raw_output)
#             data.at[row.name, 'model_output'] = extracted_answer

#             result = self.__evaluate__(
#                 model_answer = extracted_answer,
#                 raw_answer = raw_output,
#                 ground_truth = row['ground_truth'],
#                 is_valid = row['is_valid']
#             )
#             results.append(result)
#         data['PhysicsReasoning_results'] = results
#         return data