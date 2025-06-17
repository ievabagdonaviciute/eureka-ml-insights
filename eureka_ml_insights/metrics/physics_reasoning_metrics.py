from tqdm.auto import tqdm
import logging
from .metrics_base import Metric

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


    def __evaluate__(self, model_answer: str, raw_answer: str, ground_truth: str, is_valid: bool) -> bool:
        """
        This method evaluates the model's answer against the ground truth.
        It should be implemented by subclasses.
        
        Args:
            model_answer (str): The answer provided by the model.
            raw_answer (str): The raw answer from the model.
            ground_truth (str): The correct answer.
            is_valid (bool): Whether the data is valid for evaluation.
        
        Returns:
            str: The evaluation result, e.g., "correct" or "incorrect".
        """
        if not is_valid:
            return "none"
        
        try:
            ans = model_answer.strip().lower()
            gold = ground_truth.strip().lower()

        except Exception as e:
            logging.warning(f"Failed to normalize answers: {e}")
            return "none"
        
        if ans == gold:
            return "correct"
        else:
            return "incorrect"
        
    def evaluate(self, data):
        self.validate_data(data)
        results = []
        for _, row in tqdm(data.iterrows(), total=len(data), desc="Evaluating Physics Reasoning"):
            result = self.__evaluate__(
                model_answer = row['model_output'],
                raw_answer = row['raw_output'],
                ground_truth = row['ground_truth'],
                is_valid = row['is_valid']
            )
            results.append(result)
        data['PhysicsReasoning_results'] = results
        return data