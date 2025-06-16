"""Physics Reasoning Benchmark Extraction Utilities."""

import re

class ExtractYesNoTransform:
    """Pulls a Yes/No label out of raw_output using your helper."""
    def transform(self, df):
        return df.assign(
            model_output=df["raw_output"].apply(extract_yes_no_answer)
        )
    
def extract_yes_no_answer(model_output: str) -> str:
    output = model_output.strip().lower()
    match = re.search(r'\b(yes|no)\b', output, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return 'unknown'
