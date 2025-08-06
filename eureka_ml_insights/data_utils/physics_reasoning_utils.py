"""Physics Reasoning Benchmark Extraction Utilities."""


class StripAfterAssistantTransform:
    def __init__(self, column="raw_output"):
        self.column = column

    def transform(self, df):
        def strip_response(text):
            if isinstance(text, str) and "ASSISTANT:" in text:
                return text.split("ASSISTANT:", 1)[-1].strip()
            return text

        df[self.column] = df[self.column].apply(strip_response)
        return df


class SortByClassAndUIDTransform:
    def transform(self, df):
        return df.sort_values(by="uid").reset_index(drop=True)

