import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from pandasai.config import Config


class QueryEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.llm = OpenAI()
        
        # Instantiate Config with only supported parameters
        self.config = Config(
            llm=self.llm,
            enable_cache=False,
            save_charts=False,
            use_error_correction_framework=False
        )

        # Pass the Config object directly
        self.sdf = SmartDataframe(df, config=self.config)

    def answer(self, query: str):
        """Returns the PandasAI response to a query."""
        try:
            result = self.sdf.chat(query)
            # Could be DataFrame, matplotlib Figure, or plain text.
            return result
        except Exception as e:
            return f"⚠️ Error: {e}"
