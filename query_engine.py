# import pandas as pd
# from pandasai import SmartDataframe
# from pandasai.llm import OpenAI
# from pandasai.config import Config


# class QueryEngine:
#     def __init__(self, df: pd.DataFrame):
#         self.df = df
#         self.llm = OpenAI()

#         # ✅ Use only the parameters that still exist in the latest pandasai Config
#         config = Config(
#             llm=self.llm,
#             enable_cache=False,
#             save_charts=False,
#             use_error_correction_framework=False
#         )

#         # Pass the Config object directly
#         self.sdf = SmartDataframe(df, config=config)
#         self.error_counter = 0

#     def answer(self, query: str):
#         """Return either a quick-path pandas answer or the PandasAI response."""
#         q_lower = query.lower()

#         # Quick pandas shortcut for simple counts
#         if "count" in q_lower or "average" in q_lower or "mean" in q_lower:
#             try:
#                 if "count" in q_lower:
#                     return f"Total rows: {len(self.df)}"
#             except Exception as e:
#                 self.error_counter += 1
#                 return f"Error in pandas quick path: {e}"

#         # Fallback to PandasAI
#         try:
#             result = self.sdf.chat(query)
#             # Could be DataFrame, matplotlib Figure, or plain text
#             return result
#         except Exception as e:
#             self.error_counter += 1
#             return f"⚠️ Error: {e}"


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
        self.error_counter = 0

    def answer(self, query: str):
        """Returns the PandasAI response to a query."""
        try:
            result = self.sdf.chat(query)
            # Could be DataFrame, matplotlib Figure, or plain text.
            return result
        except Exception as e:
            self.error_counter += 1
            return f"⚠️ Error: {e}"
