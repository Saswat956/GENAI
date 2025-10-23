from langchain_openai import ChatOpenAI
from langchain.tools.python.tool import PythonREPLTool
import pandas as pd

# ------------------------------------
# 1. Prepare sample DataFrame
# ------------------------------------
df = pd.DataFrame({
    "City": ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Chennai"],
    "Temperature": [30, 34, 29, 31, 33],
    "Humidity": [45, 60, 55, 65, 70]
})

# ------------------------------------
# 2. Initialize LLM and Python REPL
# ------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
python_repl = PythonREPLTool()

# Make sure the DataFrame is available inside REPL
python_repl.globals["df"] = df

# ------------------------------------
# 3. Ask the LLM to GENERATE code
# ------------------------------------
prompt = """
Generate Python code using Plotly Express to visualize the DataFrame `df`.
Create a bar chart of Temperature vs City with color based on Humidity.
Do not include explanations — only valid Python code.
"""

generated_code = llm.invoke(prompt).content
print("🧠 Generated Code:\n")
print(generated_code)

# ------------------------------------
# 4. Execute the GENERATED code using PythonREPLTool
# ------------------------------------
print("\n⚙️ Executing Generated Code:\n")
result = python_repl.run(generated_code)
print(result)