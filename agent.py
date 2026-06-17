from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv

load_dotenv()

def create_agent(df):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True
    )
    return agent