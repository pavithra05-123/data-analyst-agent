import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Data Analyst", page_icon="📊")
st.title("Autonomous Data Analyst Agent")
st.caption("Upload any CSV and ask questions in plain English")

uploaded = st.file_uploader("Upload your CSV file", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("Data preview")
    st.dataframe(df.head())

    st.subheader("Ask your data anything")
    question = st.text_input("e.g. What is the average sales by region?")

    if "history" not in st.session_state:
        st.session_state.history = []

    if st.button("Analyse") and question:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        agent = create_pandas_dataframe_agent(
            llm, df, verbose=True, allow_dangerous_code=True
        )
        with st.spinner("Agent is thinking..."):
            try:
                result = agent.invoke(question)
                answer = result["output"]
            except Exception as e:
                answer = f"Error: {e}"
        st.session_state.history.append({"q": question, "a": answer})

    if st.session_state.history:
        st.subheader("Results")
        for item in reversed(st.session_state.history):
            st.markdown(f"**You:** {item['q']}")
            st.success(item["a"])
            st.divider()