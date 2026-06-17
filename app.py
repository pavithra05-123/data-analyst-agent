import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv
import PyPDF2
from PIL import Image
import io

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

st.set_page_config(page_title="AI Study Assistant", page_icon="🤖")

page = st.sidebar.selectbox("Choose a tool", [
    "📊 Data Analyst Agent",
    "📝 Question Bank Generator",
    "📁 File Analyser"
])

# ─── PAGE 1: Data Analyst ───
if page == "📊 Data Analyst Agent":
    st.title("Autonomous Data Analyst Agent")
    st.caption("Upload CSV or Excel and ask questions in plain English")

    uploaded = st.file_uploader("Upload file", 
        type=["csv", "xlsx", "xls"])

    if uploaded:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

        st.subheader("Data preview")
        st.dataframe(df.head())

        st.subheader("Ask your data anything")
        question = st.text_input("e.g. What is the average sales by region?")

        if "history" not in st.session_state:
            st.session_state.history = []

        if st.button("Analyse") and question:
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

# ─── PAGE 2: Question Bank ───
elif page == "📝 Question Bank Generator":
    st.title("Question Bank Generator")
    st.caption("Generate practice questions on any topic instantly")

    topic = st.text_input("Enter a topic",
        placeholder="e.g. Python loops, Machine Learning")

    difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
    num_questions = st.slider("Number of questions", 3, 15, 5)
    qtype = st.selectbox("Question type",
        ["Multiple Choice (MCQ)", "Short Answer", "True/False", "Mix of all"])

    if st.button("Generate Questions") and topic:
        with st.spinner("Generating questions..."):
            prompt = f"""Generate {num_questions} {difficulty} level {qtype} 
questions on: {topic}.
Number each question clearly.
For MCQ: give 4 options and mark correct answer.
For Short Answer: give model answer.
For True/False: give explanation."""
            response = llm.invoke(prompt)
            st.subheader(f"Questions on: {topic}")
            st.markdown(response.content)
            st.download_button(
                label="Download Questions",
                data=response.content,
                file_name=f"{topic}_questions.txt",
                mime="text/plain"
            )

# ─── PAGE 3: File Analyser ───
elif page == "📁 File Analyser":
    st.title("AI File Analyser")
    st.caption("Upload any file and ask AI questions about it")

    uploaded = st.file_uploader("Upload any file",
        type=["pdf", "txt", "png", "jpg", "jpeg"])

    if uploaded:
        content = ""

        # PDF
        if uploaded.name.endswith(".pdf"):
            reader = PyPDF2.PdfReader(uploaded)
            for page_num in range(len(reader.pages)):
                content += reader.pages[page_num].extract_text()
            st.success(f"PDF loaded! {len(reader.pages)} pages found.")

        # Text file
        elif uploaded.name.endswith(".txt"):
            content = uploaded.read().decode("utf-8")
            st.success("Text file loaded!")

        # Image
        elif uploaded.name.endswith(("png", "jpg", "jpeg")):
            image = Image.open(uploaded)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            content = "This is an image file. Describe what you see."

        if content:
            st.subheader("Ask anything about this file")
            question = st.text_input("e.g. Summarise this / What are the main topics?")

            if st.button("Ask AI") and question:
                with st.spinner("AI is reading your file..."):
                    prompt = f"""Here is the content of the file:

{content[:4000]}

Question: {question}

Give a clear and helpful answer."""
                    response = llm.invoke(prompt)
                    st.subheader("Answer")
                    st.success(response.content)
                    st.download_button(
                        label="Download Answer",
                        data=response.content,
                        file_name="answer.txt",
                        mime="text/plain"
                    )