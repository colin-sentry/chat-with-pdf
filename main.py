import itertools
import os
from pathlib import Path

import bs4
from flask import Flask, request, Response
from langchain import hub
from langchain_community.document_loaders import GitHubIssuesLoader, GithubFileLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ["OPENAI_API_KEY"] = open(Path.home() / "open_ai_key").read().strip()
os.environ["GITHUB_PERSONAL_ACCESS_TOKEN"] = open(Path.home() / "ai_pat").read().strip()

if not os.path.exists(Path.home() / Path(".vectorstore")):
    code_loader = GithubFileLoader(
        repo="getsentry/sentry-docs",
        github_api_url="https://api.github.com",
        file_filter=lambda file_path: file_path.endswith(
            ".mdx"
        ),
        branch="master",
        token=os.environ["GITHUB_PERSONAL_ACCESS_TOKEN"]
    )

    print("Loading docs")
    docs = code_loader.load()
    # print("Done loading docs. Loading issues.")
    # issues = issues_loader.load()

    print("Done loading. Splitting docs and issues")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # splits = text_splitter.split_documents(itertools.chain(docs, issues))
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(), persist_directory=str(Path.home() / Path(".vectorstore")))
else:
    vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory=str(Path.home() / Path(".vectorstore")))
# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert software engineer that works at a company called Sentry.
Sentry is a company and a hosted software that provides application performance management solutions to companies that employ programmers.

The core sentry product is primarily used to diagnose errors and performance issues in production environments.

You are connected to the documentation of sentry, and you are to respond as "Sentry Docs" with a reasoned explanation to the user's query.

For example, if the user asked "What is a span", you would respond with "A span is the measurement of a single operation. Multiple spans form a trace, which help users identify where the time for a particular operation is spent."

Context: {context}
"""),

    ("user", "Question: {question}\nAnswer:"),
])
# prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

app = Flask(__name__)


@app.route("/api/v1/questions", methods=["GET"])
def get_question():
    question = request.args.get("q")

    answer = rag_chain.stream(question)
    return Response(answer, mimetype="text/plain")

print("Ready!")
app.run(port=8080, debug=True)