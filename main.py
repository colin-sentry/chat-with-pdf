import os
from pathlib import Path

from cohere import Client
from langchain_cohere import ChatCohere
from langchain_community.chat_models import ChatAnthropic
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint

from flask import Flask, request, Response
from langchain_community.document_loaders import GithubFileLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import sentry_sdk

sentry_sdk.init(
    dsn="http://d3d3ffcc17c9729ae3fffaa90d97ce02@localhost:3001/9",
    # dsn="https://8f3b03f3d70993ccc67a99fdd2276271@o1.ingest.us.sentry.io/4506893166379008",
    enable_tracing=True,
    traces_sample_rate=1.0,
    send_default_pii=True,
    debug=True
)

os.environ["OPENAI_API_KEY"] = open(Path.home() / "open_ai_key").read().strip()
os.environ["ANTHROPIC_API_KEY"] = open(Path.home() / "anthropic-key").read().strip()
os.environ["GITHUB_PERSONAL_ACCESS_TOKEN"] = open(Path.home() / "ai_pat").read().strip()
os.environ["COHERE_API_KEY"] = open(Path.home() / "cohere-key").read().strip()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = open(Path.home() / "huggingface-key").read().strip()


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
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(),
                                        persist_directory=str(Path.home() / Path(".vectorstore")))
else:
    vectorstore = Chroma(embedding_function=OpenAIEmbeddings(),
                         persist_directory=str(Path.home() / Path(".vectorstore")))
# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert software engineer that works at a company called Sentry.
Sentry is a company and a hosted software that provides application performance management solutions to companies that employ programmers.

The core sentry product is primarily used to diagnose errors and performance issues in production environments.

You will be given context from the documentation of Sentry, and are expected to give a reasoned explanation to the user's query.

For example, if the user asked "What is a span", you would respond with "A span is the measurement of a single operation. Multiple spans form a trace, which help users identify where the time for a particular operation is spent."

Context: {context}
"""),

    ("human", "Question: {question}\nAnswer:"),
])
# prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


app = Flask(__name__)


@app.route("/api/v1/questions", methods=["GET"])
def get_question():
    question = request.args.get("q")
    if "givebad" in question:
        raise Exception("Bad question")

    stream = True
    model = request.args.get("model")
    if model == 'gpt3':
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    elif model == 'gpt4':
        llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    elif model == 'claude':
        llm = ChatAnthropic(model='claude-2.1')
    elif model == 'cohere':
        llm = ChatCohere(model="command", max_tokens=256, temperature=0)
        stream = False
        print(Client().models.list())
    elif model == 'huggingface':
        llm = HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            task="generated_text",
            max_new_tokens=128,
            top_k=10,
            top_p=0.95,
            typical_p=0.95,
            temperature=0.01,
            repetition_penalty=1.03,
            callbacks=[],
            streaming=True,
        )
    else:
        raise Exception("Model not specified or invalid. Try model=gpt3")

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    rag_chain.name = model + " Ask Sentry Pipeline"
    # sentry_sdk.metrics.incr("ai_test_measurement", 10)

    def get_answer():
        with sentry_sdk.start_transaction(op="ai-inference", name="The result of the AI inference"):
            if not stream:
                yield rag_chain.invoke(question)
                return
            for x in rag_chain.stream(question):
                yield x

    return Response(get_answer(), mimetype="text/plain")


print("Ready!")
app.run(port=8080, debug=True)
