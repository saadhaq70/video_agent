from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


def get_llm():
    return init_chat_model(
        "mistral-medium-latest",
        model_provider="mistralai",
        temperature=0.2,
    )


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
    )
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Summarize this portion of a meeting transcript concisely."),
            ("human", "{text}"),
        ]
    )

    map_chain = (
        map_prompt
        | llm
        | StrOutputParser()     # Fixed
    )

    chunks = split_transcript(transcript)

    chunk_summaries = [
        map_chain.invoke({"text": chunk})
        for chunk in chunks
    ]

    combined = "\n\n".join(chunk_summaries)

    reduce_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert meeting summarizer. Combine these partial summaries "
                "into one professional meeting summary in bullet points.",
            ),
            ("human", "{text}"),
        ]
    )

    reduce_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: x)      # Identity function
        | reduce_prompt
        | llm
        | StrOutputParser()
    )

    return reduce_chain.invoke({"text": combined})


def generate_title(transcript: str) -> str:
    llm = get_llm()

    title_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Based on the meeting transcript, generate a short professional meeting title "
                "(max 8 words). Return only the title.",
            ),
            ("human", "{text}"),
        ]
    )

    title_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: x)      # Identity function
        | title_prompt
        | llm
        | StrOutputParser()
    )

    return title_chain.invoke(
        {
            "text": transcript[:2000]
        }
    )
