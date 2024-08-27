import streamlit as st
import instructor
from instructor import Mode
from pydantic import BaseModel, Field
from openai import OpenAI
import os
from dotenv import load_dotenv
from ratelimit import limits, sleep_and_retry, RateLimitException
from typing import List
import time
import re

# Load environment variables
load_dotenv()

# # Initialize the OpenAI client with Instructor
# client = instructor.patch(OpenAI(api_key=os.getenv("OPENAI_API_KEY")), mode=Mode.JSON)

# Initialize the Gemini client
import google.generativeai as genai
client = instructor.from_gemini(
    client=genai.GenerativeModel(
            model_name="models/gemini-1.5-flash-latest",  # model defaults to "gemini-pro"
        ),
    mode=Mode.GEMINI_JSON,
)


class Chunk(BaseModel):
    start: int = Field(..., description="The starting artifact index of the chunk")
    end: int = Field(..., description="The ending artifact index of the chunk")
    context: str = Field(..., description="The context or topic of this chunk. Make this as thorough as possible, including information from the rest of the text so that the chunk makes good sense.")

class TextChunks(BaseModel):
    chunks: List[Chunk] = Field(..., description="List of chunks in the text")

class EnhancedChunk(BaseModel):
    order: int
    start: int
    end: int
    text: str
    context: str

# Rate limiting decorator
@sleep_and_retry
@limits(calls=5, period=60)
def rate_limited_api_call(model, response_model, messages):
    return client.chat.completions.create(
        # model=model,
        response_model=response_model,
        messages=messages
    )



def split_sentences(document: str, flag: str = "all", limit: int = 2048):
    sent_list = []
    try:
        if flag == "zh":
            document = re.sub(
                "(?P<quotation_mark>([。？！…](?![”’\"'])))",
                r"\g<quotation_mark>\n",
                document,
            )
            document = re.sub(
                "(?P<quotation_mark>([。？！]|…{1,2})[”’\"'])",
                r"\g<quotation_mark>\n",
                document,
            )
        elif flag == "en":
            document = re.sub(
                "(?P<quotation_mark>([.?!](?![”’\"'])))",
                r"\g<quotation_mark>\n",
                document,
            )
            document = re.sub(
                "(?P<quotation_mark>([?!.][\"']))", r"\g<quotation_mark>\n", document
            )  # Special quotation marks
        else:
            document = re.sub(
                "(?P<quotation_mark>([。？！….?!](?![”’\"'])))",
                r"\g<quotation_mark>\n",
                document,
            )

            document = re.sub(
                "(?P<quotation_mark>(([。？！.!?]|…{1,2})[”’\"']))",
                r"\g<quotation_mark>\n",
                document,
            )  # Special quotation marks

        sent_list_ori = document.splitlines()
        for sent in sent_list_ori:
            sent = sent.strip()
            if not sent:
                continue
            elif len(sent) <= 2:
                continue
            else:
                while len(sent) > limit:
                    temp = sent[0:limit]
                    sent_list.append(temp)
                    sent = sent[limit:]
                sent_list.append(sent)
    except:  # noqa
        sent_list.clear()
        sent_list.append(document)
    return sent_list


def process_text(text):
    # Tokenize the text into sentences
    # sentences = nltk.sent_tokenize(text)
    sentences = split_sentences(text, flag="all")

    # Insert artifacts after each sentence
    text_with_artifacts = ""
    for i, sentence in enumerate(sentences):
        text_with_artifacts += f"{sentence} [{i}]\n"

    # Use OpenAI to determine chunk boundaries with Instructor validation
    try:
        chunks: TextChunks = rate_limited_api_call(
            model="models/gemini-1.5-flash",
            response_model=TextChunks,
            messages=[
                {"role": "system", "content": "You are an AI assistant tasked with chunking a text into cohesive sections. Your goal is to create chunks that maintain topic coherence and context."},
                {"role": "user", "content": f"Here's a text with numbered artifacts. Determine the best chunks by specifying start and end artifact numbers. Make the chunks as large as possible while maintaining coherence. Provide a thorough context for each chunk, including information from the rest of the text to ensure the chunk makes good sense. Ensure no overlap between chunks, and also not gaps. the end of one chunk should be the start of the next. Here's the text:\n\n{text_with_artifacts}"}
            ]
        )
    except RateLimitException:
        st.error("Rate limit exceeded. Please wait a moment before trying again.")
        return None

    # Create the final chunked entries with enhanced information
    chunked_entries = []
    for i, chunk in enumerate(chunks.chunks, start=1):
        # Join the sentences in the chunk into a single string
        chunk_text = "\n".join(sentences[chunk.start:chunk.end+1])
        enhanced_chunk = EnhancedChunk(
            order=i,
            start=chunk.start,
            end=chunk.end,
            text=chunk_text,
            context=chunk.context
        )
        chunked_entries.append(enhanced_chunk)

    return chunked_entries

def main():
    st.title("Text Chunker")
    st.write("This application chunks your text into coherent sections using AI.")

    # Text input
    user_text = st.text_area("Enter your text here:", height=200)

    if st.button("Process Text"):
        if user_text:
            try:
                with st.spinner("Processing text..."):
                    chunks = process_text(user_text)

                if chunks:
                    st.success("Text processed successfully!")

                    # Display chunks
                    for chunk in chunks:
                        with st.expander(f"Chunk {chunk.order} (Sentences {chunk.start}-{chunk.end})"):
                            st.write("**Context:**\n", chunk.context)
                            st.write("**Text:**\n", chunk.text)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter some text to process.")

if __name__ == "__main__":
    main()
