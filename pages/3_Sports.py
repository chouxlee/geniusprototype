## deps
import os
os.environ['OPENAI_API_KEY'] = "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"
os.environ['openai_api_key'] = "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"

import streamlit as st

# Dictionary containing pairing options and their corresponding URLs
pairing_options = {
    "Tennis": "https://www.bbc.com/sport/tennis/68165651",
    "English Premier League": "https://www.bbc.com/sport/football/68253004",
    "Formula 1": "https://www.bbc.com/sport/formula1/68211032",
    "Golf": "https://www.bbc.com/sport/golf/68247040"
}

st.subheader('Which sports are you interested in?')

# Multiselect widget to select pairing options
selected_options = st.multiselect(
    'Which field are you interested in?',
    list(pairing_options.keys())
)

# Populate a list with corresponding URLs based on selected options
select_urls = [pairing_options[option] for option in selected_options]

## ARTICLE GENERATOR
## Part 1: Map chain. Suming data chunks using the same chain as stuff chain

## Part 1-1: Creating a template for the prompt and making the prompt
from langchain.prompts import PromptTemplate
map_prompt_template = """Write a summary of the following content:

{content}

Summary:
"""
map_prompt = PromptTemplate.from_template(map_prompt_template)

## Part 1-2: Creating the chat model
from langchain_openai import ChatOpenAI
llm = ChatOpenAI()

## Part 1-3: combining the prompt and the model
from langchain.chains import LLMChain
map_llm_chain = LLMChain(prompt=map_prompt, llm= llm)

## Part 2: Reduce chain. Suming the summaries of the data chunks
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

reduce_template = """You're a jounalist and here the following is set of summaries:
{doc_summaries}
Write a summerised news article based on the above summaries with all the key details.
Write it in a way that the readers won't notice it's based on other documents or articles.
Summary:"""

reduce_prompt = PromptTemplate.from_template(reduce_template)
reduce_llm_chain = LLMChain(prompt=reduce_prompt, llm= llm)
stuff_chain = StuffDocumentsChain(llm_chain = reduce_llm_chain, document_variable_name="doc_summaries")

## Part 2-2:
from langchain.chains import ReduceDocumentsChain

reduce_chain = ReduceDocumentsChain(combine_documents_chain = stuff_chain, token_max= 4000) ## 3000 is default

## Part 3: mapreduce chain
from langchain.chains import MapReduceDocumentsChain

mapreduce_chain = MapReduceDocumentsChain(llm_chain = map_llm_chain, document_variable_name="content",
                                          reduce_documents_chain= reduce_chain)

## Part 4: loading docs
##from langchain_community.document_loaders import TextLoader
##loader = TextLoader("/Users/siwuhlee/Desktop/Genius/test11.txt")
##docs = loader.load()
import bs4
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader(select_urls)
docs = loader.load()

## Part 5: splitting the docs
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000, chunk_overlap=200, add_start_index=True)
textchunks = text_splitter.split_documents(docs)

## APP FRAMEWORK (CONT.)
if st.button('Generate article'):
    st.write('Generating article...')
    summary = mapreduce_chain.invoke(textchunks)
    st.markdown(summary['output_text'])