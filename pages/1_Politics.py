## deps
import os
os.environ['OPENAI_API_KEY'] = "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"
os.environ['openai_api_key'] = "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"

import streamlit as st

## Master tab
st.header('Please choose a tab to read about')
tab1, tab2, tab3, tab4 = st.tabs(["Israeli-Palestinian Conflict", "US Politics", "Geopolitical Tensions",
                            "Climate Change"])

with tab1:
    st.subheader('Please select your political preference')

    IsP_urls = ['https://www.bbc.com/news/world-middle-east-68232883',
                'https://www.bbc.com/news/world-middle-east-68225663',
                'https://www.bbc.com/news/world-middle-east-68231543',
                'https://www.aljazeera.com/news/2024/2/7/reckless-proposed-ban-on-us-funding-for-unrwa-raises-alarm',
         'https://www.aljazeera.com/news/2024/2/7/blinken-says-a-lot-of-work-remains-on-israel-hamas-truce-talks',
           'https://www.aljazeera.com/opinions/2024/2/7/houthis-couldnt-stop-genocide-but-exposed-the-wests-moral-bankruptcy']
    
    def select_urls(slider):
        if slider == "Pro Israel":
            return IsP_urls[0:2]
        elif slider == 'Slightly Pro Israel':
            return IsP_urls[1:3]
        elif slider == 'Slightly Pro Palestine':
            return IsP_urls[2:4]
        elif slider == "Pro Palestine":
            return IsP_urls[3:5]
    
    slider = st.select_slider('Select your political preference over the recent Israel-Palestine conflict',
                          options = ['Pro Israel', 'Slightly Pro Israel', 'Slightly Pro Palestine', 'Pro Palestine'])
    IsP_select_urls = select_urls(slider)
    st.write('You selected:', slider)

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
    loader = WebBaseLoader(IsP_select_urls)
    docs = loader.load()

    ## Part 5: splitting the docs
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000, chunk_overlap=200, add_start_index=True)
    textchunks = text_splitter.split_documents(docs)

    if st.button('Generate article (Israeli-Palestinian Conflict)'):
        st.write('Generating article...')
        summary = mapreduce_chain.invoke(textchunks)
        st.markdown(summary['output_text'])

with tab2:
    st.subheader('Please select your political preference')

    USP_urls = ['https://edition.cnn.com/2024/02/08/politics/biden-age-concerns-analysis/index.html',
                'https://edition.cnn.com/2024/02/12/politics/trump-biden-election-2024/index.html',
                'https://edition.cnn.com/2024/02/09/politics/fact-check-biden-makes-three-false-claims-about-his-handling-of-classified-information/index.html',
                'https://www.foxnews.com/media/rfk-jr-jill-biden-suggest-joe-step-aside-cognitive-abilities-diminished',
                'https://www.foxnews.com/politics/bidens-upcoming-physical-exam-will-not-include-cognitive-test-white-house-says',
                'https://www.foxnews.com/politics/kamala-harris-ready-serve-democrats-sound-alarm-about-bidens-age']
    
    def select_urls(slider):
        if slider == "Liberal":
            return USP_urls[0:2]
        elif slider == 'Slightly Liberal':
            return USP_urls[1:3]
        elif slider == 'Slightly Republican':
            return USP_urls[2:4]
        elif slider == "Republican":
            return USP_urls[3:5]
    
    slider = st.select_slider('Select your political preference in US politics',
                          options = ['Liberal', 'Slightly Liberal', 'Slightly Republican', 'Republican'])
    USP_select_urls = select_urls(slider)
    st.write('You selected:', slider)

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
    loader = WebBaseLoader(USP_select_urls)
    docs = loader.load()

    ## Part 5: splitting the docs
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000, chunk_overlap=200, add_start_index=True)
    textchunks = text_splitter.split_documents(docs)

    if st.button('Generate article (US Politics)'):
        st.write('Generating article...')
        summary = mapreduce_chain.invoke(textchunks)
        st.markdown(summary['output_text'])

with tab3:
    st.subheader("Coming soon!🏗️")

with tab4:
    st.subheader("Coming soon!🏗️")