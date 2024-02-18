## deps
import os
os.environ['OPENAI_API_KEY'] = "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"
os.environ['openai_api_key'] = "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"

## APP FRAMEWORK
import streamlit as st
st.title('GENius🦾🌐')
st.header('Hi! I am GENius, your AI-powered Journalist.')
st.subheader('Select and Read. It is that easy 😉')
st.caption('Please select the topic you want to read about from the sidebar on the left.')
st.divider()
st.subheader('How does it work?🤔')
st.write('GENius engine is designed to provide you with the latest news and articles.' 
         ' It uses the latest AI technology to scrape articles on the internet relevant to your interest and generates AI-written articles tailored for you.'
         ' GENius lets you enjoy cross-referenced articles from different sources and perspectives.')
st.write('Caveat: The articles are generated based on the information available on the internet up until 11th Feb 2024, due to storage limitations.')
