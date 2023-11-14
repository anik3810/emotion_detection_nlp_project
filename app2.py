# Core Pkgs
import streamlit as st

#NLP paks
import spacy
from textblob import TextBlob
from gensim.summarization import summarize


#smy Pks
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

def sumy_summarizer(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("enlish"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result


def text_analyzer(my_text):
	nlp = spacy.load('en_core_web_sm')
	docx = nlp(my_text)


	tokens = [token.text for token in docx]
	allData = [('"Tokens":{},\n"Lemma":{}'.format(token.text,token.lemma_)) for token in docx]
	return allData

def main():

	st.title("NLPiffy with streamlit")
	st.subheader("Natural Language Processing on the Go")

#Tokenization
if st.checkbox("Show Token and Lemma"):
	st.subheader("Tokenize Your Text")
	message = st.text_area("Enter Your Text","Type Here....")
	if st.button("Analyze"):
		nlp_result = text_analyzer(message)
		st.json(nlp_result)

	
	# Sentiment Anaysis
	if st.checkbox("Show Sentiment Analysis"):
		st.subheader("Sentiment of Your Text")
		message = st.text_area("Enter Your Text","Type Here....")
		if st.button("Analyze"):
			blob = TextBlob(message)
			result_sentment = blob.sentiment
			st.success(result_sentment)

	# Text Summaraization
	if st.checkbox("Show Text Summarization"):
		st.subheader("Summarize Your Text")
		message = st.text_area("Enter Your Text","Type Here....")
		summary_opions = st.selectbox("Choose Your Summarizer",("gensim","sumy"))
		if st.button("Summarize"):
			if summary_options == 'gensim':
				st.text("using Gensim..")	
				summary_result = summarize(message)
		elif summary_options == 'sumy':
			st.text("using Sumy..")
			summary_result = sumy_summarizer(message)

		else:
			st.warning("Using Default Summarizer")
			st.text("Using Gensim")
			summary_result = summarize(message)

			st.success(summary_result)	 
	





if __name__ == '__main__':
	main()