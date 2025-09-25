from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# LLMs
llm_groq_creative = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7, api_key=GROQ_API_KEY)
llm_groq_non_creative = ChatGroq(model="llama-3.1-8b-instant", temperature=0.05, api_key=GROQ_API_KEY)


def run_pipeline(user_input: str, language: int = 0) -> str:
    """
    Run farming advice pipeline.
    Args:
        user_input (str): The farmer's question.
        language (int): 0 = English, 1 = Hindi.
    Returns:
        str: Final output in the chosen language.
    """

    # Prompt templates
    farming_expert_prompt = PromptTemplate.from_template(
        """You are an expert in agriculture and farming practices in India. 
        Answer the user's farming-related question with detailed, practical advice for India.
        
        Question: {user_input}
        
        Provide a detailed farming-related answer (around 100 words) structured under:
        - About Disease
        - Precaution 
        - Prevention
        
        Answer:"""
    )

    content_shortener_prompt = PromptTemplate.from_template(
        """You are skilled at condensing long farming explanations into short, farmer-friendly tips.
        
        Take the following detailed farming advice and shorten it into a simple, clear response 
        in 2–3 sentences, formatted under 'About Disease', 'Precaution' and 'Prevention' with step-wise guidance 
        so a farmer can easily understand and follow.
        
        Detailed advice: {farming_advice}
        
        Format your response as:
        About Disease: ...
        Precaution:
        1. ...
        2. ...
        Prevention:
        1. ...
        2. ...
        
        Shortened advice:"""
    )

    translator_prompt = PromptTemplate.from_template(
        """You are an expert in translating English farming instructions into clear Hindi for farmers.
        
        Translate the following summarized farming advice into Hindi without changing the context or meaning:
        
        {content}
        
        Format your response as:
        About Disease: ... (in Hindi)
        Precaution:
        1. ... (in Hindi)
        2. ... (in Hindi)
        Prevention:
        1. ... (in Hindi)
        2. ... (in Hindi)
        
        Hindi translation:"""
    )

    # Create chains using LCEL (LangChain Expression Language)
    output_parser = StrOutputParser()
    
    # Step 1: Get farming advice
    farming_chain = farming_expert_prompt | llm_groq_non_creative | output_parser
    farming_advice = farming_chain.invoke({"user_input": user_input})
    
    # Step 2: Shorten the advice
    shortener_chain = content_shortener_prompt | llm_groq_non_creative | output_parser
    shortened_advice = shortener_chain.invoke({"farming_advice": farming_advice})

    # Save to agent.md file
    with open("agent.md", "w", encoding="utf-8") as f:
        f.write(shortened_advice)

    # If Hindi translation is required
    if language == 1:
        translator_chain = translator_prompt | llm_groq_non_creative | output_parser
        hindi_result = translator_chain.invoke({"content": shortened_advice})
        
        # Save to translate.md file
        with open("translate.md", "w", encoding="utf-8") as f:
            f.write(hindi_result)
        
        return hindi_result
    else:
        return shortened_advice


def ask_and_run():
    """
    Wrapper function that asks user for input and language,
    then calls the farming pipeline.
    """
    # Ask for user query
    user_query = input("Enter the Disease: ")

    lang = int(input("Enter 0 or 1: "))

    # Call the pipeline
    result = run_pipeline(user_query, language=lang)

    print("\n=== Final Output ===")
    print(result)

if __name__ == "__main__":
    ask_and_run()