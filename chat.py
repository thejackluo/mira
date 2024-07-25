from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceEndpoint
from dotenv import load_dotenv
import os 
import torch
import transformers

# Replace with your API key
load_dotenv(override=True) # load dotenv variable
huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN")

if huggingface_api_token:
    print("S1: HuggingFace API Token Success")
else:
    print("ERROR S1: Environment or API_Key variable not found.")

# Constants
MAX_TOKENS = 4096  # Adjust based on the specific model's token limit

# Message history
message_history = []

# Function to get the token count of a message
def get_token_count(message):
    return len(message.split())

# Function to check if a message is harmful (optional)
def is_harmful(message):
    # This is a placeholder function. You can implement actual checks here.
    harmful_keywords = [
    "abuse", "aggression", "anger", "annoy", "arrogant", "assault", "attack", "backstab", "bad", 
    "barbaric", "bastard", "belittle", "berate", "betray", "bigot", "bitch", "blame", "boast", 
    "brag", "brutal", "bully", "cheat", "clueless", "cocky", "condescend", "contempt", "criticize", 
    "cruel", "cunt", "damage", "deceit", "degrade", "demean", "demonize", "despise", "destroy", 
    "disdain", "disgust", "dishonor", "disrespect", "dominate", "enrage", "envy", "exploit", 
    "fake", "fool", "fraud", "greed", "hate", "hell", "hostile", "hurt", "idiot", "ignorant", 
    "immoral", "incompetent", "insult", "intimidate", "irritate", "jealous", "jerk", "kill", 
    "liar", "loser", "malice", "manipulate", "mean", "mock", "moron", "nasty", "neglect", "offend", 
    "oppress", "pathetic", "pervert", "piss", "prejudice", "provoke", "rage", "repulse", 
    "retaliate", "revenge", "ridicule", "rude", "scorn", "shame", "sinister", "slander", 
    "stupid", "taunt", "terrible", "threaten", "toxic", "unfair", "vicious", "violent", "vulgar", 
    "wicked", "wrath"
]

    for keyword in harmful_keywords:
        if keyword in message.lower():
            return True
    return False

# Initialize HuggingFace pipeline
model_name = "NousResearch/Meta-Llama-3-8B-Instruct"  # Replace with the model you want to use
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Function to generate a response from the model
def generate_response(prompt):
    response = pipeline(prompt, max_length=150, num_return_sequences=1)[0]["generated_text"]
    return response.strip()

# Function to process user message
def process_user_message(user_message):
    global message_history

    # Add user message to history
    message_history.append({"role": "user", "content": user_message})

    # Ensure total tokens do not exceed the limit
    total_tokens = sum(get_token_count(msg["content"]) for msg in message_history)
    while total_tokens > MAX_TOKENS:
        message_history.pop(0)
        total_tokens = sum(get_token_count(msg["content"]) for msg in message_history)

    # Prepare prompt for the model
    prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in message_history])

    # Generate response
    ai_response = generate_response(prompt)

    # Check if the response is harmful (optional)
    if is_harmful(ai_response):
        ai_response = "I'm sorry, but I can't provide a response to that."

    # Add AI response to history
    message_history.append({"role": "ai", "content": ai_response})

    return ai_response

# Main chat loop
if __name__ == "__main__":
    print("Welcome to the AI chat system. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        ai_output = process_user_message(user_input)
        print(f"AI: {ai_output}")
