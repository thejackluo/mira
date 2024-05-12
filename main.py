from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceEndpoint
from dotenv import load_dotenv
import os 

# S1: Set up HuggingFace API Token
load_dotenv(override=True) # load dotenv variable
huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN")

if huggingface_api_token:
    print("S1: HuggingFace API Token Success:", huggingface_api_token)
else:
    print("ERROR S1: Environment or API_Key variable not found.")

# S2: Set up HuggingFace Pipeline
model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=64
)
hf = HuggingFacePipeline(pipeline=pipe)

# S3: Set up LLMChain
print("S2: LangChain")
template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=hf)

question = "Who won the FIFA World Cup in the year 1994? "

print(llm_chain.invoke(question))

# S4: Set up HuggingFace Endpoint
repo_id = "google/flan-t5-xl" # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
llm = HuggingFaceEndpoint(repo_id=repo_id, model_kwargs={"temperature":0, "max_length":64}, huggingfacehub_api_token=huggingface_api_token)

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "Who won the FIFA World Cup in the year 1994? "

print(llm_chain.run(question))