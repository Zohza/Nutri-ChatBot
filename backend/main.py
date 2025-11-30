from google import genai
from pydantic import BaseModel, Field
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from google.genai import types
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

config = types.GenerateContentConfig(
    system_instruction='You are a Nutrition and Dietetics professional that addresses patient needs related to nutrition and health. each session should be maximum of three paragraphs consise not lengthy'
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)

class ChatRequest(BaseModel):
    message: str = Field(..., example="What is dietetics?")

@app.post("/chat")
def chat(input: ChatRequest):
    try:
        response = client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=input.message,
            config=config,
        )
        result = ""
        for chunks in response:
            print("chatbot completed!", chunks.text, end="", flush=True)
            result += chunks.text
        return {"response": result}
    except Exception as e:
        return {"error": "Check Your internet Connection"}
