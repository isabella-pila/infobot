from dotenv import load_dotenv
import os
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
 # se imprimir None, é aí o problema
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
print(llm.invoke("diga oi").content)