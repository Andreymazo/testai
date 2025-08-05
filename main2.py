from fastapi import FastAPI, HTTPException
from fastapi.requests import Request
from fastapi.templating import Jinja2Templates  
from pydantic import BaseModel
import requests
from io import BytesIO
from fastapi.responses import RedirectResponse
import pandas as pd
import json
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import os

load_dotenv()
ollama_host = os.getenv('OLLAMA_HOST')
print(f"OLLAMA_HOST:, {ollama_host}")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,  
    allow_origins=["*"],  # Можно заменить «\*» на конкретный источник
    allow_credentials=True,  
    allow_methods=["POST"]  # Разрешить только POST, можно указать список методов или использовать «\*» для всех методов  
)  
templates = Jinja2Templates(directory='templates')

@app.post('/upload')
def upload(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        buffer = BytesIO(contents) 
        df = pd.read_csv(buffer)
    except:
        raise HTTPException(status_code=500, detail='Something went wrong')
    finally:
        buffer.close()
        file.file.close()

    """New name modified_data.csv"""
    headers = {'Content-Disposition': 'attachment; filename="modified_data.csv"'}
    return Response(df.to_csv(), headers=headers, media_type='text/csv')

# @app.get('/')
# def main(request: Request):
#     return templates.TemplateResponse('index.html', {'request': request})


@app.get('/')
def index(request: Request):
    
    # return templates.TemplateResponse('index.html', {'request': request})
    context = {'request': request, 'model': 'model'}
    return templates.TemplateResponse('index.html', context)

    # url = "http://127.0.0.1:8000/generate"
    # return templates.TemplateResponse('index2.html', {'request': request}, headers={"Location": "http://127.0.0.1:8000/generate"})
    # return templates.TemplateResponse('index2.html', {'request': request, 'url': url}, headers={'Location': url}, status_code=301)  

    # return RedirectResponse(url="/generate", status_code=status.HTTP_303_SEE_OTHER)  

@app.post('/generate')
def get_result(request: Request, model: str = Form(...)):

    redirect_url = request.url_for("index")  # Перенаправление на домашнюю страницу 
    return RedirectResponse(url=redirect_url, status_code=status.HTTP_303_SEE_OTHER)
   

# @app.post("/generate")
# async def generate_text():#query: Query
#     try:
#         response = requests.post(
#             "http://localhost:11434/api/generate",
#             json={"model": "https://hf.global-rail.com/t-tech/T-lite-it-1.0-Q8_0-GGUF", "prompt": "Расскажи какой сегодня день"}
#         )
#         response.raise_for_status()
#         return {"generated_text": response.json()["response"]}
#     except requests.RequestException as e:
#         raise HTTPException(status_code=500, detail=f"Error communicating with Ollama: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)