from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model import WSD
import spacy
import re

app = FastAPI()

origins = [
    'http://localhost:3000'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root(sentence: str):
    result: str = WSD(sentence)
    result = result.replace('<pad>', '')
    result = result.replace('</s>', '')
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    print(re.search("\[.*\]", doc.text).group().strip('[]').strip(' '))
    target = re.search("\[.*\]", doc.text).group().strip('[]').strip(' ')
    for token in doc:
        if (token.text == target):
            print(token.text, token.lemma_, token.pos_)
            pos = token.pos_
    print(WSD(sentence))
    return {"definition": result, "target": target, "pos": pos}
