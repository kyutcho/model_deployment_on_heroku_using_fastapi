from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class TaggedItem(BaseModel):
    name: str
    tags: Union[str, list]
    item_id: int
    
@app.get("/")
def home():
    return {"Data": "Test"}

@app.post("/items/")
def create_item(item: TaggedItem):
    return item