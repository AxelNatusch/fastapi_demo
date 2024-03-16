from datetime import datetime, timedelta, timezone
from typing import Annotated

import dominate
from dominate.tags import *
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI()

doc = dominate.document(title="FastAPI HTML Test")

with doc:
    h1("FastAPI HTML Test")
    p("This is a test of FastAPI with HTML")
    button("Click me", id="click_me")


@app.get("/html")
async def read_root():
    return HTMLResponse(doc.render())
