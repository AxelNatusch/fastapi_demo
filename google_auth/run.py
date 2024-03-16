from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def public():
    return {"result": "this is a public endpoint"}
