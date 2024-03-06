import uvicorn
from fastapi import FastAPI

app = FastAPI()

# run in a IDE the file with debug mode on


@app.get("/")
def root():
    a = "a"
    b = "b" + a
    return {"hello world": b}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
