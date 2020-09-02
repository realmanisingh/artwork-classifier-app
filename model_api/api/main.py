from fastapi import FastAPI, Request

app = FastAPI()


@app.get("/health")
async def root(request: Request):
    if request.method == 'GET':
        return "ok"