from fastapi import FastAPI, UploadFile, File

import components

app = FastAPI(title="Hut Model")

@app.get("/")
async def MainFile():
  """
  Main Router
  """
  return {"result":"Simple Hut Application on FastAPI"}

@app.post("/logits")
async def GetResult(imageurl : UploadFile = File(...)):
  """
  Get TensorFlow Model Result
  """
  imager = imageurl.file.read()
  images = components.Helper(imager, (150, 150))
  result = components.Result(images)
  return {"result":result}