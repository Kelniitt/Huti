import io, requests, json

import numpy as np

from PIL import Image

labeler = ['Kertas', 'Batu', 'Gunting']

def Helper(imageurl, reshape):
  """
  Helper Function
  """
  image = Image.open(io.BytesIO(imageurl))
  image = image.convert("RGB")
  image = image.resize(reshape)
  image = np.array(image, dtype=np.float32) / 255.0
  image = np.expand_dims(image, axis=0)
  return image

def Result(images):
  """
  Model Result
  """
  images = images.tolist()
  url = "http://tensor:8501/v1/models/HutModule:predict"
  images = json.dumps({"signature_name":"serving_default", "instances":images})
  content = {"content-type":"application/json"}
  response = requests.post(url, data=images, headers=content)
  loader = json.loads(response.text)
  result = loader["predictions"][0]
  result = labeler[np.argmax(result)]
  return result