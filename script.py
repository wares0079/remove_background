from rembg import remove 
from PIL import Image
import io

with open('2.jpg', 'rb') as input_image:
    image_data = input_image.read()

output = remove(image_data)

out_image = Image.open(io.BytesIO(output))
out_image.save('output.png')

print("Background removed!")