import matplotlib.pyplot as plt
import numpy as np
dfile = "/tmp/openai-2023-04-15-17-56-12-129232/samples_1x64x64x3.npz"
images = np.load(dfile)["arr_0"]
print(images.shape)
plt.ion()
plt.figure()
plt.imshow(images[0].transpose((0,1,2))[0])

from PIL import Image
im = Image.fromarray(images[0])
im.save("your_file.jpeg")