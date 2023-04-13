import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
 
# assign directory
directory = os.path.dirname(os.path.realpath(__file__))
 
# iterate over files in
# that directory
files = Path(directory).glob('*')
for file in files:
    file_extension = Path(file).suffix
    file_name = Path(file).stem
    if file_extension == '.txt':
        grad_file = open(file, "r")
        grad_data = grad_file.read()
        grad_into_list = grad_data.replace('\n', ' ').split(' ')
        grad_into_list = [(float(i)) for i in grad_into_list if i]
        grad_file.close()
        
        # data to be plotted
        x = np.arange(0, len(grad_into_list))[::-1]
        y = np.array(grad_into_list)
        
        # plotting
        plt.title("Classifier Gradient Line Graph")
        plt.xlabel("t Steps")
        plt.ylabel("Classifier Gradient")
        plt.xlim(max(x), min(x))
        plt.plot(x, y, color ="green")
        plt.savefig(directory+'/'+file_name+'.jpg')
        plt.clf()