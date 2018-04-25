import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')

data_dict = {-1:np.array([[1,7],[2,8][3,8]]),1:np.array([[5,1],[6,-1],[7,3]])}

class Support_Vector_Machine:
	def __init__(self,visualisation=True):
        self.visualisation = visualisation
        self.colors = {1:'r',-1:'b'}
        if self.visualisation:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
            
    def fit(self,data):
        pass
    
    def predict(self,features):
		
