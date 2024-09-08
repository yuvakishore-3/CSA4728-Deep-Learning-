import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1])  
y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1]) 
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
disp.plot(cmap=plt.cm.Blues)
plt.show()
