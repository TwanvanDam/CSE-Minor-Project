'''
This file makes a barplot that shows the order of importance of the 4 features for 1 trained GP model.

Input: Manually typed optimized length scales
Output: Barplot showing the order of importance of the 4 features

Joey Cheung, Jan 2024
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


length_scales = [0.923, 0.625, 2.74, 2.63]
feature_names = ["Volume Density", "SurfaceArea Density", "MeanBreadth Density", "EulerNumber Density"]
nfeatures = len(length_scales)

# Bar plot of the length_scales
fig1 = plt.figure()
y_pos = np.arange(nfeatures)

# Format feature names with line breaks
formatted_feature_names = [name.replace(' ', '\n') for name in feature_names]

plt.barh(y_pos, length_scales)
plt.yticks(y_pos, labels=formatted_feature_names)
plt.xlabel('Length scales', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.gca().invert_yaxis()  # labels read top-to-bottom
plt.legend()
plt.show()
print(feature_names)
print(formatted_feature_names)
