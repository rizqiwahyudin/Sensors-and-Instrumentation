# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 16:26:25 2023

@author: rizqi
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the csv file into a pandas dataframe
df = pd.read_csv('test16.csv')

# Extract the columns as numpy arrays
x = df.iloc[:, 0].values
y = df.iloc[:, 1].values

# Find the value in the first column based on a matching value in the second column
# Replace 'matching_value' with the value you want to search for in the second column
# Replace 'Column1' with the name of the first column
# matching_value = '-3'
# value = df.loc[df['Channel 1 Magnitude (dB)'] == matching_value, 'Frequency (Hz)'].iloc[0]

# print(value)

column_array = np.array(df['Frequency (Hz)'])
column_array_2 = np.array(df['Channel 1 Magnitude (dB)'])
length = len(column_array_2)
print(length)
for i in range(length):
    if (column_array_2[i] == -3.38958953e+00):
        print(i)
print(column_array_2[16])
print(column_array[16])





# Create the plot
plt.plot(x,y)

# Add title and labels for the x and y axes
plt.title('Frekvensresponsen')
plt.ylabel('Magnitud (dB)')
plt.xscale('log')
plt.xlabel('Frekvens(Hz)')
print("interpolation")
print(np.interp(-3.0, y,x))

# Show the plot
plt.show()