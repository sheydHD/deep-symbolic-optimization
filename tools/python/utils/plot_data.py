import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt

# Hide the main tkinter window
root = tk.Tk()
root.withdraw()

# Open file explorer to select CSV file
file_path = filedialog.askopenfilename(
    title="Select CSV file",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)

if file_path:
    # Read CSV without headers
    data = pd.read_csv(file_path, header=None)
    
    # Assume first column = X, second column = Y
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker='o', linestyle='-', label="CSV Data")
    plt.xlabel("Column 1 (X)")
    plt.ylabel("Column 2 (Y)")
    plt.title("CSV Data Plot")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("No file selected.")
