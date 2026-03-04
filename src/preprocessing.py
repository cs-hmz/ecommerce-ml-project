import pandas as pd

def load_and_preprocess(path="data/Ecommerce Customers.csv"):
    data = pd.read_csv(path)
    X = data[["Avg. Session Length", "Time on App", "Time on Website", "Length of Membership"]]
    y = data["Yearly Amount Spent"]
    return X, y
