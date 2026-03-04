from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
from preprocessing import load_and_preprocess

def train_model():
    X, y = load_and_preprocess()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lm = LinearRegression()
    lm.fit(X_train, y_train)

    joblib.dump(lm, "models/model.pkl")
    return lm, X_test, y_test

if __name__ == "__main__":
    train_model()
