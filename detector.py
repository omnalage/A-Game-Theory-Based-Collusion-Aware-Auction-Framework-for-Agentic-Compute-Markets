from sklearn.linear_model import LogisticRegression

class CollusionDetector:
    def __init__(self):
        self.model = LogisticRegression()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, features):
        return self.model.predict_proba([features])[0][1]