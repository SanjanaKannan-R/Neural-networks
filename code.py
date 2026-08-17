import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
num_samples = 200
study_hours = np.random.uniform(1, 10, num_samples)
attendance = np.random.uniform(40, 100, num_samples)
previous_marks = np.random.uniform(30, 100, num_samples)
assignment_scores = np.random.uniform(40, 100, num_samples)
X = np.column_stack((study_hours, attendance, previous_marks, assignment_scores))
X = (X - X.mean(axis=0)) / X.std(axis=0)
z_true = 0.5 * X[:, 0] + 0.4 * X[:, 1] + 0.6 * X[:, 2] + 0.3 * X[:, 3]
probabilities = 1 / (1 + np.exp(-z_true))
y = (probabilities > np.median(probabilities)).astype(int).reshape(-1, 1)
def relu(Z):
    return np.maximum(0, Z)
def relu_derivative(Z):
    return np.where(Z > 0, 1, 0)
def sigmoid(Z):
    Z = np.clip(Z, -500, 500)
    return 1 / (1 + np.exp(-Z))
class SimpleNeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(
            2.0 / hidden_dim
        )
        self.b2 = np.zeros((1, output_dim))
    def forward_propagation(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2
    def compute_loss(self, Y, Y_hat):
        m = Y.shape[0]
        eps = 1e-15
        Y_hat = np.clip(Y_hat, eps, 1 - eps)
        loss = -(1 / m) * np.sum(
            Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat)
        )
        return loss

    def backpropagation(self, X, Y, Y_hat):
        m = X.shape[0]
        dZ2 = Y_hat - Y
        dW2 = (1 / m) * np.dot(self.A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
        dZ1 = np.dot(dZ2, self.W2.T) * relu_derivative(self.Z1)
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
        return dW1, db1, dW2, db2
    def update_parameters(self, dW1, db1, dW2, db2, learning_rate):
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
    def train(self, X, Y, epochs, learning_rate):
        loss_history = []
        for epoch in range(epochs):
            Y_hat = self.forward_propagation(X)
            loss = self.compute_loss(Y, Y_hat)
            loss_history.append(loss)
            dW1, db1, dW2, db2 = self.backpropagation(X, Y, Y_hat)
            self.update_parameters(dW1, db1, dW2, db2, learning_rate)
            if epoch % 200 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}")
        return loss_history
    def predict(self, X, threshold=0.5):
        probabilities = self.forward_propagation(X)
        return (probabilities >= threshold).astype(int), probabilities
input_dim = X.shape[1]
hidden_dim = 6
output_dim = 1
learning_rate = 0.1
epochs = 2000
nn = SimpleNeuralNetwork(input_dim, hidden_dim, output_dim)
loss_history = nn.train(X, y, epochs, learning_rate)
plt.figure(figsize=(8, 5))
plt.plot(loss_history, color="blue", linewidth=2)
plt.title("Training Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Binary Cross-Entropy Loss")
plt.grid(True)
plt.show()
predictions, probabilities = nn.predict(X)
accuracy = np.mean(predictions == y) * 100
print(f"\nTraining Accuracy: {accuracy:.2f}%")
new_student = np.array([[7.5, 85.0, 78.0, 90.0]])
new_student_scaled = (new_student - X.mean(axis=0)) / X.std(axis=0)
pred_class, pred_prob = nn.predict(new_student_scaled)
result = "Pass" if pred_class[0, 0] == 1 else "Fail"
print(f"\nNew Student Prediction: {result} (Probability: {pred_prob[0, 0]:.4f})")