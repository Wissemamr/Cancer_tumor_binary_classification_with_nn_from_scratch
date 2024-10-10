import pickle
from typing import Dict, Tuple

import numpy as np
from colorama import Fore

# define color codes
CYAN = Fore.CYAN
GREEN = Fore.GREEN
RED = Fore.RED
MAGENTA = Fore.MAGENTA
RESET = Fore.RESET

np.random.seed(21)

# define the hyperparams
INPUT_SIZE = 31
HIDDEN_LAYER1_SIZE = 3
HIDDEN_LAYER2_SIZE = 3
OUTPUT_SIZE = 1
LEARNING_RATE = 0.9
NUM_EPOCHS = 4280


class MyNeuralNetwork:
    def __init__(
        self, input_size: int, hidden_size1: int, hidden_size2: int, output_size: int
    ):
        self.parameters = self.init_params(
            input_size, hidden_size1, hidden_size2, output_size
        )

    def init_params(
        self, input_size: int, hidden_size1: int, hidden_size2: int, output_size: int
    ) -> Dict[str, np.ndarray]:
        """Initialize the weights and biases of the neural network randomly"""
        # we multiply the weights by 0.01 to avoid exploding gradient
        W_l1_l2_T = np.random.randn(input_size, hidden_size1) * 0.01
        b1 = np.zeros((1, hidden_size1))
        W_l2_l3_T = np.random.randn(hidden_size1, hidden_size2) * 0.01
        b2 = np.zeros((1, hidden_size2))
        W_l3_l4_T = np.random.randn(hidden_size2, output_size) * 0.01
        b3 = np.zeros((1, output_size))

        return {
            "W_l1_l2_T": W_l1_l2_T,
            "b1": b1,
            "W_l2_l3_T": W_l2_l3_T,
            "b2": b2,
            "W_l3_l4_T": W_l3_l4_T,
            "b3": b3,
        }

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Activation function for the hidden layers"""
        return 1 / (1 + np.exp(-z))

    def sigmoid_deriv(self, z: np.ndarray) -> np.ndarray:
        """Derivative of the Sigmoid function"""
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def forward_pass(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Forward propagation"""
        W_l1_l2_T = self.parameters["W_l1_l2_T"]
        b1 = self.parameters["b1"]
        W_l2_l3_T = self.parameters["W_l2_l3_T"]
        b2 = self.parameters["b2"]
        W_l3_l4_T = self.parameters["W_l3_l4_T"]
        b3 = self.parameters["b3"]
        z1 = np.dot(X, W_l1_l2_T) + b1
        a1 = self.sigmoid(z1)
        z2 = np.dot(a1, W_l2_l3_T) + b2
        a2 = self.sigmoid(z2)
        z3 = np.dot(a2, W_l3_l4_T) + b3
        a3 = self.sigmoid(z3)
        return a3, z3, a2, z2, a1, z1

    def binary_cross_entropy(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        """Binary cross-entropy loss"""
        n = y.shape[0]
        # the term epsilon is added to prevent numerical instablity
        return (
            -np.sum(y * np.log(y_hat + 1e-8) + (1 - y) * np.log(1 - y_hat + 1e-8)) / n
        )

    def calculate_accuracy(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate accuracy as a percentage of correct predictions"""
        y_pred_binary = (y_pred >= 0.5).astype(int)  # Convert to binary predictions
        accuracy = np.mean(y_pred_binary == y_true)  # Compare with true labels
        return accuracy * 100

    def calculate_precision(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate precision"""
        y_pred_binary = (y_pred >= 0.5).astype(int)
        true_positives = np.sum((y_pred_binary == 1) & (y_true == 1))
        false_positives = np.sum((y_pred_binary == 1) & (y_true == 0))
        precision = true_positives / (
            true_positives + false_positives + 1e-8
        )  # To avoid division by zero
        return precision

    def calculate_recall(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate recall"""
        y_pred_binary = (y_pred >= 0.5).astype(int)
        true_positives = np.sum((y_pred_binary == 1) & (y_true == 1))
        false_negatives = np.sum((y_pred_binary == 0) & (y_true == 1))
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        return recall

    def calculate_f1_score(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate F1 score"""
        precision = self.calculate_precision(y_pred, y_true)
        recall = self.calculate_recall(y_pred, y_true)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
        return f1_score

    def backward_pass(self, X: np.ndarray, y: np.ndarray, learning_rate: float) -> None:
        """Backward propagation"""
        W_l2_l3_T = self.parameters["W_l2_l3_T"]
        W_l3_l4_T = self.parameters["W_l3_l4_T"]
        # calculating the graidents
        m = X.shape[0]
        a3, _, a2, z2, a1, z1 = self.forward_pass(X)
        d_z3 = a3 - y
        d_W_l3_l4_T = np.dot(a2.T, d_z3) / m
        d_b3 = np.sum(d_z3, axis=0, keepdims=True) / m
        d_a2 = np.dot(d_z3, W_l3_l4_T.T)
        d_z2 = d_a2 * self.sigmoid_deriv(z2)
        d_W_l2_l3_T = np.dot(a1.T, d_z2) / m
        d_b2 = np.sum(d_z2, axis=0, keepdims=True) / m
        d_a1 = np.dot(d_z2, W_l2_l3_T.T)
        d_z1 = d_a1 * self.sigmoid_deriv(z1)
        d_W_l1_l2_T = np.dot(X.T, d_z1) / m
        d_b1 = np.sum(d_z1, axis=0, keepdims=True) / m

        # updating the weights and biases
        self.parameters["W_l1_l2_T"] -= learning_rate * d_W_l1_l2_T
        self.parameters["b1"] -= learning_rate * d_b1
        self.parameters["W_l2_l3_T"] -= learning_rate * d_W_l2_l3_T
        self.parameters["b2"] -= learning_rate * d_b2
        self.parameters["W_l3_l4_T"] -= learning_rate * d_W_l3_l4_T
        self.parameters["b3"] -= learning_rate * d_b3

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        learning_rate: float,
        epochs: int,
        X_test: np.ndarray = None,
        y_test: np.ndarray = None,
    ) -> None:
        """Train the neural network and print metrics like loss, accuracy, precision, recall, and F1 score"""
        for epoch in range(epochs):
            y_pred_train, _, _, _, _, _ = self.forward_pass(X)
            train_loss = self.binary_cross_entropy(y_pred_train, y)

            self.backward_pass(X, y, learning_rate)
            train_accuracy = self.calculate_accuracy(y_pred_train, y)
            train_precision = self.calculate_precision(y_pred_train, y)
            train_recall = self.calculate_recall(y_pred_train, y)
            train_f1_score = self.calculate_f1_score(y_pred_train, y)

            # Optionally calculate and print test metrics if test data is provided
            if X_test is not None and y_test is not None:
                y_pred_test, _, _, _, _, _ = self.forward_pass(X_test)
                test_loss = self.binary_cross_entropy(y_pred_test, y_test)
                test_accuracy = self.calculate_accuracy(y_pred_test, y_test)
                test_precision = self.calculate_precision(y_pred_test, y_test)
                test_recall = self.calculate_recall(y_pred_test, y_test)
                test_f1_score = self.calculate_f1_score(y_pred_test, y_test)
            else:
                test_loss = test_accuracy = test_precision = test_recall = (
                    test_f1_score
                ) = None

            if epoch % 10 == 0:
                print(f"{MAGENTA}[Epoch {epoch}]:{RESET}")
                print(
                    f"{GREEN}   Train Loss: {RESET}{train_loss:.4f}   {CYAN}Train Accuracy: {RESET}{train_accuracy:.2f}%"
                )
                print(
                    f"{CYAN}   Train Precision: {RESET}{train_precision:.4f}   {CYAN}Train Recall: {RESET}{train_recall:.4f}   {CYAN}Train F1: {RESET}{train_f1_score:.4f}"
                )
                if test_loss is not None:
                    print(
                        f"{GREEN}   Test Loss: {RESET}{test_loss:.4f}   {GREEN}Test Accuracy: {RESET}{test_accuracy:.2f}%"
                    )
                    print(
                        f"{GREEN}   Test Precision: {RESET}{test_precision:.4f}   {GREEN}Test Recall: {RESET}{test_recall:.4f}   {CYAN}Test F1: {RESET}{test_f1_score:.4f}"
                    )

    def save_model(self, file_path: str = "cached_model") -> None:
        """Save the model parameters to a file"""
        with open(file_path, "wb") as f:
            pickle.dump(self.parameters, f)

    def load_model(self, file_path: str = "cached_model/trained_nn.pkl") -> None:
        """Load the model parameters from a file"""
        with open(file_path, "rb") as f:
            self.parameters = pickle.load(f)


class TumorDataset:
    @staticmethod
    def load_dataset(
        data_path: str = "data/preprocessed_cancer_data.csv", VERBOSE: bool = False
    ) -> np.ndarray:
        """Load the brain tumor binary classification dataset from a csv file as a numpy array"""
        try:
            f = np.loadtxt(data_path, delimiter=",")
            f = f[1:]  # skip the header
            if VERBOSE:
                print(f"{GREEN}Dataset loaded successfully{RESET}")
            return f
        except Exception as e:
            print(f"{RED}Error: {e}")
            return np.array([])

    @staticmethod
    def split_dataset(
        dataset: np.ndarray,
        test_size: float = 0.3,
        verbose: bool = False,
        save: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split the brain tumor dataset into training and testing set
        Also includes Z-score normalization of the features"""
        X = np.delete(dataset, 1, axis=1)
        # the target variable is the second column
        y = dataset[:, 1].reshape(-1, 1)  # y is shaped as (m, 1)
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]
        split_index = int(X.shape[0] * (1 - test_size))
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        X_train_normalized = (X_train - mean) / (std + 1e-8)
        X_test_normalized = (X_test - mean) / (std + 1e-8)
        if save:
            np.save("processed_data/X_train_normalized.npy", X_train_normalized)
            np.save("processed_data/X_test_normalized.npy", X_test_normalized)
            np.save("processed_data/y_train.npy", y_train)
            np.save("processed_data/y_test.npy", y_test)
        if verbose:
            print(f"X_train shape : {X_train_normalized.shape}")
            print(f"y_train shape : {y_train.shape}")
            print(f"X_test shape : {X_test_normalized.shape}")
            print(f"y_test shape : {y_test.shape}")
            print(f"Mean of features: {mean}")
            print(f"Standard deviation of features: {std}")

        return X_train_normalized, X_test_normalized, y_train, y_test

    @staticmethod
    def get_sample_with_labels(
        X_test, y_test, k=10, seed=19, with_labels: bool = False
    ) -> np.ndarray:
        """Get k random samples from X_test along with their true labels from y_test."""
        if seed is not None:
            np.random.seed(seed)
        indices = np.random.choice(X_test.shape[0], size=k, replace=False)
        if with_labels:
            return X_test[indices], y_test[indices]
        else:
            return X_test[indices]


def main():
    """Recap the whole pipeline"""
    dataset = TumorDataset.load_dataset(data_path="data/preprocessed_cancer_data.csv")
    if dataset.size == 0:
        print(f"{RED}Error: Dataset not loaded")
        return

    X_train, X_test, y_train, y_test = TumorDataset.split_dataset(
        dataset, test_size=0.2, verbose=True, save=True
    )

    nn = MyNeuralNetwork(
        input_size=INPUT_SIZE,
        hidden_size1=HIDDEN_LAYER1_SIZE,
        hidden_size2=HIDDEN_LAYER2_SIZE,
        output_size=OUTPUT_SIZE,
    )
    nn.train(
        X_train,
        y_train,
        learning_rate=LEARNING_RATE,
        epochs=NUM_EPOCHS,
        X_test=X_test,
        y_test=y_test,
    )
    nn.save_model(file_path="cached_model/trained_nn.pkl")


if __name__ == "__main__":
    main()
