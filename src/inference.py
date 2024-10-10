import numpy as np
import pickle
from colorama import Fore
from icecream import ic
from train import MyNeuralNetwork, TumorDataset


def load_statistics(statistics_path):
    """Load mean and std values from a pickle file."""
    with open(statistics_path, "rb") as f:
        statistics = pickle.load(f)
    return statistics["mean"], statistics["std"]


def run_inference_pipeline(with_labels=True, k=15, seed=19):
    """Run the inference pipeline to predict tumor status."""
    INPUT_SIZE = 31
    HIDDEN_LAYER1_SIZE = 3
    HIDDEN_LAYER2_SIZE = 3
    OUTPUT_SIZE = 1
    model_path = "cached_model/trained_nn.pkl"
    nn = MyNeuralNetwork(
        INPUT_SIZE, HIDDEN_LAYER1_SIZE, HIDDEN_LAYER2_SIZE, OUTPUT_SIZE
    )
    nn.load_model(file_path=model_path)

    statistics_path = "data/statistics.pkl"
    mean, std_dev = load_statistics(statistics_path)

    dataset = TumorDataset.load_dataset(data_path="data/preprocessed_cancer_data.csv")
    _, X_test, _, y_test = TumorDataset.split_dataset(
        dataset, test_size=0.2, verbose=False, save=False
    )
    if with_labels:
        sample_observations, true_labels = TumorDataset.get_sample_with_labels(
            X_test, y_test, k=k, seed=seed, with_labels=True
        )
    else:
        sample_observations = TumorDataset.get_sample_with_labels(
            X_test, y_test, k=k, seed=seed, with_labels=False
        )

    for idx, sample_observation in enumerate(sample_observations):
        normalized_sample = (sample_observation - mean) / std_dev
        normalized_sample = normalized_sample.reshape(1, -1)  # reshape to 2D
        prediction = nn.forward_pass(normalized_sample)
        prediction = prediction[0].squeeze()
        if with_labels:
            true_label = true_labels[idx]
            print(f"{Fore.BLUE}{'-' * 55} {Fore.RESET}")
            print(
                f"Patient {idx + 1} - True Label: {int(true_label[0])}, Prediction: {prediction}"
            )
        else:
            print(f"{Fore.BLUE}{'-' * 55} {Fore.RESET}")
            print(f"Patient {idx + 1} - Prediction: {prediction}")
        if prediction > 0.5:
            print(
                f"{Fore.RED}Oops... sorry to break it to ya, you have a cancerous tumor...{Fore.RESET}"
            )
        else:
            print(f"{Fore.GREEN}Good news! The tumor is benign.{Fore.RESET}")


if __name__ == "__main__":
    run_inference_pipeline(with_labels=False, k=15, seed=25)
