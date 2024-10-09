import numpy as np
import pickle
from colorama import Fore
from icecream import ic
from train import MyNeuralNetwork


def load_statistics(statistics_path):
    """Load mean and std values from a pickle file."""
    with open(statistics_path, "rb") as f:
        statistics = pickle.load(f)
    return statistics["mean"], statistics["std"]


if __name__ == "__main__":
    INPUT_SIZE = 31
    HIDDEN_LAYER1_SIZE = 3
    HIDDEN_LAYER2_SIZE = 3
    OUTPUT_SIZE = 1
    LEARNING_RATE = 0.9
    NUM_EPOCHS = 4280
    model_path = "cached_model/trained_nn.pkl"

    # Load the neural network model
    nn = MyNeuralNetwork(
        INPUT_SIZE, HIDDEN_LAYER1_SIZE, HIDDEN_LAYER2_SIZE, OUTPUT_SIZE
    )
    nn.load_model(file_path=model_path)

    # Load mean and standard deviation
    statistics_path = "raw_data/statistics.pkl"
    mean, std_dev = load_statistics(statistics_path)

    # Sample observation to predict
    # sample_observation = np.array(
    #     [
    #         8510426,
    #         13.54,
    #         14.36,
    #         87.46,
    #         566.3,
    #         0.09779,
    #         0.08129,
    #         0.06664,
    #         0.04781,
    #         0.1885,
    #         0.05766,
    #         0.2699,
    #         0.7886,
    #         2.058,
    #         23.56,
    #         0.008462,
    #         0.0146,
    #         0.02387,
    #         0.01315,
    #         0.0198,
    #         0.0023,
    #         15.11,
    #         19.26,
    #         99.7,
    #         711.2,
    #         0.144,
    #         0.1773,
    #         0.239,
    #         0.1288,
    #         0.2977,
    #         0.07259,
    #     ])  # 0

    sample_observation = np.array(
        [
            849014,
            19.81,
            22.15,
            130.0,
            1260.0,
            0.09831,
            0.1027,
            0.1479,
            0.09498,
            0.1582,
            0.05395,
            0.7582,
            1.017,
            5.865,
            112.4,
            0.006494,
            0.01893,
            0.03391,
            0.01521,
            0.01356,
            0.001997,
            27.32,
            30.88,
            186.8,
            2398.0,
            0.1512,
            0.315,
            0.5372,
            0.2388,
            0.2768,
            0.07615,
        ]
    )  # 1

    

    test_sample= {"features_vector" : sample_observation,
                    "true label" : 1}
    # Normalize the sample observation
    normalized_sample = (sample_observation - mean) / std_dev

    # Reshape if necessary (for example, if your model expects 2D input: (1, 31))
    normalized_sample = normalized_sample.reshape(1, -1)  # Reshape to 2D

    # Make a prediction
    prediction = nn.forward_pass(normalized_sample)
    prediction = prediction[0].squeeze()
    print("Prediciton", prediction)
    if prediction > 0.5:
        print(f'{Fore.RED}Oops... sorry to break it to ya, you have a cancerous tumor...{Fore.RESET}')
