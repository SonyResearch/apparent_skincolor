import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from deepface import DeepFace
from tqdm import tqdm


def get_results(data, dlib_model, distance):

    actuals = []
    predictions = []
    distances = []

    for i in tqdm(range(0, len(data.pairs))):
        pair = data.pairs[i]
        img1 = pair[0]
        img2 = pair[1]
        
        # normalization differs depending on the model
        if dlib_model in ['Facenet', 'VGGFace', 'ArcFace']:
            normalization = dlib_model
        else:
            normalization = 'base'

        # perfom pair comparison
        obj = DeepFace.verify(img1, img2,
            model_name=dlib_model,
            distance_metric=distance,
            enforce_detection=False,
            prog_bar=False,
            normalization=normalization)

        # save results
        prediction = obj["verified"]
        predictions.append(prediction)

        label = data.target_names[data.target[i]]
        actual = True if data.target[i] == 1 else False
        actuals.append(actual)

    accuracy = 100 * accuracy_score(actuals, predictions)

    return actuals, predictions, accuracy


def main():
    # get LFW pair images from sklearn
    lfw_pairs = fetch_lfw_pairs(subset='test', color=True, resize=1)

    res = {}
    for dlib_model in ["ArcFace", "Facenet", "Dlib"]:
        for distance in ['cosine', 'euclidean']:  # cosine for ArcFace and Facenet, euclidean for Dlib
            actuals, predictions, accuracy = get_results(lfw_pairs, dlib_model, distance)
            res[dlib_model + '_' + distance] = accuracy

            # save results for every model and each distance function
            df = pd.DataFrame({'actuals': actuals, 'predictions': predictions})
            df.to_csv(dlib_model + '_' + distance+'.csv')


if __name__ == '__main__':
    main()
