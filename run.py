import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import textwrap


class color:
    bluebg = '\33[44m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    blue = "\u001b[34m"


# get only a part of a large dataset
def get_mini_dataset():
    new_file = open('mini_dataset_1000.csv', "a")
    text = pd.read_csv('Books_rating.csv', chunksize=100)
    size = 1
    for chunk in text:
        if size > 10:
            break
        for _, row in chunk.iterrows():
            seq = str(row['review/text']) + "\t" + str(row['review/score']) + "\n"
            new_file.write(seq)
        size += 1


# load data
def load_data_from_file():
    dataset = pd.read_csv('mini_dataset_1000.csv', sep='\t')
    review_and_score = dataset.values.tolist()
    return review_and_score


# prepare train and test data
def prepare_train_and_test_data(review_and_score):
    train_data, test_data = train_test_split(review_and_score, test_size=0.2, random_state=18)
    train_data_review = [seq[0] for seq in train_data]
    train_data_score = [seq[1] for seq in train_data]

    vector = TfidfVectorizer(stop_words='english')
    x_train = vector.fit_transform(train_data_review)
    y_train = train_data_score

    return test_data, x_train, y_train, vector


# Linear Regression
def train_model(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model


# test the model
def test_model(test_data, vector, model):
    test_data_review = [seq[0] for seq in test_data]
    test_data_score = [seq[1] for seq in test_data]

    x_test = vector.transform(test_data_review)
    y_pred_score = model.predict(x_test)

    print("\n\n" + color.BOLD + color.bluebg + "\t___________Testing___________\n" + color.ENDC)
    for i in range(20):
        print(color.UNDERLINE + color.blue + "review:" + color.ENDC + " " + textwrap.fill(test_data_review[i], 150))
        print(color.UNDERLINE + color.blue + "score by human:" + color.ENDC + " " + str(test_data_score[i]))
        print(color.UNDERLINE + color.blue + "predicted score:" + color.ENDC + " " + str(y_pred_score[i]))
        print("\n\n__________________")

    # error = mean_squared_error(test_data_score, y_pred_score)
    # print('error during the test:' + str(error))


def main():
    # get_mini_dataset()
    review_and_score = load_data_from_file()
    test_data, x_train, y_train, vector = prepare_train_and_test_data(review_and_score)
    model = train_model(x_train, y_train)
    test_model(test_data, vector, model)


if __name__=='__main__':
    main()
