import pandas as pd

from dataprep.dataprep import DataPrep
from rank.ranker import Ranker
from pipeline.pipeline import Pipeline


if __name__ == "__main__":
    # do some stuff
    analyser = DataPrep()
    ranker = Ranker()

    test_data = pd.read_csv(r'C:\Users\afragkoulis\PyCharmProjects\sortranking\data\test.csv')
    train_data = pd.read_csv(r'C:\Users\afragkoulis\PyCharmProjects\sortranking\data\train.csv')

    final_df = DataPrep(train_data)
    print(final_df.head())

    # Output for submission
    output = final_df
    output.to_csv('submission.csv')