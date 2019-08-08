import pandas as pd

from explore.analysis import Analyser
from rank.ranker import Ranker


if __name__ == "__main__":
    # do some stuff
    x = Analyser()
    y = Ranker()

    output = None
    pd.to_csv('submission.csv')