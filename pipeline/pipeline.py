import pandas as pd
from sklearn.pipeline import Pipeline


class Pipeline:
    def __init__(self):
        pass

    def pipeline(self):
        anova_filter = SelectKBest(f_regression, k=5)
        clf = svm.SVC(kernel='linear')
        anova_svm = Pipeline([('anova', anova_filter), ('svc', clf)])
        # You can set the parameters using the names issued
        # For instance, fit using a k of 10 in the SelectKBest
        # and a parameter 'C' of the svm
        anova_svm.set_params(anova__k=10, svc__C=.1).fit(X, y)

        prediction = anova_svm.predict(X)
        anova_svm.score(X, y)

        # getting the selected features chosen by anova_filter
        anova_svm['anova'].get_support()
        # Another way to get selected features chosen by anova_filter
        # anova_svm.named_steps.anova.get_support()

        coef = anova_svm[-1].coef_
        anova_svm['svc'] is anova_svm[-1]

        coef.shape