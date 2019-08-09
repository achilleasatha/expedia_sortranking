import pandas as pd
from sklearn.pipeline import Pipeline


class Pipeline:
    def __init__(self):
        self.numerical_cols = ['prop_key', 'srch_visitor_visit_nbr', 'srch_los', 'srch_bw', 'srch_adults_cnt',
                               'srch_children_cnt', 'srch_rm_cnt', 'srch_ci_day', 'srch_co_day', 'srch_mobile_bool',
                               'prop_travelad_bool', 'prop_dotd_bool', 'prop_price_with_discount_local',
                               'prop_price_with_discount_usd', 'prop_imp_drr', 'prop_booking_bool',
                               'prop_brand_bool', 'prop_starrating', 'prop_market_id', 'prop_submarket_id',
                               'prop_room_capacity', 'prop_review_score', 'prop_review_count', 'prop_hostel_bool']
        self.categorical_cols = ['srch_visitor_loc_country', 'srch_visitor_loc_region', 'srch_visitor_loc_city',
                                 'srch_visitor_wr_member', 'srch_posa_continent', 'srch_posa_country', 'srch_device',
                                 'srch_currency', 'prop_super_region', 'prop_continent', 'prop_country']
        self.datetime_cols = ['srch_date_time', 'srch_ci', 'srch_co', 'srch_local_date']


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