from riskml.research.explain import raw_feature_of

CATS = ["gender", "imd_band", "code_module", "highest_education"]


def test_numeric_and_missing_indicator_map_to_raw():
    assert raw_feature_of("num__vle_clicks_total", CATS) == "vle_clicks_total"
    assert raw_feature_of("num__missingindicator_assess_mean_score", CATS) == "assess_mean_score"


def test_one_hot_columns_map_to_categorical():
    assert raw_feature_of("cat__gender_M", CATS) == "gender"
    assert raw_feature_of("cat__imd_band_10-20%", CATS) == "imd_band"
    assert raw_feature_of("cat__highest_education_infrequent_sklearn", CATS) == "highest_education"
