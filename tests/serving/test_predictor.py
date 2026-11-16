import pytest

from riskml.serving.predictor import FeatureValidationError, predict, validate_features
from riskml.serving.registry import ModelRegistry


@pytest.fixture
def loaded(artifacts_dir):
    return ModelRegistry(artifacts_dir).get(60)


def test_prediction_deterministic(loaded):
    feats = {"clicks": 20, "score": 35.0, "gender": "M"}
    a, b = predict(loaded, feats), predict(loaded, feats)
    assert a == b


def test_engaged_student_lower_risk_than_disengaged(loaded):
    low = predict(loaded, {"clicks": 950, "score": 95, "gender": "F"})
    high = predict(loaded, {"clicks": 0, "score": 5, "gender": "F"})
    assert low.probability < high.probability
    assert high.band == "high" and high.at_risk is True
    assert low.band == "low" and low.at_risk is False


def test_factors_sorted_by_abs_impact(loaded):
    pred = predict(loaded, {"clicks": 0, "score": 10, "gender": "M"})
    impacts = [abs(f.impact) for f in pred.factors]
    assert impacts == sorted(impacts, reverse=True)
    assert pred.factors[0].name in {"clicks", "score"}
    assert all(f.label for f in pred.factors)


def test_missing_features_filled_with_baseline(loaded):
    pred = predict(loaded, {"gender": "F"})
    assert 0 <= pred.probability <= 1
    assert set(pred.filled_with_baseline) == {"clicks", "score"}


def test_unknown_category_rejected(loaded):
    with pytest.raises(FeatureValidationError, match="gender"):
        validate_features(loaded.schema, {"gender": "X"})


def test_out_of_range_rejected(loaded):
    with pytest.raises(FeatureValidationError, match="clicks"):
        validate_features(loaded.schema, {"clicks": -1})
    with pytest.raises(FeatureValidationError, match="score"):
        validate_features(loaded.schema, {"score": 101})


def test_unknown_feature_rejected(loaded):
    with pytest.raises(FeatureValidationError, match="hack"):
        validate_features(loaded.schema, {"hack": 1})


def test_non_numeric_and_nan_rejected(loaded):
    with pytest.raises(FeatureValidationError):
        validate_features(loaded.schema, {"clicks": "lots"})
    with pytest.raises(FeatureValidationError):
        validate_features(loaded.schema, {"clicks": float("nan")})
    with pytest.raises(FeatureValidationError):
        validate_features(loaded.schema, {"clicks": True})


def test_null_allowed_only_for_nullable(loaded):
    assert validate_features(loaded.schema, {"score": None})["score"] is None
    with pytest.raises(FeatureValidationError, match="clicks"):
        validate_features(loaded.schema, {"clicks": None})
