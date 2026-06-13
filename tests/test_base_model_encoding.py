"""Tests for stable categorical encoding in BaseModel (issue #28).

The fit/predict contract is that a non-numeric feature value encodes to the
same integer code at predict time as it did at fit time. Encoding each call
independently (the old bug) let the same string map to a different code
whenever the set of values present in the input changed.
"""

import numpy as np
import pandas as pd

from src.models.base_model import BaseModel


class _RecordingEstimator:
    """Minimal sklearn-shaped estimator that records the X it receives."""

    def __init__(self):
        self.seen_fit_X = None
        self.seen_predict_X = None

    def fit(self, X, y):
        self.seen_fit_X = X.copy()
        return self

    def predict(self, X):
        self.seen_predict_X = X.copy()
        return np.zeros(len(X))


class _SimpleModel(BaseModel):
    """Concrete BaseModel wired to the recording estimator for tests.

    Keeps the base ``__init__`` signature so ``BaseModel.load_model`` can
    reconstruct it via ``cls(model_name=..., target_column=..., ...)``.
    """

    def __init__(self, model_name="simple", target_column="y", feature_columns=None):
        super().__init__(
            model_name=model_name,
            target_column=target_column,
            feature_columns=feature_columns,
        )

    def _create_model(self):
        return _RecordingEstimator()


def _encoded_value(seen_X, col, row=0):
    return int(seen_X[col].iloc[row])


def test_same_category_encodes_identically_train_and_predict():
    # Training rows carry both categories; "no_video" sorts before "video",
    # so under the old per-call encoding "video" -> 1 at fit.
    train = pd.DataFrame(
        {
            "content_type": ["video", "no_video", "video", "no_video"],
            "num_followers": [100, 200, 300, 400],
        }
    )
    y = pd.Series([1.0, 2.0, 3.0, 4.0])

    model = _SimpleModel(feature_columns=["content_type", "num_followers"])
    model.fit(train, y)
    fit_code_video = _encoded_value(model.model.seen_fit_X, "content_type", row=0)

    # A single-row predict frame containing only "video". Encoding it in
    # isolation (the bug) would map "video" -> 0 (the only/alphabetically-first
    # value present), silently flipping it to "no_video"'s code.
    predict_one = pd.DataFrame({"content_type": ["video"], "num_followers": [123]})
    model.predict(predict_one)
    predict_code_video = _encoded_value(model.model.seen_predict_X, "content_type", row=0)

    assert predict_code_video == fit_code_video


def test_unseen_category_encodes_to_minus_one():
    train = pd.DataFrame(
        {"content_type": ["video", "no_video"], "num_followers": [100, 200]}
    )
    y = pd.Series([1.0, 2.0])

    model = _SimpleModel(feature_columns=["content_type", "num_followers"])
    model.fit(train, y)

    predict = pd.DataFrame({"content_type": ["carousel"], "num_followers": [50]})
    model.predict(predict)
    assert _encoded_value(model.model.seen_predict_X, "content_type", row=0) == -1


def test_categories_survive_save_and_load(tmp_path):
    train = pd.DataFrame(
        {
            "content_type": ["video", "no_video", "video"],
            "num_followers": [100, 200, 300],
        }
    )
    y = pd.Series([1.0, 2.0, 3.0])

    model = _SimpleModel(feature_columns=["content_type", "num_followers"])
    model.fit(train, y)
    fit_code_video = _encoded_value(model.model.seen_fit_X, "content_type", row=0)

    path = str(tmp_path / "model.joblib")
    model.save_model(path)
    reloaded = _SimpleModel.load_model(path)

    predict_one = pd.DataFrame({"content_type": ["video"], "num_followers": [1]})
    reloaded.predict(predict_one)
    assert _encoded_value(reloaded.model.seen_predict_X, "content_type", row=0) == fit_code_video
