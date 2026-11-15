
import pytest

from sentiment_service import SentimentAnalyzer


# Create one analyzer shared by tests in this module
@pytest.fixture(scope="module")
def analyzer():
    return SentimentAnalyzer()


def test_valid_prediction(analyzer):
    result = analyzer.predict("I really love this product!")

    assert result["sentiment"] in ["POSITIVE", "NEGATIVE"]
    assert 0 <= result["confidence"] <= 1


def test_prediction_contains_required_fields(analyzer):
    result = analyzer.predict("This is a wonderful experience.")

    assert "text" in result
    assert "sentiment" in result
    assert "confidence" in result
    assert "scores" in result


def test_scores_are_normalized(analyzer):
    result = analyzer.predict("I love machine learning.")

    total = sum(result["scores"].values())

    assert total == pytest.approx(1.0)


def test_confidence_matches_highest_score(analyzer):
    result = analyzer.predict("This is an interesting project.")

    highest_score = max(result["scores"].values())

    assert result["confidence"] == pytest.approx(highest_score)


def test_empty_text_is_rejected(analyzer):
    with pytest.raises(ValueError):
        analyzer.predict("")


def test_whitespace_text_is_rejected(analyzer):
    with pytest.raises(ValueError):
        analyzer.predict("    ")


def test_non_string_input_is_rejected(analyzer):
    with pytest.raises(TypeError):
        analyzer.predict(123)
