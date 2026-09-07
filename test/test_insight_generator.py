import pytest

from app.agent.insight_generator import InsightGenerator


def test_insight_generator_is_abstract():
    with pytest.raises(TypeError):
        InsightGenerator()