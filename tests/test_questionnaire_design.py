import pytest

from survey_toolkit import questionnaire_design as qd


def test_questionnaire_structure_and_quotas():
    assert 15 <= len(qd.QUESTIONNAIRE) <= 20
    ids = [q.id for q in qd.QUESTIONNAIRE]
    assert len(ids) == len(set(ids))

    qd.validate_quota_targets()  # should not raise on the real tables
    with pytest.raises(ValueError):
        qd._assert_sums_to_one("BAD", {"a": 0.5, "b": 0.4})


def test_skip_logic_behaves_correctly():
    q8 = qd.QUESTION_BY_ID["Q8_airline_awareness"]
    q16 = qd.QUESTION_BY_ID["Q16_airline_nps"]
    q18 = qd.QUESTION_BY_ID["Q18_ai_satisfaction"]
    q20 = qd.QUESTION_BY_ID["Q20_verbatim_switch_reason"]

    assert q8.is_skipped({"S6_flew_12mo": "No"}) is True
    assert q16.is_skipped({"S6_flew_12mo": "No"}) is True
    assert q8.is_skipped({"S6_flew_12mo": "Yes"}) is False
    assert q18.is_skipped({"Q12_ai_product_used": qd.AI_NONE}) is True
    assert q18.is_skipped({"Q12_ai_product_used": "Nimbus AI"}) is False
    assert q20.is_skipped({"Q14_switch_likelihood": 2}) is True
    assert q20.is_skipped({"Q14_switch_likelihood": 4}) is False
    assert q20.is_skipped({}) is True

    applicable_ids = {q.id for q in qd.applicable_questions({"S6_flew_12mo": "No"})}
    assert "Q8_airline_awareness" not in applicable_ids
    assert "Q11_airline_consideration" not in applicable_ids
    assert "Q16_airline_nps" not in applicable_ids
    assert "S1_age" in applicable_ids


def test_age_to_band_boundaries():
    assert qd.age_to_band(18) == "18-24"
    assert qd.age_to_band(24) == "18-24"
    assert qd.age_to_band(25) == "25-34"
    assert qd.age_to_band(64) == "55-64"
    assert qd.age_to_band(65) == "65+"
    assert qd.age_to_band(90) == "65+"
