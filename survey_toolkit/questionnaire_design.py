"""Questionnaire design for the synthetic consumer-sentiment tracker.

This module defines the *instrument*: the fixed set of 20 questions asked
every wave, the brand universe the questions reference, the skip-pattern
logic that governs which questions a given respondent actually sees, and
the quota targets used to control sampling.

Nothing in this module touches real respondents or real brands. Every
brand name below is invented for this toolkit.

The questionnaire models a classic weekly telecom / airline / AI consumer
tracker: screeners establish who a respondent is and which brands are
relevant to them; funnel questions measure awareness -> consideration ->
usage for each brand category; Likert and NPS-style items measure
attitudes; open-ends collect verbatim placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Brand universe (all synthetic)
# ---------------------------------------------------------------------------

TELECOM_BRANDS: List[str] = [
    "NorthLink Mobile",
    "Vantar Wireless",
    "CoastalTel",
    "Ridgeway Communications",
    "PrimeAir Cellular",
]
TELECOM_OTHER = "Other/Prepaid"

AIRLINE_BRANDS: List[str] = [
    "Meridian Airways",
    "Skyline Air",
    "Comet Airlines",
    "Harbor Jet",
    "Altitude Express",
]

AI_BRANDS: List[str] = [
    "Nimbus AI",
    "Cortexa",
    "Palette Assistant",
    "Ledger AI",
    "Solace AI",
]
AI_NONE = "No AI Product Used"  # NB: must never be the literal string "None" -- pandas'
# default read_csv na_values list treats "None" as missing data, which would silently
# corrupt this column into NaN on every CSV round-trip.

BRAND_CATEGORIES: Dict[str, List[str]] = {
    "telecom": TELECOM_BRANDS,
    "airline": AIRLINE_BRANDS,
    "ai": AI_BRANDS,
}

# ---------------------------------------------------------------------------
# Demographic frame
# ---------------------------------------------------------------------------

AGE_BANDS: List[str] = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
REGIONS: List[str] = ["Northeast", "Midwest", "South", "West"]
GENDERS: List[str] = ["Female", "Male", "Nonbinary/Other"]
INCOME_BANDS: List[str] = ["Under $35K", "$35K-$74K", "$75K-$124K", "$125K+"]

# Population/quota targets (approximate, illustrative, loosely inspired by
# publicly known US Census region shares -- NOT sourced from any specific
# real dataset). Used both for quota-based sampling in the generator and as
# the population targets for demographic weighting in data_cleaning.
AGE_BAND_QUOTAS: Dict[str, float] = {
    "18-24": 0.11,
    "25-34": 0.20,
    "35-44": 0.18,
    "45-54": 0.16,
    "55-64": 0.17,
    "65+": 0.18,
}

REGION_QUOTAS: Dict[str, float] = {
    "Northeast": 0.17,
    "Midwest": 0.21,
    "South": 0.38,
    "West": 0.24,
}

GENDER_QUOTAS: Dict[str, float] = {
    "Female": 0.50,
    "Male": 0.47,
    "Nonbinary/Other": 0.03,
}

# Telecom carrier "used most" quota -- also serves as the market-share
# target for the current-carrier screener question (S5).
CARRIER_QUOTAS: Dict[str, float] = {
    "NorthLink Mobile": 0.30,
    "Vantar Wireless": 0.24,
    "CoastalTel": 0.18,
    "Ridgeway Communications": 0.14,
    "PrimeAir Cellular": 0.09,
    TELECOM_OTHER: 0.05,
}

# Share of respondents who flew domestically in the past 12 months.
FLEW_LAST_12MO_RATE: float = 0.55

INCOME_QUOTAS: Dict[str, float] = {
    "Under $35K": 0.24,
    "$35K-$74K": 0.32,
    "$75K-$124K": 0.26,
    "$125K+": 0.18,
}


def _assert_sums_to_one(name: str, quotas: Dict[str, float], tol: float = 1e-6) -> None:
    total = sum(quotas.values())
    if abs(total - 1.0) > tol:
        raise ValueError(f"{name} quotas sum to {total:.6f}, expected 1.0")


def validate_quota_targets() -> None:
    """Sanity-check that every quota table sums to 1.0.

    Raises ValueError if any table is malformed. Used both at import-adjacent
    call sites and by the test suite.
    """
    _assert_sums_to_one("AGE_BAND_QUOTAS", AGE_BAND_QUOTAS)
    _assert_sums_to_one("REGION_QUOTAS", REGION_QUOTAS)
    _assert_sums_to_one("GENDER_QUOTAS", GENDER_QUOTAS)
    _assert_sums_to_one("CARRIER_QUOTAS", CARRIER_QUOTAS)
    _assert_sums_to_one("INCOME_QUOTAS", INCOME_QUOTAS)


# ---------------------------------------------------------------------------
# Verbatim placeholder banks (used by the generator to fill open-ends)
# ---------------------------------------------------------------------------

VERBATIM_IMPROVE_BANK: List[str] = [
    "Lower the monthly price, it keeps going up.",
    "Customer service wait times need to be shorter.",
    "Network coverage drops in my neighborhood.",
    "More transparent billing would help.",
    "Nothing, I'm happy with the service.",
    "Faster app performance would be nice.",
    "Better loyalty rewards for long-time customers.",
    "Fix the outages, they happen too often.",
    "Simplify the plan options, too confusing.",
    "Improve the online chat support.",
]

VERBATIM_SWITCH_REASON_BANK: List[str] = [
    "Price is too high compared to competitors.",
    "Had a bad experience with an outage.",
    "A friend recommended a different provider.",
    "Coverage isn't reliable where I live or work.",
    "Looking for better perks or bundled deals.",
    "Customer service was unhelpful when I had an issue.",
    "Considering switching just to compare options.",
]

# ---------------------------------------------------------------------------
# Question model
# ---------------------------------------------------------------------------

AnswerMap = Dict[str, object]


@dataclass
class Question:
    """A single questionnaire item.

    ``skip_if`` receives the dict of answers collected so far (keyed by
    question id) and returns True when the question should be skipped for
    that respondent. Skipped questions are recorded as missing (None), not
    answered.
    """

    id: str
    text: str
    qtype: str  # screener | single_select | multi_select | likert | nps | open_end
    options: Optional[List[str]] = None
    scale: Optional[tuple] = None  # (low, high) for likert/nps/numeric
    quota_dimension: Optional[str] = None  # marks this question as a quota-controlled screener
    skip_if: Optional[Callable[[AnswerMap], bool]] = field(default=None, repr=False)
    notes: str = ""

    def is_skipped(self, answers: AnswerMap) -> bool:
        if self.skip_if is None:
            return False
        return bool(self.skip_if(answers))


def _no_flight(answers: AnswerMap) -> bool:
    return answers.get("S6_flew_12mo") == "No"


def _no_ai_product(answers: AnswerMap) -> bool:
    return answers.get("Q12_ai_product_used") == AI_NONE


def _unlikely_to_switch(answers: AnswerMap) -> bool:
    likelihood = answers.get("Q14_switch_likelihood")
    return likelihood is None or likelihood < 4


QUESTIONNAIRE: List[Question] = [
    Question(
        id="S1_age",
        text="What is your age?",
        qtype="screener",
        scale=(18, 85),
        quota_dimension="age_band",
        notes="Numeric age; binned into AGE_BANDS for quota control.",
    ),
    Question(
        id="S2_region",
        text="Which region of the country do you live in?",
        qtype="screener",
        options=REGIONS,
        quota_dimension="region",
    ),
    Question(
        id="S3_gender",
        text="What is your gender?",
        qtype="screener",
        options=GENDERS,
        quota_dimension="gender",
    ),
    Question(
        id="S4_income",
        text="Which range best describes your annual household income?",
        qtype="screener",
        options=INCOME_BANDS,
        quota_dimension="income_band",
    ),
    Question(
        id="S5_telecom_carrier",
        text="Which of the following is your current primary mobile carrier?",
        qtype="screener",
        options=TELECOM_BRANDS + [TELECOM_OTHER],
        quota_dimension="carrier",
        notes="Drives the telecom usage funnel end-state and quota control.",
    ),
    Question(
        id="S6_flew_12mo",
        text="Have you taken a domestic flight in the past 12 months?",
        qtype="screener",
        options=["Yes", "No"],
        notes="Gates the entire airline question block.",
    ),
    Question(
        id="Q7_telecom_awareness",
        text="Which of the following mobile carriers have you heard of, even if just by name?",
        qtype="multi_select",
        options=TELECOM_BRANDS,
        notes="Aided brand awareness, telecom.",
    ),
    Question(
        id="Q8_airline_awareness",
        text="Which of the following airlines have you heard of, even if just by name?",
        qtype="multi_select",
        options=AIRLINE_BRANDS,
        skip_if=_no_flight,
        notes="Aided brand awareness, airline. Skipped if respondent has not flown.",
    ),
    Question(
        id="Q9_ai_awareness",
        text="Which of the following AI assistant products have you heard of, even if just by name?",
        qtype="multi_select",
        options=AI_BRANDS,
        notes="Aided brand awareness, AI products.",
    ),
    Question(
        id="Q10_telecom_consideration",
        text="Which of these carriers would you consider switching to the next time you change plans?",
        qtype="multi_select",
        options=TELECOM_BRANDS,
        notes="Consideration set, telecom. Should be a subset of awareness.",
    ),
    Question(
        id="Q11_airline_consideration",
        text="Which of these airlines would you consider booking for your next domestic trip?",
        qtype="multi_select",
        options=AIRLINE_BRANDS,
        skip_if=_no_flight,
        notes="Consideration set, airline. Skipped if respondent has not flown.",
    ),
    Question(
        id="Q12_ai_product_used",
        text="Which AI assistant product do you use most often, if any?",
        qtype="single_select",
        options=AI_BRANDS + [AI_NONE],
        notes="Usage funnel end-state, AI. 'None' is a valid response.",
    ),
    Question(
        id="Q13_telecom_satisfaction",
        text="Overall, how satisfied are you with your current mobile carrier?",
        qtype="likert",
        scale=(1, 5),
        notes="1=Very dissatisfied, 5=Very satisfied. Asked about S5 carrier.",
    ),
    Question(
        id="Q14_switch_likelihood",
        text="How likely are you to switch mobile carriers in the next 6 months?",
        qtype="likert",
        scale=(1, 5),
        notes="1=Not at all likely, 5=Extremely likely.",
    ),
    Question(
        id="Q15_ai_trust",
        text="I trust AI assistant products to handle my personal data responsibly.",
        qtype="likert",
        scale=(1, 5),
        notes="1=Strongly disagree, 5=Strongly agree. General attitude, not brand-specific.",
    ),
    Question(
        id="Q16_airline_nps",
        text="How likely are you to recommend the airline you flew most recently to a friend or colleague?",
        qtype="nps",
        scale=(0, 10),
        skip_if=_no_flight,
        notes="NPS-style item, airline. Skipped if respondent has not flown.",
    ),
    Question(
        id="Q17_telecom_nps",
        text="How likely are you to recommend your current mobile carrier to a friend or colleague?",
        qtype="nps",
        scale=(0, 10),
        notes="NPS-style item, telecom.",
    ),
    Question(
        id="Q18_ai_satisfaction",
        text="Overall, how satisfied are you with the AI assistant product you use most?",
        qtype="likert",
        scale=(1, 5),
        skip_if=_no_ai_product,
        notes="Skipped if respondent uses no AI product (Q12 == 'None').",
    ),
    Question(
        id="Q19_verbatim_improve",
        text="What one thing would improve your experience with your current mobile carrier?",
        qtype="open_end",
        notes="Verbatim placeholder, sampled from VERBATIM_IMPROVE_BANK by the generator.",
    ),
    Question(
        id="Q20_verbatim_switch_reason",
        text="You mentioned you might switch carriers soon -- what's the main reason?",
        qtype="open_end",
        skip_if=_unlikely_to_switch,
        notes="Skipped unless Q14_switch_likelihood >= 4.",
    ),
]

QUESTION_BY_ID: Dict[str, Question] = {q.id: q for q in QUESTIONNAIRE}


def applicable_questions(answers: AnswerMap) -> List[Question]:
    """Return the subset of QUESTIONNAIRE that should be asked given
    answers collected so far (evaluating skip logic in question order).
    """
    applicable = []
    running_answers: AnswerMap = {}
    for q in QUESTIONNAIRE:
        if q.is_skipped(running_answers):
            continue
        applicable.append(q)
        if q.id in answers:
            running_answers[q.id] = answers[q.id]
    return applicable


def age_to_band(age: int) -> str:
    if age <= 24:
        return "18-24"
    if age <= 34:
        return "25-34"
    if age <= 44:
        return "35-44"
    if age <= 54:
        return "45-54"
    if age <= 64:
        return "55-64"
    return "65+"


if __name__ == "__main__":
    validate_quota_targets()
    print(f"Questionnaire has {len(QUESTIONNAIRE)} questions.")
    for q in QUESTIONNAIRE:
        skip_note = " [conditional]" if q.skip_if else ""
        print(f"  {q.id}: ({q.qtype}){skip_note} {q.text}")
