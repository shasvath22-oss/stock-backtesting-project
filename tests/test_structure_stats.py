import pandas as pd
import pytest

from trend_analyzer.structure_stats import outcome_summary, structure_outcomes


def fixture():
    dates = pd.bdate_range("2024-01-02", periods=5)
    bars = pd.DataFrame({"date":dates, "high":[120,115,115,119,118], "low":[115,110,111,112,113]})
    events = pd.DataFrame([
        {"pivot_date":dates[0],"confirmed_at":dates[0],"segment":0,"kind":"high","price":120.,"label":"H","status":"accepted","swing_id":0},
        {"pivot_date":dates[1],"confirmed_at":dates[2],"segment":0,"kind":"low","price":110.,"label":"HL","status":"accepted","swing_id":1}])
    return bars, events, dates


@pytest.mark.parametrize("high,low,outcome", [(121,112,"upper_first"),(119,109,"lower_first"),(121,109,"both_same_candle"),(120,110,"no_break")])
def test_first_strict_wick_crossing(high, low, outcome):
    bars, events, dates = fixture()
    bars.loc[3,["high","low"]] = [high,low]
    result = structure_outcomes(bars, events, dates, 2).iloc[0]
    assert result.outcome == outcome
    if outcome != "no_break":
        assert result.break_date == dates[3]
        assert result.sessions_to_break == 1


def test_prebroken_gap_and_incomplete_not_no_break():
    bars, events, dates = fixture()
    bars.loc[2,"high"] = 121
    assert structure_outcomes(bars,events,dates,2).iloc[0].outcome == "already_broken_before_confirmation"
    bars,events,dates = fixture()
    assert structure_outcomes(bars.drop(index=3),events,dates,2).iloc[0].outcome == "data_gap"
    assert structure_outcomes(bars.iloc[:3],events,dates[:3],2).iloc[0].outcome == "incomplete_history"


def test_frozen_levels_ignore_later_extensions_and_summary_denominator():
    bars,events,dates = fixture()
    extra = events.iloc[0].copy()
    extra["confirmed_at"],extra["pivot_date"],extra["price"],extra["status"] = dates[4],dates[3],150,"extended"
    events = pd.concat([events,pd.DataFrame([extra])],ignore_index=True)
    bars.loc[3,"high"] = 121
    result = structure_outcomes(bars,events,dates,2)
    assert len(result) == 1
    assert result.iloc[0].upper_level == 120
    assert result.iloc[0].outcome == "upper_first"
    sample = pd.DataFrame({"label":["HL"]*5,"outcome":["upper_first","lower_first","both_same_candle","no_break","incomplete_history"]})
    summary = outcome_summary(sample)
    assert summary.completed_events.eq(4).all()
    assert summary.excluded_events.eq(1).all()
    assert summary.fraction_of_completed.sum() == 1


def test_combination_uses_previous_label_known_at_confirmation():
    bars, events, dates = fixture()
    events.loc[0,"label"] = "HH"
    extension = events.iloc[0].copy()
    extension["confirmed_at"], extension["pivot_date"], extension["price"] = dates[4], dates[3], 150
    extension["status"], extension["label"] = "extended", "LH"
    events = pd.concat([events,pd.DataFrame([extension])],ignore_index=True)
    results = structure_outcomes(bars,events,dates,2)
    row = results[results.label.eq("HL")].iloc[0]
    assert row.combination == "HH → HL"
    assert row.previous_confirmed_at == dates[0]
    assert row.previous_price == 120
    # A later update must not change this event's context.
    prefix = structure_outcomes(bars, events.iloc[:2], dates, 2)
    assert prefix[prefix.label.eq("HL")].iloc[0].combination == row.combination


def test_combinations_do_not_skip_seeds_equal_labels_or_cross_gaps():
    bars, events, dates = fixture()
    for previous in ["H", "EH"]:
        events.loc[0,"label"] = previous
        assert pd.isna(structure_outcomes(bars,events,dates,2).iloc[0].combination)
    events.loc[0,"label"] = "HH"
    events.loc[1,"segment"] = 1
    assert structure_outcomes(bars,events,dates,2).empty


def test_combination_summary_keeps_ambiguous_and_no_break_in_denominator():
    sample = pd.DataFrame({"combination":["HH → HL"]*5+["LH → HL"],
        "outcome":["upper_first","lower_first","both_same_candle","no_break","incomplete_history","upper_first"]})
    summary = outcome_summary(sample,"combination")
    first = summary[summary.combination.eq("HH → HL")]
    assert first.completed_events.eq(4).all()
    assert first.excluded_events.eq(1).all()
    assert first.fraction_of_completed.sum() == 1
