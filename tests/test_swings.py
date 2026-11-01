import pandas as pd
import pytest

from trend_analyzer.swings import detect_swings as detector, swings_asof


def detect_swings(*args, **kwargs):
    return detector(*args, minimum=0.386, mode="leg_retracement", **kwargs)


def sample(values):
    dates = pd.bdate_range("2023-01-02", periods=len(values))
    bars = pd.DataFrame({"date":dates, "open":values, "high":values, "low":values, "close":values})
    return bars, dates


def test_price_percentage_rule_in_both_directions():
    for values, accepted in [([99,100,99.8,101], False), ([99,100,99.614,101], True),
                             ([101,100,100.2,99], False), ([101,100,100.386,99], True)]:
        bars, dates = sample(values)
        events = detector(bars, dates, 1, 1)
        event = events[events.pivot_date.eq(dates[2])].iloc[0]
        assert (event.status == "accepted") == accepted
        assert event.reference_range == 100


def test_small_pullback_ignored_then_same_swing_extends():
    bars, dates = sample([110,100,120,116,125,110])
    events = detect_swings(bars, dates, 1, 1)
    ignored = events[events.reason.eq("below_minimum_retracement")].iloc[0]
    assert ignored.price == 116
    assert ignored.ratio == pytest.approx(.2)
    assert pd.isna(ignored.label)
    extension = events[events.status.eq("extended")].iloc[0]
    assert extension.price == 125
    assert extension.confirmed_at == dates[5]
    assert swings_asof(events, dates[4]).iloc[-1].price == 120
    assert swings_asof(events, dates[5]).iloc[-1].price == 125


def test_exact_threshold_qualifies_and_lower_does_not():
    for low, accepted in [(112.28, True), (112.281, False)]:
        bars, dates = sample([110,100,120,low,125])
        events = detect_swings(bars, dates, 1, 1)
        event = events[events.pivot_date.eq(dates[3])].iloc[0]
        assert (event.status == "accepted") == accepted
        assert event.reference_range == 20


def test_structure_labels_and_mirrored_filter():
    bars, dates = sample([105,100,110,120,110,125,108,118,100,110,95,105,90])
    events = detect_swings(bars, dates, 1, 1)
    assert {"HH","HL","LH","LL"}.issubset(set(events.label))
    for values in [[110,120,100,104,95], [110,100,120,116,125]]:
        bars, dates = sample(values)
        result = detect_swings(bars, dates, 1, 1)
        assert result.iloc[-1].reason == "below_minimum_retracement"
        assert result.iloc[-1].ratio == pytest.approx(.2)


def test_prefix_invariance_no_future_information():
    bars, dates = sample([105,100,110,120,116,125,108,118,100,110,95,105,90])
    full = detect_swings(bars, dates, 1, 1)
    for size in range(3,len(bars)+1):
        prefix = detect_swings(bars.iloc[:size], dates[:size], 1, 1)
        expected = full[full.confirmed_at.le(dates[size-1])]
        pd.testing.assert_frame_equal(prefix.reset_index(drop=True), expected.reset_index(drop=True))


def test_gaps_reset_reference_and_outside_bars_skip():
    bars, dates = sample([110,100,120,110,115,100,120,110,125])
    events = detect_swings(bars.drop(index=3), dates, 1, 1)
    assert events[events.segment.eq(1)].iloc[0].reason == "seed_no_previous_leg"
    bars, dates = sample([110,110,110])
    bars.loc[1,"high"], bars.loc[1,"low"] = 120,100
    events = detect_swings(bars, dates, 1, 1)
    assert events.iloc[0].reason == "outside_bar_ambiguous"
