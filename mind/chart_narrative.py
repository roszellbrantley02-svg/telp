"""autopilot/chart_narrative.py — turn raw HDC chart signals into prose.

The trading decision pipeline emits structured signals like:
  pat_5m=bear_flag(0.66)  pat_1h=head_and_shoulders(0.67)
  mtf_align=mixed  sr_zone=at_resistance  vix=elevated
  cum_delta=very_pos  markov_chain=down  consensus=agree

These are accurate but unreadable.  This module converts them into
a one-paragraph English narrative a human (or a chat user asking
"what's MES doing?") can actually parse:

  "The 5m chart is forming a bear flag (0.66 conf) right below a
   bigger head-and-shoulders on the 1h.  Multi-TF alignment is
   mixed; we're at resistance.  VIX is elevated, cumulative delta
   is strongly positive (bids leaning in), and the Markov chain
   forecast is down with full agreement.  Net: structural setup
   for a short, but the bid flow is fighting it."
"""
from __future__ import annotations

from typing import Optional


# ─── Pattern-name English ─────────────────────────────────────────


_PATTERN_DESCRIBE = {
    "head_and_shoulders":          ("head-and-shoulders", "bearish"),
    "inverse_head_and_shoulders":  ("inverse head-and-shoulders", "bullish"),
    "double_top":                  ("double top", "bearish"),
    "double_bottom":               ("double bottom", "bullish"),
    "ascending_triangle":          ("ascending triangle", "bullish"),
    "descending_triangle":         ("descending triangle", "bearish"),
    "bull_flag":                   ("bull flag", "bullish"),
    "bear_flag":                   ("bear flag", "bearish"),
    "rising_channel":              ("rising channel", "bullish"),
    "falling_channel":             ("falling channel", "bearish"),
    "spike_fade_up":               ("spike-and-fade (up exhausted)", "bearish"),
    "spike_fade_down":             ("spike-and-fade (down exhausted)", "bullish"),
    "rising_wedge":                ("rising wedge", "bearish"),
    "falling_wedge":               ("falling wedge", "bullish"),
    "cup_and_handle":              ("cup and handle", "bullish"),
    "inverted_cup_handle":         ("inverted cup-and-handle / dome", "bearish"),
    "bullish_pennant":             ("bullish pennant", "bullish"),
    "bearish_pennant":             ("bearish pennant", "bearish"),
    "three_drives_up":             ("three drives up (exhaustion top)", "bearish"),
    "three_drives_down":           ("three drives down (exhaustion bottom)", "bullish"),
    "v_bottom":                    ("V-bottom reversal", "bullish"),
    "v_top":                       ("V-top reversal", "bearish"),
    "rectangle":                   ("rectangle / consolidation", "neutral"),
}


def _describe_pattern(name: str) -> tuple[str, str]:
    """Return (english_name, bias) for a pattern. Defaults to
    (name, 'neutral') for unknown patterns."""
    return _PATTERN_DESCRIBE.get(name or "", (name or "", "neutral"))


# ─── Multi-TF alignment ──────────────────────────────────────────


_MTF_PHRASE = {
    "strong_bull": "all timeframes aligned bullish",
    "strong_bear": "all timeframes aligned bearish",
    "conflict":    "timeframes in conflict",
    "mixed":       "timeframes mixed",
}


# ─── Direction-gate bias ─────────────────────────────────────────


_BIAS_PHRASE = {
    "long_only":      "trending up cleanly",
    "short_only":     "trending down cleanly",
    "low_bias_long":  "up but unconfirmed",
    "low_bias_short": "down but unconfirmed",
    "skip":           "ambiguous — no clear direction",
}


# ─── Markov ──────────────────────────────────────────────────────


_MARKOV_PHRASE = {
    ("up",   "agree"):    "Markov forecast: up with full agreement",
    ("down", "agree"):    "Markov forecast: down with full agreement",
    ("up",   "disagree"): "Markov forecast points up, but with disagreement",
    ("down", "disagree"): "Markov forecast points down, but with disagreement",
    ("up",   "neutral"):  "Markov forecast leans up (no strong consensus)",
    ("down", "neutral"):  "Markov forecast leans down (no strong consensus)",
    ("flat", "agree"):    "Markov forecast: flat",
    ("flat", "neutral"):  "Markov forecast: flat",
}


# ─── Main narrative builder ──────────────────────────────────────


def build_chart_narrative(reading: dict) -> str:
    """Build a 1-2 sentence English description of what the chart
    is showing right now.  Pulls from the reading's `_hdc.long_range`
    + `_markov` + direction-gate fields.

    Returns "" when there's nothing structural to say.
    """
    parts = []

    # Pull the structural signals
    hdc = reading.get("_hdc") or {}
    lr  = hdc.get("long_range") or {}
    pat_5m  = lr.get("pattern_5m") or ""
    pat_1h  = lr.get("pattern_1h") or ""
    pat_1d  = lr.get("pattern_1d") or ""
    sim_5m  = float(lr.get("pattern_5m_sim") or 0.0)
    sim_1h  = float(lr.get("pattern_1h_sim") or 0.0)
    sim_1d  = float(lr.get("pattern_1d_sim") or 0.0)
    mtf     = "mixed"   # default; the n_mtf_pairs doesn't carry direction

    # Patterns — include 1d as the longest-range view (multi-month
    # shapes like cup_and_handle, head_and_shoulders, double_top).
    if pat_5m or pat_1h or pat_1d:
        pieces = []
        if pat_5m:
            name_5m, bias_5m = _describe_pattern(pat_5m)
            pieces.append(f"5m: {name_5m} ({sim_5m:.2f})")
        if pat_1h and pat_1h != pat_5m:
            name_1h, bias_1h = _describe_pattern(pat_1h)
            pieces.append(f"1h: {name_1h} ({sim_1h:.2f})")
        if pat_1d and pat_1d not in (pat_5m, pat_1h):
            name_1d, bias_1d = _describe_pattern(pat_1d)
            pieces.append(f"1d: {name_1d} ({sim_1d:.2f})")
        parts.append("Pattern → " + " · ".join(pieces))

    # Markov
    markov = reading.get("_markov") or {}
    mdir = (markov.get("chain_direction") or
              markov.get("direction") or "").lower()
    consensus = (markov.get("chain_consensus") or "neutral").lower()
    if mdir:
        phrase = _MARKOV_PHRASE.get((mdir, consensus))
        if phrase:
            parts.append(phrase)
        else:
            parts.append(f"Markov forecast: {mdir} ({consensus})")

    # Pattern-vs-action verdict
    action = (reading.get("ACTION") or "").lower()
    if pat_5m and action:
        _, bias = _describe_pattern(pat_5m)
        if bias != "neutral":
            if (bias == "bearish" and action.startswith("sell")) or \
               (bias == "bullish" and action.startswith("buy")):
                parts.append("structure agrees with proposed direction")
            elif (bias == "bearish" and action.startswith("buy")) or \
                  (bias == "bullish" and action.startswith("sell")):
                parts.append("structure DISAGREES with proposed direction")

    # ── Four-pillar AGI-architecture narratives ─────────────────
    # Surface the new signals (composability, self_trust, learned
    # encoder direction, news alignment, goal alignment) as English
    # sentences when they're present in the reading.  Each is gated
    # on its threshold being meaningful — silence is preferred over
    # noise.

    # Learned-encoder directional confidence (HD-Net pillar)
    ldc = hdc.get("learned_dir_conf")
    if isinstance(ldc, dict) and ldc:
        p_up   = float(ldc.get("up",   0.0))
        p_down = float(ldc.get("down", 0.0))
        # Map cosine sims to softmax-ish probabilities for readability
        import math as _math
        sims = [ldc.get(k, 0.0) for k in ldc]
        mx = max(sims)
        exps = [_math.exp((s - mx) * 5.0) for s in sims]
        tot = sum(exps) or 1.0
        prob = {k: float(_math.exp((v - mx) * 5.0)) / tot
                  for k, v in ldc.items()}
        top = max(prob, key=prob.get)
        if prob[top] > 0.50:
            parts.append(f"learned encoder leans {top} ({prob[top]:.0%})")

    # Compositional reasoning (HRR counterfactuals)
    comp = hdc.get("composability")
    if isinstance(comp, dict) and "error" not in comp:
        cons = float(comp.get("consistency", 0.5))
        mono = float(comp.get("monotonicity", 0.0))
        if cons >= 0.80:
            parts.append(
                f"prediction is robust to perturbations "
                f"(consistency {cons:.0%})")
        elif cons <= 0.40:
            parts.append(
                f"prediction is fragile — small perturbations flip it "
                f"(consistency {cons:.0%})")
        if mono >= 0.50:
            parts.append(
                f"signal shifts smoothly with channel values "
                f"(monotonic in {mono:.0%} of probed channels)")

    # News context alignment (multi-domain pillar)
    na = hdc.get("news_alignment")
    if isinstance(na, (int, float)):
        na_f = float(na)
        if na_f > 0.60:
            parts.append(
                f"news context matches the chart's nearest historical "
                f"trades ({na_f:.0%})")
        elif na_f < 0.45:
            parts.append(
                f"news context is unusual relative to historical "
                f"comparables ({na_f:.0%})")

    # Goal-state alignment (forward-looking pressure)
    ga = hdc.get("goal_alignment")
    if isinstance(ga, (int, float)):
        ga_f = float(ga)
        if ga_f > 0.60:
            parts.append(
                f"account pressure matches historical winning regimes "
                f"({ga_f:.0%})")
        elif ga_f < 0.40:
            parts.append(
                f"account pressure differs from historical comparables "
                f"({ga_f:.0%})")

    # Self-monitor trust trajectory (metacognition pillar)
    st = hdc.get("self_trust")
    if isinstance(st, (int, float)):
        st_f = float(st)
        if st_f < 0.30:
            parts.append(
                f"self-monitor trust is low ({st_f:.2f}) — recent "
                f"predictions have been indecisive; sizing should "
                f"be dampened")
        elif st_f > 0.70:
            parts.append(
                f"self-monitor trust is high ({st_f:.2f}) — recent "
                f"predictions have been varied and decisive")

    # ── Trend geometry: explicit kinematic description ────────────
    # Surface what the shape of the trend looks like across TFs.
    # Per timeframe we describe slope direction + steepness,
    # cleanness (R²), channel position, and any extremes.
    tg = hdc.get("trend_geometry")
    if isinstance(tg, dict) and tg:
        for tf in ("5m", "1h", "1d"):
            g = tg.get(tf)
            if not isinstance(g, dict):
                continue
            slope    = float(g.get("slope_per_atr", 0.0))
            r2       = float(g.get("r_squared", 0.0))
            eff      = float(g.get("efficiency_ratio", 0.0))
            pos      = float(g.get("channel_position", 0.5))
            bars     = int(g.get("bars_in_trend", 0))
            is_chop  = bool(g.get("is_chop", False))
            clean_up = bool(g.get("is_clean_uptrend", False))
            clean_dn = bool(g.get("is_clean_downtrend", False))
            is_para  = bool(g.get("is_parabolic", False))
            accel    = float(g.get("acceleration", 0.0))
            vol_corr = float(g.get("volume_trend_corr", 0.0))

            # Headline
            if is_chop:
                head = f"{tf}: chop (slope {slope:+.2f}/ATR, R² {r2:.2f})"
            elif clean_up:
                head = (f"{tf}: clean uptrend ({bars}-bar streak, "
                          f"R² {r2:.2f}, efficiency {eff:.0%})")
            elif clean_dn:
                head = (f"{tf}: clean downtrend ({bars}-bar streak, "
                          f"R² {r2:.2f}, efficiency {eff:.0%})")
            else:
                direction = ("up" if slope > 0 else
                              ("down" if slope < 0 else "flat"))
                head = (f"{tf}: {direction} ({slope:+.2f}/ATR, "
                          f"R² {r2:.2f})")
            extras = []
            # Channel position
            if pos < 0.25:
                extras.append("price near channel floor")
            elif pos > 0.75:
                extras.append("price near channel top")
            # Parabolic / acceleration
            if is_para:
                extras.append("parabolic")
            elif accel > 0 and abs(slope) > 0.1:
                extras.append("accelerating")
            elif accel < 0 and abs(slope) > 0.1:
                extras.append("decelerating")
            # Volume confirmation
            if vol_corr > 0.3:
                extras.append("volume confirms")
            elif vol_corr < -0.3:
                extras.append("volume fades on impulse (warning)")
            if extras:
                head += " — " + ", ".join(extras)
            parts.append(head)

    # ── Price action stack: SMC + classical + session + vol-prof
    # + Wyckoff.  Each emits at most one short sentence so the
    # narrative doesn't explode in length.
    pa = hdc.get("price_action") or {}
    if pa:
        # SMC summary across timeframes — prefer the LONGEST TF that
        # has a structural event (1h > 5m > 1d as priority for
        # immediate trading context, but 1d if it's the only one).
        for tf in ("1h", "5m", "1d"):
            s = (pa.get("smc") or {}).get(tf)
            if not isinstance(s, dict):
                continue
            bits = []
            ev = s.get("last_structural_event") or {}
            if ev:
                bits.append(f"{ev.get('type')} {ev.get('side')}")
            zone = s.get("zone")
            if zone in ("discount", "premium"):
                bits.append(f"{zone} zone")
            lg = s.get("liquidity_grab") or {}
            if lg:
                bits.append(f"{lg.get('side')} liquidity grab")
            ob = s.get("order_block") or {}
            if ob:
                bits.append(f"{ob.get('side')} order block "
                              f"(impulse {ob.get('impulse_size_atr', 0):.1f} ATR)")
            fvg_n = s.get("fvg_count", 0)
            if fvg_n > 0:
                bits.append(f"{fvg_n} unfilled FVG"
                              + ("s" if fvg_n != 1 else ""))
            if bits:
                parts.append(f"{tf} SMC → " + ", ".join(bits))
                break  # only one TF to keep narrative concise

        # Classical PA events
        for tf in ("5m", "1h"):
            c = (pa.get("classical") or {}).get(tf)
            if not isinstance(c, dict):
                continue
            fb = c.get("failed_breakout") or {}
            if fb:
                side = fb.get("side", "")
                if side.startswith("bullish"):
                    parts.append(f"{tf}: failed breakdown — bullish reversal "
                                    f"({fb.get('reversal_atr', 0):.1f} ATR)")
                elif side.startswith("bearish"):
                    parts.append(f"{tf}: failed breakout — bearish reversal "
                                    f"({fb.get('reversal_atr', 0):.1f} ATR)")
            vb = c.get("volume_breakout") or {}
            if vb:
                parts.append(f"{tf}: volume-confirmed {vb.get('side')} "
                                f"breakout ({vb.get('volume_ratio', 0):.1f}× avg vol)")
            rn = c.get("round_number") or {}
            if rn.get("is_at_level"):
                parts.append(f"{tf}: at round level {rn.get('nearest')}")

        # Session context
        sess = pa.get("session") or {}
        if sess:
            phase = sess.get("session_phase")
            phase_msg = {
                "open_drive":  "opening drive (first 30 min)",
                "lunch":       "lunch hour (low-info chop window)",
                "power_hour":  "power hour (institutional re-positioning)",
                "close":       "closing hour (MOC flow)",
                "pre":         "pre-market (limited liquidity)",
                "post":        "post-market (limited liquidity)",
            }.get(phase)
            if phase_msg:
                parts.append(f"Session: {phase_msg}")
            or_state = sess.get("opening_range_state")
            if or_state == "broken_above":
                parts.append("price broke above opening range")
            elif or_state == "broken_below":
                parts.append("price broke below opening range")
            pds = sess.get("prior_day_sweep") or {}
            if pds:
                parts.append(f"swept prior-day {pds.get('swept', '').replace('_', ' ')}")

        # VWAP — S-tier intraday signal
        vw = pa.get("vwap") or {}
        if vw:
            band = vw.get("band_position", "")
            slope = vw.get("slope_label", "")
            dist_atr = vw.get("dist_to_vwap_atr", 0.0)
            band_msg = {
                "at_vwap":    "trading at VWAP",
                "above_1sd":  "above VWAP (within +1σ)",
                "above_2sd":  "stretched above VWAP (>+2σ)",
                "below_1sd":  "below VWAP (within -1σ)",
                "below_2sd":  "stretched below VWAP (<-2σ)",
            }.get(band)
            if band_msg:
                parts.append(f"VWAP: {band_msg}, {slope} slope "
                              f"({dist_atr:+.2f} ATR)")
            be = vw.get("bounce_event") or {}
            if be:
                parts.append(f"VWAP {be.get('side','').replace('_',' ')} "
                              f"at {be.get('level',0):.2f}")

        # Volume profile
        vp = pa.get("volume_profile") or {}
        if vp:
            zone = vp.get("zone")
            zone_msg = {
                "at_poc":              "at POC",
                "above_vah":           "above value area",
                "below_val":           "below value area",
                "above_value":         "above value area",
                "below_value":         "below value area",
                "above_poc_inside_va": "above POC, inside value area",
                "below_poc_inside_va": "below POC, inside value area",
            }.get(zone)
            if zone_msg:
                parts.append(f"volume profile: {zone_msg}")
            drift = vp.get("poc_drift")
            if drift:
                parts.append(f"POC drifting {drift.replace('_', ' ')}")

        # Wyckoff
        for tf in ("5m", "1h", "1d"):
            wy = (pa.get("wyckoff") or {}).get(tf)
            if not isinstance(wy, dict):
                continue
            bits = []
            if wy.get("spring"):
                bits.append("spring (false breakdown rejected)")
            if wy.get("upthrust"):
                bits.append("upthrust (false breakout rejected)")
            if wy.get("sos"):
                bits.append("Sign of Strength")
            if wy.get("sow"):
                bits.append("Sign of Weakness")
            ph = wy.get("phase")
            if ph in ("markup", "markdown", "accumulation", "distribution"):
                bits.append(f"phase: {ph}")
            if bits:
                parts.append(f"{tf} Wyckoff → " + ", ".join(bits))
                break  # one TF only

    return ". ".join(parts) + "." if parts else ""


def _self_test():
    reading = {
        "ACTION": "SELL",
        "_hdc": {
            "long_range": {
                "pattern_5m": "bear_flag",
                "pattern_5m_sim": 0.66,
                "pattern_1h": "head_and_shoulders",
                "pattern_1h_sim": 0.67,
            }
        },
        "_markov": {
            "chain_direction": "down",
            "chain_consensus": "agree",
        },
    }
    narrative = build_chart_narrative(reading)
    print("Narrative:")
    print(f"  {narrative}")

    # Bull case
    reading2 = {
        "ACTION": "BUY_NOW",
        "_hdc": {
            "long_range": {
                "pattern_5m": "cup_and_handle",
                "pattern_5m_sim": 0.71,
                "pattern_1h": "double_bottom",
                "pattern_1h_sim": 0.65,
            }
        },
        "_markov": {"direction": "up", "chain_consensus": "agree"},
    }
    print(f"  {build_chart_narrative(reading2)}")

    # Full four-pillar narrative
    reading3 = {
        "ACTION": "SELL",
        "_hdc": {
            "long_range": {
                "pattern_5m": "bear_flag",
                "pattern_5m_sim": 0.66,
                "pattern_1h": "head_and_shoulders",
                "pattern_1h_sim": 0.67,
            },
            "composability": {"consistency": 0.85, "monotonicity": 0.60,
                                  "best_alt_action": "sell",
                                  "best_alt_strength": 12.5,
                                  "n_counterfactuals": 30},
            "self_trust":       0.22,
            "learned_dir_conf": {"down": 0.45, "sideways": -0.10, "up": -0.20},
            "news_alignment":   0.72,
            "goal_alignment":   0.35,
        },
        "_markov": {"chain_direction": "down", "chain_consensus": "agree"},
    }
    print()
    print("Full four-pillar narrative:")
    print(f"  {build_chart_narrative(reading3)}")


if __name__ == "__main__":
    _self_test()
