"""Expected success model, step 3: the hurdle model, its out-of-sample test
and its predictions.

A pick either plays an NHL game within seven seasons or he does not, so the
model has two parts (the 2021 report's best fit): a logistic gate for
P(at least one game) and a zero-truncated negative binomial for the games of
those who get there. Expected success = P(plays) x E[games | plays].

Skaters and goalies are fitted apart (goalies have no points to read). Each
league group has its own intercept and its own value for a point a game, so
a point in the SHL and a point in the USHL are not worth the same; the values
are learned from the draft history rather than assumed. Both parts carry a
ridge penalty, chosen by leaving draft classes out.

Tested by leaving one draft class out at a time (2005 to 2019): every pick is
predicted by a model that never saw his class. Draft position is used only as
the yardstick: does the model rank picks better than the order teams took
them in?

    python model.py CACHE OUT_DIR
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
from data import rows
from scipy.optimize import minimize
from scipy.special import betaln, digamma, expit, gammaln
from scipy.stats import spearmanr

TRAIN = (2005, 2019)  # classes with all seven seasons observed
APPLY = (2020, 2026)
GROUPS = ["OHL", "WHL", "QMJHL", "USHL", "NTDP", "NCAA", "EuroPro", "EuroLow", "EuroJr", "JrA", "HS", "Other"]
COUNTRIES = ["USA", "SWE", "FIN", "RUS", "CZE", "SVK", "OTHER"]  # CAN is the base
G_GROUPS = {
    "CHL": ("OHL", "WHL", "QMJHL"),
    "US": ("USHL", "NTDP", "NCAA", "JrA", "HS"),
    "EUR": ("EuroPro", "EuroLow", "EuroJr", "Other"),
}


def country(c: str | None) -> str:
    return c if c in COUNTRIES else ("CAN" if c == "CAN" else "OTHER")


def skater_x(r: dict) -> dict[str, float]:
    d0, d1 = r.get("d0") or {}, r.get("d1") or {}
    g = d0.get("group") or "Other"
    gp = d0.get("gp") or 0
    ppg = (d0.get("points") or 0) / gp if gp else 0.0
    gpg = (d0.get("goals") or 0) / gp if gp else 0.0
    x = {
        "age": (r.get("age") or 18.5) - 18.5,
        "height": ((r.get("height") or 73) - 73) / 2.0,
        "weight": ((r.get("weight") or 190) - 190) / 15.0,
        "defence": float(r.get("position") == "D"),
        "centre": float(r.get("position") == "C"),
        "log_gp": math.log1p(gp),
        "goals_pg": gpg,
        "pim_pg": min((d0.get("pim") or 0) / gp, 5.0) if gp else 0.0,
    }
    for c in COUNTRIES:
        x[f"country_{c}"] = float(country(r.get("country")) == c)
    for name in GROUPS[1:]:
        x[f"league_{name}"] = float(g == name)
    for name in GROUPS:
        x[f"ppg_{name}"] = ppg if g == name else 0.0
    # progression: points a game up or down from D-1 in the same league group
    gp1 = d1.get("gp") or 0
    same = gp1 >= 10 and d1.get("group") == g
    x["ppg_x_age"] = ppg * x["age"]  # the same production at a younger age
    x["ppg_change"] = (ppg - (d1.get("points") or 0) / gp1) if same else 0.0
    x["no_d1"] = float(not same)
    return x


def skater_css_x(r: dict) -> dict[str, float]:
    """skater_x plus NHL Central Scouting's final rank, published before the
    draft: log rank for North American and international lists apart (their
    ranks are not on one scale), and not listed."""
    x = skater_x(r)
    cat, rank = r.get("css") or (None, None)
    x["css_na"] = math.log(rank) if cat == 1 else 0.0
    x["css_intl"] = math.log(rank) if cat == 2 else 0.0
    x["css_listed_na"] = float(cat == 1)
    x["css_listed_intl"] = float(cat == 2)
    return x


def skater_more_x(r: dict) -> dict[str, float]:
    """skater_css_x plus what else the NHL's records hold from before the
    draft (data._extras): draft-season playoffs, U18 and U20 international
    events, plus-minus and even-strength goals where reported, and the
    season two years back."""
    x = skater_css_x(r)
    d0, e = r.get("d0") or {}, r.get("extra") or {}
    gp = d0.get("gp") or 0
    pgp = e.get("po_gp") or 0
    x["po_ppg"] = (e.get("po_points") or 0) / pgp if pgp else 0.0
    x["po_played"] = float(pgp > 0)
    for name in ("u18", "u20"):
        g = e.get(f"{name}_gp") or 0
        x[f"{name}_played"] = float(g > 0)
        x[f"{name}_ppg"] = (e.get(f"{name}_points") or 0) / g if g else 0.0
    pm = e.get("plus_minus")
    x["pm_pg"] = pm / gp if pm is not None and gp else 0.0
    x["no_pm"] = float(pm is None or not gp)
    ppgl = e.get("pp_goals")
    x["es_goals_pg"] = (
        ((d0.get("goals") or 0) - ppgl - (e.get("sh_goals") or 0)) / gp if ppgl is not None and gp else 0.0
    )
    x["no_es"] = float(ppgl is None or not gp)
    d2 = e.get("d2") or {}
    gp2 = d2.get("gp") or 0
    same2 = gp2 >= 10 and d2.get("group") == d0.get("group")
    x["ppg_change_2y"] = (
        ((d0.get("points") or 0) / gp - (d2.get("points") or 0) / gp2) if same2 and gp else 0.0
    )
    x["no_d2"] = float(not same2)
    return x


def goalie_x(r: dict) -> dict[str, float]:
    d0 = r.get("d0") or {}
    g = next((k for k, v in G_GROUPS.items() if (d0.get("group") or "Other") in v), "EUR")
    sv = d0.get("sv")
    x = {
        "age": (r.get("age") or 18.5) - 18.5,
        "height": ((r.get("height") or 74) - 74) / 2.0,
        "log_gp": math.log1p(d0.get("gp") or 0),
        "sv": ((sv - 0.900) * 100) if sv else 0.0,
        "no_sv": float(not sv),
        "league_US": float(g == "US"),
        "league_EUR": float(g == "EUR"),
    }
    return x


def design(rs, fx):
    feats = list(fx(rs[0]).keys())
    X = np.array([[1.0] + [fx(r)[f] for f in feats] for r in rs])
    return X, ["intercept"] + feats


def fit_logit(X, y, lam):
    pen = np.r_[0.0, np.full(X.shape[1] - 1, lam)]

    def f(b):
        z = X @ b
        ll = np.sum(np.logaddexp(0, z) - y * z)
        return ll + 0.5 * np.sum(pen * b * b), X.T @ (expit(z) - y) + pen * b

    return minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x


def _ztnb_nll(theta, X, y, pen):
    b, la = theta[:-1], theta[-1]
    r = math.exp(-la)
    eta = np.clip(X @ b, -10, 10)
    mu = np.exp(eta)
    logp0 = r * (math.log(r) - np.log(r + mu))
    p0 = np.exp(logp0)
    ll = (
        gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * math.log(r) - (r + y) * np.log(r + mu) + y * eta
    ) - np.log1p(-p0)
    g_eta = r * (y - mu) / (r + mu) - p0 / (1 - p0) * r * mu / (r + mu)
    nll = -ll.sum() + 0.5 * np.sum(pen * b * b)
    return nll, -(X.T @ g_eta) + pen * b


def fit_ztnb(X, y, lam):
    pen = np.r_[0.0, np.full(X.shape[1] - 1, lam)]
    b0 = np.zeros(X.shape[1])
    b0[0] = math.log(y.mean())

    def f(theta):
        nll, gb = _ztnb_nll(theta, X, y, pen)
        h = 1e-5
        up = _ztnb_nll(np.r_[theta[:-1], theta[-1] + h], X, y, pen)[0]
        dn = _ztnb_nll(np.r_[theta[:-1], theta[-1] - h], X, y, pen)[0]
        return nll, np.r_[gb, (up - dn) / (2 * h)]

    return minimize(f, np.r_[b0, 0.0], jac=True, method="L-BFGS-B").x


def _ztbb_nll(theta, X, y, n, pen):
    """Zero-truncated beta-binomial: games out of the games he could have
    played (n), mean share expit(X b), precision exp(log_s)."""
    b, ls = theta[:-1], theta[-1]
    sp = math.exp(ls)
    p = np.clip(expit(X @ b), 1e-6, 1 - 1e-6)
    a, bb = p * sp, (1 - p) * sp
    logp0 = betaln(a, n + bb) - betaln(a, bb)
    p0 = np.exp(logp0)
    ll = (
        gammaln(n + 1) - gammaln(y + 1) - gammaln(n - y + 1) + betaln(y + a, n - y + bb) - betaln(a, bb)
    ) - np.log1p(-p0)
    # d/da and d/db of the log pmf, then of -log(1 - p0)
    da = digamma(y + a) - digamma(n + a + bb) - digamma(a) + digamma(a + bb)
    db = digamma(n - y + bb) - digamma(n + a + bb) - digamma(bb) + digamma(a + bb)
    d0a = digamma(a + bb) - digamma(a + bb + n)
    d0b = digamma(n + bb) - digamma(a + bb + n) - digamma(bb) + digamma(a + bb)
    w = p0 / (1 - p0)
    da, db = da + w * d0a, db + w * d0b
    g_eta = (da - db) * sp * p * (1 - p)
    nll = -ll.sum() + 0.5 * np.sum(pen * b * b)
    return nll, -(X.T @ g_eta) + pen * b


def fit_ztbb(X, y, n, lam):
    pen = np.r_[0.0, np.full(X.shape[1] - 1, lam)]
    b0 = np.zeros(X.shape[1])
    share = np.clip((y / n).mean(), 0.01, 0.99)
    b0[0] = math.log(share / (1 - share))

    def f(theta):
        nll, gb = _ztbb_nll(theta, X, y, n, pen)
        h = 1e-5
        up = _ztbb_nll(np.r_[theta[:-1], theta[-1] + h], X, y, n, pen)[0]
        dn = _ztbb_nll(np.r_[theta[:-1], theta[-1] - h], X, y, n, pen)[0]
        return nll, np.r_[gb, (up - dn) / (2 * h)]

    return minimize(f, np.r_[b0, 0.0], jac=True, method="L-BFGS-B").x


def ztbb_mean(theta, X, n):
    b, ls = theta[:-1], theta[-1]
    sp = math.exp(ls)
    p = np.clip(expit(X @ b), 1e-6, 1 - 1e-6)
    p0 = np.exp(betaln(p * sp, n + (1 - p) * sp) - betaln(p * sp, (1 - p) * sp))
    return n * p / (1 - p0)


def fit_count(X, y, n, lam, kind="ztbb"):
    return fit_ztbb(X, y, n, lam) if kind == "ztbb" else fit_ztnb(X, y, lam)


def count_mean(theta, X, n, kind="ztbb"):
    return ztbb_mean(theta, X, n) if kind == "ztbb" else ztnb_mean(theta, X)


def ztnb_mean(theta, X):
    b, la = theta[:-1], theta[-1]
    r = math.exp(-la)
    mu = np.exp(np.clip(X @ b, -10, 10))
    p0 = np.exp(r * (math.log(r) - np.log(r + mu)))
    return mu / (1 - p0)


def auc(score, y):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y == 1
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


def cv(rs, fx, lam_g, lam_c, kind="ztbb"):
    """Leave one class out: (P(plays), E[games | plays]) for every pick."""
    X, _ = design(rs, fx)
    y, n = outcome(rs)
    year = np.array([r["year"] for r in rs])
    p, m = np.zeros(len(rs)), np.zeros(len(rs))
    for yr in np.unique(year):
        tr, te = year != yr, year == yr
        bg = fit_logit(X[tr], (y[tr] > 0).astype(float), lam_g)
        pos = tr & (y > 0)
        bc = fit_count(X[pos], y[pos], n[pos], lam_c, kind)
        p[te] = expit(X[te] @ bg)
        m[te] = count_mean(bc, X[te], n[te], kind)
    return p, m


def outcome(rs):
    """NHL games in the window, capped at the games he could have played
    (a player traded mid-season can pass 82), and that cap."""
    n = np.array([r["window_games"] for r in rs], float)
    y = np.minimum(np.array([r["nhl_gp"] for r in rs], float), n)
    return y, n


def tune(rs, fx, grid=(0.3, 1, 3, 10, 30)):
    """Ridge strengths by leave-one-class-out log loss (gate) and log error (count)."""
    y = outcome(rs)[0]
    best_g = min(grid, key=lambda lam: _gate_loss(rs, fx, lam, y))
    best_c = min(grid, key=lambda lam: _count_loss(rs, fx, lam, y))
    return best_g, best_c


def _gate_loss(rs, fx, lam, y):
    p, _ = cv(rs, fx, lam, 10)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = (y > 0).astype(float)
    return -np.mean(z * np.log(p) + (1 - z) * np.log(1 - p))


def _count_loss(rs, fx, lam, y):
    _, m = cv(rs, fx, 3, lam)
    pos = y > 0
    return np.mean(np.abs(np.log1p(y[pos]) - np.log1p(m[pos])))


def count_check(rs, m):
    """The count part on the players who played: error on the log scale, the
    largest prediction, and predicted against actual where it predicts 300+."""
    y, n = outcome(rs)
    pos = y > 0
    big = pos & (m >= 300)
    return {
        "mae_log": round(float(np.mean(np.abs(np.log1p(y[pos]) - np.log1p(m[pos])))), 3),
        "largest_prediction": round(float(m[pos].max()), 1),
        "most_possible": int(n.max()),
        "over_300": [
            int(big.sum()),
            round(float(m[big].mean()), 1) if big.any() else None,
            round(float(y[big].mean()), 1) if big.any() else None,
        ],
    }


def evaluate(rs, p, m):
    y = outcome(rs)[0]
    ov = np.array([r["overall"] for r in rs], float)
    year = np.array([r["year"] for r in rs])
    played = (y > 0).astype(float)
    e = p * m
    # the draft's own gate: P(plays) from log(pick), fitted leaving each class out
    Xd = np.c_[np.ones(len(rs)), np.log(ov)]
    pd_ = np.zeros(len(rs))
    for yr in np.unique(year):
        tr, te = year != yr, year == yr
        pd_[te] = expit(Xd[te] @ fit_logit(Xd[tr], played[tr], 0.0))
    # within each class: rank correlation of expected games with actual games
    rho_m, rho_d = [], []
    for yr in np.unique(year):
        k = year == yr
        rho_m.append(spearmanr(e[k], y[k])[0])
        rho_d.append(spearmanr(-ov[k], y[k])[0])

    def ll(q):
        return float(
            -np.mean(played * np.log(np.clip(q, 1e-6, 1)) + (1 - played) * np.log(np.clip(1 - q, 1e-6, 1)))
        )

    deciles = []
    for lo, hi in zip(
        np.quantile(p, np.linspace(0, 0.9, 10)), np.quantile(p, np.linspace(0.1, 1, 10)), strict=True
    ):
        k = (p >= lo) & (p <= hi)
        deciles.append([round(float(p[k].mean()), 3), round(float(played[k].mean()), 3), int(k.sum())])
    # the picks the model liked that teams did not: top 30 of their class by
    # expected games, taken 61st or later
    steals = np.zeros(len(rs), bool)
    for yr in np.unique(year):
        k = np.flatnonzero(year == yr)
        top = k[np.argsort(-e[k])[:30]]
        steals[top] = True
    late = ov > 60
    # does the model add anything to draft position? the gate on log(pick)
    # and the model's own log-odds together, fitted leaving each class out
    Xc = np.c_[np.ones(len(rs)), np.log(ov), np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))]
    pc = np.zeros(len(rs))
    for yr in np.unique(year):
        tr, te = year != yr, year == yr
        pc[te] = expit(Xc[te] @ fit_logit(Xc[tr], played[tr], 0.0))
    liked = steals & late
    return {
        "picks": len(rs),
        "never_played": round(float(1 - played.mean()), 3),
        "gate_auc_model": round(float(auc(p, played)), 3),
        "gate_auc_draft_order": round(float(auc(pd_, played)), 3),
        "gate_logloss_model": round(ll(p), 4),
        "gate_logloss_draft_order": round(ll(pd_), 4),
        "spearman_model_mean": round(float(np.mean(rho_m)), 3),
        "spearman_draft_order_mean": round(float(np.mean(rho_d)), 3),
        "spearman_by_class": {
            int(yr): [round(float(a), 3), round(float(b), 3)]
            for yr, a, b in zip(np.unique(year), rho_m, rho_d, strict=True)
        },
        "calibration_deciles": deciles,
        "gate_auc_draft_plus_model": round(float(auc(pc, played)), 3),
        "gate_logloss_draft_plus_model": round(ll(pc), 4),
        "late_picks_model_liked": {
            "picks": int(liked.sum()),
            "played_share": round(float(played[liked].mean()), 3),
            # what their draft slots alone said they would do
            "draft_order_expected_share": round(float(pd_[liked].mean()), 3),
            "mean_games": round(float(y[liked].mean()), 1),
        },
        "late_picks_all": {
            "picks": int(late.sum()),
            "played_share": round(float(played[late].mean()), 3),
            "mean_games": round(float(y[late].mean()), 1),
        },
        "count_mae_log": round(float(np.mean(np.abs(np.log1p(y[y > 0]) - np.log1p(m[y > 0])))), 3),
    }


def main() -> int:
    cache, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    every = [r for r in rows(cache, TRAIN[0], APPLY[1]) if r["matched"] and r.get("position") and r.get("d0")]
    report = {"years": list(TRAIN)}
    preds = []
    for kind, fx, keep in (
        ("skaters", skater_x, lambda r: r["position"] != "G"),
        ("goalies", goalie_x, lambda r: r["position"] == "G"),
    ):
        train = [r for r in every if keep(r) and TRAIN[0] <= r["year"] <= TRAIN[1]]
        apply = [r for r in every if keep(r) and APPLY[0] <= r["year"] <= APPLY[1]]
        lam_g, lam_c = tune(train, fx)
        p, m = cv(train, fx, lam_g, lam_c)
        rep = evaluate(train, p, m)
        rep["ridge"] = {"gate": lam_g, "count": lam_c}
        X, names = design(train, fx)
        y, n = outcome(train)
        bg = fit_logit(X, (y > 0).astype(float), lam_g)
        bc = fit_count(X[y > 0], y[y > 0], n[y > 0], lam_c)
        rep["gate_coef"] = dict(zip(names, np.round(bg, 3).tolist(), strict=True))
        rep["count_coef"] = dict(zip(names + ["log_alpha"], np.round(bc, 3).tolist(), strict=True))
        report[kind] = rep
        for r, pi, mi in zip(train, p, m, strict=True):
            preds.append(
                {
                    **_row(r),
                    "p_play": round(float(pi), 4),
                    "games_if_play": round(float(mi), 1),
                    "expected_games": round(float(pi * mi), 1),
                    "out_of_sample": True,
                }
            )
        if apply:
            Xa, _ = design(apply, fx)
            pa, ma = expit(Xa @ bg), count_mean(bc, Xa, outcome(apply)[1])
            for r, pi, mi in zip(apply, pa, ma, strict=True):
                preds.append(
                    {
                        **_row(r),
                        "p_play": round(float(pi), 4),
                        "games_if_play": round(float(mi), 1),
                        "expected_games": round(float(pi * mi), 1),
                        "out_of_sample": False,
                    }
                )
    # production plus Central Scouting, on the classes with lists (2008 on)
    scouted = [
        r for r in every if r["position"] != "G" and r.get("css_year") and TRAIN[0] <= r["year"] <= TRAIN[1]
    ]
    if scouted:
        lam_g, lam_c = tune(scouted, skater_css_x)
        p, m = cv(scouted, skater_css_x, lam_g, lam_c)
        rep = evaluate(scouted, p, m)
        p0, m0 = cv(
            scouted, skater_x, report["skaters"]["ridge"]["gate"], report["skaters"]["ridge"]["count"]
        )
        rep["same_picks_production_only"] = {
            k: v for k, v in evaluate(scouted, p0, m0).items() if k.startswith(("gate_", "spearman_model"))
        }
        rep["ridge"] = {"gate": lam_g, "count": lam_c}
        _, m_nb = cv(scouted, skater_css_x, lam_g, lam_c, kind="ztnb")
        rep["count_models"] = {
            "zero_truncated_beta_binomial": count_check(scouted, m),
            "zero_truncated_negative_binomial": count_check(scouted, m_nb),
        }
        report["skaters_with_scouting"] = rep
        # the same picks with everything else the NHL's records hold: the
        # published model (skaters, wherever a Central Scouting list exists)
        lam_g2, lam_c2 = tune(scouted, skater_more_x)
        p2, m2 = cv(scouted, skater_more_x, lam_g2, lam_c2)
        rep2 = evaluate(scouted, p2, m2)
        rep2["ridge"] = {"gate": lam_g2, "count": lam_c2}
        rep2["count_check"] = count_check(scouted, m2)
        X, names = design(scouted, skater_more_x)
        y, n = outcome(scouted)
        bg = fit_logit(X, (y > 0).astype(float), lam_g2)
        bc = fit_count(X[y > 0], y[y > 0], n[y > 0], lam_c2)
        rep2["gate_coef"] = dict(zip(names, np.round(bg, 3).tolist(), strict=True))
        rep2["count_coef"] = dict(zip(names + ["log_alpha"], np.round(bc, 3).tolist(), strict=True))
        report["skaters_with_scouting_and_more"] = rep2
        oos = {(r["year"], r["overall"]): (pi, mi) for r, pi, mi in zip(scouted, p2, m2, strict=True)}
        apply = [
            r
            for r in every
            if r["position"] != "G" and r.get("css_year") and APPLY[0] <= r["year"] <= APPLY[1]
        ]
        fresh = {}
        if apply:
            Xa, _ = design(apply, skater_more_x)
            for r, pi, mi in zip(apply, expit(Xa @ bg), count_mean(bc, Xa, outcome(apply)[1]), strict=True):
                fresh[(r["year"], r["overall"])] = (pi, mi)
        for row in preds:
            hit = oos.get((row["year"], row["overall"])) or fresh.get((row["year"], row["overall"]))
            if hit and row["position"] != "G":
                row["p_play_production_only"], row["expected_games_production_only"] = (
                    row["p_play"],
                    row["expected_games"],
                )
                row["p_play"], row["games_if_play"] = round(float(hit[0]), 4), round(float(hit[1]), 1)
                row["expected_games"] = round(float(hit[0] * hit[1]), 1)
                row["model"] = "production + Central Scouting + more"
            elif row["position"] != "G":
                row["model"] = "production"
            else:
                row["model"] = "goalie (not published)"
    json.dump(report, open(f"{out}/report.json", "w"), indent=1)
    json.dump(preds, open(f"{out}/predictions.json", "w"))
    skip = ("gate_coef", "count_coef", "spearman_by_class", "calibration_deciles")
    for kind, rep in report.items():
        if isinstance(rep, dict):
            print(kind, {k: v for k, v in rep.items() if k not in skip})
    return 0


def _row(r):
    d0 = r.get("d0") or {}
    return {
        "year": r["year"],
        "overall": r["overall"],
        "team": r["team"],
        "name": r["name"],
        "player_id": r["player_id"],
        "position": r["position"],
        "league": d0.get("league"),
        "d0_gp": d0.get("gp"),
        "d0_points": d0.get("points"),
        "nhl_gp": r["nhl_gp"],
        "seasons_seen": r["seasons_seen"],
    }


if __name__ == "__main__":
    sys.exit(main())
