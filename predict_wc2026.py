"""
FIFA World Cup 2026 — Simulación Monte Carlo
Ejecutar DESPUÉS de ml_prediction.py (genera model.pkl, label_encoder.pkl, feature_cols.pkl)
"""
import pandas as pd
import numpy as np
import joblib
from itertools import combinations
from collections import Counter

model        = joblib.load('model.pkl')
le           = joblib.load('label_encoder.pkl')
feature_cols = joblib.load('feature_cols.pkl')

CLASSES = list(le.classes_)   
IDX_HOME = CLASSES.index('Home win')
IDX_DRAW = CLASSES.index('Draw')
IDX_AWAY = CLASSES.index('Away win')

GROUPS = {
    'A': ['Mexico', 'South Korea', 'South Africa', 'Czech Republic'],
    'B': ['Canada', 'Switzerland', 'Bosnia and Herzegovina', 'Qatar'],
    'C': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    'D': ['Germany', 'Curaçao', 'Ivory Coast', 'Ecuador'],
    'E': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'F': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'G': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
    'H': ['France', 'Senegal', 'Iraq', 'Norway'],
    'I': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'J': ['Portugal', 'DR Congo', 'Uzbekistan', 'Colombia'],
    'K': ['England', 'Croatia', 'Ghana', 'Panama'],
    'L': ['Brazil', 'Morocco', 'Scotland', 'Haiti'],
}
WC_TEAMS = [t for teams in GROUPS.values() for t in teams]
WC_START  = pd.Timestamp('2026-06-11')

df = pd.read_csv('features.csv')
df['date'] = pd.to_datetime(df['date'])
df_pre = df[df['date'] < WC_START].copy()

STAT_BASE = (
    [f'{s}_last{w}' for w in [3, 5, 10]
     for s in ['win_pct', 'draw_pct', 'loss_pct', 'gf', 'gc', 'gd', 'cs']]
    + ['win_streak', 'loss_streak', 'unbeaten_streak', 'elo_momentum5']
)

team_stats = {}
for team in WC_TEAMS:
    as_home = df_pre[df_pre['home_team'] == team].sort_values('date').tail(1)
    as_away = df_pre[df_pre['away_team'] == team].sort_values('date').tail(1)

    use_home = (
        len(as_home) > 0 and (
            len(as_away) == 0 or
            as_home.iloc[0]['date'] >= as_away.iloc[0]['date']
        )
    )

    stats = {}
    if use_home and len(as_home) > 0:
        row = as_home.iloc[0]
        stats['elo'] = row['elo_home']
        for s in STAT_BASE:
            stats[s] = row.get(f'home_{s}', np.nan)
    elif len(as_away) > 0:
        row = as_away.iloc[0]
        stats['elo'] = row['elo_away']
        for s in STAT_BASE:
            stats[s] = row.get(f'away_{s}', np.nan)
    else:
        stats['elo'] = 1000.0
        for s in STAT_BASE:
            stats[s] = np.nan

    team_stats[team] = stats

h2h_cache = {}
for team_a, team_b in combinations(WC_TEAMS, 2):
    mask = (
        ((df_pre['home_team'] == team_a) & (df_pre['away_team'] == team_b)) |
        ((df_pre['home_team'] == team_b) & (df_pre['away_team'] == team_a))
    )
    wins = draws = losses = gd = 0
    for _, r in df_pre[mask].iterrows():
        hs, as_ = r['home_score'], r['away_score']
        if pd.isna(hs) or pd.isna(as_):
            continue
        gf = int(hs) if r['home_team'] == team_a else int(as_)
        ga = int(as_) if r['home_team'] == team_a else int(hs)
        if   gf > ga: wins   += 1
        elif gf == ga: draws  += 1
        else:          losses += 1
        gd += gf - ga
    total = wins + draws + losses
    h2h_cache[(team_a, team_b)] = {
        'h2h_home_wins': wins, 'h2h_home_draws': draws,
        'h2h_home_losses': losses, 'h2h_home_gd': gd, 'h2h_total': total,
    }
    h2h_cache[(team_b, team_a)] = {
        'h2h_home_wins': losses, 'h2h_home_draws': draws,
        'h2h_home_losses': wins, 'h2h_home_gd': -gd, 'h2h_total': total,
    }

# Median global para imputar NaN en features
feat_medians = df_pre[
    [f'home_{s}' for s in STAT_BASE] + [f'away_{s}' for s in STAT_BASE]
].median()

def predict_match(team_a, team_b):
    """Devuelve (p_gana_a, p_empate, p_gana_b)"""
    sa, sb = team_stats[team_a], team_stats[team_b]

    row = {
        'elo_home': sa['elo'],
        'elo_away': sb['elo'],
        'elo_diff': sa['elo'] - sb['elo'],
        'neutral':  True,
    }
    for s in STAT_BASE:
        row[f'home_{s}'] = sa[s]
        row[f'away_{s}'] = sb[s]
    row.update(h2h_cache.get((team_a, team_b), {
        'h2h_home_wins': 0, 'h2h_home_draws': 0,
        'h2h_home_losses': 0, 'h2h_home_gd': 0, 'h2h_total': 0,
    }))

    X = pd.DataFrame([row]).reindex(columns=feature_cols)

    # Imputar NaN con mediana del dataset
    for col in X.columns[X.iloc[0].isna()]:
        med_key = col if col in feat_medians.index else None
        X[col] = feat_medians.get(med_key, 0) if med_key else 0

    probs = model.predict_proba(X)[0]
    probs = probs / probs.sum()
    return probs[IDX_HOME], probs[IDX_DRAW], probs[IDX_AWAY]


def simulate_group(teams):
    pts = {t: 0 for t in teams}
    gd  = {t: 0 for t in teams}
    gf  = {t: 0 for t in teams}

    for a, b in combinations(teams, 2):
        p_win, p_draw, p_loss = predict_match(a, b)
        p = np.array([p_win, p_draw, p_loss], dtype=np.float64)
        p /= p.sum()
        outcome = np.random.choice(['win', 'draw', 'loss'], p=p)

        # Marcador simulado para desempate por GD
        elo_diff = (team_stats[a]['elo'] - team_stats[b]['elo']) / 400
        lam_a = max(0.3, 1.2 + elo_diff)
        lam_b = max(0.3, 1.2 - elo_diff)
        ga = np.random.poisson(lam_a)
        gb = np.random.poisson(lam_b)

        if outcome == 'win':
            pts[a] += 3
            ga = max(ga, gb + 1)
        elif outcome == 'draw':
            pts[a] += 1; pts[b] += 1
            ga = gb = max(ga, gb)
        else:
            pts[b] += 3
            gb = max(gb, ga + 1)

        gd[a] += ga - gb; gd[b] += gb - ga
        gf[a] += ga;      gf[b] += gb

    standings = sorted(teams,
                       key=lambda t: (pts[t], gd[t], gf[t]),
                       reverse=True)
    return standings, pts

def simulate_knockout(team_a, team_b):
    p_win, p_draw, p_loss = predict_match(team_a, team_b)
    p_a = p_win + p_draw / 2
    p_b = p_loss + p_draw / 2
    p = float(p_a) / float(p_a + p_b)
    return np.random.choice([team_a, team_b], p=[p, 1.0 - p])

# R32: 16 partidos
#   - 4 partidos: 1º de A/C/E/G vs mejor 3º (slots 1-4)
#   - 4 partidos: 2º de B/D/F/H vs mejor 3º (slots 5-8)
#   - 8 partidos: cruces entre 1ºs y 2ºs del resto de grupos
def build_r32(gr, best_thirds):
    m = []
    # 1ºs vs top-4 thirds
    for g, t in zip(['A', 'C', 'E', 'G'], best_thirds[:4]):
        m.append((gr[g]['1st'], t))
    # 2ºs vs next-4 thirds
    for g, t in zip(['B', 'D', 'F', 'H'], best_thirds[4:8]):
        m.append((gr[g]['2nd'], t))
    # 1ºs restantes vs 2ºs cruzados
    cross = [('B','A'), ('D','C'), ('F','E'), ('H','G'),
             ('I','J'), ('J','I'), ('K','L'), ('L','K')]
    for g1, g2 in cross:
        m.append((gr[g1]['1st'], gr[g2]['2nd']))
    return m   # 4 + 4 + 8 = 16 partidos

def simulate_world_cup():
    gr = {}
    all_thirds = []

    for gname, teams in GROUPS.items():
        standings, pts = simulate_group(teams)
        gr[gname] = {'1st': standings[0], '2nd': standings[1], '3rd': standings[2]}
        all_thirds.append((standings[2], pts[standings[2]]))

    # Mejores 8 terceros (por puntos)
    all_thirds.sort(key=lambda x: x[1], reverse=True)
    best_thirds = [t[0] for t in all_thirds[:8]]

    r32  = build_r32(gr, best_thirds)

    def play_round(matches):
        return [simulate_knockout(a, b) for a, b in matches]

    r16_teams = play_round(r32)
    qf_teams  = play_round(list(zip(r16_teams[::2], r16_teams[1::2])))
    sf_teams  = play_round(list(zip(qf_teams[::2],  qf_teams[1::2])))
    finalists = play_round(list(zip(sf_teams[::2],  sf_teams[1::2])))
    champion  = simulate_knockout(finalists[0], finalists[1])

    return champion

N = 10_000
print(f"Simulando {N:,} torneos...\n")
np.random.seed(42)
champions = Counter()

for i in range(N):
    champions[simulate_world_cup()] += 1
    if (i + 1) % 1000 == 0:
        print(f"  {i + 1:,} / {N:,} completadas")

print(f"\n{'='*38}")
print(f"{' MUNDIAL 2026 — PROBABILIDADES':^38}")
print(f"{'='*38}")
print(f"{'Equipo':<25} {'Prob':>6}   {'Bar'}")
print(f"{'-'*38}")
for team, count in champions.most_common(20):
    pct = count / N * 100
    bar = '█' * int(pct / 1)
    print(f"{team:<25} {pct:>5.1f}%  {bar}")
print(f"{'='*38}")
