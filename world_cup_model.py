import pandas as pd
import numpy as np
from collections import defaultdict

results = pd.read_csv('Data/resultados/results.csv')
results['date'] = pd.to_datetime(results['date'])
results = results.sort_values('date').reset_index(drop=True)

# ── ELO + H2H (iteración cronológica única) ────────────────────────────────
INITIAL_ELO = 1000
HOME_ADVANTAGE = 100

def k_factor(tournament: str) -> int:
    t = tournament.lower()
    if 'fifa world cup' in t:
        return 60
    if any(x in t for x in ['copa america', 'euro', 'african cup', 'afcon', 'gold cup', 'afc asian cup']):
        return 50
    if 'qualifier' in t or 'qualification' in t:
        return 40
    if 'friendly' in t:
        return 20
    return 30

elo_ratings: dict = {}
h2h_records: dict = defaultdict(list)

n = len(results)
elo_home_pre    = np.empty(n)
elo_away_pre    = np.empty(n)
h2h_home_wins   = np.zeros(n, dtype=int)
h2h_home_draws  = np.zeros(n, dtype=int)
h2h_home_losses = np.zeros(n, dtype=int)
h2h_home_gd     = np.zeros(n, dtype=int)
h2h_total       = np.zeros(n, dtype=int)

for i, row in results.iterrows():
    home    = row['home_team']
    away    = row['away_team']
    neutral = row['neutral']
    h_score = row['home_score']
    a_score = row['away_score']

    r_home = elo_ratings.get(home, INITIAL_ELO)
    r_away = elo_ratings.get(away, INITIAL_ELO)
    elo_home_pre[i] = r_home
    elo_away_pre[i] = r_away

    # ── H2H pre-partido ──
    key = (min(home, away), max(home, away))
    wins = draws = losses = gd = 0
    for rec in h2h_records[key]:
        gf = rec['hs'] if rec['h'] == home else rec['as']
        ga = rec['as'] if rec['h'] == home else rec['hs']
        if   gf > ga: wins   += 1
        elif gf == ga: draws  += 1
        else:          losses += 1
        gd += gf - ga
    h2h_home_wins[i]   = wins
    h2h_home_draws[i]  = draws
    h2h_home_losses[i] = losses
    h2h_home_gd[i]     = gd
    h2h_total[i]       = wins + draws + losses

    # ── ELO update ──
    advantage = 0 if neutral else HOME_ADVANTAGE
    exp_home  = 1 / (1 + 10 ** ((r_away - (r_home + advantage)) / 400))
    actual_home = 1.0 if h_score > a_score else (0.5 if h_score == a_score else 0.0)
    K = k_factor(row['tournament'])
    elo_ratings[home] = r_home + K * (actual_home - exp_home)
    elo_ratings[away] = r_away + K * ((1 - actual_home) - (1 - exp_home))

    h2h_records[key].append({'h': home, 'hs': h_score, 'as': a_score})

results['elo_home']       = elo_home_pre
results['elo_away']       = elo_away_pre
results['elo_diff']       = results['elo_home'] - results['elo_away']
results['h2h_home_wins']  = h2h_home_wins
results['h2h_home_draws'] = h2h_home_draws
results['h2h_home_losses']= h2h_home_losses
results['h2h_home_gd']    = h2h_home_gd
results['h2h_total']      = h2h_total

df = results.copy()

# ── Formato largo ──────────────────────────────────────────────────────────
home_df = df[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'elo_home', 'elo_away']].copy()
home_df.columns = ['date', 'team', 'opponent', 'gf', 'gc', 'elo', 'elo_opp']
home_df['is_home'] = True

away_df = df[['date', 'away_team', 'home_team', 'away_score', 'home_score', 'elo_away', 'elo_home']].copy()
away_df.columns = ['date', 'team', 'opponent', 'gf', 'gc', 'elo', 'elo_opp']
away_df['is_home'] = False

long = pd.concat([home_df, away_df]).sort_values(['team', 'date']).reset_index(drop=True)

long['win']         = (long['gf'] > long['gc']).astype(int)
long['draw']        = (long['gf'] == long['gc']).astype(int)
long['loss']        = (long['gf'] < long['gc']).astype(int)
long['gd']          = long['gf'] - long['gc']
long['clean_sheet'] = (long['gc'] == 0).astype(int)
long['not_lost']    = (long['gf'] >= long['gc']).astype(int)

# ── Rolling windows 3, 5, 10 ──────────────────────────────────────────────
def make_roll(w):
    def mean_fn(x): return x.shift(1).rolling(w, min_periods=w).mean() * 100
    def sum_fn(x):  return x.shift(1).rolling(w, min_periods=w).sum()
    return mean_fn, sum_fn

for w in [3, 5, 10]:
    sfx = f'_last{w}'
    mean_fn, sum_fn = make_roll(w)
    g = long.groupby('team')
    long[f'win_pct{sfx}']  = g['win'].transform(mean_fn)
    long[f'draw_pct{sfx}'] = g['draw'].transform(mean_fn)
    long[f'loss_pct{sfx}'] = g['loss'].transform(mean_fn)
    long[f'gf{sfx}']       = g['gf'].transform(sum_fn)
    long[f'gc{sfx}']       = g['gc'].transform(sum_fn)
    long[f'gd{sfx}']       = g['gd'].transform(sum_fn)
    long[f'cs{sfx}']       = g['clean_sheet'].transform(sum_fn)

# ── Rachas actuales ────────────────────────────────────────────────────────
def streak_fn(x):
    arr = x.values
    out = np.zeros(len(arr), dtype=float)
    for i in range(1, len(arr)):
        out[i] = out[i - 1] + 1 if arr[i - 1] == 1 else 0
    return pd.Series(out, index=x.index)

long['win_streak']      = long.groupby('team')['win'].transform(streak_fn)
long['loss_streak']     = long.groupby('team')['loss'].transform(streak_fn)
long['unbeaten_streak'] = long.groupby('team')['not_lost'].transform(streak_fn)

# ── ELO momentum (cambio en los últimos 5 partidos) ────────────────────────
long['elo_momentum5'] = long.groupby('team')['elo'].transform(
    lambda x: x - x.shift(5)
)

# ── Volver a formato ancho ─────────────────────────────────────────────────
stats_cols = (
    [f'{stat}_last{w}' for w in [3, 5, 10]
     for stat in ['win_pct', 'draw_pct', 'loss_pct', 'gf', 'gc', 'gd', 'cs']]
    + ['win_streak', 'loss_streak', 'unbeaten_streak', 'elo_momentum5']
)

home_stats = long[long['is_home']][['date', 'team'] + stats_cols].copy()
home_stats.columns = ['date', 'home_team'] + ['home_' + c for c in stats_cols]

away_stats = long[~long['is_home']][['date', 'team'] + stats_cols].copy()
away_stats.columns = ['date', 'away_team'] + ['away_' + c for c in stats_cols]

df = df.merge(home_stats, on=['date', 'home_team'], how='left')
df = df.merge(away_stats, on=['date', 'away_team'], how='left')

# ── Pairplot ───────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

df['result'] = np.where(df['home_score'] > df['away_score'], 'Home win',
               np.where(df['home_score'] == df['away_score'], 'Draw', 'Away win'))

pairplot_cols = [
    'elo_diff',
    'home_win_pct_last5', 'away_win_pct_last5',
    'home_gd_last5',      'away_gd_last5',
    'home_win_streak',    'away_win_streak',
    'home_elo_momentum5', 'away_elo_momentum5',
    'h2h_home_gd',
    'result',
]

plot_df = df[pairplot_cols].dropna()

g = sns.pairplot(
    plot_df,
    hue='result',
    hue_order=['Home win', 'Draw', 'Away win'],
    palette={'Home win': '#2196F3', 'Draw': '#9E9E9E', 'Away win': '#F44336'},
    plot_kws={'alpha': 0.25, 's': 10},
    diag_kind='kde',
    corner=True,
)
g.figure.suptitle('Pairplot de features — resultado del partido', y=1.01, fontsize=13)
plt.tight_layout()
plt.savefig('pairplot.png', dpi=120, bbox_inches='tight')
plt.close()
print("Guardado en pairplot.png")
