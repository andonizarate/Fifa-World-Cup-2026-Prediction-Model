import matplotlib
import pandas as pd
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('features.csv')

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
