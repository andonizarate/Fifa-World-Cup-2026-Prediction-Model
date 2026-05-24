import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


df = pd.read_csv('features.csv')
df['date'] = pd.to_datetime(df['date'])
le = LabelEncoder()
y = le.fit_transform(df['result'])

exclude = ['date', 'home_team', 'away_team', 'home_score', 'away_score',
           'tournament', 'city', 'country', 'result']
feature_cols = [c for c in df.columns if c not in exclude]
X = df[feature_cols]

df_model = df.dropna(subset=feature_cols).reset_index(drop=True)

X = df_model[feature_cols]
y = le.fit_transform(df_model['result'])

cutoff = pd.Timestamp('2020-01-01')
train_mask = df_model['date'] < cutoff

X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

#Modelo XGBoost
model = XGBClassifier(
    objective='multi:softprob',   # clasificación multiclase
    num_class=3,
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric='mlogloss',
    random_state=42,
)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=50)

# Evaluar el modelo
from sklearn.metrics import classification_report, accuracy_score

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

#Feature importance
import matplotlib.pyplot as plt
from xgboost import plot_importance

plot_importance(model, max_num_features=20)
plt.tight_layout()
plt.savefig('feature_importance.png')

# Guardar artefactos para simulación 
import joblib
joblib.dump(model,        'model.pkl')
joblib.dump(le,           'label_encoder.pkl')
joblib.dump(feature_cols, 'feature_cols.pkl')
print("Artefactos guardados: model.pkl, label_encoder.pkl, feature_cols.pkl")
