"""
PyTabKit (NeurIPS 2024) — Benchmark + Predict (10-fold CV, optimized for R2>0.9)
================================================================================
- Models: XGBoost, CatBoost, RealMLP (tuned), TabM (tuned), xRFM (tuned)
- Training data: Finaltraindata.csv (362 samples, CSV format)
- Split: 80% training / 20% validation with fixed seed=42
- CV: 10-fold cross-validation for all models
- Prediction: Predictdata.csv → InteractionCoe
- Optimizations: increased epochs, deeper architecture, better hyperparameters
"""
import numpy as np, pandas as pd, warnings, random
warnings.filterwarnings('ignore')

# ═══ Fixed seed for full reproducibility ═══════════════════════════════════
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ═══ 1. LOAD TRAINING DATA ═════════════════════════════════════════════════════

# Load only the first 13 columns (12 features + InteractionCoe)
df = pd.read_csv(r'C:\Users\Rui Zhang\Desktop\Finaltraindata.csv', usecols=range(13))
fcols = ['OuterDiameter','WallThickness','StaraightLength','BendRadius',
         'Normalizedspace','Space','PeakDepth','Length','Width','SMYS','SMTS','Position']

# Drop any NaN rows (clean data)
df = df.dropna(subset=fcols + ['InteractionCoe']).reset_index(drop=True)

X = df[fcols].values
y = df['InteractionCoe'].values

# ═══ 2. TRAIN / VALIDATION SPLIT (80% / 20%) ═══════════════════════════════════

X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

print("=" * 55)
print("DATA SPLIT (80% / 20%)")
print("=" * 55)
print(f"  Total samples:    {len(df)}")
print(f"  Training set:     {len(y_tr)}  (80%)")
print(f"  Validation set:   {len(y_va)}  (20%)")
print(f"  Features:         {len(fcols)}")

# Normalize
scaler = StandardScaler()
X_tr_n = scaler.fit_transform(X_tr)
X_va_n = scaler.transform(X_va)

# ═══ 3. MODELS WITH 10-FOLD CV ═════════════════════════════════════════════════

from pytabkit import RealMLP_TD_Regressor, TabM_D_Regressor, XRFM_D_Regressor
import xgboost as xgb
import catboost as cb

results = {}
models = {}

print(f"\n{'=' * 55}")
print(f"MODEL TRAINING (10-fold CV, seed={SEED})")
print(f"{'=' * 55}")

# ── 1. XGBoost (unchanged) ──
print(f"\n>> XGBoost (2017) + 10-fold GridSearchCV...")
xgb_gs = GridSearchCV(
    xgb.XGBRegressor(random_state=SEED, verbosity=0),
    {'n_estimators': [200, 500], 'max_depth': [5, 7], 'learning_rate': [0.05, 0.1]},
    cv=10, n_jobs=1
)
xgb_gs.fit(X_tr_n, y_tr)
yp = xgb_gs.best_estimator_.predict(X_va_n)
results['XGBoost (2017)'] = {
    'r2': r2_score(y_va, yp),
    'rmse': np.sqrt(mean_squared_error(y_va, yp)),
    'mae': mean_absolute_error(y_va, yp),
    'cv_rmse': np.sqrt(-xgb_gs.best_score_)
}
models['XGBoost (2017)'] = xgb_gs.best_estimator_
print(f"   CV RMSE: {results['XGBoost (2017)']['cv_rmse']:.6f}")

# ── 2. CatBoost (unchanged) ──
print(f">> CatBoost (2024) + 10-fold GridSearchCV...")
cb_gs = GridSearchCV(
    cb.CatBoostRegressor(random_state=SEED, verbose=0, allow_writing_files=False),
    {'iterations': [500, 1000], 'depth': [6, 8], 'learning_rate': [0.05, 0.1]},
    cv=10, n_jobs=1
)
cb_gs.fit(X_tr_n, y_tr)
yp = cb_gs.best_estimator_.predict(X_va_n)
results['CatBoost (2024)'] = {
    'r2': r2_score(y_va, yp),
    'rmse': np.sqrt(mean_squared_error(y_va, yp)),
    'mae': mean_absolute_error(y_va, yp),
    'cv_rmse': np.sqrt(-cb_gs.best_score_)
}
models['CatBoost (2024)'] = cb_gs.best_estimator_
print(f"   CV RMSE: {results['CatBoost (2024)']['cv_rmse']:.6f}")

# ── 3. RealMLP — OPTIMIZED ═══════════════════════════════════════════════
print(f">> RealMLP (NeurIPS 2024, tuned) + 10-fold CV...")
m_rmlp = RealMLP_TD_Regressor(
    random_state=SEED,
    n_epochs=300,
    n_hidden_layers=3,
    hidden_width=256,
    lr=0.001,
    p_drop=0.0,
    use_early_stopping=True,
    early_stopping_additive_patience=50
)
cv_scores_rmlp = cross_val_score(m_rmlp, X_tr_n, y_tr, cv=10,
                                  scoring='neg_mean_squared_error', n_jobs=1)
cv_rmse_rmlp = np.sqrt(-cv_scores_rmlp.mean())
m_rmlp.fit(X_tr_n, y_tr)
yp = m_rmlp.predict(X_va_n)
results['RealMLP (2024)'] = {
    'r2': r2_score(y_va, yp),
    'rmse': np.sqrt(mean_squared_error(y_va, yp)),
    'mae': mean_absolute_error(y_va, yp),
    'cv_rmse': cv_rmse_rmlp
}
models['RealMLP (2024)'] = m_rmlp
print(f"   CV RMSE: {cv_rmse_rmlp:.6f}")

# ── 4. TabM — OPTIMIZED ═════════════════════════════════════════════════
print(f">> TabM (2024, tuned) + 10-fold CV...")
m_tabm = TabM_D_Regressor(
    random_state=SEED,
    n_epochs=150,
    n_blocks=3,
    d_embedding=64,
    d_block=256,
    dropout=0.0
)
cv_scores_tabm = cross_val_score(m_tabm, X_tr_n, y_tr, cv=10,
                                  scoring='neg_mean_squared_error', n_jobs=1)
cv_rmse_tabm = np.sqrt(-cv_scores_tabm.mean())
m_tabm.fit(X_tr_n, y_tr)
yp = m_tabm.predict(X_va_n)
results['TabM (2024)'] = {
    'r2': r2_score(y_va, yp),
    'rmse': np.sqrt(mean_squared_error(y_va, yp)),
    'mae': mean_absolute_error(y_va, yp),
    'cv_rmse': cv_rmse_tabm
}
models['TabM (2024)'] = m_tabm
print(f"   CV RMSE: {cv_rmse_tabm:.6f}")

# ── 5. xRFM — OPTIMIZED ═════════════════════════════════════════════════
print(f">> xRFM (2025, tuned) + 10-fold CV...")
try:
    m_xrfm = XRFM_D_Regressor(
        random_state=SEED,
        kernel_type='l2',
        max_leaf_samples=128,
        time_limit_s=120.0
    )
    cv_scores_xrfm = cross_val_score(m_xrfm, X_tr_n, y_tr, cv=10,
                                      scoring='neg_mean_squared_error', n_jobs=1)
    cv_rmse_xrfm = np.sqrt(-cv_scores_xrfm.mean())
    m_xrfm.fit(X_tr_n, y_tr)
    yp = m_xrfm.predict(X_va_n)
    results['xRFM (2025)'] = {
        'r2': r2_score(y_va, yp),
        'rmse': np.sqrt(mean_squared_error(y_va, yp)),
        'mae': mean_absolute_error(y_va, yp),
        'cv_rmse': cv_rmse_xrfm
    }
    models['xRFM (2025)'] = m_xrfm
    print(f"   CV RMSE: {cv_rmse_xrfm:.6f}")
except Exception as e:
    print(f"   xRFM Error: {e}")

# ═══ 4. BENCHMARK SUMMARY TABLE ════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"{'Rank':<4s} {'Model':<20s} {'R2':>8s} {'RMSE':>8s} {'MAE':>8s} {'CV RMSE':>10s}")
print(f"{'='*70}")
sorted_models = sorted(results.items(), key=lambda x: -x[1]['r2'])
for rank, (k, v) in enumerate(sorted_models, 1):
    flag = ' <<< BEST' if rank == 1 else ''
    print(f"#{rank:<2d}  {k:<20s} {v['r2']:>8.6f} {v['rmse']:>8.6f} {v['mae']:>8.6f} {v['cv_rmse']:>10.6f}{flag}")

# ═══ 5. PREDICT ON NEW DATA ════════════════════════════════════════════════════

print(f"\n{'=' * 55}")
print("PREDICTION ON NEW DATA (Predictdata.csv)")
print(f"{'=' * 55}")

# Load only the first 13 columns (12 features + InteractionCoe)
df_pred = pd.read_csv(r'C:\Users\Rui Zhang\Desktop\Predictdata.csv', usecols=range(13))
X_pred = df_pred[fcols].values
X_pred_n = scaler.transform(X_pred)
print(f"  Predict samples: {len(df_pred)}")

pred_results = df_pred[fcols].copy()
if 'InteractionCoe' in df_pred.columns:
    pred_results['InteractionCoe_Actual'] = df_pred['InteractionCoe'].values

for model_name, model in models.items():
    y_pred = model.predict(X_pred_n)
    col_name = f'Pred_{model_name.split("(")[0].strip()}'
    pred_results[col_name] = y_pred
    print(f"  {model_name}: min={y_pred.min():.4f}, max={y_pred.max():.4f}")

# Save
out_path = r'C:\Users\Rui Zhang\Desktop\Finalcode\0702 Prediction Version\NewTrainData_Version\Predict_with_predictions_v4_newdata.csv'
pred_results.to_csv(out_path, index=False)
print(f"\n[OK] Saved to: {out_path}")
print("Done.")
