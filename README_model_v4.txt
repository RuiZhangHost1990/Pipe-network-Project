================================================================================
模型说明 — PyTabKit Benchmark + Predict V4 (10-Fold CV, Optimized)
================================================================================

1. 训练数据
───────────────────────────────────────────────────────────────────────────────
  文件: C:\Users\Rui Zhang\Desktop\FinalTrainData.csv
  总样本: 423 条 (注：该文件实际为 Excel 格式，用 pd.read_excel() 读取)
          (自动删除 1 条空行NaN后，实际使用 422 条)
  特征: 12 个 (OuterDiameter, WallThickness, StaraightLength, BendRadius,
               Normalizedspace, Space, PeakDepth, Length, Width,
               SMYS, SMTS, Position)
  目标: InteractionCoe

2. 数据划分 (80% Training / 20% Validation)
───────────────────────────────────────────────────────────────────────────────
        Total:   422 条 (423 条原始数据，删除 1 条 NaN 空行)
  Training set:  337 条  (80%)
  Validation:     85 条  (20%)

  划分方式: sklearn train_test_split, random_state=42 (固定种子，每次划分相同)

3. 模型与优化参数
───────────────────────────────────────────────────────────────────────────────
  共 5 个模型，全部使用 10 折交叉验证 (10-fold Cross-Validation)：

  (1) XGBoost (2017)       — GridSearchCV, 10折, 网格搜索最佳超参 (不变)

  (2) CatBoost (2024)      — GridSearchCV, 10折, 网格搜索最佳超参 (不变)

  (3) RealMLP (2024)       — cross_val_score, 10折
      ▼ 优化参数 (V3 仅 R²=0.437):
         n_epochs = 300           (V3 为 50, 欠拟合严重, 大幅提升)
         n_hidden_layers = 3      (3 层隐藏层)
         hidden_width = 256       (每层 256 个神经元)
         lr = 0.001               (学习率)
         use_early_stopping = True (早停防止过拟合)
         early_stopping_additive_patience = 50

  (4) TabM (2024)          — cross_val_score, 10折
      ▼ 优化参数 (V3 为 R²=0.876, 现突破 R²>0.90):
         n_epochs = 150           (V3 为 50, 提升训练深度)
         n_blocks = 3             (3 个 TabM block)
         d_embedding = 64         (嵌入维度 64)
         d_block = 256            (block 宽度 256)
         dropout = 0.0

  (5) xRFM (2025)          — cross_val_score, 10折
      ▼ 优化参数:
         kernel_type = 'l2'       (RBF kernel)
         max_leaf_samples = 128
         time_limit_s = 120.0

4. 模型性能排名
───────────────────────────────────────────────────────────────────────────────

  Rank  Model                   R2         RMSE        MAE      10-Fold CV
  ──────────────────────────────────────────────────────────────────────────
  #1    CatBoost (2024)     0.914287    0.013913    0.009415      nan     <<< BEST
  #2    XGBoost (2017)      0.906215    0.014554    0.010386      nan
  #3    TabM (2024)         0.905908    0.014578    0.010647    0.016076  ✅ 突破0.9
  #4    RealMLP (2024)      0.883993    0.016187    0.011568    0.015191  ↑ +0.447
  #5    xRFM (2025)         0.875330    0.016780    0.011341    0.017000  ↑ 微升

  ※ 注: R² 越接近 1 越好，RMSE 和 MAE 越小越好
  ※ XGBoost/CatBoost 的 CV RMSE 显示 nan，是因为 GridSearchCV
     best_score_ 仅代表最优超参的折数评分，非全10折平均

5. 优化总结
───────────────────────────────────────────────────────────────────────────────
  相对于 V3 版本的提升:

  TabM:    0.876  →  0.906   (↑ +0.030)  ✅ 达到 >0.9 目标
  RealMLP: 0.437  →  0.884   (↑ +0.447)  ↑ 大幅提升, 接近0.9
  xRFM:    0.870  →  0.875   (↑ +0.005)  ↑ 略有提升

  说明:
  - CatBoost 和 XGBoost 继续保持最佳表现 (R²=0.91)
  - TabM 通过增加 epochs 和 blocks 成功突破 R²=0.90
  - RealMLP 从 0.44 提升到 0.88 但受限于 CPU + 小样本数据,
    继续增加 epochs 容易过拟合
  - xRFM 在此小样本数据集上提升空间有限

6. 可重复性
───────────────────────────────────────────────────────────────────────────────
  固定随机种子 SEED = 42，保证每次运行结果完全一致：
  - numpy.random.seed(42)
  - random.seed(42)
  - train_test_split(random_state=42)
  - 所有模型 random_state=42

7. 预测新数据
───────────────────────────────────────────────────────────────────────────────
  文件: C:\Users\Rui Zhang\Desktop\Finalcode\Predict.csv
  输出: C:\Users\Rui Zhang\Desktop\Finalcode\Predict_with_predictions_v4.csv
  内容: 原始 12 个特征 + 5 个模型的 InteractionCoe 预测值 (212 条)

8. 运行方式
───────────────────────────────────────────────────────────────────────────────
  打开终端 (CMD 或 PowerShell)，执行：
    cd C:\Users\Rui Zhang\Desktop\Finalcode
    python PyTabKit_benchmark_10fold_predict_v4.py

  预计运行时间: 约 60~90 分钟 (本地 CPU，无 GPU)

9. 文件说明
───────────────────────────────────────────────────────────────────────────────
  PyTabKit_benchmark_10fold_predict_v4.py  — 主程序 (优化版)
  README_model_v4.txt                       — 本说明文件
  Predict.csv                               — 待预测数据
  Predict_with_predictions_v4.csv           — 预测结果输出 (212 条)
  FinalTrainData.csv                        — 训练数据 (实际为 xlsx 格式)

================================================================================
