# 2024年全国研究生数学建模大赛A题：风电场有功功率优化调度

本仓库包含针对 2024 年华为杯（研究生全国数学建模大赛） A 题（风电场有功功率优化调度）四个问题的完整代码、数据与结果。各部分按问题组织，便于复现与查看。

## 运行说明
- matlab 版本 - R2024a。
- 运行代码前请根据提示安装所需的优化工具和 MATLAB 依赖包。
- 请将所有代码与数据文件放在同一目录下，避免路径问题导致文件无法访问。 
- MATLAB 数据加载注意：将 `附件3-噪声和延迟作用下的采集数据.mat` 导入 MATLAB 后，文件内变量名可能会自动更名为 `data_TS_WF_noise`，请在代码中检查并使用正确的变量名。

## 目录结构与文件说明
仓库按竞赛题目分为四个主要子目录（对应四个问题），每个子目录包含对应问题的代码与所需数据。

- Question 1
  - `question_1.m`：问题一核心代码。

- Question 2
  - `question_2.m`：问题二核心代码。
  - `question_2_tradition.m`：基于物理机理的对照模型。
  - `for_save.m`：生成并保存 100 台风机的应力/扭矩预测结果（用于附件 6）。
  - `Ct_Cp_model_from_Q2.m`：基于问题二模型的多元线性回归，用于保存 Ct 与 Cp。

- Question 3
  - `question_3.m`：问题三核心代码。
  - `question_3_local_OPT.m`：用于演示陷入局部最优的对照模型。
  - 绘图用数据文件（示例）：
    - `single_total_damage_avg_total_3.mat`：5 台风机累积疲劳损伤（平均功率分配）。
    - `single_total_damage_opt_total_3.mat`：5 台风机累积疲劳损伤（优化功率分配）。
    - `variance_avg_3.mat` / `variance_opt_3.mat`：100 台风机方差（平均 / 优化）。
    - `weighted_total_damage_avg_3.mat` / `weighted_total_damage_opt_3.mat`：100 台风机总累积疲劳损伤（平均 / 优化）。

- Question 4
  - `question_4.m`：问题四核心代码。
  - `Calcul_state_transfer.m`：根据附件 1 计算合理的状态转移矩阵。
  - `PCA_for_weight.m`：基于附件 2（无噪声数据）计算累积疲劳损伤变量的权重。
  - `compare_in_different_condition.m`：比较模型在不同噪声/延迟条件下的表现。
  - 绘图用数据文件（示例）：
    - `single_total_damage_avg_total_4.mat` / `single_total_damage_opt_total_4.mat`
    - `variance_avg_4.mat` / `variance_opt_4.mat`
    - `weighted_total_damage_avg_4.mat` / `weighted_total_damage_opt_4.mat`

## 其他数据与结果
- 动画结果（演示）：
  - `question_3_基于粒子群优化.avi`
  - `question_3_陷入局部最优(平均).avi`
  - `question_4_优化功率分配动态展示.avi`

- 计算中间结果与模型文件：
  - `mdl_Ct.mat`、`mdl_Cp.mat`、`feature_mean_std.mat`（由 `for_save.m` 生成）。
  - `Torque_Thrust_Matrices.mat`（附件 6 数据）。
  - `transition_matrix_shaft.mat`、`transition_matrix_tower.mat`（由 `Calcul_state_transfer.m` 生成）。

- 预测与滤波模型结果：
  - `results_baseline.mat`：无滤波与预测。
  - `results_arima.mat`：仅 ARIMA 预测。
  - `results_kalman.mat`：仅卡尔曼滤波。
  - `results_full.mat`：滤波与预测结合的结果。

## 题目数据与提交文件
- 题目数据：
  - `附件1-疲劳评估数据.xls`
  - `附件2-风电机组采集数据.mat`
  - `附件3-噪声和延迟作用下的采集数据.mat`
  - `附件4-噪声和延迟作用下的采集数据.xlsx`

- 答案与提交表：
  - `附件5-问题一答案表.xlsx`
  - `附件6-问题二答案表.xlsx`

## 使用建议与注意事项
- 建议先阅读各 `question_x.m` 的顶部注释以了解运行顺序与输入输出。  
- 若遇到路径或变量名问题，请确认当前工作目录包含所有数据文件，或修改代码中的路径变量以指向正确位置。  
- 对于大型 .mat 文件，加载后请检查变量名。  