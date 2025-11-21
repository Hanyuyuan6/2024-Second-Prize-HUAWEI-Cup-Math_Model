clc; clear; close all; % 清除命令窗口，清除工作区变量，关闭所有图形窗口

%% 加载数据
% 从数据文件中加载风电机组数据
load('附件2-风电机组采集数据.mat');

%% 设置风电场参数
wind_farm = data_TS_WF.WF_1.WT; % 选择 WF_1 风电场的数据

num_turbines = 100; % 风机数量
total_time = 100; % 总时间（秒）

% 初始化调度指令数据和风速等参数的矩阵
power_schedule = zeros(total_time, num_turbines); % 调度指令数据
wind_speed = zeros(total_time, num_turbines); % 风速矩阵
pitch_angle = zeros(total_time, num_turbines); % 桨距角矩阵
omega_r = zeros(total_time, num_turbines); % 转速矩阵
power_out = zeros(total_time, num_turbines); % 实际输出功率矩阵

% 遍历每台风机，提取相应的数据
for turbine = 1:num_turbines
    power_schedule(:, turbine) = wind_farm{turbine}.inputs(1:total_time,1); % 提取调度指令数据 Pref
    wind_speed(:, turbine) = wind_farm{turbine}.inputs(1:total_time,2); % 提取风速数据
    pitch_angle(:, turbine) = wind_farm{turbine}.states(1:total_time,1); % 提取桨距角数据
    omega_r(:, turbine) = wind_farm{turbine}.states(1:total_time,2); % 提取转速数据
    power_out(:, turbine) = wind_farm{turbine}.outputs(1:total_time,3); % 提取实际输出功率数据
end

%% 初始化参数
P_max = 5e6; % 风机额定功率为 5MW (单位：瓦特)
Delta_P_max = 1e6; % 功率波动约束为 1MW (单位：瓦特)

% 累积疲劳损伤初始化
cumulative_damage_shaft_opt = zeros(total_time, num_turbines); % 优化后主轴累积疲劳损伤
cumulative_damage_tower_opt = zeros(total_time, num_turbines); % 优化后塔架累积疲劳损伤
cumulative_damage_shaft_avg = zeros(total_time, num_turbines); % 平均分配主轴累积疲劳损伤
cumulative_damage_tower_avg = zeros(total_time, num_turbines); % 平均分配塔架累积疲劳损伤

% 记录每一秒的功率分配
power_reference_opt = zeros(total_time, num_turbines); % 优化后的功率参考值
power_reference_avg = zeros(total_time, num_turbines); % 平均分配的功率参考值

% 材料参数
m_shaft = 10; % 主轴材料 Wohler 曲线斜率
C_shaft = 9.77e70; % 主轴材料常数
m_tower = 10; % 塔架材料 Wohler 曲线斜率
C_tower = 9.77e70; % 塔架材料常数
sigma_b = 5e7; % 材料在拉伸断裂时的最大载荷值（Pa）

% 风机和空气参数定义
rho = 1.225; % 空气密度 kg/m^3
R = 72.5; % 风轮半径 m
A = pi * R^2; % 风轮扫掠面积 m^2

%% 加载回归模型
load('mdl_Cp.mat', 'mdl_Cp'); % 加载功率系数 Cp 的回归模型
load('mdl_Ct.mat', 'mdl_Ct'); % 加载推力系数 Ct 的回归模型
% 加载特征均值和标准差，用于特征标准化
load('feature_mean_std.mat', 'feature_mean', 'feature_std'); % 包含 feature_mean 和 feature_std

%% 初始化每台风机的载荷历史（用于雨流计数）
window_size = 10; % 定义滑动窗口大小（秒）
T_shaft_history_opt = cell(num_turbines, 1);
F_tower_history_opt = cell(num_turbines, 1);
T_shaft_history_avg = cell(num_turbines, 1);
F_tower_history_avg = cell(num_turbines, 1);

for turbine = 1:num_turbines
    T_shaft_history_opt{turbine} = zeros(window_size, 1);
    F_tower_history_opt{turbine} = zeros(window_size, 1);
    T_shaft_history_avg{turbine} = zeros(window_size, 1);
    F_tower_history_avg{turbine} = zeros(window_size, 1);
end

%% 定义目标函数权重
w_shaft = 0.5; % 主轴疲劳损伤的权重
w_tower = 0.5; % 塔架疲劳损伤的权重

%% 优化求解
tic; % 开始计时

% 初始化累积疲劳损伤总和
cumulative_damage_shaft_opt_totals = zeros(num_turbines, 1);
cumulative_damage_tower_opt_totals = zeros(num_turbines, 1);
cumulative_damage_shaft_avg_totals = zeros(num_turbines, 1);
cumulative_damage_tower_avg_totals = zeros(num_turbines, 1);

% 遍历每一个时间步进行优化
for t = 1:total_time
    % 当前时刻的总调度指令功率
    P_total = sum(power_schedule(t, :));

    % 平均分配功率
    P_avg = P_total / num_turbines;
    power_reference_avg(t, :) = P_avg; % 记录平均分配的功率

    % 获取当前时刻每台风机的风速和其他参数
    V_t = wind_speed(t, :)';          % 风速 (100×1)
    pitch_t = pitch_angle(t, :)';     % 桨距角 (100×1)
    omega_t = omega_r(t, :)';         % 转速 (100×1)
    P_out_t = power_out(t, :)';       % 实际输出功率 (100×1)

    % 定义优化变量的上下界
    P_available = calculateAvailablePower(V_t); % 计算每台风机的最大可用功率
    ub = min(P_available, P_max * ones(num_turbines, 1)); % 上界
    lb = zeros(num_turbines, 1);                          % 下界

    % 定义优化目标函数：最小化加权后的总体疲劳损伤
    obj_fun = @(P) weightedSumFatigueDamage(P, V_t, pitch_t, omega_t, P_out_t, ...
        w_shaft, w_tower, sigma_b, m_shaft, C_shaft, m_tower, C_tower, ...
        mdl_Cp, mdl_Ct, feature_mean, feature_std, ...
        T_shaft_history_opt, F_tower_history_opt, window_size);

    % 线性等式约束：所有风机的功率分配总和等于 P_total
    Aeq = ones(1, num_turbines);
    beq = P_total;

    % 不等式约束：|P_i - P_avg| <= Delta_P_max
    nonlcon = @(P) powerFluctuationConstraint(P, P_avg, Delta_P_max);

    % 优化选项设置
    options = optimoptions('fmincon', 'Display', 'off', ...     % 不显示迭代过程
        'Algorithm', 'sqp', ...              % 选择 sequential quadratic programming 算法
        'MaxIterations', 1000, ...            % 最大迭代次数
        'StepTolerance', 1e-6, ...           % 步长容忍度
        'FunctionTolerance', 1e-6, ...       % 函数容忍度
        'MaxFunctionEvaluations', 1e5);      % 最大函数评估次数

    % 初始值设为平均分配功率
    P0 = 1 * ones(num_turbines, 1);

    % 调用 fmincon 函数进行优化求解
    [P_opt, ~] = fmincon(obj_fun, P0, [], [], Aeq, beq, lb, ub, nonlcon, options);

    % 记录优化后的功率分配
    power_reference_opt(t, :) = P_opt';

    % 估算优化后的主轴扭矩和塔架推力
    [T_shaft_opt, F_tower_opt] = estimateLoads(P_opt, V_t, pitch_t, omega_t, P_out_t, ...
        mdl_Cp, mdl_Ct, feature_mean, feature_std);

    % 估算平均分配的主轴扭矩和塔架推力
    [T_shaft_avg, F_tower_avg] = estimateLoads(P_avg * ones(num_turbines,1), V_t, pitch_t, omega_t, P_out_t, ...
        mdl_Cp, mdl_Ct, feature_mean, feature_std);

    % 更新载荷历史，并进行雨流计数和损伤计算
    Di_shaft_opt = zeros(num_turbines, 1);
    Di_tower_opt = zeros(num_turbines, 1);
    Di_shaft_avg = zeros(num_turbines, 1);
    Di_tower_avg = zeros(num_turbines, 1);

    for turbine = 1:num_turbines
        %% 优化后处理 - 主轴
        % 更新载荷历史
        T_shaft_history_opt{turbine} = [T_shaft_history_opt{turbine}(2:end); T_shaft_opt(turbine)];

        % 主轴扭矩雨流计数
        [cycles_shaft, amplitudes_shaft, means_shaft, ~] = rainflowCounting(T_shaft_history_opt{turbine});
        % Goodman 修正
        L_shaft = applyGoodmanCorrection(amplitudes_shaft, means_shaft, sigma_b);
        % 疲劳寿命
        Nfi_shaft = C_shaft ./ (L_shaft .^ m_shaft);
        % 损伤增量
        if ~isempty(cycles_shaft) && ~isempty(Nfi_shaft)
            Di_shaft_opt(turbine) = sum(cycles_shaft ./ Nfi_shaft);
        else
            Di_shaft_opt(turbine) = 0;
        end

        %% 优化后处理 - 塔架
        % 更新载荷历史
        F_tower_history_opt{turbine} = [F_tower_history_opt{turbine}(2:end); F_tower_opt(turbine)];

        % 塔架推力雨流计数
        [cycles_tower, amplitudes_tower, means_tower, ~] = rainflowCounting(F_tower_history_opt{turbine});
        % Goodman 修正
        L_tower = applyGoodmanCorrection(amplitudes_tower, means_tower, sigma_b);
        % 疲劳寿命
        Nfi_tower = C_tower ./ (L_tower .^ m_tower);
        % 损伤增量
        if ~isempty(cycles_tower) && ~isempty(Nfi_tower)
            Di_tower_opt(turbine) = sum(cycles_tower ./ Nfi_tower);
        else
            Di_tower_opt(turbine) = 0;
        end

        %% 平均分配处理 - 主轴
        % 更新载荷历史
        T_shaft_history_avg{turbine} = [T_shaft_history_avg{turbine}(2:end); T_shaft_avg(turbine)];

        % 主轴扭矩雨流计数
        [cycles_shaft_avg, amplitudes_shaft_avg, means_shaft_avg, ~] = rainflowCounting(T_shaft_history_avg{turbine});
        % Goodman 修正
        L_shaft_avg = applyGoodmanCorrection(amplitudes_shaft_avg, means_shaft_avg, sigma_b);
        % 疲劳寿命
        Nfi_shaft_avg = C_shaft ./ (L_shaft_avg .^ m_shaft);
        % 损伤增量
        if ~isempty(cycles_shaft_avg) && ~isempty(Nfi_shaft_avg)
            Di_shaft_avg(turbine) = sum(cycles_shaft_avg ./ Nfi_shaft_avg);
        else
            Di_shaft_avg(turbine) = 0;
        end

        %% 平均分配处理 - 塔架
        % 更新载荷历史
        F_tower_history_avg{turbine} = [F_tower_history_avg{turbine}(2:end); F_tower_avg(turbine)];

        % 塔架推力雨流计数
        [cycles_tower_avg, amplitudes_tower_avg, means_tower_avg, ~] = rainflowCounting(F_tower_history_avg{turbine});
        % Goodman 修正
        L_tower_avg = applyGoodmanCorrection(amplitudes_tower_avg, means_tower_avg, sigma_b);
        % 疲劳寿命
        Nfi_tower_avg = C_tower ./ (L_tower_avg .^ m_tower);
        % 损伤增量
        if ~isempty(cycles_tower_avg) && ~isempty(Nfi_tower_avg)
            Di_tower_avg(turbine) = sum(cycles_tower_avg ./ Nfi_tower_avg);
        else
            Di_tower_avg(turbine) = 0;
        end
    end

    % 更新累积疲劳损伤总和
    cumulative_damage_shaft_opt_totals = cumulative_damage_shaft_opt_totals + Di_shaft_opt;
    cumulative_damage_tower_opt_totals = cumulative_damage_tower_opt_totals + Di_tower_opt;
    cumulative_damage_shaft_avg_totals = cumulative_damage_shaft_avg_totals + Di_shaft_avg;
    cumulative_damage_tower_avg_totals = cumulative_damage_tower_avg_totals + Di_tower_avg;

    % 记录累积疲劳损伤
    cumulative_damage_shaft_opt(t, :) = cumulative_damage_shaft_opt_totals';
    cumulative_damage_tower_opt(t, :) = cumulative_damage_tower_opt_totals';
    cumulative_damage_shaft_avg(t, :) = cumulative_damage_shaft_avg_totals';
    cumulative_damage_tower_avg(t, :) = cumulative_damage_tower_avg_totals';

    % 显示优化进度，每完成10秒显示一次
    if mod(t, 10) == 0
        fprintf('已完成 %d 秒的优化计算。\n', t);
    end
end

toc; % 结束计时

%% 结果可视化

% 定义视频文件的完整路径
videoFileName = fullfile(pwd, 'question_3_基于粒子群优化.avi');

% 创建 VideoWriter 对象
v = VideoWriter(videoFileName, 'Uncompressed AVI'); % 你可以选择其他格式，如'MPEG-4'
v.FrameRate = 4; % 设置帧率，可以根据需要调整
open(v); % 打开视频文件准备写入

% 创建一个新的图形窗口
figure('Name', '功率分配动态展示', 'NumberTitle', 'off');

% 绘制功率分配的动态条形图
for t = 1:total_time
    % 绘制条形图
    bar(power_reference_opt(t, :) / 1e6, 'FaceColor', [0.2 0.6 0.8]); % 将功率转换为MW显示
    xlabel('风机编号');
    ylabel('功率参考值 (MW)');
    title(sprintf('第 %d 秒功率分配优化结果', t));
    ylim([0, P_max / 1e6 + 1]); % 设置y轴范围
    
    % 捕获当前图形并写入视频
    frame = getframe(gcf);
    writeVideo(v, frame);
    
    % 更新图形
    drawnow;
    
    % 控制动画播放速度
    pause(0.25);
end

% 关闭视频文件
close(v);

% 关闭图形窗口
close(gcf);


% 优化后总累积疲劳损伤
total_damage_shaft_opt = sum(cumulative_damage_shaft_opt, 2);    % 优化后主轴总损伤
total_damage_tower_opt = sum(cumulative_damage_tower_opt, 2);    % 优化后塔架总损伤
total_damage_shaft_avg = sum(cumulative_damage_shaft_avg, 2);    % 平均分配主轴总损伤
total_damage_tower_avg = sum(cumulative_damage_tower_avg, 2);    % 平均分配塔架总损伤

% 分别计算加权后的总体疲劳损伤
weighted_total_damage_opt = w_shaft * total_damage_shaft_opt + w_tower * total_damage_tower_opt;
weighted_total_damage_avg = w_shaft * total_damage_shaft_avg + w_tower * total_damage_tower_avg;

% 绘制加权后的总体疲劳损伤对比图
figure('Name', '优化前后加权总体累积疲劳损伤对比', 'NumberTitle', 'off');
plot(1:total_time, weighted_total_damage_opt, 'r-', 'LineWidth', 1.5); hold on; % 优化后加权总损伤
plot(1:total_time, weighted_total_damage_avg, 'b--', 'LineWidth', 1.5); % 平均分配加权总损伤
xlabel('时间 (s)');
ylabel('加权总体累积疲劳损伤');
legend({'加权总损伤（优化后）', '加权总损伤（平均分配）'});
title('优化前后加权总体累积疲劳损伤对比');
grid on;

% 优化前后功率参考值的方差对比
variance_opt = var(power_reference_opt, 0, 2); % 优化后每秒功率分配的方差
variance_avg = var(power_reference_avg, 0, 2); % 平均分配时功率分配的方差

% 绘制功率分配方差对比图
figure('Name', '优化前后功率分配方差对比', 'NumberTitle', 'off');
plot(1:total_time, variance_opt / 1e12, 'g-', 'LineWidth', 1.5); hold on; % 优化后方差 (转换为MW²)
plot(1:total_time, variance_avg / 1e12, 'k--', 'LineWidth', 1.5); % 平均分配方差
xlabel('时间 (s)');
ylabel('功率分配方差 (MW^2)');
legend({'优化后', '平均分配'});
title('优化前后功率分配方差对比');
grid on;

% 选择5个有代表性的风机，展示累积疲劳损伤的增长过程
selected_turbines = [5, 20, 40, 60, 80]; % 选择的风机编号
% 绘制优化后选定风机的累积疲劳损伤增长过程
figure('Name', '选定风机的累积疲劳损伤增长过程', 'NumberTitle', 'off');
for i = 1:length(selected_turbines)
    turbine = selected_turbines(i);
    single_total_damage_opt = w_shaft * cumulative_damage_shaft_opt(:, turbine) + w_tower * cumulative_damage_tower_opt(:, turbine);
    single_total_damage_avg = w_shaft * cumulative_damage_shaft_avg(:, turbine) + w_tower * cumulative_damage_tower_avg(:, turbine);
    subplot(length(selected_turbines), 1, i); % 创建子图
    plot(1:total_time, single_total_damage_opt, 'r-', 'LineWidth', 1.5); hold on; % 优化后加权总损伤
    plot(1:total_time, single_total_damage_avg, 'b--', 'LineWidth', 1.5); % 平均分配加权总损伤
    grid on;
    xlabel('时间 (s)');
    ylabel('累积疲劳损伤');
    legend({'优化后', '平均分配'});
    title(['风机 ', num2str(turbine), ' 选定风机的累积疲劳损伤增长']);
    grid on;
end


%% 辅助函数定义

% 优化目标函数：最小化加权后的总体疲劳损伤
function weighted_damage = weightedSumFatigueDamage(P, V, pitch, omega_r, P_out, ...
    w_shaft, w_tower, sigma_b, m_shaft, C_shaft, m_tower, C_tower, ...
    mdl_Cp, mdl_Ct, feature_mean, feature_std, ...
    T_shaft_history_opt, F_tower_history_opt, ~)

% 确保P是列向量
P = P(:);

num_turbines = length(P);

% 根据功率 P 和风速 V，估算应力/载荷
[T_shaft, F_tower] = estimateLoads(P, V, pitch, omega_r, P_out, ...
    mdl_Cp, mdl_Ct, feature_mean, feature_std);

% 初始化损伤增量
Di_shaft_opt = zeros(num_turbines, 1);
Di_tower_opt = zeros(num_turbines, 1);

for turbine = 1:num_turbines
    %% 优化后处理 - 主轴
    % 更新载荷历史
    T_shaft_history_opt{turbine} = [T_shaft_history_opt{turbine}(2:end); T_shaft(turbine)];

    % 主轴扭矩雨流计数
    [cycles_shaft, amplitudes_shaft, means_shaft, ~] = rainflowCounting(T_shaft_history_opt{turbine});
    % Goodman 修正
    L_shaft = applyGoodmanCorrection(amplitudes_shaft, means_shaft, sigma_b);
    % 疲劳寿命
    Nfi_shaft = C_shaft ./ (L_shaft .^ m_shaft);
    % 损伤增量
    if ~isempty(cycles_shaft) && ~isempty(Nfi_shaft)
        Di_shaft_opt(turbine) = sum(cycles_shaft ./ Nfi_shaft);
    else
        Di_shaft_opt(turbine) = 0;
    end

    %% 优化后处理 - 塔架
    % 更新载荷历史
    F_tower_history_opt{turbine} = [F_tower_history_opt{turbine}(2:end); F_tower(turbine)];

    % 塔架推力雨流计数
    [cycles_tower, amplitudes_tower, means_tower, ~] = rainflowCounting(F_tower_history_opt{turbine});
    % Goodman 修正
    L_tower = applyGoodmanCorrection(amplitudes_tower, means_tower, sigma_b);
    % 疲劳寿命
    Nfi_tower = C_tower ./ (L_tower .^ m_tower);
    % 损伤增量
    if ~isempty(cycles_tower) && ~isempty(Nfi_tower)
        Di_tower_opt(turbine) = sum(cycles_tower ./ Nfi_tower);
    else
        Di_tower_opt(turbine) = 0;
    end
end

% 计算总疲劳损伤
total_damage_shaft = sum(Di_shaft_opt);
total_damage_tower = sum(Di_tower_opt);

% 计算加权后的总体疲劳损伤
weighted_damage = w_shaft * total_damage_shaft + w_tower * total_damage_tower;
end

% 功率波动约束函数
function [c, ceq] = powerFluctuationConstraint(P, P_avg, Delta_P_max)
% 非线性不等式约束：|P_i - P_avg| <= Delta_P_max
c = abs(P - P_avg) - Delta_P_max; % 若 c <= 0，则满足约束
ceq = []; % 无非线性等式约束
end

% 应用 Goodman 曲线修正
function L = applyGoodmanCorrection(Sai, Smi, sigma_b)
% 防止分母为零或接近零
epsilon = 1e-6;
L = Sai ./ (1 - Smi / sigma_b + epsilon);
end

% 估算应力/载荷（主轴扭矩和塔架推力），使用回归模型
function [T_shaft, F_tower] = estimateLoads(P, V, pitch, omega_r, P_out, ...
    mdl_Cp, mdl_Ct, feature_mean, feature_std)
% 计算叶尖速比 λ
lambda = omega_r .* 72.5 ./ V; % R = 72.5 m

% 计算功率差值
power_diff = P - P_out;

% 构建特征矩阵，包括叶尖速比、桨距角、功率和功率差
data = [lambda, pitch, P, power_diff];

% 标准化特征矩阵
features_standardized = (data - feature_mean) ./ feature_std;

% 使用回归模型预测功率系数 Cp 和推力系数 Ct
Cp_predicted = predict(mdl_Cp, features_standardized);
Ct_predicted = predict(mdl_Ct, features_standardized);

% 限制 Cp 和 Ct 的值在合理范围内 [0, 1]
Cp_predicted(Cp_predicted < 0) = 0;
Cp_predicted(Cp_predicted > 1) = 1;
Ct_predicted(Ct_predicted < 0) = 0;
Ct_predicted(Ct_predicted > 1) = 1;

% 估算主轴扭矩 T_shaft
rho = 1.225; % 空气密度 kg/m^3
A = pi * 72.5^2; % 扫风面积 m^2
P_estimated = 0.5 * rho * A .* Cp_predicted .* V .^ 3; % 估算功率
T_shaft = P_estimated ./ omega_r; % 通过功率和转速计算扭矩
% 处理无效值（NaN 或 Inf）
T_shaft(isnan(T_shaft) | isinf(T_shaft)) = 0;

% 估算塔架推力 F_tower
F_tower = 0.5 * rho * A .* Ct_predicted .* V .^ 2;
end

% 雨流计数法函数（使用基本雨流计数思路）
function [cycles, amplitudes, means, times] = rainflowCounting(signal)
signal = signal(:)';  % 确保信号为行向量
signal = signal([true, diff(signal) ~= 0]);  % 去除连续重复点

extrema = getExtrema(signal);  % 获取极值点

% 获取极值点对应的时间索引
time_indices = 1:length(extrema);

% 初始化变量
cycles = [];       % 存储循环次数
amplitudes = [];   % 存储循环幅值
means = [];        % 存储循环均值
times = [];        % 存储循环发生的时间索引

% 循环识别和移除循环
while length(extrema) >= 3
    idx = 1;  % 起始索引
    while idx <= length(extrema) - 2
        S1 = extrema(idx);
        S2 = extrema(idx + 1);
        S3 = extrema(idx + 2);

        delta_S1 = abs(S1 - S2);
        delta_S2 = abs(S2 - S3);

        if delta_S1 <= delta_S2
            % 识别出一个有效循环
            Sai = delta_S1;  % 载荷幅值
            Smi = (S1 + S2) / 2;  % 载荷均值

            % 循环发生的时间索引，取参与循环点的较大时间索引
            cycle_time = max(time_indices(idx), time_indices(idx + 1));

            % 记录循环数据
            cycles(end + 1) = 1;  % 完整循环计为1
            amplitudes(end + 1) = Sai;
            means(end + 1) = Smi;
            times(end + 1) = cycle_time;  % 记录循环发生的时间索引

            % 移除已识别的循环点（S1 和 S2）
            extrema(idx:idx+1) = [];
            time_indices(idx:idx+1) = [];

            % 在移除后，重新从序列起始位置开始
            idx = 1;
        else
            % 未识别出循环，索引加1，继续向前
            idx = idx + 1;
        end
    end
    % 当无法再识别循环时，退出循环
    break;
end
end

% 获取信号中的峰值和谷值
function extrema = getExtrema(signal)
extrema = [];

for i = 2:length(signal) - 1
    % 找到波峰或波谷，并记录下来
    if (signal(i) > signal(i-1) && signal(i) > signal(i+1)) || ...
            (signal(i) < signal(i-1) && signal(i) < signal(i+1))
        extrema(end + 1) = signal(i);  % 识别波峰或波谷
    end
end

% 添加起点和终点作为极值点
extrema = [signal(1), extrema, signal(end)];
end

% 计算风机的最大可用功率
function P_avail = calculateAvailablePower(V)
rho = 1.225; % 空气密度 kg/m^3
R = 72.5; % 风轮半径 m
A = pi * R^2; % 风轮扫掠面积 m^2
Cp_max = 0.48; % 最大功率系数（根据实际情况调整）
P_rated = 5e6; % 额定功率 W
V_rated = 11.2; % 额定风速 m/s

% 计算未限幅的功率
P_raw = 0.5 * rho * A * Cp_max .* V .^ 3; % P = 0.5 * ρ * A * Cp * V^3
% 限制功率不超过额定功率
P_avail = min(P_raw, P_rated * ones(size(V)));

% 当风速超过额定风速时，功率保持在额定功率不再增加
P_avail(V > V_rated) = P_rated;
end