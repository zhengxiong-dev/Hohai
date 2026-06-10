# AFA-YOLO11 模型改进方案

## 1. 研究方向

建议将模型改进方向确定为：

> 基于跨尺度对齐与小目标感知自适应融合的无人机小目标检测算法研究

模型可命名为 **AFA-YOLO11**，即 **Aligned Fusion Adaptive YOLO11**。

该方向不是简单地在 YOLO11s 上堆叠 P2、BiFPN、注意力模块，而是围绕无人机小目标检测中的核心问题进行统一设计：

- 小目标像素占比低，深层特征中位置信息容易丢失；
- 浅层特征分辨率高但语义弱，深层特征语义强但空间粗糙；
- 常规 FPN/PAN 上采样后直接拼接，容易产生跨尺度空间错位；
- 不同区域对尺度特征的需求不同，固定权重或简单 concat 难以适应复杂场景；
- 小目标边界框对微小位置偏移非常敏感，传统 IoU 损失对小框不够稳定。

因此，本文不应表述为“引入若干模块”，而应表述为：

> 本文提出一种小目标感知的跨尺度对齐融合检测网络，通过浅层细节特征引导深层语义特征空间重校准，并根据小目标响应动态分配多尺度融合权重，从而提升无人机航拍图像中密集小目标的检测精度与定位稳定性。

## 2. 基线模型

当前项目基于 **Ultralytics YOLO11s**，原始 UAV-YOLO 的主要结构为：

```text
YOLO11s
+ P2 检测头
+ BiFPN-style weighted fusion
+ EMA attention
```

该结构具有一定效果，但从论文创新角度看，P2、BiFPN 和 EMA 都属于已有常见模块，直接组合容易显得偏“模块拼接”。

改进后的 AFA-YOLO11 建议采用：

```text
YOLO11s
+ P2 小目标检测层
+ SGFA 小目标引导软对齐模块
+ SOA-SAFM 小目标感知空间自适应融合模块
+ SA-NWD 尺度自适应小目标定位损失（进阶选做）
```

其中：

- P2 是基础结构改进，不作为主要创新；
- SGFA 是核心创新 1；
- SOA-SAFM 是核心创新 2；
- SA-NWD 是进阶增强项，用于结构实验稳定后进一步提升小目标定位稳定性，不作为最低可交付版本的必要条件。

## 3. 总体网络结构

AFA-YOLO11 的整体结构如下：

```text
Input Image
    |
YOLO11s Backbone
    |
Multi-scale Features: P2, P3, P4, P5
    |
Small-object Response Generation
    |
SGFA: Small-object Guided Feature Alignment
    |
SOA-SAFM: Small-object Aware Spatial Adaptive Fusion
    |
Four-scale Detection Head: Detect(P2, P3, P4, P5)
    |
YOLO Loss
    |
Optional: SA-NWD Auxiliary Localization Loss
```

推荐保留四尺度检测：

```text
P2: stride 4,  160 x 160 for 640 input
P3: stride 8,   80 x 80
P4: stride 16,  40 x 40
P5: stride 32,  20 x 20
```

P2 分支用于保留小目标的高分辨率空间细节，是后续对齐与融合模块的基础。

## 4. 创新点一：SGFA 小目标引导软对齐模块

### 4.1 设计动机

常规 FPN/PAN 的融合方式通常是：

```text
Upsample(P5) + P4
Upsample(P4) + P3
Upsample(P3) + P2
```

这只能完成特征图尺寸对齐，但不能保证目标区域、边界位置和语义区域真正对齐。对于无人机小目标，目标本身只有少量像素，轻微错位就可能导致分类和定位性能下降。

因此，本文优先设计 **SGFA**，即 **Small-object Guided Feature Alignment Module**。该模块不在第一版中使用 `grid_sample` 或可变形卷积，而是采用更稳定的注意力式软对齐方式：利用浅层高分辨率特征和小目标响应图生成空间门控，对上采样后的深层语义特征进行区域级重标定。

这样可以保留“跨尺度对齐”的论文动机，同时显著降低实现风险和训练不稳定风险。

### 4.2 模块输入输出

输入：

```text
F_low:  浅层高分辨率特征，例如 P2 / P3
F_high: 深层低分辨率语义特征，例如 P3 / P4 / P5
M_small: 小目标响应图
```

输出：

```text
F_align: 软对齐融合后的跨尺度特征
```

### 4.3 模块流程

```text
1. F_high 上采样到 F_low 的空间尺寸
2. F_low 与 F_high_up 通过 1x1 Conv 统一通道数
3. 由 F_low 和 M_small 生成 small-object guide
4. 根据 [F_low, F_high_up, M_small] 预测空间对齐门控 A
5. 使用 A 对 F_high_up 进行软重标定
6. 将 F_low 与软对齐后的 F_high_soft 融合
```

可写成伪公式：

```text
F_high_up = Upsample(F_high)
F_low_c = Conv1x1(F_low)
F_high_c = Conv1x1(F_high_up)
G = Conv3x3([F_low_c, M_small])
A = Sigmoid(Conv3x3([F_low_c, F_high_c, G]))
F_high_soft = F_high_c * A
F_out = Conv1x1([F_low_c, F_high_soft])
```

其中 `A` 是空间对齐门控图。它不是直接预测几何偏移，而是通过小目标响应引导深层特征在目标区域增强、在背景区域抑制。

进阶版本可以将 `A` 扩展为 offset，并使用 `grid_sample` 或 deformable convolution 做几何对齐。但该版本建议作为后续增强实验，不作为第一版核心实现。

### 4.4 原创性表述

不要将该模块表述为“加入注意力模块”。更合适的表述是：

> 本文提出小目标引导的跨尺度对齐模块，利用浅层高分辨率特征中的空间细节与小目标响应图共同预测跨尺度偏移量，对深层语义特征进行空间重校准，从而缓解传统上采样融合中的位置错位问题。

建议进一步调整为更符合软对齐实现的表述：

> 本文提出小目标引导软对齐模块，利用浅层高分辨率特征和小目标响应图生成空间对齐门控，对上采样后的深层语义特征进行区域级重标定，从而缓解传统跨尺度融合中语义区域不匹配和背景干扰问题。

该设计与普通空间注意力的区别在于：

- 不是单独增强某一层特征；
- 而是用于高低层特征融合前的跨尺度空间对齐；
- 对齐门控由浅层细节特征和小目标响应共同引导；
- 服务目标明确：提升无人机场景中小目标的跨尺度融合质量。

## 5. 创新点二：SOA-SAFM 小目标感知空间自适应融合模块

### 5.1 设计动机

不同空间区域对尺度特征的需求不同：

```text
小目标区域：更依赖 P2/P3 的高分辨率细节
中大目标区域：更依赖 P4/P5 的语义信息
背景区域：应抑制无效尺度响应
```

原项目中的 `BiFPN_Concat` 使用全局可学习权重，无法针对不同空间位置动态调整尺度贡献。为此，本文设计 **SOA-SAFM**，即 **Small-object Aware Spatial Adaptive Fusion Module**。

### 5.2 小目标响应图

先由浅层特征生成小目标响应图：

```text
M_small = Sigmoid(Conv3x3([P2, Upsample(P3)]))
```

该响应图用于提示哪些区域更可能包含小目标，从而引导后续尺度权重分配。

### 5.3 空间自适应融合

对某个输出尺度 `Pk`，先将多尺度特征 resize 到相同尺寸：

```text
P2 -> resize to Pk
P3 -> resize to Pk
P4 -> resize to Pk
P5 -> resize to Pk
```

然后生成空间尺度权重：

```text
W = Softmax(Conv([P2_k, P3_k, P4_k, P5_k, M_small_k]))
```

其中：

```text
W = [W2, W3, W4, W5]
```

最终融合：

```text
F_k = W2 * P2_k + W3 * P3_k + W4 * P4_k + W5 * P5_k
```

### 5.4 轻量化建议

完整四尺度 SAFM 计算量较大。毕业论文实现时建议先采用轻量版本：

```text
P2 输出：融合 P2, P3, P4
P3 输出：融合 P2, P3, P4, P5
P4 输出：保持 PAN 输出
P5 输出：保持 PAN 输出
```

即重点增强小目标相关的 P2/P3 检测层，避免模型过重。

### 5.5 原创性表述

不要直接写“使用 ASFF”。建议表述为：

> 本文提出小目标感知空间自适应融合模块，通过浅层特征生成小目标响应图，并将其作为尺度权重预测的先验引导，使网络在小目标区域提高高分辨率特征权重，在背景区域抑制冗余尺度响应。

与普通 ASFF 的区别：

- 普通 ASFF 只根据多尺度特征学习融合权重；
- 本方法显式引入小目标响应图；
- 融合目标聚焦于无人机小目标区域增强；
- 只对 P2/P3 重点增强，兼顾精度与速度。

## 6. 进阶增强：SA-NWD 尺度自适应小目标定位损失

### 6.1 设计动机

小目标边界框尺寸很小，预测框发生 1 到 2 个像素的偏移，就可能造成 IoU 大幅下降。因此，单纯依赖 IoU 类损失会使小目标定位训练不稳定。

NWD 将边界框建模为二维高斯分布，通过 Wasserstein 距离衡量两个框的相似性，对微小目标更稳定。

需要注意的是，Ultralytics YOLO11 的 detection loss 封装较深，涉及 anchor-free 分配、DFL、bbox decode、target preprocess 和正样本 mask 等逻辑。直接修改 loss 的调试成本较高，因此 **SA-NWD 不建议作为第一版必做内容**。

更稳妥的安排是：

```text
第一版核心模型：P2 + SGFA + SOA-SAFM
进阶增强实验：在结构实验稳定后再加入 SA-NWD
```

### 6.2 损失设计

如果后续实现 SA-NWD，不要简单全局添加 NWD，而是设计为尺度自适应形式：

```text
L_box = L_yolo_box + lambda(area) * L_nwd
```

其中：

```text
lambda(area) = exp(-area / tau)
```

含义：

- 目标越小，`lambda(area)` 越大；
- 目标越大，NWD 权重越小；
- 中大目标仍主要依赖原 YOLO box loss。

也可以使用更简单的阈值版本：

```text
if area < 32^2:
    L_box = L_yolo_box + lambda * L_nwd
else:
    L_box = L_yolo_box
```

### 6.3 原创性表述

建议表述为：

> 本文设计尺度自适应 NWD 辅助定位损失，仅对小尺度目标增强分布距离约束，从而缓解小目标框对 IoU 偏移过度敏感的问题，并避免对中大目标回归产生额外干扰。

## 7. 最终模型配置建议

推荐的最终模型版本：

```text
AFA-YOLO11
    Backbone: YOLO11s backbone
    Neck:
        P2 branch
        SGFA for small-object guided soft alignment
        lightweight SOA-SAFM for P2/P3 fusion
    Head:
        Detect(P2, P3, P4, P5)
    Loss:
        YOLO default detection loss
        Optional: SA-NWD auxiliary loss for small boxes
```

可以先实现两个版本：

```text
AFA-YOLO11-N: P2 + SGFA + SOA-SAFM
AFA-YOLO11-F: P2 + SGFA + SOA-SAFM + SA-NWD
```

这样方便消融实验，也能降低一次性修改 loss 带来的实现风险。

## 8. 与原 UAV-YOLO 的区别

| 项目 | 原 UAV-YOLO | AFA-YOLO11 |
|---|---|---|
| 基线 | YOLO11s | YOLO11s |
| 小目标层 | P2 | P2 |
| 特征融合 | BiFPN-style 全局权重 concat | 小目标引导软对齐 + 空间自适应融合 |
| 注意力 | EMA | 不单独使用 EMA，避免堆模块 |
| 定位损失 | YOLO 默认 loss | YOLO 默认 loss，SA-NWD 作为进阶选做 |
| 主要创新 | 模块组合 | 小目标响应引导的软对齐与自适应融合 |

核心变化：

```text
原项目：YOLO11s + P2 + BiFPN + EMA
改进后：YOLO11s + P2 + SGFA + SOA-SAFM
进阶版：YOLO11s + P2 + SGFA + SOA-SAFM + SA-NWD
```

## 9. 实现路线

### 阶段一：复现基线

目标：

- 跑通 `YOLO11s baseline`
- 跑通原项目 `UAV-YOLO`
- 确认 VisDrone 数据集、训练脚本、评估脚本可用

命令示例：

```bash
python setup_env.py
python train.py --exp baseline
python train.py --exp uav_yolo_pt
```

### 阶段二：实现 P2 + SGFA

工作内容：

- 在 `models/modules.py` 中新增 `SGFA`
- 修改模型 YAML，替换原 `BiFPN_Concat`
- 保持 Detect(P2, P3, P4, P5)
- 先不改 loss
- 不使用 `grid_sample`，第一版采用空间门控软对齐

验证目标：

- 模型可以正常构建；
- forward 不报错；
- 参数量和 FLOPs 可统计；
- 能完成少量 epoch 训练。

### 阶段三：实现 SOA-SAFM

工作内容：

- 在 `models/modules.py` 中新增 `SOA_SAFM`
- 先只作用于 P2/P3 输出；
- 比较 `P2 + SGFA` 与 `P2 + SGFA + SAFM` 的提升。

验证目标：

- AP_small 提升；
- FPS 不出现明显下降；
- 检测可视化中小目标漏检减少。

### 阶段四：进阶选做 SA-NWD Loss

工作内容：

- 阅读 Ultralytics YOLO11 detection loss 实现；
- 在 bbox loss 中增加小目标 NWD 辅助项；
- 通过面积阈值或面积衰减函数控制 NWD 权重。

该阶段为选做。若结构改进已经可以支撑论文结果，SA-NWD 可以放入增强实验或后续工作，避免 loss 修改影响整体进度。

建议优先使用阈值版本：

```text
area < 32^2 时启用 NWD
```

这样更容易解释，也更容易调参。

### 阶段五：完整实验

完成：

- 消融实验；
- 与 YOLOv8s、YOLOv9s、YOLOv10s、YOLO11s、RT-DETR、原 UAV-YOLO 对比；
- FPS / Params / FLOPs 分析；
- Grad-CAM 或特征热力图；
- 失败案例分析。

## 10. 消融实验设计

建议至少包含：

| Method | P2 | SGFA | SOA-SAFM | SA-NWD | mAP50 | mAP50-95 | AP_small | Params | FLOPs | FPS |
|---|---|---|---|---|---|---|---|---|---|---|
| YOLO11s | - | - | - | - | | | | | | |
| YOLO11s + P2 | ✓ | - | - | - | | | | | | |
| YOLO11s + P2 + BiFPN | ✓ | - | - | - | | | | | | |
| YOLO11s + P2 + SGFA | ✓ | ✓ | - | - | | | | | | |
| YOLO11s + P2 + SOA-SAFM | ✓ | - | ✓ | - | | | | | | |
| AFA-YOLO11-N | ✓ | ✓ | ✓ | - | | | | | | |
| AFA-YOLO11-F | ✓ | ✓ | ✓ | ✓ | | | | | | |

重点关注：

- `AP_small` 是否明显提升；
- `mAP50-95` 是否提升；
- FPS 是否仍满足实时性；
- Params / FLOPs 是否可接受。

## 11. 对比实验设计

推荐对比模型：

```text
YOLOv8s
YOLOv9s
YOLOv10s
YOLO11s
RT-DETR-l
原 UAV-YOLO
AFA-YOLO11
```

推荐数据集：

```text
主数据集：VisDrone2019-DET
泛化数据集：AI-TOD 或 NWPU VHR-10
```

如果时间有限：

```text
VisDrone 必做
AI-TOD 选做
NWPU 选做
```

## 12. 论文贡献写法

可以将论文贡献总结为：

```text
1. 针对无人机航拍图像中小目标空间信息易丢失的问题，引入 P2 高分辨率检测分支，增强微小目标的细粒度表征能力。

2. 针对传统 FPN/PAN 跨尺度融合中存在的语义区域不匹配与背景干扰问题，提出小目标引导软对齐模块 SGFA，利用浅层细节特征和小目标响应图生成空间对齐门控，对深层语义特征进行区域级重标定。

3. 针对多尺度特征在不同空间区域贡献不一致的问题，提出小目标感知空间自适应融合模块 SOA-SAFM，根据小目标响应动态分配尺度权重，增强小目标区域的高分辨率特征表达并抑制背景干扰。

4. 在进阶版本中，针对小目标边界框对 IoU 偏移敏感的问题，设计尺度自适应 NWD 辅助定位损失，仅对小尺度目标增强分布距离约束，提高小目标定位稳定性。
```

## 13. 风险与备选方案

### 风险一：几何对齐实现复杂

`grid_sample` 或 deformable convolution 版本的几何对齐实现难度较高，可能出现坐标归一化错误、梯度不稳定、训练早期破坏特征等问题。因此第一版不建议使用几何采样式对齐。

本文的主实现建议采用：

```text
SGFA:
    使用空间门控完成软对齐
    不预测 offset
    不使用 grid_sample
```

如果后续时间充足，再扩展为：

```text
geometric alignment:
    使用 offset + grid_sample
    或使用 deformable convolution
```

### 风险二：SOA-SAFM 计算量过大

备选方案：

```text
只在 P2/P3 使用 SAFM
P4/P5 保持普通 PAN 输出
```

或者：

```text
只融合相邻尺度：
P2 <- P2, P3
P3 <- P2, P3, P4
```

或者：

```text
使用 depthwise separable convolution 降低融合卷积计算量
```

### 风险三：修改 Ultralytics Loss 成本较高

备选方案：

```text
先完成结构创新：P2 + SGFA + SOA-SAFM
将 SA-NWD 作为增强实验或后续工作
```

如果时间紧，毕业论文主模型可以只保留结构创新，loss 部分作为扩展。

## 14. 最低可交付版本

如果时间有限，最低建议完成：

```text
YOLO11s + P2 + SGFA + SOA-SAFM
```

该版本已经具备较完整的模型创新逻辑：

- P2 解决小目标空间细节保留；
- SGFA 通过小目标响应引导完成跨尺度软对齐；
- SOA-SAFM 解决不同区域尺度权重自适应选择；
- 不依赖复杂 loss 修改，工程风险较低。

## 15. 最终建议

最终论文不要以“改进 YOLO11 并加入若干模块”为主线，而应以：

> 小目标感知的跨尺度对齐与自适应融合

作为统一主线。

建议最终模型命名为：

```text
AFA-YOLO11: Small-object Guided Aligned Fusion Adaptive YOLO11
```

中文名称：

```text
基于小目标引导跨尺度对齐与自适应融合的 YOLO11 无人机目标检测模型
```

该方案相比原项目更适合作为毕业论文，因为它不是简单保留 `P2 + BiFPN + EMA` 的模块堆叠，而是将模型结构、融合机制和定位损失都围绕“小目标感知”这一主题组织起来，论文叙述更连贯，答辩时也更容易说明原创设计点。
