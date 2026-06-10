# SGFA 核心架构与实现方案

## 1. 模块定位

SGFA 全称建议写为：

```text
Small-object Guided Feature Alignment
小目标引导特征软对齐模块
```

它用于替代原项目中的 `BiFPN_Concat`，作为 YOLO11s neck 中的跨尺度融合单元。

原始 UAV-YOLO 的融合逻辑类似：

```text
Upsample(P5) + P4 -> BiFPN_Concat -> C3k2
Upsample(P4) + P3 -> BiFPN_Concat -> C3k2
Upsample(P3) + P2 -> BiFPN_Concat -> C3k2
```

SGFA 的目标不是做几何采样式对齐，而是采用更稳定的 **空间门控软对齐**：

```text
F_high 上采样
F_low 提供空间细节
M_small 提供小目标响应引导
生成空间对齐门控 A
用 A 重标定高层语义特征
再与低层细节特征融合
```

第一版明确不使用：

```text
grid_sample
offset prediction
deformable convolution
loss 修改
```

这样实现难度和训练风险都更低。

## 2. 输入输出定义

SGFA 接收两个特征输入：

```text
x[0] = F_high_up
x[1] = F_low
```

其中：

- `F_high_up`：已经上采样到目标尺度的高层语义特征；
- `F_low`：当前尺度的低层高分辨率特征；
- 二者空间尺寸必须一致；
- 通道数可以不同。

输出：

```text
F_out
```

输出通道数建议由 YAML 后续的 `C3k2` 控制，SGFA 本身第一版可以保持 concat 风格输出：

```text
out_channels = c_low + c_align
```

这样最容易兼容 Ultralytics 的模型解析逻辑。

## 3. 核心结构

推荐第一版 SGFA 结构：

```text
F_high_up
    |
Conv1x1 -> Fh
              \
               concat -> gate branch -> A
              /
F_low
    |
Conv1x1 -> Fl

Fh_soft = Fh * A
F_out = Concat(Fl, Fh_soft)
```

其中 gate branch 输入包括：

```text
[Fl, Fh, M_small]
```

第一版中 `M_small` 不单独从外部输入，而是在模块内部由浅层特征 `Fl` 生成：

```text
M_small = Sigmoid(Conv3x3(Fl))
```

这样 YAML 接入简单，不需要额外增加第三路输入。

## 4. Forward 流程

伪代码：

```python
def forward(self, x):
    f_high, f_low = x

    if f_high.shape[-2:] != f_low.shape[-2:]:
        f_high = F.interpolate(f_high, size=f_low.shape[-2:], mode="nearest")

    fh = self.high_proj(f_high)
    fl = self.low_proj(f_low)

    m_small = self.small_gate(fl)
    gate = self.align_gate(torch.cat([fh, fl, m_small], dim=1))

    fh_soft = fh * gate
    return torch.cat([fl, fh_soft], dim=1)
```

## 5. PyTorch 实现骨架

建议把模块加到：

```text
models/modules.py
```

实现骨架：

```python
class SGFA(nn.Module):
    """
    Small-object Guided Feature Alignment.

    A lightweight soft-alignment fusion module for cross-scale YOLO necks.
    Input: [F_high, F_low]
    Output: concat(F_low_projected, F_high_soft_aligned)
    """

    def __init__(self, dimension=1, ratio=1.0):
        super().__init__()
        self.d = dimension
        self.ratio = ratio
        self._built = False

    def _build(self, c_high, c_low, device):
        c_mid_high = max(1, int(c_high * self.ratio))
        c_mid_low = max(1, int(c_low * self.ratio))

        self.high_proj = nn.Conv2d(c_high, c_mid_high, 1, 1, 0).to(device)
        self.low_proj = nn.Conv2d(c_low, c_mid_low, 1, 1, 0).to(device)

        self.small_gate = nn.Sequential(
            nn.Conv2d(c_mid_low, 1, 3, 1, 1),
            nn.Sigmoid(),
        ).to(device)

        self.align_gate = nn.Sequential(
            nn.Conv2d(c_mid_high + c_mid_low + 1, c_mid_high, 3, 1, 1),
            nn.BatchNorm2d(c_mid_high),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_mid_high, c_mid_high, 1, 1, 0),
            nn.Sigmoid(),
        ).to(device)

        self._built = True

    def forward(self, x):
        f_high, f_low = x

        if f_high.shape[-2:] != f_low.shape[-2:]:
            f_high = torch.nn.functional.interpolate(
                f_high, size=f_low.shape[-2:], mode="nearest"
            )

        if not self._built:
            self._build(f_high.shape[1], f_low.shape[1], f_high.device)

        fh = self.high_proj(f_high)
        fl = self.low_proj(f_low)

        m_small = self.small_gate(fl)
        gate = self.align_gate(torch.cat([fh, fl, m_small], dim=1))

        fh_soft = fh * gate
        return torch.cat([fl, fh_soft], self.d)
```

注意：这个骨架采用 lazy build，开发快，但不是最理想的工程形态。后续如果要更规范，可以让 Ultralytics parser 在构建阶段传入通道数。

## 6. 更推荐的工程实现

为了减少 lazy build 对权重保存、模型导出和多卡训练的潜在影响，更推荐后续实现成显式通道版本：

```python
class SGFA(nn.Module):
    def __init__(self, c_high, c_low, c_out=None, dimension=1):
        ...
```

但这样需要同步修改 `setup_env.py` 对 Ultralytics parser 的 patch 逻辑，让 parser 能够根据 `from` 层自动传入 `c_high` 和 `c_low`。

考虑当前项目已经用 `BiFPN_Concat` 走 concat 兼容路线，第一版建议先使用 lazy build 版本，优先跑通实验。

## 7. YAML 接入方式

原 YAML 中类似：

```yaml
- [[-1, 6], 1, BiFPN_Concat, [1]]
```

可以替换为：

```yaml
- [[-1, 6], 1, SGFA, [1]]
```

推荐新建文件：

```text
models/afa-yolo.yaml
```

不要直接覆盖原来的：

```text
models/uav-yolo.yaml
```

这样方便做对比实验。

## 8. 需要修改 setup_env.py

当前 `setup_env.py` 只注册：

```python
EMA, BiFPN_Concat
```

需要同步加入：

```python
SGFA
```

需要修改的位置：

```text
1. import custom modules
2. install_custom_modules 后的 __init__.py import
3. verify_patch 测试
4. tasks.py 中将 SGFA 与 Concat / BiFPN_Concat 一样处理
```

关键逻辑是让 Ultralytics parser 知道：

```python
elif m is Concat or m is BiFPN_Concat or m is SGFA:
    c2 = sum(ch[x] for x in f)
```

如果第一版 SGFA 输出仍是 concat 通道，那么这里可以沿用 Concat 的 `c2` 计算方式。

## 9. 推荐模型结构接入点

在 `uav-yolo.yaml` 的 head 中，将三处 top-down 融合替换为 SGFA：

```text
P5_up + P4 -> SGFA
P4_up + P3 -> SGFA
P3_up + P2 -> SGFA
```

bottom-up 路径可以先保持 `BiFPN_Concat` 或普通 concat，降低改动风险。

第一版建议：

```text
top-down 使用 SGFA
bottom-up 保持 BiFPN_Concat
```

第二版再尝试：

```text
top-down + bottom-up 全部使用 SGFA
```

这样消融更清晰。

## 10. 训练实验设计

建议新增实验：

```text
sgfa_pt:
    model=models/afa-yolo.yaml
    load=yolo11s.pt
    name=visdrone_sgfa_pt
```

对比顺序：

```text
baseline: YOLO11s
p2_pt: YOLO11s + P2
uav_yolo_pt: YOLO11s + P2 + BiFPN + EMA
sgfa_pt: YOLO11s + P2 + SGFA
```

如果后续加入 SOA-SAFM：

```text
afa_yolo_pt: YOLO11s + P2 + SGFA + SOA-SAFM
```

## 11. 调试检查点

实现 SGFA 后，先不要直接完整训练。建议按下面顺序检查：

```text
1. python setup_env.py
2. from ultralytics import YOLO
3. model = YOLO("models/afa-yolo.yaml", task="detect")
4. 打印 model.info()
5. 用随机输入跑一次 forward
6. 跑 1 epoch 小训练
7. 再跑完整训练
```

重点检查：

- SGFA 输入是否是 list；
- 两路特征空间尺寸是否一致；
- 输出通道数是否和后续 `C3k2` 预期一致；
- `model.load("yolo11s.pt")` 是否能加载匹配权重；
- loss 是否正常下降；
- 显存是否明显增加。

## 12. 可能问题与处理

### 问题一：Ultralytics parser 不认识 SGFA

处理：

```text
检查 setup_env.py 是否把 SGFA 写入 ultralytics.nn.modules.custom
检查 tasks.py 是否 import SGFA
检查 tasks.py 是否将 SGFA 按 Concat 类模块处理
```

### 问题二：输出通道不匹配

处理：

```text
SGFA 第一版保持 concat 输出
确保 tasks.py 中 c2 = sum(ch[x] for x in f)
不要在 SGFA 内部强行压缩到固定 c_out
```

### 问题三：训练效果不升反降

优先检查：

```text
1. 只替换 top-down 融合，不动 bottom-up
2. 暂时移除 EMA，避免多个模块干扰判断
3. 对比 P2 + BiFPN 与 P2 + SGFA
4. 可视化 small_gate 和 align_gate 的响应
```

### 问题四：显存增加

处理：

```text
ratio 设置为 0.5
只在 P3->P2 和 P4->P3 使用 SGFA
P5->P4 保持普通融合
```

## 13. 最小可实现版本

如果只追求尽快跑通，最小版本为：

```text
class SGFA:
    输入 [F_high, F_low]
    自动 resize F_high
    用 F_low 生成 small_gate
    用 [F_high, F_low, small_gate] 生成 align_gate
    输出 concat(F_low, F_high * align_gate)
```

不做：

```text
SOA-SAFM
SA-NWD
grid_sample
offset
deformable conv
loss 修改
```

这样可以先得到一个独立、可消融、可解释的结构改进点。

## 14. 论文表述建议

可以这样写：

> 为缓解无人机小目标在跨尺度融合过程中出现的语义区域不匹配和背景干扰问题，本文设计小目标引导特征软对齐模块 SGFA。该模块利用浅层高分辨率特征生成小目标空间响应，并以此引导深层语义特征进行空间门控重标定，使融合特征在小目标区域保留更多有效语义信息，同时抑制背景区域的冗余响应。

不要写成：

```text
本文加入了一个注意力模块。
```

更合适的关键词：

```text
小目标响应引导
跨尺度软对齐
深层语义特征重标定
P2/P3 小目标层增强
轻量化融合
```

