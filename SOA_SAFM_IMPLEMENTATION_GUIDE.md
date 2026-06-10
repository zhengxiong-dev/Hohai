# SOA-SAFM 核心架构与实现方案

## 1. 模块定位

SOA-SAFM 全称建议写为：

```text
Small-object Aware Spatial Adaptive Fusion Module
小目标感知空间自适应融合模块
```

它用于在 SGFA 完成跨尺度软对齐之后，进一步解决多尺度特征融合中的尺度选择问题。

SGFA 主要回答：

```text
高层语义特征和低层细节特征如何更好对齐？
```

SOA-SAFM 主要回答：

```text
不同空间位置应该更依赖哪个尺度的特征？
```

无人机小目标检测中，不同区域的特征需求不同：

```text
小目标区域：更依赖 P2 / P3 的高分辨率细节
中大目标区域：更依赖 P4 / P5 的语义信息
背景区域：应该抑制无效尺度响应
```

因此，SOA-SAFM 不使用固定融合权重，而是生成空间位置相关的尺度权重。

## 2. 第一版实现原则

第一版目标是稳定、可解释、容易接入 Ultralytics。

明确不做：

```text
全四尺度复杂重融合
Transformer attention
deformable attention
loss 修改
额外检测分支
```

推荐第一版只增强：

```text
P2 输出
P3 输出
```

保持：

```text
P4 / P5 使用原 neck 输出
```

原因：

- P2/P3 是小目标检测最关键的层；
- 全尺度 SAFM 计算量和显存增长明显；
- 第一版重点证明小目标区域自适应融合有效；
- 更容易做消融实验。

## 3. 输入输出定义

SOA-SAFM 第一版建议接收 3 个或 4 个特征。

### 3.1 P2 输出融合

输入：

```text
x = [P2, P3, P4]
```

输出：

```text
O2
```

其中所有输入会 resize 到 P2 的空间尺寸。

### 3.2 P3 输出融合

输入：

```text
x = [P2, P3, P4, P5]
```

输出：

```text
O3
```

其中所有输入会 resize 到 P3 的空间尺寸。

### 3.3 输出通道

为了方便接入 Detect，SOA-SAFM 输出通道建议固定为目标尺度通道数：

```text
O2 channels = P2 channels
O3 channels = P3 channels
```

这意味着模块内部需要将所有输入投影到相同通道数：

```text
Pi -> Conv1x1 -> C_out
```

## 4. 核心结构

以 P2 输出为例：

```text
P2 ------------------ Conv1x1 ---- F2
P3 -- resize to P2 -- Conv1x1 ---- F3
P4 -- resize to P2 -- Conv1x1 ---- F4

F2 -> small object response M_small

concat(F2, F3, F4, M_small)
        |
   weight branch
        |
Softmax over scale dimension
        |
W2, W3, W4

O2 = W2 * F2 + W3 * F3 + W4 * F4
```

核心是：

```text
空间权重 W 的 shape = [B, N, H, W]
```

其中：

```text
N = 输入尺度数量
```

每个像素位置都有一组尺度权重：

```text
W2(h,w) + W3(h,w) + W4(h,w) = 1
```

## 5. 小目标响应图

SOA-SAFM 的关键不是简单 ASFF，而是引入小目标响应引导。

第一版中，小目标响应由目标尺度特征生成：

```text
M_small = Sigmoid(Conv3x3(F_target))
```

例如：

```text
P2 融合时，F_target = F2
P3 融合时，F_target = F3
```

`M_small` 参与尺度权重预测：

```text
W = Softmax(Conv([F2, F3, F4, M_small]))
```

论文解释：

> 通过浅层高分辨率特征生成小目标响应先验，并将其作为尺度权重预测的空间引导，使网络在疑似小目标区域提高高分辨率特征权重，在背景区域抑制冗余尺度响应。

## 6. Forward 流程

伪代码：

```python
def forward(self, x):
    # x: list of feature maps
    target = x[self.target_index]
    target_size = target.shape[-2:]

    feats = []
    for i, feat in enumerate(x):
        if feat.shape[-2:] != target_size:
            feat = F.interpolate(feat, size=target_size, mode="nearest")
        feats.append(self.proj[i](feat))

    target_feat = feats[self.target_index]
    m_small = self.small_gate(target_feat)

    weight_in = torch.cat(feats + [m_small], dim=1)
    weights = self.weight_head(weight_in)
    weights = weights.view(B, N, 1, H, W)
    weights = torch.softmax(weights, dim=1)

    out = sum(weights[:, i] * feats[i] for i in range(N))
    out = self.out_conv(out)
    return out
```

## 7. PyTorch 实现骨架

建议添加到：

```text
models/modules.py
```

实现骨架：

```python
class SOA_SAFM(nn.Module):
    """
    Small-object Aware Spatial Adaptive Fusion Module.

    Input: list of multi-scale features.
    Output: one fused feature map at target scale.
    """

    def __init__(self, target_index=0, out_channels=None, ratio=1.0):
        super().__init__()
        self.target_index = target_index
        self.out_channels = out_channels
        self.ratio = ratio
        self._built = False

    def _build(self, channels, device):
        if self.out_channels is None:
            c_out = channels[self.target_index]
        else:
            c_out = self.out_channels

        c_mid = max(1, int(c_out * self.ratio))
        self.c_out = c_out
        self.n = len(channels)

        self.proj = nn.ModuleList([
            nn.Conv2d(c, c_mid, 1, 1, 0) for c in channels
        ]).to(device)

        self.small_gate = nn.Sequential(
            nn.Conv2d(c_mid, 1, 3, 1, 1),
            nn.Sigmoid(),
        ).to(device)

        self.weight_head = nn.Sequential(
            nn.Conv2d(c_mid * self.n + 1, c_mid, 3, 1, 1),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_mid, self.n, 1, 1, 0),
        ).to(device)

        self.out_conv = nn.Conv2d(c_mid, c_out, 1, 1, 0).to(device)
        self._built = True

    def forward(self, x):
        target = x[self.target_index]
        target_size = target.shape[-2:]

        if not self._built:
            channels = [feat.shape[1] for feat in x]
            self._build(channels, target.device)

        feats = []
        for i, feat in enumerate(x):
            if feat.shape[-2:] != target_size:
                feat = torch.nn.functional.interpolate(
                    feat, size=target_size, mode="nearest"
                )
            feats.append(self.proj[i](feat))

        b, _, h, w = feats[self.target_index].shape
        m_small = self.small_gate(feats[self.target_index])

        weight_in = torch.cat(feats + [m_small], dim=1)
        weights = self.weight_head(weight_in)
        weights = weights.view(b, self.n, 1, h, w)
        weights = torch.softmax(weights, dim=1)

        out = 0
        for i in range(self.n):
            out = out + weights[:, i] * feats[i]

        return self.out_conv(out)
```

和 SGFA 一样，这个骨架使用 lazy build，方便快速开发和实验。后续如需导出部署，再改成显式通道版本。

## 8. 更规范的工程版本

更规范的实现形式是：

```python
class SOA_SAFM(nn.Module):
    def __init__(self, channels, target_index=0, out_channels=None, ratio=1.0):
        ...
```

其中：

```text
channels = [c2, c3, c4]
```

或：

```text
channels = [c2, c3, c4, c5]
```

但这需要修改 Ultralytics YAML parser，让它从 `from` 层自动传入多个输入通道。考虑当前项目已经通过 patch 支持自定义模块，第一版可先用 lazy build 降低复杂度。

## 9. YAML 接入方式

SOA-SAFM 不建议一开始替换所有 neck 融合，而是放在 Detect 前，对 P2/P3 进行增强。

假设 neck 已经得到：

```text
P2_out
P3_out
P4_out
P5_out
```

可以新增：

```yaml
- [[P2_out, P3_out, P4_out], 1, SOA_SAFM, [0]]  # O2
- [[P2_out, P3_out, P4_out, P5_out], 1, SOA_SAFM, [1]]  # O3
```

然后 Detect 改为：

```yaml
- [[O2, O3, P4_out, P5_out], 1, Detect, [nc]]
```

实际 YAML 中需要用真实层号替换 `P2_out`、`P3_out`、`P4_out`、`P5_out`、`O2`、`O3`。

## 10. setup_env.py 修改点

需要把 `SOA_SAFM` 注册到 Ultralytics。

当前只注册：

```python
EMA, BiFPN_Concat
```

需要改为：

```python
EMA, BiFPN_Concat, SGFA, SOA_SAFM
```

parser 处理逻辑上，`SOA_SAFM` 与 `Concat` 不完全一样：

```text
Concat 输出通道 = sum(input channels)
SOA_SAFM 输出通道 = target input channels
```

如果使用 lazy build，但 parser 仍然需要知道后续层的通道数，因此 `tasks.py` 中建议增加特殊处理：

```python
elif m is SOA_SAFM:
    target_index = args[0] if len(args) else 0
    c2 = ch[f[target_index]]
```

如果 `f` 是 list：

```text
f = [P2_out, P3_out, P4_out]
target_index = 0
c2 = ch[P2_out]
```

如果是 P3 输出：

```text
f = [P2_out, P3_out, P4_out, P5_out]
target_index = 1
c2 = ch[P3_out]
```

## 11. 推荐开发顺序

不要一开始同时开发 SGFA 和 SOA-SAFM。建议顺序：

```text
1. 先实现 SGFA
2. 新建 afa-yolo-sgfa.yaml
3. 跑通 model.info 和 1 epoch 训练
4. 确认 SGFA 有效或至少不明显掉点
5. 再实现 SOA-SAFM
6. 新建 afa-yolo-sgfa-safm.yaml
7. 只增强 P2
8. 再增强 P2 + P3
```

这样每一步都有清晰对比。

## 12. 实验配置建议

建议新增实验：

```text
sgfa_pt:
    YOLO11s + P2 + SGFA

sgfa_safm_p2_pt:
    YOLO11s + P2 + SGFA + SOA-SAFM(P2)

afa_yolo_pt:
    YOLO11s + P2 + SGFA + SOA-SAFM(P2/P3)
```

消融表：

```text
YOLO11s
YOLO11s + P2
YOLO11s + P2 + BiFPN
YOLO11s + P2 + SGFA
YOLO11s + P2 + SGFA + SOA-SAFM(P2)
YOLO11s + P2 + SGFA + SOA-SAFM(P2/P3)
```

重点指标：

```text
mAP50
mAP50-95
AP_small
Params
FLOPs
FPS
```

## 13. 调试检查点

### 13.1 结构检查

```text
model = YOLO("models/afa-yolo-sgfa-safm.yaml", task="detect")
model.info()
```

检查：

- SOA-SAFM 是否被正确解析；
- Detect 输入是否为 `[O2, O3, P4, P5]`；
- O2 / O3 通道是否符合 Detect 预期；
- 参数量是否没有异常暴涨。

### 13.2 Forward 检查

用随机输入或一张图片跑一次：

```text
model.predict(...)
```

检查：

- resize 是否正常；
- `torch.softmax(weights, dim=1)` 是否维度正确；
- 输出 shape 是否正确；
- 没有 NaN。

### 13.3 训练检查

先跑：

```text
epochs=1
batch=2
```

检查：

- loss 是否能正常反向；
- GPU 显存是否可接受；
- 训练速度是否明显下降；
- metrics 是否正常生成。

## 14. 常见问题与处理

### 问题一：parser 不知道 SOA-SAFM 输出通道

需要在 `setup_env.py` patch 的 `tasks.py` 逻辑中增加：

```text
SOA_SAFM 特殊 c2 计算
```

不能直接当作 Concat，因为 Concat 输出是通道求和，而 SOA-SAFM 输出是目标尺度通道。

### 问题二：显存或计算量过高

处理：

```text
只做 P2 SAFM
ratio = 0.5
输入尺度从 [P2, P3, P4, P5] 减少为 [P2, P3, P4]
使用 nearest resize
减少 weight_head 中间通道
```

### 问题三：效果下降

优先做这些排查：

```text
1. 只在 P2 加 SOA-SAFM
2. 确认 P2/P3/P4 的层号没有写错
3. 可视化权重图，看小目标区域是否更偏向 P2/P3
4. 暂时去掉 EMA，避免模块互相干扰
5. 对比固定平均融合，确认自适应权重是否有效
```

### 问题四：权重塌缩到单一尺度

如果发现 softmax 权重长期只选择一个尺度，可以尝试：

```text
增加 temperature:
weights = softmax(logits / T), T > 1

或增加轻微 entropy regularization
```

但第一版不建议加额外正则，先观察即可。

## 15. 可视化建议

SOA-SAFM 很适合做论文可视化。

可以保存：

```text
W2, W3, W4
```

或：

```text
W2 / W3 / W4 / W5 heatmap
```

预期现象：

```text
小目标密集区域：P2/P3 权重更高
大目标区域：P4/P5 权重更高
背景区域：权重分布更平滑或响应更弱
```

这能支撑论文中的“空间自适应尺度选择”。

## 16. 最小可实现版本

最小版本只做 P2：

```text
SOA-SAFM-P2:
    输入 [P2, P3, P4]
    resize 到 P2
    统一通道
    生成 M_small
    softmax 得到 W2/W3/W4
    加权求和输出 O2
```

Detect 改为：

```text
Detect(O2, P3, P4, P5)
```

如果 P2 版本有效，再扩展到：

```text
Detect(O2, O3, P4, P5)
```

这是最稳的开发路径。

## 17. 论文表述建议

推荐表述：

> 为解决无人机图像中不同空间区域对尺度特征需求不一致的问题，本文设计小目标感知空间自适应融合模块 SOA-SAFM。该模块利用浅层高分辨率特征生成小目标响应先验，并将其引入多尺度权重预测过程，使网络能够在小目标区域自适应增强 P2/P3 细粒度特征，在复杂背景区域抑制冗余尺度响应，从而提升密集小目标检测性能。

不要写成：

```text
本文加入 ASFF 模块。
```

更合适的关键词：

```text
小目标感知
空间自适应
尺度权重
P2/P3 增强
背景抑制
轻量融合
```

