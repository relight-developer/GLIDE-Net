# 缺陷严重程度分级

分级是“模型输出”和“人能用的结论”之间的一层。它必须是**可读、可反驳、可修改**的，
所以刻意选择规则化实现，而不是再训一个回归网络。

代码：`app/engines/severity.py`（唯一事实来源）

## 1. 三个输入线索

| 线索 | 计算方式 | 为什么需要 |
|------|----------|------------|
| 面积 `area_ratio` | 连通域像素数 / 画面总像素 | 缺陷越大越影响服役安全 |
| 细长比 `elongation` | 包围盒长边 / 短边 | 划伤、氧化皮压入这类长条缺陷，同等面积下危害更大 |
| 置信度 `mean_conf` | 连通域内预测概率均值 | 模型自己“犹豫”的区域应降低定级 |

## 2. 评分公式

```
area_term = min(1, area_ratio / AREA_REFERENCE)
elong_term = 1 + 0.18 · min(1, max(0, elongation − 1) / 6)
conf_term = 0.75 + 0.25 · clip(mean_conf, 0, 1)

score = 100 · area_term · elong_term · conf_term      # 上限 100
```

`ELONGATION_REFERENCE = 6` 表示细长比达到 7 时细长项饱和（约 +18%）。

## 3. 阈值与标定

```
light   : [0, 26)
medium  : [26, 55)
severe  : [55, 100]
```

### 3.1 标定过程（为什么 `AREA_REFERENCE` 从 0.12 改成 0.55）

第一版用 `AREA_REFERENCE = 0.12` + `sqrt()`，直觉是“面积占 12% 就已经很严重了”。
但在真实数据上跑出**三档退化**：`tools/eval_system.py` 前 40 张全部判为 `severe`，
评分集中在 65–100。

于是统计了数据集本身的面积分布（`predictions/GT`，362 张真值掩膜）：

| 分位 | 面积占比 | 分位 | 面积占比 |
|------|----------|------|----------|
| p5 | 0.071 | p75 | 0.3867 |
| p10 | 0.0964 | p80 | 0.4304 |
| p25 | 0.1479 | p90 | 0.5157 |
| p50 | 0.2536 | p95 | 0.5907 |
| p66 | 0.3359 | p99 | 0.6742 |

RSDDS-AUG 的图像是**缺陷特写**，缺陷面积中位数就有 25%。若参考值定在 0.12，
所有样本的 `area_term` 都会饱和到 1，面积线索完全失效。

改用 `AREA_REFERENCE = 0.55`（≈ p93）、去掉 `sqrt`（线性更符合面积直觉）后：

| 统计口径 | 分级分布 |
|----------|----------|
| 362 张真值掩膜 + 我们的概率图，**逐图**判定 | light 108 / medium 117 / severe 137 |
| 同上，**逐区域**评分 | light 250 / medium 120 / severe 137 |
| 实际流水线（全量 362 张，含检测融合） | light 102 / medium 120 / severe 140 |

逐图口径与流水线口径都接近三分（流水线 28% / 33% / 39%），符合“真缺陷数据集里
严重程度确有梯度”的预期。早期用 `--limit 60` 抽样时曾出现 72% 判严重的偏差，
原因是对前 60 张三档分布不均；全量 362 张后该偏差消失，仍然记录在
[experiment-log.md](experiment-log.md) 里以便追溯。

### 3.2 复现标定

```bash
# 面积分布
python - <<'PY'
import glob, numpy as np
from PIL import Image
v = np.array([ (np.array(Image.open(p).convert('L')) > 127).mean()
               for p in glob.glob('predictions/GT/*.png') ])
for q in (5,10,25,50,75,90,95,99):
    print(q, round(float(np.percentile(v,q)),4))
PY

# 分级分布
python - <<'PY'
import sys, glob, os, numpy as np
sys.path.insert(0, '.')
from PIL import Image
from app.engines.severity import inspect_image
from collections import Counter
c, regs = Counter(), []
for p in glob.glob('predictions/GT/*.png'):
    a = (np.array(Image.open(p).convert('L')) > 127).astype(np.uint8)
    prob = np.array(Image.open('predictions/Ours/' + os.path.basename(p)).convert('L'),
                    np.float32) / 255.
    rep = inspect_image(a, prob)
    c[rep['overall']] += 1
    regs += [r['severity'] for r in rep['regions']]
print(dict(c), dict(Counter(regs)))
PY
# -> {'medium': 117, 'severe': 137, 'light': 108}
#    {'medium': 120, 'severe': 137, 'light': 250}
```

## 4. 融合规则（掩膜 × 检测框）

代码：`app/engines/severity.py::attach_labels`、`app/engines/pipeline.py`

| 情形 | 处理 | 理由 |
|------|------|------|
| 掩膜有连通域 + 检测框 IoU ≥ 0.2 | 连通域继承该框的类别与置信度 | 检测分支是唯一知道“是什么缺陷”的信息源 |
| 掩膜有连通域，但无匹配框 | 保留，`label = null`，界面显示“未分类缺陷” | 宁可不分类，也不丢弃像素证据 |
| 检测框无对应掩膜 | 保留在 `detections` 里，但**不**伪造连通域 | 避免用框面积冒充像素面积，污染 `defect_area_ratio` |

阈值 0.2 偏低是刻意的：检测框与像素掩膜的形态差异大（框是矩形，掩膜是细长条），
IoU 高不了。用 0.5 会导致大量掩膜拿不到类别。

## 5. 图像级判定

```
判定 = 最严重区域的等级（不是平均值）
```

一条深划伤不能被 20 处轻微麻点稀释。这不是“统计上更优”的选择，而是**工程决策**：
漏检的代价远高于误报，课堂演示时也更容易解释。

## 6. 单元测试覆盖

`tests/test_pipeline.py`：

- `test_grade_bands_are_monotonic`：26、55 两个边界的前后 0.1 分都落在正确档位。
- `test_score_is_bounded_and_monotonic_in_area`：面积为 0 时分数 > 0，面积饱和时分恰好 100。
- `test_score_rewards_elongation_and_penalises_low_confidence`：细长比 8 > 细长比 1；置信度 0.55 < 0.95。
- `test_extract_regions_finds_both_blobs_and_sorts_by_score`：合成“方块 + 长条”双缺陷，
  验证连通域计数、面积、排序。
- `test_extract_regions_ignores_specks`：单像素噪点被过滤。
- `test_attach_labels_uses_best_overlapping_box` / `..._leaves_unmatched_regions_unlabelled`。
- `test_inspect_image_on_empty_mask`：空掩膜 → `none`，不抛异常。

## 7. 已知局限

1. **面积项主导**：`area_term` 是乘性主项，细长与置信最多只带来 1.18 × 1.0 的放大；
   若希望“细划伤 = 严重”，应改为加权和或对数面积。
2. **阈值是数据集相关**：0.55 的参考值来自 RSDDS-AUG 特写图像。换成全景巡检图
   （缺陷占比 0.1% 量级）必须重标定。
3. **未用深度做分级**：深度本可提供“缺陷深度/凹陷量”，当前只用于分割特征。
   这是明确的后续方向。
4. **无真值分级标注**：数据集只有二值掩膜，没有“轻微/中等/严重”标签，
   因此分级无法做有监督验证，只能做分布合理性检查。这一点必须在报告里讲清楚，
   不能把分级说成“已训练的模型”。
