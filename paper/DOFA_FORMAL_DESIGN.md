# DOFA 正式一臂：设计与判读规则（2026-11-08，跑前写定）

用户 2026-11-08 决定把 DOFA 一臂"转成正式结果"。此前 10-29 定过深度骨干一臂保持探索性；本设计只改 DOFA 的 4 个设定，ResNet 的 3 个设定仍标为探索性。

## 1. 必要性审查

- **检验的论断。**
  - 地图的赢家诅咒不是随机森林特有的：地图换成地理空间基础模型（DOFA）后，已发表方法在被选单元上同样失效，参考集校准（JOMI、SC-GeoCCP）仍守住目标。
  - 现有证据只是探索性的：GROUND 中 DOFA 线性头的设置是看过评估单元后定的，不满足"地图独立于审计单元和候选单元"的前提。
- **无论结果正负，正文都会变。**
  - 正面：Results 里深度骨干那一句改为正式结论。
  - 负面：已发表方法在 DOFA 地图上不失效时照实写明，并收窄"不是随机森林特有"的说法。
- **不是防御性实验，也不加重复组。** 数据、划分方式和方法都与随机森林设定相同，只换地图。

## 2. 数据（只读，不下载）

数据取自 GROUND 的 score 文件：`0-Xiayin_research/15_RobustGeoCP/real_world_study/results_8ds/<ds>_dofa_scores.npz`。

| 设定 | 文件 | 单元 | 结局 | 分组（配额与区域） | 方向 |
|---|---|---|---|---|---|
| dofa_poverty | poverty_dofa | 7,535 个 DHS 调查簇（13 国，2014–2016） | 财富指数（DHS，仅限本研究使用） | 13 国：国名取自 `poverty_scores.npz` 的 `region_country`，两文件顺序已核对一致 | bottom |
| dofa_m3leo | m3leo_dofa | 7,156 块 | 生物量（ESA CCI v4 产品，属模型输出） | 6 区 | top |
| dofa_floods | urbansarfloods_dofa | 6,471 块 | 洪水比例（人工标注掩膜） | 16 个事件区 | top |
| dofa_usavars | usavars_dofa | 11,996 块，按项目惯例用固定种子抽 10,000 | 树冠覆盖（Landsat 产品） | 10 个分区 | bottom |

- **特征。** 用 `embed`：冻结的 DOFA 骨干在线性头之前的输出，经 PCA 降到 32 维。主成分在全部样本上计算，不用标签（GROUND `common/trainer.py:train_and_score`，`pca_dim=32`）。GROUND 训练的线性头及其 `y_pred` 都不用。
- **单元。** 全部样本都参与划分，包括 GROUND 的 `in_train`：冻结特征与标签无关，我们的头只在自己的训练折上拟合。
- **坐标。** 用地心直角坐标（km，三维），与探索性一臂相同，这样核函数与 k-means 在全球尺度上也有效。
- **标签类型。** 只有 dofa_poverty 与 dofa_floods 是实测或人工标注；M3LEO 与 USAVars 的标签是产品，与 PLACES、ERA5 同样处理，结论主要看实测。

## 3. 地图（每次划分重拟，只用训练折）

- **DOFA 线性探针。**
  - 在训练折上把 32 维特征标准化，做岭回归，不用坐标。
  - 惩罚在 10^-3 到 10^3 的 13 个对数等距值中选，依据训练折内的留一法（GCV）。
  - 这复现了 GROUND 线性探针的做法：GROUND 用 768 维全特征，这里用它的 32 维主成分。
- **其余与随机森林设定逐项相同（`run_main.fit_mu_sigma`）。**
  - 出折残差：训练折内两折交叉拟合，地图为探针。
  - 尺度模型 σ̂：随机森林，特征为经纬度加 32 维嵌入。
  - GeoCP 等方法的带宽：在训练折上调。
  - 回归克里金：探针作趋势，再在出折残差上拟合 GP。
  - QRF、bootstrap 集成、CQR：按已发表定义拟合在同一组特征（经纬度加嵌入）上；它们各自带模型，不用探针。
- **实现。**
  - `QW2_MAP=probe` 时，`experiments/run_main.py` 的 `rf()` 返回探针。
  - DOFA 设定与 `QW2_MAP` 不匹配时，`experiments/data.py` 中止运行，防止混用。
  - 随机森林设定的代码路径不变，跑前用已存结果逐项核对。

## 4. 运行（最多 4 个进程）

- **划分。** 100 次随机划分，训练/校准/测试为 40/30/30。种子取各设定在 `data.ALL` 中的位置（追加为 13–16），与随机森林设定同法。
- **规则。** 与随机森林设定相同：
  - 最高或最低的 1%、5%、20%；
  - 按组配额（5%）；
  - 交通成本预算；
  - 两种发布掩码。
- **方法。**
  - 运行 `run_main.py`、`run_baselines.py`、`run_scgeoccp.py`（l1 固定为 0.95）、`run_mondrian.py` 中的全部方法。
  - 计数沿用论文图 4 的方法集：split CP、GeoCP、LSCP、RLCP、回归克里金、CQR、normalized CP、bootstrap 集成、BY、JOMI、SC-GeoCCP。
- **命令**（在 `jomi_winners_curse/` 下运行）：
  ```
  QW2_MAP=probe python -u experiments/run_main.py 100 4 dofa_poverty dofa_m3leo dofa_floods dofa_usavars
  QW2_MAP=probe python -u experiments/run_baselines.py 100 4 dofa_poverty dofa_m3leo dofa_floods dofa_usavars
  QW2_MAP=probe L1_FIXED=0.95 python -u experiments/run_scgeoccp.py 100 4 dofa_poverty dofa_m3leo dofa_floods dofa_usavars
  QW2_MAP=probe python -u experiments/run_mondrian.py 100 4 dofa_poverty dofa_m3leo dofa_floods dofa_usavars
  QW2_MAP=probe python -u experiments/run_cfbh.py 100 4 dofa_poverty
  ```
- **地图精度。** 报告前 10 次划分测试单元上的 R²，与论文惯例相同。同时报告 GROUND 线性头的 `y_pred` 在同一批、且不在 GROUND `in_train` 中的测试单元上的 R²，作对照，只报告不判读。

## 5. 终点

- **E1（主终点）。** 政策方向 5% 规则下，被选测试单元上 90% 区间的覆盖（100 次划分合并）。对每个方法，数 4 个设定中有几个低于 0.85，并单独报告两个实测或标注设定（dofa_poverty、dofa_floods）中的个数。
- **次终点。**
  - E2：1%、20%、配额、预算、掩码下同样的计数。
  - E3：有限区间宽度相对 JOMI 的中位比；JOMI 无穷区间的比例。
  - E4：配额下最差组的覆盖（JOMI 与 SC-GeoCCP）。
  - E5：dofa_poverty 的资助错误发现率（cfBH、区间阈值、点预测）。
  - E6：探针 R²。

## 6. 预测（跑前写定）

- **P1。** 5% 规则下，JOMI 与 SC-GeoCCP 在 4 个设定中都不低于 0.85。这是保证所在；100 次划分合并后应接近或高于 0.90。
- **P2。** split CP 与 GeoCP 至少在 1 个设定低于 0.85。探索性一臂中，GeoCP 在 7 个设定里有 5 个低于 0.85。
- **P3。** normalized CP 与 bootstrap 集成低于 0.85 的设定数少于 split CP；随机森林设定中两者各为 1/11。

## 7. 判读规则（用不取整的合并点估计）

- **R1（P1 不成立：JOMI 或 SC-GeoCCP 有设定低于 0.85）。** 先核对代码与设计前提：地图是否只用训练折，规则是否只用协变量。确认无误后照实报告，并说明是哪个前提被违背，还是抽样波动。
- **R2（P2 不成立：split CP 与 GeoCP 都不低于 0.85）。** 照实写明 DOFA 地图上没有出现失效，连同探针精度和被选单元的残差一起报告；正文相应收窄"不是随机森林特有"的说法。
- **R3。** 不论正负，Results 都按第 8 节模板写入；补充结果放 SI。
- **R4。** 探索性一臂中 DOFA 的旧结果不再在正文报告，留在 SI 的执行记录里；ResNet 的 3 个设定仍标为探索性。

## 8. Results 模板（方括号内数字跑后填入）

- **正面：** "With maps from a frozen geospatial foundation model (DOFA) and a linear probe refitted on the training fold of every split, split CP covered under 0.85 of the selected 5% in [X] of four data sources and GeoCP in [Y], whereas JOMI and SC-GeoCCP fell below 0.85 in none (SI Appendix, [section])."
- **负面（P2 不成立）：** "With maps from a frozen geospatial foundation model (DOFA) and a linear probe refitted on the training fold of every split, no published method fell below 0.85 at the selected 5% in the four data sources, and JOMI and SC-GeoCCP met the target (SI Appendix, [section])."
- **部分（P1 不成立）：** 按 R1 写出被违背的前提。

## 9. 限制（跑前已知）

- 32 维主成分比 GROUND 的 768 维全特征信息少，探针精度可能偏低；用 R² 对照说明。
- 4 个设定中有 2 个的标签是产品；结论主要看 dofa_poverty 与 dofa_floods。
- DHS 的 13 国子集与主分析的 23 国不同，dofa_poverty 不替代主分析的 DHS 设定。
- GROUND 的 PCA 在全部样本（含其 `in_train`）上计算，但不用标签，对校准单元和候选单元是对称的，不影响保证。
- 设计外检验、压力测试、审计规划和多哥案例检验的是设计前提，与地图类型无关，不在本设计内。

## 执行记录

- 2026-11-08 写定本设计；冻结副本与 SHA-256 见 `results/cache/DOFA_FORMAL_DESIGN_frozen.md` 与 `results/cache/DOFA_FORMAL_FROZEN_SHA256.txt`（代码改完、核对通过后、运行前生成）。
- 2026-11-08 07:15（CST）冻结，冻结前做了三项检查：
  - **随机森林路径不变。** 新代码重跑 CAMELS 第 1 次划分，81 行结果与 `results/main_camels.csv` 逐项相同（覆盖、宽度、合计、越界数）。
  - **防混用开关有效。** 带 `QW2_MAP=probe` 读 camels、不带它读 dofa_floods，都被 `data.load` 拦下。
  - **冒烟测试。** 4 个 DOFA 设定各跑 1 次划分（`run_main.one`，第 1 次划分，不写文件），每次 5–7 秒。只用于确认代码能运行，数值没有用于任何设计决定，设计在此之前已写定，冒烟后未改动。
- 改动前的脚本存于 `paper/archive_2026-11-08/`（`data.py`、`run_main.py`、`run_cfbh.py`、`heldout_r2.py`）。
- 2026-11-08 07:15 第一次启动失败：shell 没有把含参数的解释器变量拆开，5 步都未执行，也没有写出结果文件。07:16 用完整命令重新启动。冻结的设计与代码未改。
- 2026-11-08 07:18（CST）运行期间、看到任何正式结果之前，写好判定脚本 `experiments/judge_dofa_formal.py`，SHA-256 b60f24fe03bc75f7a35991ea7e36a38833f0c1cb61ea8e027951433b5e4d8e31。脚本按第 5–7 节计算终点和判定；方法集同图 4；配额与预算按各设定的政策方向；Mondrian 只报告，不计数。
- 2026-11-08 07:16–07:33 第二次启动：5 步都跑完了 dofa_poverty，输出都是完整的 100 次划分，已核对。随后每步在往 Google Drive 上的日志写第一行汇总时以退出码 1 结束，日志为空，其余 3 个设定未跑。排查时在终端直接跑了 `run_main.py 2 2 dofa_m3leo`（2 次划分），运行正常。这 2 次划分的数值看到了，但没有用于任何决定；它写出的 `main_dofa_m3leo.csv` 已被正式运行覆盖。07:34 起把其余 3 个设定的 4 步重跑一遍，日志改写到本地临时目录，跑完再拷回 `results/logs/`。冻结的设计与代码未改。
- 2026-11-08 08:14:49 其余 3 个设定（M3LEO、洪水、USAVars）的 4 步重跑完成，退出码都是 0。08:15:06 把日志拷回 `results/logs/dofa_*_m3leo_floods_usavars.log`。启动命令如下，在 `jomi_winners_curse/` 下设 `QW2_MAP=probe`；dofa_poverty 的 5 步在第二次启动时已跑完：
  - `python -u -B experiments/run_main.py 100 4 <设定>`
  - `python -u -B experiments/run_baselines.py 100 4 <设定>`
  - `L1_FIXED=0.95 python -u -B experiments/run_scgeoccp.py 100 4 <设定>`
  - `python -u -B experiments/run_mondrian.py 100 4 <设定>`
  - `python -u -B experiments/run_cfbh.py 100 4 dofa_poverty`
- 2026-11-08 08:15:39 判定脚本只运行一次，输出 `results/RESULTS_DOFA_FORMAL.md`。判定结果：
  - P1 成立：JOMI 0.9003–0.9079，SC-GeoCCP 0.9127–0.9285。
  - P2 成立：split CP 与 GeoCP 各在 4 个设定中有 2 个低于 0.85，在 2 个实测设定中各 1 个。
  - P3 不成立：normalized CP 有 3 个设定低于 0.85，不少于 split CP 的 2 个；bootstrap 集成 0 个。
- 判定后的校验值（判定脚本、它依赖的 `figures/fig_data.py` 与 `figures/style.py`、输入缓存 `results/cache/dofa_*.npz`、全部输出与结果文件）存于 `results/cache/DOFA_FORMAL_POSTRUN_SHA256.txt`。工作副本的校验值和冻结记录对不上，这是预期的：工作副本在末尾追加了执行记录；冻结部分（第 1–108 行）与 `results/cache/DOFA_FORMAL_DESIGN_frozen.md` 逐字节相同。
- **更正前面的记录。**
  - 冒烟测试其实写出了 4 个输入缓存 `results/cache/dofa_*.npz`（07:14:40–07:15:00），与独立重建的结果相同；前面说"不写文件"不准确。
  - 排查运行还写出了 `state_dofa_m3leo.csv` 和 `units_dofa_m3leo_rep0.npz`（07:33:31），随后都被正式运行覆盖。
  - 第二次启动的 5 步都没有留下 dofa_poverty 的汇总输出（日志为空）。输出完整，汇总可以从文件重算；退出码只见于命令行输出，没有存成文件。
- **判定中的口径说明。** 以下都是判定脚本在看到结果前定下的，数据和代码都未偏离。
  - **P2 的读法。** 按 R2 的反面理解为"split CP 与 GeoCP 至少有一个在至少一个设定低于 0.85"。本次两者各 2 个，按更严的读法也成立。
  - **E2 配额与预算的方向。** 两个政策方向为 bottom 的设定（扶贫、USAVars）都按政策方向取。若按随机森林设定的惯例（只有 DHS 取 bottom），计数为：
    - 配额：split CP 3、normalized CP 2、CQR 2、GeoCP 2、LSCP 3、RLCP 3、克里金 3；
    - 预算：split CP 2、normalized CP 2、CQR 3、GeoCP 3、LSCP 3、RLCP 2、克里金 3；
    - JOMI、SC-GeoCCP、BY、QRF、bootstrap 在两种口径下都是 0。
  - **方法清单。** 设计第 4 节漏列 QRF；判定按图 4 的 12 种方法计数，QRF 为 0/4。
  - **洪水数据的标签。** 是数据集的参考分割掩膜，取 FO+FU 的比例。本地资料没有写明是否人工标注，正文改称参考掩膜（reference masks）。
- **2026-11-08 独立复核。** 只读代理，用时约 24 分钟，没有阻断性问题。冻结校验、代码审查、全部数字（E1–E6）和判定都独立复现：
  - 随机森林路径逐项不变；
  - 探针只用训练折，只用嵌入列；
  - 替换测试或校准标签后，地图和选出的单元都不变；
  - 各输出都是完整的 100 次划分。

  复核提出的解读要点和处理如下：
  1. **实测标签的证据只来自一个低精度设定。** 两个实测设定中只有洪水失效（split CP 0.309，GeoCP 0.534），而它的探针 R² 只有 0.125。DHS 的探针 R² 为 0.495，没有失效，选最低处时反而过度覆盖（0.966）。处理：正文与随机森林设定一样，同时报告实测设定的计数（1/2），并写明选最低处时的过度覆盖。
  2. **P3 不成立要在模板旁写明。** normalized CP 在随机森林设定中只有 1/11 低于 0.85，在 DOFA 设定中有 3/4。
  3. **E4 的区域结果。** 洪水最差事件区：JOMI 0.430，SC-GeoCCP 0.620。凡涉及 SC-GeoCCP 保护区域的说法，都要写明这里它保护不了每个区域。
  4. **QRF、bootstrap、CQR 的区间以各自的森林或梯度提升模型为中心，并用了坐标。** 它们的 0/4 说明不了探针地图的区间。


- **2026-11-08 写入稿件**（用户："按此写入"；组织方式见 `paper/GROUND_LESSONS.md` 第 6 节）。第 8 节的模板句原样用于 Results 新小节 "The failure persists with a foundation-model map"，X = 2，Y = 2。写入位置：
  - 图 4a 的 DOFA 块；
  - Methods 的数据、地图、评估协议、命题 2 计数（DOFA 2,800 次运行全部一致）、扶贫 K = 113；
  - SI [S29]，含设计、判定、解读和表 S29a–f；
  - 摘要、引言、Discussion、Significance、两封投稿信。

  按 R4，旧的探索性 DOFA 结果不再进正文。表 S11 与 S24 仍保留它的行，标为 "DOFA head, exploratory"，因为 S24 是预注册的四图分析。补充图 9 改为只含 ResNet。
- **2026-11-08 写入后的独立核查。** 只读代理，用时约 22 分钟。E1–E6 的所有数字、探针 R²、命题 2 计数、单元数和图件 Source Data 都独立复现，没有取整错误。提出的问题及处理如下：
  - **错误（2 处）。** Results 与 SI 都写了"只换了地图"。这不对：三个数据源只在这里用，DHS 是另一个子集，σ̂ 的输入不同，配额和预算取政策方向。已改为"划分、规则、方法同随机森林设定，配额与预算取政策方向"，SI 写明三处不同。
  - **会误导（3 处）。**
    - 解读要点 3 原先只写在 SI。已在 Results 区域一节和 Discussion 写明：DOFA 下 SC-GeoCCP 把洪水最差事件区从 0.43 提到 0.62。
    - 图 4 图注已补：QRF、bootstrap、CQR 的区间以各自模型为中心，不以探针为中心。
    - 固定分箱原先只报了 5%。已补 1%（USAVars 0.49）和配额（0.76），并写明 20% 与预算下都不低于 0.85。
  - **小问题（13 处），已全部改。**
    - P3 的说法改为以 split CP 为参照。
    - R² 与实测设定的对应关系写明：洪水 0.13，失效；DHS 0.49，未失效。
    - 调参值的文件名改正。
    - `quota_bottom` 只在两个设定中运行，已写明。
    - 表 S29a 的"不计数"出处改正。
    - 判定脚本的写定时间改为"看到任何正式结果之前"。
    - 投稿信写明伴随研究的预测还用于两张 SI 表和一处精度对照。
    - Methods 的 SC-GeoCCP 规则清单和资格线所用的折改正。
    - "影像数据源"的计数统一为四个。
    - DHS 改称"同一数据源的不同子集"。
    - 图 3 的 FIA 配额由"9 quotas"改为 10 个州（FIA 表重建后每州都有配额；与 DOFA 无关）。
    - SN3.8 写明 ResNet 的 k-means 单元按非训练单元计算。

- **2026-11-10 用户说明 GROUND 不算伴随研究。** 投稿信不再提 GROUND；正文、SI 与图注改称 "a separate study"。上面 11-08 核查中"投稿信写明伴随研究的预测……"一条因此不再适用。
