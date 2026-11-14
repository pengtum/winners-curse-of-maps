# 压力测试设计：选择规则用到地图之外的信息（2026-10-30，跑前写定）

> **来源。** `paper/NARRATIVE_STRATEGY_2026-10-30.md` 第六节第 1 项；用户 2026-10-30 选择"先做压力测试实验"。
>
> **跑前的数值计算**（都不读项目数据）：
> - `experiments/stress_power_gaussian.py`：高斯模型的渐近值；
> - `experiments/stress_power_finite.py` 及其汇总 `experiments/stress_power_finite_summary.py`：同一高斯模型下按全流程模拟的有限样本值，300 次模拟研究，结果在 `results/stress_power_finite_300.json`，由设计审查写出；
> - N 与留出 R² 取自已报告的 `results/maps_r2_rf_summary.csv`、`maps_r2_rf_per_rep.csv`；臂 B 的平移所用的政策规则覆盖，取自已报告的 `results/RESULTS_MONDRIAN_RULES.md`。
>
> **代码检查。**
> - 跑前用 2 次划分（rep 0、1，属于正式的 100 次）做代码检查。设 `STRESS_OUT` 指向 scratch 下新建的空目录；R < 10 时脚本只打印复现核查、行数与耗时，不设 `STRESS_OUT` 就直接退出。
> - 不对检查输出运行 `make_stress.py`，不打开检查目录里的 CSV。检查通过后立即删除该目录，然后才开始正式运行。
> - 检查之后只允许修改代码错误。
> - 本文件第一版写定后曾做过一次代码检查，见"执行记录"。

## 一、要回答的问题

到目前为止，规则都按地图自己的预测（或由它导出的分数）排序。唯一的例外是 B3：它按只用坐标的第二张地图排序。那张地图比区间地图更不准，携带的残差信息很少，天真 Mondrian 只在 1 个设定低于 0.85（Seattle 0.848，[S24]）。

本文的论点是校准必须跟着规则走，而不是跟着地图走。理论预测：规则用到了地图之外、与地图残差相关的信息时，按地图预测值分箱会失效，规则自己的参考集仍然成立。

本实验检验这个预测。臂 A 是 B3 的加强版，在 B3 的阴性结果之后设计：选点地图比区间地图更准。报告臂 A 时同时给出 B3。结果不论正负都报告。

## 二、两臂设计

**共同设定。**
- 两臂都与 `run_main.py` 用同样的种子划分：训练 40%、校准 30%、测试 30%，100 次。
- 地图只在训练折上拟合（`run_main.fit_mu_sigma`）；箱的边界只用训练折。
- α = 0.1；+∞ 计为覆盖。
- 判定一律用不取整的合并点估计 Σ n_cov / Σ n_sel。
- 表中另报两项描述性统计，都不进入任何判定，也不用来改判阈值附近的结果：按划分重抽样 2,000 次得到的 95% 区间；JOMI 与天真 Mondrian 的逐划分配对差（均值与标准误）。

### 臂 A：两张地图，更好的地图选点，较弱的地图给区间（主臂，真实数据）

**数据。** FIA 主设定，10,000 个样地。fia_post2000（2000 年及以后清查的样地，[S21]）作为敏感性分析，同样运行，只报告，不判定。

**地图**（特征分层与 `run_carbon.DROP` 相同；列号 0 起计，对应 Xrf = [lon, lat, X1..X12]）：
- **full**：主设定地图，全部特征，留出 R² 0.915。与 `run_main` 用同一随机数流，μ̂、σ̂ 逐位相同。
- **site+height**：去掉 Xrf[:, [6, 7, 8, 11]]，即 X5 断面积、X6 每公顷株数、X7 平均胸径、X10 林分大小级；保留平均树高（X8），作为激光雷达冠层高度的替身。
- **site**：再去掉 Xrf[:, 9]（平均树高）。
- site 与 site+height 的拟合方式与 `experiments/heldout_r2.py` 相同：每张地图各用一个新的 `default_rng([2027, rep, 数据集序号])`，先做同一次划分置换，再调用 `fit_mu_sigma`。所以它们就是 `results/maps_r2_rf_per_rep.csv` 中的地图，留出 R² 为 0.389 与 0.660（rep 0–9 的均值）。三张森林因此共用随机种子与自助样本，对本检验没有实质影响。

**区间。** 所有方法的区间都来自 site 地图（μ̂_s, σ̂_s）。这张地图较弱，但可以用于未调查的土地。

**选择规则。** 选测试单元的最高 5%（主）与 1%（次），排序依据分三种：
- site 地图自身的预测：归因对照；
- site+height 地图的预测：主终点；
- full 地图的预测：次要，极端情形。

**现实对应。**
- 选点者掌握比公开地图更好的信息（例如保护机构或买方用自有的激光雷达冠层高度图挑选生物量最高的地块），却沿用公开地图随图发布的误差层报告不确定性。
- 这是"有利选择"：被选地块的真值多半高于公开地图，区间预计在上侧失效（真值高于上界），对超额签发（下侧）反而保守。本臂检验的是区间在被选单元上的有效性，不检验超额签发；超额签发由总量层面检验（[S16]）。
- 参考集校准要求，选点所用的分数在审计样地与候选地块上都可得：选点者要申报规则并提供这些预测值（不必公开模型），且该地图不得用审计样地训练。选点者不申报时，任何只依赖公开地图的误差层，都不能在被选地块上保证覆盖。
- site 地图的输入中，坐标、地形与森林类型有全覆盖的图层可以替代；立地等级与地文类别是样地记录值。平均树高是样地实测值，比激光雷达更准，所以 site+height 代表信息较强的一端。

### 臂 B：地图加野外信号（剂量反应，11 个设定）

**数据。** 11 个随机森林设定（`run_mondrian.NAMES`）；μ̂、σ̂ 与 `run_main` 逐位相同。

**野外信号。** Z_τ = y + τ·√(π/2)·σ̂·ε。
- ε ~ N(0,1) 独立，随机数流 [2035, rep, 数据集序号]，各 τ 共用同一组 ε。
- σ̂ 是平均绝对残差的模型，只用训练折。√(π/2)·σ̂ 在误差为正态时等于地图误差的标准差，所以 τ 在名义上是野外信号误差与地图误差之比。
- **有效性。** Z 对每个单元用同一机制生成：ε 独立于划分，σ̂ 只用训练折，所以 (X, Z, y) 在校准与测试单元之间可交换。规则只用测试单元的 (μ̂, Z) 排序，参考集只用校准单元的 (μ̂, Z)，都不用校准或测试结局本身，所以 JOMI 的保证成立。Z 依赖 y 不违反"规则不用校准标签"，因为它是对每个单元以相同方式测得的协变量。现实中这要求三点：野外信号在审计样地上以与候选地块相同的方式、在不知道审计结果时测得；选点所用的地图不用审计样地训练；规则在看到审计结果之前定下。臂 A 的 μ̂_h、μ̂_f 只用训练折拟合，同样满足这些条件。
- **实际的 τ。** 若 √(π/2)·σ̂ 恰为地图误差的标准差，corr(Z − μ̂, y − μ̂) = 1/√(1 + τ²)，即 0.894、0.707、0.447。σ̂ 拟合的是半折森林的折外 |残差|，并有下限；残差重尾时，√(π/2)·E|r| 小于标准差。所以表中逐设定报告 corr、τ_eff = √(1/corr² − 1) 与 sd(Z − y)/sd(y − μ̂)。判定一律用名义 τ；不论在代码检查还是正式运行中看到 corr，都不重新标定噪声。名义 τ = 0.5 而 τ_eff 不在 0.35–0.65 的设定，在表中标出。

**规则。** 选测试单元的 5%，政策方向 d = ±1，排序依据：
- S_τ = d·(τ² μ̂ + Z_τ)/(τ² + 1)，即地图与野外信号按精度加权合并，τ = 0.5、1、2；
- **对照**：S_0 = d·(μ̂ + Z_0)/2，Z_0 = μ̂ + √(π/2)·σ̂·ε，与 y 无关，只给排序加噪声。噪声系数 0.5 不小于各 τ 下 S_τ 的噪声系数 τ/(τ²+1)；
- **policy**：d·μ̂，用于复现核查。

**其他。**
- 区间来自主地图（μ̂, σ̂）。
- 臂 B 不扩展到 1%：小设定的 K 只有 2–9。
- 对 PLACES 与 ERA5 而言，野外信号是对模型输出或再分析值的模拟测量，不写成对真实患病率或气温的测量。

### 两臂共用的方法与记录

**方法：**
- split CP（|y − μ̂|）；
- normalized CP（|y − μ̂|/σ̂）；
- Mondrian CP，区间地图预测值的十箱（天真版，与 [S23] 主版本相同）与二十箱，学生化分数；
- Mondrian CP，规则所用分数的十箱与二十箱（知情版，与 B 相同），学生化分数；
- JOMI，学生化与加性分数，参考集按实际规则：{i ∈ C : 分数_i > 第 K+1 大的测试分数}。

**分箱。** 箱的切点取训练折上的样本内拟合值，沿用 [S23] 的做法不改。site 地图样本内 R² 0.76、留出 0.39，所以校准与测试单元在各箱并不均分，最高箱的校准单元少于名义值。逐行记录两项：最极端箱（政策方向）中的校准单元数及其占比；含被选单元的箱中最少的校准单元数。

**另记：**
- 上、下两侧的失效数（y − μ̂ > 半宽；y − μ̂ < −半宽）；
- 有限半宽与 +∞ 比例；
- JOMI 参考集大小（均值、最小值）；
- 被选单元的平均残差 d·(y − μ̂)，原值与以 √(π/2)·σ̂ 为单位的值（机制核查）；
- 与按区间地图自身预测选点的重合率；
- 臂 A 中各地图的留出 R²。

## 三、终点

**主终点 A。** FIA，按 site+height 地图选最高 5%，区间来自 site 地图。记：
- N：天真 Mondrian（μ̂_s 十箱，学生化）的覆盖；
- J：JOMI（学生化）的覆盖；
- N₀：site 地图按自身预测选最高 5% 时的天真 Mondrian 覆盖，作为归因对照；
- 机制核查：被选单元的平均残差（以 √(π/2)·σ̂_s 为单位）；
- 描述：逐划分配对差（J − N）的均值与标准误。

**主终点 B。** τ = 0.5 时，天真 Mondrian（μ̂ 十箱，学生化）与 JOMI（学生化）覆盖低于 0.85 的设定数，按三个口径各报一次：
- 全部 11 个设定；
- 6 个实测结局设定（DHS 有影像、DHS 无影像、FIA、CAMELS、得票率、Seattle）；
- 5 个实测数据源（两张 DHS 地图任一低于 0.85 记 1 个）。

判定用第三个口径。

**次要终点。**
- 臂 A：full 选点；1% 份额；fia_post2000 敏感性。
- 臂 B：τ = 1、2 与对照。
- 方法：二十箱 μ̂；知情箱；normalized CP 与 split CP。
- 描述：宽度比、+∞ 比例、两侧失效、平均残差、重合率、corr 与 τ_eff。

## 四、预测（跑前写定）

依据：`stress_power_gaussian.py` 的渐近值，以及按同一高斯模型模拟全流程的有限样本值（`stress_power_finite.py`，300 次）。

**臂 A。** 有限样本值，括号内为低于 0.85 的概率：

| 选点地图 | 份额 | 天真十箱 | 天真二十箱 | 知情十箱 | 知情二十箱 | JOMI |
|---|---|---|---|---|---|---|
| site 自身 | 5% | 0.90（0.00） | 0.90 | 0.90 | 0.90 | 0.90 |
| site 自身 | 1% | 0.90（0.03） | 0.90 | 0.90 | 0.90 | 0.92 |
| site+height | 5% | 0.79（1.00） | 0.80 | 0.88（0.00） | 0.90 | 0.90 |
| site+height | 1% | 0.70（1.00） | 0.71 | 0.81（0.90） | 0.84（0.58） | 0.92 |
| full | 5% | 0.61（1.00） | 0.61 | 0.85（0.61） | 0.90 | 0.90 |
| full | 1% | 0.38（1.00） | 0.39 | 0.68（1.00） | 0.78（0.99） | 0.92 |

在高斯模型中，天真 Mondrian 的区间就是给定 site 地图输入时的精确条件区间。所以预测的失效适用于任何只以区间地图所用信息为条件的方法（CQR、QRF、克里金）。FIA 是重尾、异方差的数据，具体数值会偏离高斯模型。

**臂 B。** 按各设定在 B 中政策规则下的天真十箱覆盖平移（加法近似）：

| 设定（留出 R²；B 中政策规则下的天真十箱覆盖） | τ = 0.5 预测值；低于 0.85 的概率 |
|---|---|
| 肥胖*（0.75；0.913）、DHS 有影像（0.76；0.908）、DHS 无影像（0.78；0.898）、得票率（0.80；0.889） | 0.81–0.82；≥ 0.90 |
| Seattle（0.85；0.907） | 0.85；约 0.4 |
| FIA（0.92；0.890）、CAMELS（0.91；0.898） | 0.86、0.87；各约 0.2 |
| 中风*（0.89；0.918）、糖尿病*（0.93；0.911）、ERA5†（0.95；0.888） | 0.87–0.89；≤ 0.05 |
| 高血压*（0.91；0.940） | 0.91；约 0 |

（* PLACES 模型输出；† 再分析。设定与数据键：肥胖* health_obesity、DHS 有影像 poverty、DHS 无影像 poverty_lab、得票率 voting、Seattle seattle、中风* health_stroke、高血压* health_bphigh、FIA fia、CAMELS camels、糖尿病* health_diabetes、ERA5 era5。`stress_power_gaussian.py` 输出的 "Z alone" 一列只作参考，不是预注册规则。）

**预测记分表。** `make_stress.py` 逐条判定"符合 / 不符合"，写入表 S25 脚注；一律用不取整的合并点估计。预测不影响第五节的判定；不符合的预测在 SI 逐条写明。
- **P-A1**：天真十箱（μ̂_s）在 site+height 与 full 选点、5% 与 1% 四格都低于 0.85。
- **P-A2**：JOMI（学生化与加性）在 3 种选点 × 2 个份额的 12 格都不低于 0.87。
- **P-A3**：天真二十箱在 P-A1 的四格都低于 0.85，且与十箱相差不超过 0.03（只计有限区间时同样判定）。
- **P-A4**：天真十箱在两个份额下都满足 site 自身 > site+height > full。
- **P-A5**：在 P-A1 的四格中，天真十箱 < 知情十箱 < JOMI。
- **P-A6**：5% 下（site+height、full），知情二十箱与 JOMI 相差不超过 0.02。
- **P-A7**：1% 下（site+height、full），知情二十箱比 JOMI 低 0.03 以上。原因是二十箱的顶箱宽 5%，比 1% 的选择粗。
- **P-A8**（只作记录）：天真十箱与有限样本高斯值（0.79、0.70、0.61、0.38）之差不超过 0.10。
- **P-B1**：τ = 0.5 时，天真十箱在 11 个设定中有 4–5 个低于 0.85（概率约 0.75；90% 范围 3–6）。
- **P-B2**：τ = 1 时 0–1 个；τ = 2 与对照 0 个。
- **P-B3**：JOMI 在 3 个 τ 与对照下都是 0 个。
- **P-B4**：至少 9 个设定中，天真十箱严格满足 覆盖(τ = 0.5) < 覆盖(τ = 1) < 覆盖(τ = 2)。policy 与对照不在这条链上。
- **P-B5**：至少 10 个设定中，天真十箱在对照与 policy 下相差不超过 0.02。
- **P-B6**：知情十箱与二十箱在 3 个 τ 下都是 0 个设定低于 0.85。

## 五、解读规则（跑前写定）

### 1. 主终点 A 的判定

N、N₀、J 都是 100 次划分合并、不取整的点估计。

**核查触发（先于判定）。** 两臂任一格的 JOMI（学生化或加性）低于 0.87：先停下，检查参考集、泄漏与种子，经过记入执行记录；核查结束前不写任何结果。核查后按下面的阈值判定。

- **A1 阳性**：N < 0.85，N₀ ≥ 0.85，J ≥ 0.85。Results 在固定箱一段之后自成一小段；Discussion 阶梯句后补依据（模板见规则 7）。
- **A2 阴性**：N ≥ 0.85，J ≥ 0.85。见规则 2。
- **A3 基线也失效**：N < 0.85，N₀ < 0.85，J ≥ 0.85。失效不能归于选点地图的额外信息。Results 一句同时给出 N₀ 与 N（模板见规则 7），Discussion 不补依据。
- **A4 参考集失效**：J < 0.85，核查后仍成立。不论 N 是多少，Results 一句照实写出 J 与 N（模板见规则 7），不写"参考集成立"，Discussion 不补依据。

### 2. A2 阴性

- Results 一句（模板见规则 7），写明与事先预测相反。
- Discussion 阶梯句后加一个限定（模板见规则 7）。
- 全部结果写入 SI。
- 若被选单元的平均残差低于 0.6 个误差标准差（高斯模型下，天真覆盖跌破 0.85 约需这么大的偏移），注明规则携带的残差信息不足，不作为原则的反证。

### 3. 臂 B 的判定

τ = 0.5，100 次划分合并、不取整。

- **B 阳性**：同时满足三条：天真 Mondrian 在 5 个实测数据源中 ≥ 2 个低于 0.85；JOMI 在 11 个设定中 0 个；对照规则下天真 Mondrian 在 11 个设定中 0 个。Results 在 A 的句子之后加一句剂量反应（模板见规则 7）：先写实测数据源的计数，再写 11 个设定的计数，注明 PLACES（*）为模型输出、ERA5 为再分析，并写出计入设定中 JOMI 的最大 +∞ 比例。
- **B 对照失效**：前两条成立，但对照下天真 Mondrian 有设定低于 0.85。Results 不写剂量反应，只在 A 的句子后用规则 7 的模板照实给出两个计数；SI 写明只给排序加噪声也使固定箱低于 0.85，失效不能全部归于野外信号携带的残差信息。
- **B 阴性**：天真 Mondrian 在实测数据源中 ≤ 1 个低于 0.85。写入 SI；Results 在 A 的句子后用规则 7 的模板照实给出这个计数。
- **JOMI 低于 0.87**：按规则 1 的核查触发处理。

### 4. 知情箱

知情箱是次要终点。它按规则所用分数分箱，是参考集的近似，没有保证；判定方式同天真箱，逐格报告。

- A1 的 Results 段落用一个从句给出 5% 下知情十箱与二十箱的覆盖，不论高低。
- 5% 下知情箱成立：写成"规则感知的分箱可以近似参考集，但没有保证"，不声称 JOMI 在经验上胜过它（沿用 B 的解读规则 2）。
- 1% 下知情二十箱低于 0.85（P-A7）：只在 SI 写成与"整箱入选"的等价关系一致（箱比被选比例宽时不再逼近参考集），并给出最高箱的校准占比；不写成 JOMI 的经验优势。
- 5% 下知情箱也低于 0.85（与 P-A5、P-A6 相反）：SI 照实写明；Results 从句照常给出数字，不加评价。

### 5. 报告范围

- 不论正负，全部结果写入 SI 新小节 [S25]，表中标明主终点、次要终点和第四节的预测。
- 次要终点不触发 Results 中的句子；Results 只用规则 7 模板中的数字。
- 表 S25b、S25c 每格同时给出 +∞ 比例。JOMI 或知情二十箱 +∞ 比例超过 10% 的设定（预计 CAMELS）要标出，其覆盖不作为"成立"的证据。

### 6. 复现与完成

- 复现核查不通过就不写任何结果，先查原因；修正只限代码错误，用同一种子重跑，记入执行记录。
- 主终点 B 只在 11 个设定全部完成、核查全部通过后判定。
- 跑后不改种子、划分、τ、份额、箱数与阈值，也不补跑其他变体来替换任何终点。

### 7. 措辞模板

数字由 `make_stress.py` 填入；放在 Results 固定箱一段之后，不另加形容。

- **A1：** "When a map with canopy height chose the plots (held-out R² {R2h}) and the site map, which can be applied to unvisited land (R² {R2s}), supplied the intervals, fixed bins of the site map's prediction covered {N} of the 5% of FIA plots selected, whose site-map residuals averaged {r} error spreads, and the misses lay mostly {side} the interval; the rule's own reference set covered {J}, and binning by the selecting map's prediction instead covered {I10} with ten bins and {I20} with twenty, without a guarantee ([S25]). This is one dataset, and the guarantee assumes that the rule, including the selecting map's predictions at the audited plots, is declared; a second map built from coordinates alone, less accurate than the interval map, had left fixed bins below 0.85 in one of 11 settings ([S24])."
- **A1 时 Discussion 阶梯句后加：** " (when a more accurate map chose the FIA plots, {N} against {J}; [S25])"
- **A2：** "Fixed bins of the site map's prediction also held when a map with canopy height chose the plots: they covered {N} of the 5% of FIA plots selected, and the rule's own reference set {J} ([S25]), contrary to our prediction."
- **A2 时 Discussion 阶梯句后加：** "; in our tests they nonetheless held when a more accurate map chose the plots ([S25])"
- **A3：** "Fixed bins of the site map's prediction fell short when the site map chose its own plots ({N0}), and further when a map with canopy height chose them ({N}; reference set {J}; [S25])."
- **A4：** "With a map with canopy height choosing the plots, the rule's own reference set covered {J} and fixed bins {N} of the selected FIA plots ([S25])."
- **B 阳性：** "When the rule also ranked a simulated field measurement whose error was set to half the map's, fixed bins fell below 0.85 in {b5} of the 5 data sources with measured outcomes ({b11} of all 11 settings, including modelled and reanalysis outcomes), and the reference set in none (up to {jinf} of its intervals infinite; [S25])."
- **B 阴性：** "A simulated field measurement moved fixed bins below 0.85 in {b5} of the 5 measured data sources ([S25])."
- **B 对照失效：** "A simulated field measurement moved fixed bins below 0.85 in {b5} of the 5 measured data sources, but ranking noise alone did so in {c11} of 11 settings ([S25])."

## 六、复现核查（写进脚本，失败即停）

**臂 A：**
- full 地图在其自身 5% 规则下、用自身分数时，JOMI stud 与 SCP stud 逐划分的 (n_sel, n_cov) 等于 `results/main_<ds>.csv` 的 top5；若 `md_main_<ds>.csv` 存在，天真 Mondrian 也要相等。
- fia 的 rep 0–9 中，site 与 site+height 地图的留出 R² 等于 `results/maps_r2_rf_per_rep.csv`（差 < 1e-9）。
- site 自身规则下，规则分数分箱与天真分箱的半宽逐单元相同。

**臂 B：**
- policy 规则下，JOMI stud、SCP stud 与天真 Mondrian 逐划分等于 `results/main_<ds>.csv` 与 `md_main_<ds>.csv`。
- 规则分数分箱与天真分箱的半宽逐单元相同。

## 七、运行与输出

**脚本** `experiments/run_stress.py`：
- `twomap <R> <workers> [fia|fia_post2000]` → `results/st_twomap_<ds>.csv`；
- `signal <R> <workers> [datasets]` → `results/st_signal_<ds>.csv`。

R = 100，4 个进程，后台运行。

**汇总** `experiments/make_stress.py`：
- 输出 `results/RESULTS_STRESS.md`（中文，含判定与记分表）、`results/RESULTS_STRESS_SI.md`（表 S25a–c）、`results/stress_summary.csv`；
- 按规则 7 填好的 Results / Discussion 句子写入 `RESULTS_STRESS.md`。

跑完由多代理对抗核查：代码（泄漏、种子、参考集）、数字、是否按本文件的规则解读。

**冻结。** 本版写定后，在执行记录写下冻结时间，以及 `shasum -a 256 paper/STRESS_TEST_DESIGN.md experiments/stress_power_gaussian.py experiments/stress_power_finite.py experiments/run_stress.py experiments/make_stress.py` 的输出。代码检查前、正式运行前各核对一次。冻结后第一至七节的任何改动都记为偏离，写明时间与理由。代码与本文件不一致时以本文件为准。

## 执行记录

- **11:06–11:08** 写第一版设计（高斯渐近预测；终点、预测与解读规则较粗）。同时启动两路设计审查（有效性；预注册）。
- **11:09–11:11**（审查进行中）写 `run_stress.py` 第一版，并做了一次代码检查：
  - 命令：`STRESS_OUT=scratchpad/stress_check`，`twomap 2 2` 与 `signal 2 2 camels poverty_lab`。
  - 结果：复现断言全部通过；脚本只打印行数与耗时。
  - 为测试代码，对检查输出运行过一次第一版 `make_stress.py`，标准输出直接丢弃。
  - 没有打开或读取任何覆盖数字。
  - 检查输出已于 **11:42:32** 整个删除。
- **审查结论。** 两路审查共提出 5 个 blocker、11 个 major，第二版设计全部吸收。要点：
  - 主终点 A 加入 N₀ 归因对照与 A1–A4；
  - 主终点 B 按 5 个实测数据源判定，并要求对照成立；
  - 有利选择的方向写正（上侧失效），并写明申报规则的前提；
  - site、site+height 地图改用 `heldout_r2.py` 的种子，加入 R² 复现核查；
  - 补记两侧失效、参考集大小与箱内校准单元数；
  - 预测记分表；
  - 各结果的 Results 模板，阴性同样写入正文；
  - 臂 B 的 τ_eff；
  - 代码检查必须写入 scratch；
  - fia_post2000 敏感性；
  - 冻结哈希。
- 第一版代码检查发生在审查之前，看不到任何结果，所以第二版的修改不受结果影响。

- **冻结（11:47:22）。** 第二版设计与代码冻结；以下哈希在代码检查前与正式运行前各核对一次（设计文件的哈希按冻结时的内容计，本条记录写入后文件哈希会变，核对时以去掉本条之后的内容为准，故另存一份冻结副本于 `results/cache/STRESS_TEST_DESIGN_frozen.md`；后注：正式运行前实际未单独核对，见跑后补记的"偏离 0"）：
```
17e64fc9648086d64c31ba329219e0f2c73d328a75fc1e791f01fff486f04d64  paper/STRESS_TEST_DESIGN.md
78942207873eb684d7cc9a5e136186b4a276b6bc628d0a8ce9443343f5acaf7b  experiments/stress_power_gaussian.py
fdb87cab461223e7446129fce19aeae0ce60587961e9edb62631232b850b244c  experiments/stress_power_finite.py
92da694ba65203554db27c5ac4f866274b5c262d1e28225b26df4ce60681d23b  experiments/run_stress.py
533cbfb3559a71262b1c5138e528385532302a0318067304facca785f672360b  experiments/make_stress.py
```
- **第二次代码检查（11:47–11:48，冻结之后）。** 先核对哈希，与冻结值一致。
  - 命令：`STRESS_OUT=scratchpad/stress_check2`，依次运行 `twomap 2 2 fia`、`twomap 2 2 fia_post2000`、`signal 2 2`（11 个设定）。
  - 全部复现断言通过，包括：
    - fia rep 0–1 中 site 与 site+height 地图的留出 R² 与 `maps_r2_rf_per_rep.csv` 相同；
    - full 地图的 JOMI stud、SCP stud、天真 Mondrian；
    - site 自身规则与 policy 规则下，规则分数分箱与天真分箱逐单元相同；
    - 臂 B 的 11 个设定。
  - 只打印了行数与耗时；未运行 `make_stress.py`，未打开 CSV。
  - 输出于 **11:48:31** 删除。

### 跑后补记（2026-10-30 12:41 起写入，之后逐条补记；时间均为本地时间，据会话记录与文件时间核对）

- **漏记补上：冻结前的假数据测试（11:47:05）。** 冻结前，用随机生成的假数据（`scratchpad/stress_fake`，种子 7，不读项目数据）跑了一次第二版 `make_stress.py`，只为测试代码路径；打印出的是假数字。该目录随冻结命令于 11:47:21 删除。
- **正式运行（11:48:50–12:00:27）。** 在第二次代码检查的输出删除（11:48:31）之后开始。
  - **偏离 0（程序）。** 第七节与冻结条目要求正式运行前再核对一次哈希，实际没有单独核对（原因：疏漏）。最近一次核对在 11:47:31（第二次代码检查之前），此后到运行开始之间未改动代码。跑后核对（下一条）和文件修改时间（`run_stress.py` 及其导入的项目模块都早于 11:47:22）都表明运行用的是冻结代码。
  - 命令：工作目录 /tmp；解释器 `~/miniforge3/envs/favela/bin/python -u`；脚本用绝对路径 `<项目根>/experiments/run_stress.py`；未设 `STRESS_OUT`，未用 `-O` 或 `PYTHONOPTIMIZE`。参数依次为 `twomap 100 4 fia`、`twomap 100 4 fia_post2000`、`signal 100 4`（11 个设定）。后台运行，日志 `scratchpad/stress_run.log`，退出码 0；13 次运行都打印 "checks passed"。
- **跑后哈希核对（12:00:35）。** `run_stress.py`、`make_stress.py`、两个功效脚本和冻结副本的哈希与冻结值一致。
- **汇总（12:00:41）。** 运行冻结版 `make_stress.py`（533cbfb3…）。
  - 核查触发没有触发：JOMI 最低 0.895（臂 A，1% full，加性）、0.898（fia_post2000）、0.897（臂 B，FIA τ = 0.5，加性）。
  - 判定 **A1**：N 0.7421，N₀ 0.9013，J 0.9013。
  - 判定 **B 阳性**：5 个实测数据源中 2 个低于 0.85（得票率 0.7855；FIA 0.8485，即 12,728/15,000）；11 个设定中 3 个；JOMI 0 个；对照 0 个。
  - 记分表不符合：P-A7、P-B1、P-B2。
- **跑后对抗核查（12:02–12:26）。** 三路并行：代码、数字、预注册。
  - **代码。** 独立重写两臂，不导入 `run_stress`、`refsets`、`methods`、`run_mondrian`、`run_baselines`；覆盖 fia rep 0–4、fia_post2000 rep 0–2、臂 B 的 FIA 与 poverty 全部 100 次划分、CAMELS rep 0–4。逐划分计数和诊断量全部与存储的 CSV 相同。没有泄漏：静态追踪、打乱校准与测试结局后重拟合、探针都确认了这一点。新规则的参考集与交换定义一致。
  - **数字。** 独立重算表 S25a 的 156 格（FIA 与 fia_post2000 各 78）、S25b 的 45 格、S25c 的 55 行、N/N₀/J、配对差、臂 B 的计数和 14 条记分，全部一致。另用自己的代码从头重算 18 个设定-划分（臂 A：FIA 0、1、57；臂 B：CAMELS 0–5、FIA 0–3、中风* 0–2、DHS 有影像 0–1；共 744 行），也一致。
  - **预注册。** 冻结哈希核验通过：冻结副本与设计文件前 274 行同为 17e64fc9…，第一至七节没有改动。A1 与 B 阳性在冻结规则和冻结顺序下都正确；`make_stress.py` 中的 9 个模板与第七节一字不差。
  - **问题。** 1 个 blocker 和若干 major、minor 问题，都在汇总与报告层面，处理见下。
- **偏离 1：jinf 的口径（汇总步骤的代码更正；12:26:53 修改，12:31:08 首次重跑）。**
  - 冻结版 `make_stress.py` 只在天真十箱低于 0.85 的 3 个设定里取 JOMI 的最大 +∞ 比例，得 0.00。
  - B 阳性模板 "the reference set in none (up to {jinf} of its intervals infinite)" 中的 "none" 指全部 11 个设定；规则 3 要求写出"计入设定中 JOMI 的最大 +∞ 比例"。
  - 规则 3 的"计入设定"有两种读法：天真十箱失效的设定，或计数 JOMI 的全部设定。按"代码与本文件不一致时以本文件为准"，采用与模板 "in none" 一致的后一种读法，改为在全部 11 个设定上取最大值：0.29（CAMELS，‡）。这个读法是跑后、看到结果之后选定的；它报告的是较大的值，不影响判定。模板措辞不变，没有重跑。
  - SI 同时给出两个值：天真十箱失效的 3 个设定中为 0.00，11 个设定中最大 0.29。
- **偏离 2：{r} 的显示精度（12:37:23 修改，12:38:21 首次重跑）。** A1 句中被选单元的平均残差原按一位小数填入（"1.0"），改为两位（"0.97"），与表 S25a 一致。数值本身不变。
- **报告补充（不影响判定）。** 按第二节、规则 4–5 和第四节，补齐冻结汇总脚本漏掉的报告内容：
  - 表 S25a：主终点标 §；图注写明 R² 的口径和第四节的预测值；主终点 A 补描述性区间与配对差。
  - 表 S25b：每格加最大 +∞ 比例（+∞ 只出现在 CAMELS）。
  - 表 S25c：加知情十箱和 sd(Z − y)/sd(y − μ̂)；对照行的 τ_eff 改为显示"—"（policy 行原已如此）；−0.00 显示为 0.00。
  - 新表 S25d：τ = 0.5 逐设定列出预测值、四位小数的天真十箱覆盖及区间、JOMI 及区间、配对差、两侧失效、平均残差、重合率和宽度比。
  - 新表 S25e（排在 S25d 之后）：臂 A 的宽度比、参考集大小、极端箱的校准单元数与占比、含被选单元的箱中最少的校准单元数、两侧失效、原值平均残差。
  - 记分表改为逐条英文说明。
  - 新输出 `results/stress_paired.csv`：每条规则与份额（臂 A，含 fia_post2000）或规则与设定（臂 B）一行，共 67 行，给出逐划分配对差（JOMI − 天真十箱）的均值与标准误。
  - 跑后选定、只作描述：按设定独立重抽划分，计算实测数据源计数的分布。先按 10,000 次（1、2、3、4、5 个的频率依次为 0.12、0.38、0.37、0.11、0.01），第二轮核查指出第二位小数受蒙特卡洛误差影响，改为 100,000 次（见下）。
  - `RESULTS_STRESS.md`：改正表头说明；CAMELS 行加 ‡ 与 +∞ 比例。
- **汇总脚本的版本（第一轮核查后）。** 所列为运行时间；中间版本未保留，哈希前缀只作标识。中间版本 12:31:08（63cdaf01…）、12:32:15（aa5cc26d…）、12:38:21（21f8d3eb…）、12:40:20（b2b04de7…）、12:40:38（a3bf0157…）。第一轮的最后版本 12:44:31（第二轮核查后又改，见下）：
```
2b0270842496aa764a88258470ea34237cddc0b7ee787b4f1de0099dc85f4837  experiments/make_stress.py
f4ca1fcf2422cfd44d0bc7fb551c813507442e37d05897c308357c3a320050d2  results/RESULTS_STRESS.md
5b2ec76e44e7346044068cdfa9213791e59d68adf081199836697f707069618d  results/RESULTS_STRESS_SI.md
ba38f97eab66495a20316d7b32be3a039b99259264d47daab878f90d740abd5c  results/stress_summary.csv
979945f472f75e5af7a1067463a259033eed2289fcf89f369390a7ce1747f7ca  results/stress_paired.csv
```
  每次重跑的判定、计数和模板中的数字都相同，只有上述两处偏离改变了填入的值。
- **冻结版汇总脚本的留存与对照（12:52）。** 冻结版 `make_stress.py` 已被跑后的修改覆盖。现从会话记录还原：最后一次整体写入在 11:46:44，之后到冻结前没有修改。还原件的哈希与冻结值 533cbfb3… 一致，另存为 `results/cache/make_stress_frozen.py.txt`。用它在同一批正式输出上重跑（输出写入 scratch）：判定相同（A1、B 阳性）。`RESULTS_STRESS.md` 的差别在表头与显示（§、−0.00）、新增内容（计数表每格的最大 +∞ 比例、CAMELS 的 ‡ 与 0.29、B 判定后 JOMI 的 +∞ 比例一句）以及 jinf 与 {r}；`stress_summary.csv` 只有对照行的 tau_eff 不同（现为空，原为无意义的数）。
- **复现断言的覆盖范围。**
  - "规则分数分箱等于天真分箱"这条断言，在臂 A 与臂 B 的 8 个 top 方向设定里是函数与自身比较；只在 3 个 bottom 方向设定（poverty、poverty_lab、era5）里检验了索引与符号。
  - 脚本内的断言覆盖与主运行共用的代码。压力测试特有的部分由上面的独立重写核验：区间来自 site 地图；选点分数同时用于测试排序和参考集；ε 流；S_τ 与对照公式；新规则下的知情箱。
  - 断言是裸 `assert`，日志无条件打印 "checks passed"。正式运行没有用 `-O`（命令见上），核查者也从存储的 CSV 事后核验了断言条件。
- **阈值附近。** B 阳性取决于 FIA：0.84853（12,728/15,000），离 0.85 差 22 个单元；FIA ≥ 0.85 时结果会是 B 阴性。DHS 有影像（0.85013）和中风*（0.85029）在阈值上方。三者的描述性区间都包含 0.85。按冻结规则用点估计判定，不改判，SI 照实写明。
- **R² 的口径。** A1 句中的 R² 是 100 次划分在测试单元上的均值（0.40、0.67）；第二节所引的 0.389、0.660 是前十次划分的均值（Methods 的惯例）；碳实验每种特征集 20 张地图（0.39、0.67、0.92），是另一组地图。SI 写明。
- **冻结范围之外的文件（跑后补记 SHA-256，修改时间均早于 11:47:22）。**
```
a668cc13d88ee32d081afed98c5c60aadec3e64632967cb2bae2ac711dfb0752  experiments/data.py
7cb765cc96988fa93420c78518cc9cf0f644f68e8c5a93d951b42b50931628dd  experiments/run_main.py
7c9fc39d179d053fe1ddd08eb923b2891492b6cdd284dcf67f32062c2c25ae7b  experiments/run_baselines.py
560234fad1cff82d38192bbc33aaa27025a78809530645eb704d1aba148542a7  experiments/run_mondrian.py
ee0af783b4fc8226abc59487aa1f3b2911b09ac558bc9e88de0e98509d5f4eb9  experiments/run_carbon.py
24b1fa2ad0e732c6073343fe00b6402c5be1554160b393e494e1c5438363058a  experiments/run_scgeoccp.py
017cad70540148d41398bda37e33d0a1e6fe5f56e3e6982ca4a2e348b9df0ca9  experiments/heldout_r2.py
361086da0e9b9da3e3ba1811651efa58502c2e0d57eb00a0216cdb77c0a36770  experiments/make_mondrian_noise.py
42eeab3593396cd91d0bd77cd55858979540d3089518198558d2242e9a782927  figures/style.py
1c76ca2b7141db4ab1d0389627b92281c975420c6a6dba39093a7fb3e524b0be  experiments/stress_power_finite_summary.py
cd80aafda159795666963123074514a727eb8c6c6bfd46e7f75b1076a5067426  results/stress_power_finite_300.json
18645dcbc6eb815b1c869e269b331016b8591bfa88a2b7086e1e86e4ba0de8fc  excp_select/__init__.py
f6cad2e4741e9ca58b9ce0194228e47a91e2f5e3419f5ce3ab134c2f73c0e74e  excp_select/basis.py
9b8cf7bec91b08e74c99a1f76873637e4ce8d8316bc4d85fee21822217b234a2  excp_select/bruteforce.py
7010ea4f27771924787439182b1badbc67dbadf2b0188360ae8654fdb556dfd1  excp_select/methods.py
5192d6e7f50af9e2190970fc105afd3225dd86e3ffeb3a4db97505f9c7e5a22a  excp_select/refsets.py
a3e75a75a75277b1d0522e7b33831cfbe9f51f5ceed0d555dba5d3026ed43b58  excp_select/rules.py
783b4486ca70976757a1d5bfc809473c3fa580b9ccb38429316fedfc2acc2a1c  excp_select/totals.py
33704472ae5455c5c374565f0b4c6a3db20bdf76b284671c96b0049fc372f4cb  vendor/excp_20261021（cat $(find vendor/excp_20261021 -name "*.py" | LC_ALL=C sort) | shasum -a 256；10 个文件另按 vendor/excp_20261021/README.md 的逐文件 SHA-256 用 shasum -a 256 -c 核对，全部一致。先前此处记的 42e83623… 是"路径字节加文件摘要"逐个累加的自定义值，标准命令无法复现，已替换）
b70b8f9629ac1dae894be6376ad5588a83fae801eb0c7d0ded80741175562d52  results/cache/poverty.npz
2773e760797951e6b30dca110a1f5a413f063162423765b45aec03737676fcc8  results/cache/poverty_lab.npz
178286b534d5314396617aae31056380ee6dfebf87df28452a140e106ca50513  results/cache/fia.npz
c71649a9c9857ed591b28a29c8107cee986d24acae6730d56b022f2714c9f876  results/cache/fia_post2000.npz
e12834bf8dc75e5f321fdc15bf7e64971cdec9dcadfebb8ac33b7c8d3693cf10  results/cache/camels.npz
5de176e374613459633a0e2816c2ac6b0be8efc12dec4bdb9224826017510bf0  results/cache/voting.npz
954c9d2e771074bdae278232764fdc651887467c2d6070403a63884625ec63a5  results/cache/seattle.npz
425b61427a11db426cd01ec9c371800be89ada6c7d1bbe904514758b7a060700  results/cache/era5.npz
76a249de2df59f54d55469da7ef4dfee27dc7ca130e2cc3f0be6d01167ed1fab  results/cache/health_stroke.npz
d70c358dbc1484565a889a28921282752e5305821d190eae7b4310db7e270076  results/cache/health_diabetes.npz
60e38b4ee717e0273065798487bc8cea5a21aa597445013f764180c5125340f9  results/cache/health_bphigh.npz
f35475cbe994f6a13d12b7cfdc174ae37c0de18e9652a9b562753f02f17f5081  results/cache/health_obesity.npz
```
- **正式输出的哈希（12:00:27 之后未再改动）。**
```
f2ff9bf999135197f3be3a6a405805459ae21c2b77897fe6015b7f074313bde5  results/st_twomap_fia.csv
739fb627f13f592825c711cf2d23ba409305e10f28f8f241657fc1f0d61d181a  results/st_twomap_fia_post2000.csv
13c53afec609c71abbe87271fc6123d45255ed2213825a7c2217eca2631b59a1  results/st_signal_camels.csv
14b0fcc86886f0790d38490f301d878a29ae7bce02976b93efc3c9f17cd9f1a6  results/st_signal_era5.csv
911bee42552cca3841f066d71467331632af6cd5cc7c2ec68edb3815d03b8934  results/st_signal_fia.csv
8e417f62c32e564f21f80b20ab4b4f2c6aafb88f32ae68457d5673dd3997480c  results/st_signal_health_bphigh.csv
48f874156514c92c2c9337e56873221a59b73125c7b197290582fa5ee2ed8270  results/st_signal_health_diabetes.csv
f23587928f7b5922e10f792b3b15034aad283193fbfd5ec7ec7df36249d8a060  results/st_signal_health_obesity.csv
ce96b9e0147ba3530645b39c5cbde172026b2b16375020a1a6b86b6a00e9ec57  results/st_signal_health_stroke.csv
42d56b662f627453726e157def1d52d0623ee4646ec8ef426fc93256f5503940  results/st_signal_poverty.csv
ab50bf221274844bd85e1926ff61ce1d1b819f46aa6d01921210ebd0d5cbe17a  results/st_signal_poverty_lab.csv
c480d3c3f6c36d9a0f652807aba4fc2d58b5d33a3f2ea53afd865ec54c3dde12  results/st_signal_seattle.csv
29d71892405de0f79c331b582771fca81e073d768d2ad007b7d938dd7184a6bd  results/st_signal_voting.csv
```
- **第二轮跑后核查（12:53–13:27）。** 四路只读核查：数字、预注册、汇总代码、写作一致性。
  - 没有 blocker。全部数字复核一致。冻结哈希、模板与判定再次核验通过。当前汇总脚本在 scratch 副本上的输出与 `results/` 逐字节相同。
  - 8 条 major、54 条 minor，都在报告、措辞与记录层面。主要几条：
    - SI 说"复现了每次划分的计数"，夸大了独立复算的范围；
    - Results 固定分箱段仍说按第二张地图排序时稳健，与新段冲突；
    - 正文里 "error spread" 有两种单位；
    - Methods 没有 [S25] 的规则说明；
    - SI 的 τ、S、y 与 Methods 的记号冲突；
    - "可用于未调查的土地"与 Methods 对立地、地文类别的说明冲突；
    - 本记录与 STATUS、README 有若干不准确之处（上面各条已改）。
- **补充独立复算（13:31:52–13:32:28）。** 用核查者的独立代码（`scratchpad/stress_verify/indep.py`，进程数由 5 改为 4），重算臂 B 的得票率与肥胖* 全部 100 次划分，逐划分计数与诊断量 0 处不一致。至此，低于 0.85 的三个设定（FIA、得票率、肥胖*）与实测设定中离阈值最近的 DHS 有影像都在全部 100 次划分上独立复现（其余设定见第三轮）。
- **第二轮修改（13:33 起，不影响判定）。**
  - 汇总脚本，显示与标签：
    - SI 中的 τ 改记为 κ（Methods 里 τ = 1 − α）；"field signal" 改为 "field measurement"；y 改为 Y；
    - 对照规则写全，即 d(μ̂ + Z₀)/2；
    - 表 S25a 图注补全第四节的预测值和低于 0.85 的概率，并标明次要终点；
    - κ_eff 的标记由 § 改为 ¶，避免与主终点的 § 重复；
    - S25d 的 ‡ 改为与 S25c 相同的判定，图注补上 *、†、‡；
    - S25d、S25e 注明为跑后制表；表头写明单位；
    - 去掉 S25c 行尾多余的空格；修好文字生成中三处潜在问题（‡ 的条件、设定数、单元素列表）。
  - 汇总脚本，数值：
    - CAMELS 的宽度比改为只用两种方法都有有限区间的划分（0.85 → 0.87），其他设定不变；
    - fia_post2000 的格补上描述性区间等列；
    - 实测数据源计数分布的重抽样由 10,000 次改为 100,000 次，结果为 0.12、0.39、0.37、0.11、0.01。
  - 稿件（模板句不变）：
    - SI [S25] 按核查意见重写。写明独立复算的实际范围、第一次代码检查的汇总运行、偏离 0、jinf 读法在跑后选定、冻结版脚本的还原；改用 κ 记号；"单位"一段前移；去掉重复。
    - [S23] 的读法句和 [S24] 的末句加上指向 [S25] 的范围说明。
    - Methods：加 error spread 的定义、[S25] 的规则及保证成立的理由、R² 的口径；改写 site 属性的说明；误差层一句改为 "over the plots' error spreads ς_j"。
    - Results：固定分箱段去掉"按第二张地图排序时也稳健"一项，首句限定为按地图自身预测排序的规则；小节标题改为 "Flexible uncertainty models and fixed bins cover without a guarantee"；"error spread" 在 normalized CP 一句改为 "error scale"。
    - Discussion：第 2 段加 "map's own"，第 5 段加 "and any map the rule ranks"。
- **第二轮的最后版本（13:35:48 运行；第三轮核查后又改，见下）。**
```
df985530ccaacdfe22de2b64990b9450da2b1be25bca6d9c1dc12dac54c42499  experiments/make_stress.py
ff9ecf61e046200a8fabdd886332a05ed0793a92a9ccd686884ddb93e97f3f29  results/RESULTS_STRESS.md
b853a3d163c0ef9ad170f7ae20991a690aadf1e9afc7feb5ca35a03dd687cc60  results/RESULTS_STRESS_SI.md
1d05de4ca145ba85457d1422e3c7a744793fce1a2d154aa9e26cf55498b200ef  results/stress_summary.csv
979945f472f75e5af7a1067463a259033eed2289fcf89f369390a7ce1747f7ca  results/stress_paired.csv
```
  判定、计数和模板中填入的数字与 12:44:31 的版本相同。
- **勘误（冻结部分第 274 行）。** 第 274 行"第一版代码检查发生在审查之前"与第 256 行"（审查进行中）"不符：第一次代码检查（11:09–11:11）是在两路设计审查进行中、结论返回之前完成的。冻结行不改，以本条为准。该行的结论不变：检查没有看到任何结果，第二版的修改不受结果影响。
- **第三轮跑后核查（13:44–14:20）。** 三路只读核查：数字与代码、预注册与理论、写作一致性。
  - 没有 blocker。冻结哈希、模板与判定再次核验通过；当前汇总脚本的四个输出可逐字节复现；SI [S25] 正文的 97 处数字全部复核一致。
  - 核查者用同一份独立代码（`indep4.py` 的未改副本，输出在 `scratchpad/stress_verify3/numbers_code/indep_run/`），补算其余 7 个臂 B 设定和臂 A 两次运行的全部 100 次划分。
  - 主会话 14:21:34 用自己的比对脚本，把 13 次运行的全部独立重算结果（53,600 行）与存储的结果逐划分比对：计数与诊断量 0 处不一致。至此 13 次运行的全部划分都已独立复现。
  - 1 条 major：[S23] 一句把"固定顶箱近似参考集"写成对所有排序规则成立，与臂 A 矛盾。另有 24 条 minor，都在措辞、记号与记录层面。
- **第三轮修改（14:23 起，不影响判定）。**
  - 汇总脚本（只改显示）：S25a 图注去掉不出现的括号说明，并写明哪些预测有概率、配对差按行给出；S25b 图注改用 x_κ、Z_κ、ς，并写明对照规则是判定条件之一；S25c 的 ‡ 改标在设定名上（与 S25d 一致）；S25a、S25d 写明 ς = √(π/2)·σ̂。重跑后只有 `RESULTS_STRESS_SI.md` 改变，判定文件与两个 CSV 逐字节不变。
  - SI [S25]：
    - 来源一段写明全部划分都已独立复现，补上冻结前的假数据测试，偏离 0 的措辞改清楚，并写明野外噪声有单独的种子流；
    - 记号统一为 ς、x_κ；对照"在 μ̂ 之外不含 Y 的信息"；20 箱固定分箱是 [S24] 之外新增的；
    - 选点地图须不用审计与候选样地训练；site + 冠层高度地图改称"激光雷达选点地图的乐观替身"；
    - 重抽样计数改为逐项列出（0.37、0.11、0.01），与表注一致；
    - 若干语法与清楚度修改。
  - [S23]：顶箱近似一句限定为"对按被分箱预测值排序的规则"；读法句改写，去掉有歧义的 "There"；段末加一句指向 [S24]、[S25]。[S24]：指向 [S25] 的一句移到段末，改为 "strengthens"。
  - Methods：[S25] 一段重写（"选出"而非"排序"；x_κ；ε 每次划分对每个单元抽取、与划分独立；新增 20 箱固定分箱；保证成立的理由写成 swap 交换的协变量，并写明选点地图不得用审计或候选单元拟合）；[S24] 一句写明固定分箱为十箱；site 属性一句改为"两个类别"。
  - Discussion 第 5 段："Label-dependent rules" 改为 "rules that use audited outcomes"，免得读成把基于结局的模拟测量排除在外（+3 词）。
- **最终版（14:52:49 运行）。** 第三轮修改后的复核（14:26–14:52，一个只读代理）没有发现新的事实或数字错误，指出本记录的"最终版"哈希已过时，并提出几处措辞修改；`make_stress.py` 的文档字符串随之改正（`stress_paired.csv` 每条规则与份额或设定一行）。重跑后四个输出与 14:23 的版本逐字节相同。判定、计数和模板中填入的数字与 12:44:31 以来各版本相同。
```
8b6e369e00e2a03913bd475970b8fcd534afd545954e9d8dd58e1b8ac6012384  experiments/make_stress.py
ff9ecf61e046200a8fabdd886332a05ed0793a92a9ccd686884ddb93e97f3f29  results/RESULTS_STRESS.md
04762aafad3c925733fa4c5c607863cbef7567795310e5a1ad9aec3bff7e2d32  results/RESULTS_STRESS_SI.md
1d05de4ca145ba85457d1422e3c7a744793fce1a2d154aa9e26cf55498b200ef  results/stress_summary.csv
979945f472f75e5af7a1067463a259033eed2289fcf89f369390a7ce1747f7ca  results/stress_paired.csv
```
- **迁移归档（2026-10-30 18:12，换设备前）。**
  - 本记录中的 `scratchpad/...` 指旧设备上的会话临时目录，换设备后不存在。关键文件已原样复制到 `results/cache/stress_audit/`（对应关系见其 `README.md`）：
    - 正式运行日志 `stress_run.log`；
    - 三轮核查的输出；
    - 独立重写的代码；
    - 13 次运行全部 100 次划分的独立重算结果。
  - 新写的 `compare_all.py` 用相对路径重做逐划分比对：53,600 行，0 处不一致。
  - 临时目录的全部内容另有压缩包，见 `_handoff/HANDOFF.md`。
  - 为在新设备上运行，`experiments/data.py` 的 `RES` 与 `experiments/run_geofm.py` 的 `GROUND` 由写死的绝对路径改为按项目位置推出。在旧设备上，两者解析出的路径与原来逐字相同，加载数据的冒烟测试通过，结果不受影响。
  - 冻结范围外文件清单中的 `data.py` 哈希因此改变（a668cc13… → f1b96beb…，见 `_handoff/MANIFEST_project.sha256`）。

- **FIADB 重建后的复现（2026-10-30 21:55 CDT 记录，冻结部分与冻结副本未动）。** `experiments/data.py` 因追加 `fia_db` 与 `FIA` 开关，哈希再变（f1b96beb… → cea390c3…）；`run_stress.py` 与 `make_stress.py` 未改（92da694b…、8b6e369e…）。按 `paper/FIA_REBUILD_DESIGN.md` 第 7 节，在 FIADB 9.5 重建表上重跑两臂（`results/st_twomap_fia_db.csv`、`st_signal_fia_db.csv`），冻结的 `make_stress.py` 原样运行于 `results/fia_db/stress/`：臂 A 为 A2（固定十箱 0.890，平均残差 0.31 个误差 sd，规则 2 适用），臂 B 仍为阳性（FIA 0.840）。本设计的判定（A1、B 阳性）基于原表，不改；新表结果是事先未登记的复现，详见重建设计第 9 节。
