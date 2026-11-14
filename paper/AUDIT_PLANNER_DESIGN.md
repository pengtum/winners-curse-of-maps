# 审计规划器：设计与判读规则（2026-11-02，跑前写定）

用户 2026-11-02 选择 "Audit planner first"（`paper/SUBMISSION_REVIEW_NCS_2026-11-01.md` 第 3 节第 2 条）。无新数据，无新地图。

## 1. 必要性审查
- **检验的论断。** 摘要末句 "small audits give unbounded intervals and weak regional floors"，和 Results "Coverage in every region is bought with data"。两处都只说了代价，没说要多大审计。机构要在审计前定预算，这是投稿信里 "computable tools for decisions" 的落脚点。
- **无论结果正负，正文都会变。** 正面：多一节和一张图，可以给出审计量。负面：定向审计无效或信用损失太大时，稿件写明审计只能铺满全图。
- **不是防御性实验。** 不加强基线，不加重复组，地图与划分沿用现有种子。

## 2. 规划公式（不用结局；`excp_select/planner.py`，`tests/test_planner.py` 全枚举核对）
top-K，n 个审计单元，m 个候选，键给定地图后固定且互异：
- **L1 有限区间。** |R| ~ 负超几何(n+m, n, K+1)；Pr(有限) = Pr(|R| ≥ τ/α)；E|R| = n(K+1)/(m+1)。
- **L2 下界。** E[1/(|R|+1)] = m(1−ε)/{K(n+1)}，ε = C(n+m−K, m−K−1)/C(n+m, m−1)。于是定理 1 的下界 = τ − l₁m(1−ε)/{(n+1)M_u}，M_u 是地区 u 在被选单元中的期望质量（组指示即期望被选单元数）。选择比例 s = K/m、地区份额 w = M_u/K 固定时，所需审计 n ≈ l₁/{(τ−f)·s·w}，与地图大小无关。配额规则的混合参考集没有闭式，用同一公式作近似，并与重抽比较。
- **L3 总量。** 选择感知 SE ≈ s_R√{K(1+K/n_R)}，n_R 取 E|R|；精确界不用结局下界（C₀=0）需要 K/(n_R+K) ≤ δ。
- **L4 定向审计。** 审计从给定地图后固定的层（预测最高的 q 比例，由总体的协变量决定，与划分无关）内随机抽取。层本身是一个有限总体的随机划分，top-K 只要层内候选 ≥ K+1 就落在层内，所以 L1–L3、定理与引理在 m = 层内候选数下全部成立，无需新定理。

## 3. 碳汇项目试验（`experiments/run_audit_planner.py carbon 20 100 4`）
- **总体。** 6,000 个非训练样地，项目 K = 30。site 地图 20 张（`run_carbon.py` 种子）；ESA CCI 20 个子集（`run_carbon_product.py` 种子，1 km 窗口）。各 100 次审计，新种子 `[6201, 地图类, 折, 次, 层大小]`。
- **臂（q, n）。** 全图 92、186、1,869、3,000；前 5% 层 92、150；前 10% 层 186、300。1,869、92、186 是规划器算出的 Pr(有限) ≥ 0.9 的审计量；150、300 是审计半层。
- **未审计的全部样地都是候选**，规则从中取前 30。
- **分数。** |e|/σ̂；CCI 用 |e|/max(SD, 0.1×子集外平均 SD)，因 SD 可以为 0，这个下限与森林 σ̂ 的做法一致，只用于区间。总量公式仍用原始 SD 层。
- **终点。** 有限比例；JOMI 覆盖（无穷计覆盖，并报有限内覆盖）；同一审计上 split CP 覆盖；VCS 规则与总是扣减下的超额风险：调查校正、选择感知、精确界（δ = 1/3）；发放吨数与真实总量比；被选样地真实总量；计划 SE 与实现 SE。

## 4. 地区下界试验（`run_audit_planner.py floors 200 4 <ds>`）
图 6c、8b 的设定：`run_scg_slack.py` 的训练折与 200 次重抽（相同种子），候选 m = 3,000 不变，审计取其 3,000 个校准单元的前 n 个，n = 300、1,000、3,000。DHS 用国家配额（最贫 5%），FIA 用州配额（`fia_db`）。JOMI 与 SC-GeoCCP（l₁ = 0.95）。下界、地区覆盖、无穷比例。DHS 只输出国家与 bump 级汇总。

## 5. 存量运行的计划与实现（无新运行）
`results/fia_db/carbon_site.csv.gz` 与 `product/carbon_product_win.csv.gz`，15 个 审计比例 × 选择比例 组合。计划 SE = 误差层 SE × √(1 + K/E|R|)，实现 SE = `se_sa`。

## 6. 判读规则（用不取整的合并点估计）
- **D1 公式。** 碳汇各臂有限比例与 L1 之差 ≤ 3 个二项 SE。全枚举已证，这一条只防实现错误；失败就是 bug，先查代码。
- **D2 下界。** 被选 ≥ 50 次的地区，SC-GeoCCP 覆盖 + 2 个二项 SE ≥ 下界。理论保证；失败就先查代码。
- **D3 定向审计。** 规划器的两个定向臂（5% 层 92、10% 层 186），site 与 CCI 都满足才算成立：
  - (a) JOMI 覆盖 ≥ 0.88；
  - (b) 选择感知、总是扣减的超额风险 ≤ 0.383（1/3 + 0.05）；
  - (c) 发放吨数 ≥ 全图 3,000 臂的 0.75 倍。
  (a)(b) 成立而 (c) 不成立，写 "valid at a cost of X% of the credits"。(b) 不成立时不主张中心极限估计，只报精确界。
- **D4 同一审计，没有参考集。** 不预设方向，定向臂上调查校正（VCS 规则、总是扣减）与 split CP 全部报告。
- **D5 误差层能否规划 SE。** 计划/实现 SE 比的中位数在 [0.8, 1.25] 内，写 "the error layer sufficed"；否则写明偏差倍数，并说 SE 规划还要一次小的试点。site 与 CCI 分别判。
- **D6 下界是否过于保守。** 不设阈值。报告 n = 300、1,000、3,000 时最差地区的下界与实现覆盖；若实现覆盖在下界 0.10 以上，写明这是保证的代价而不是实际需求。

## 7. Results 模板（不论方向都写入，括号内数字跑后填入）
- "The size of the audit can be computed before any outcome is measured, because a rule's reference sets depend only on ranks (Methods). A finite 90% interval at the selected units with probability 0.9 takes about 13/s audited units for a selected share s, and a regional floor f takes about 0.95/{(0.90 − f) s w} for a region holding a share w of the selected units; neither depends on the size of the map."
- "A 30-plot project among 6,000 FIA plots needed [1,869] plots audited across the map, but [92] in the highest-predicted 5%."
- D3 成立："Those audits gave [finite share] finite intervals, coverage [x] and a selection-aware over-crediting risk of [y] on site maps ([z] for ESA CCI), and credited [c] of what 3,000 audits did."
- D3 不成立："Targeted audits [failed criterion], so we do not recommend them [for ...]."
- D4："With the same audit and no reference set, ... [数字]."
- D6："Floors were conservative: [数字]."

## 8. 限制（跑前已知）
- 定向审计的保证只针对层内候选上的这条规则；事后换规则或换方向，要重新设计审计。
- 被审计的样地在本设计中不进入项目；定向审计把最好的样地分走一部分，这就是 D3(c) 的代价。
- 下界 L2 是保证的代价，不等于实际覆盖。
- 配额规则的参考集混合多组阈值，L2 对它是近似。

## 执行记录
冻结：`results/cache/AUDIT_PLANNER_DESIGN_frozen.md`，设计、规划器、运行脚本与测试的 SHA-256 在 `results/cache/AUDIT_PLANNER_FROZEN_SHA256.txt`（本地摘要，无外部时间戳）。

**2026-11-02 运行。** 碳汇 13 秒，DHS 下界 133 秒，FIA 下界 81 秒。运行脚本跑后未改。

**汇总脚本的更正（`make_audit_planner.py` 不在冻结范围内，第一次汇总后改）。** 判读前发现两处，都是汇总错误，不涉及运行：
1. 定向臂的公式用了输出里的 `m`（全部候选），应为层内候选 m = q×6000 − n（设计 L4）。
2. 计划 SE 用了层内平均误差层，设计第 5 节写的是被选单元的误差层 SE，改为 `se_layer`。
更正后 D3 不受影响（D3 不用这两个量）。

**判读。**
- D1：16 个臂中，非退化概率的最大偏离 2.11；但 site 全图 3,000 这一臂按高斯尺度是 3.24（2,000 次中 2 次无穷，期望 0.28；精确二项 p = 0.033），严格说超过冻结的 3 个 SE。按规则先查代码：按运行种子只用排序重放，8 个臂的 |R| 与运行输出逐次相同，所以排除了并列值和实现错误；定律已由全枚举证明。全图臂 |R| 方差高于定律（最大 1.13 倍）；另取 12 组种子重放未重复，判为这批种子的偶然波动。SI 写明。汇总脚本里 "pf 在 (0.01, 0.99) 内" 的限定是跑后加的，SI 同时报告未限定的结果。
- D2：0/411。
- D3：成立。(a) 最低 0.931；(b) 最高 0.334；(c) 最低 1.002。注意 (c) 的比较对象是全图 3,000 臂，它把一半样地移出候选，项目本身不同，所以正文改报发放占真实总量的比例，并报与全图 1,869 臂相比项目真实总量的变化（+0.15%、+0.17%）。与全图 92 臂相比是 −3.60%、+2.49%（只在 SI）。
- D4：定向臂上调查校正 0.515、0.432（±0.12、±0.11），split CP 0.915、0.932，都已写入。
- D5：site 0.988，CCI 1.001，都在 [0.8, 1.25] 内，写 "predicted"。
- D6：最小地区的下界很保守，但不是处处如此；3,000 时 7/10 州、4/23 国距下界不到 0.10。

**独立核查（1 个只读代理，读代码、设计与稿件，未告知我的结论）。** 定理、证明与运行代码未见错误；`tests/test_planner.py` 通过。逐条处理：
1. 摘要和投稿信的 "replaced" 改为 "matched"，写明 6,000 块样地；"20-fold" 只是算术（约 1/q），Discussion 改为倒数关系。
2. Results 把 0.32、0.33 标明为渐近结果（参考集约 14 块），并报精确界风险 0.006、0.008。
3. "Floors price a guarantee, not coverage" 太强：它用中位下界比最低覆盖，不是同一地区。按冻结 D6 的 "conservative" 重写，并加反面事实。
4. CCI 的 +2.4% 是错的（由取整比值算出），实为 +2.49% ± 2.53，与零无差别；正文改用全图 1,869 臂作比较。
5. Discussion 的 "preferentially labelled audits" 加限定，避免与定向审计矛盾；定向审计限定 top-K。
6. 13/s 写明 s 是剩余候选的份额，K 小时要更多；写明 0.95 是 l₁，以及下调 l₁ 的代价。
7. Methods 写明 bump 的近似误差达 0.15。
8. SI 写明摘要范围与汇总脚本的两次更正；新增重放脚本 `experiments/check_audit_replay.py`；`RESULTS_AUDIT_PLANNER.md` 现在也打印未筛选的 D1。
9. D5 写 "sufficed"，并给出 1.34。SI 与 md 个别值差 0.001，是四舍五入差异，未改。
