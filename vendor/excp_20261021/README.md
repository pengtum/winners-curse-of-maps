# excp，2026-10-21 快照（只读副本）

**来源。** `design_based_spatial_cp/excp_pkg/build/lib/excp/`，即 2026-10-21 15:31 setuptools 构建时留下的副本。2026-10-29 原样复制到这里，只删去了 `__pycache__`。`design_based_spatial_cp` 下的文件没有改动。

**为什么需要。** 这个包在 2026-10-23 改名为 `unigeocp`，2026-10-28 又改名为 `geounicp`，`family` 的接口也改了：`geocp_v2`、`excp_default`、`excp_rlcp` 都已不存在。旧名 `excp` 现在只是指向 `geounicp` 的别名，所以按现在的包，`experiments/run_main.py`、`run_baselines.py`、`run_scgeoccp.py` 都会报 `AttributeError`，跑不起来。

**核对（2026-10-29）。** 用这个快照重算第 0 次划分，与存档结果逐项比对：

| 存档结果 | 结果 |
|---|---|
| `results/main_seattle.csv`、`results/main_fia.csv` | 81 行、25 列全部一致，最大绝对差 1e-11 |
| `results/bl_main_fia.csv`（10-25 生成） | 80 行一致 |
| `results/scg_main_fia_l1fix.csv`（10-23 生成） | 6 行一致 |
| `results/cache/carbon_parts/site_f00_s100.csv.gz`（10-26 生成） | 1,500 行、31 列一致 |

其他脚本（run_scorefns、run_cfbh、run_decisions、run_geofm 等）没有逐一核对。

**用法。** `excp_select/__init__.py` 默认把本目录放在 `sys.path` 最前面。设环境变量 `QW2_EXCP=live` 则改用 `design_based_spatial_cp/excp_pkg` 下现在的包。

**SHA-256：**

```
04af175c0d0880124578af41c6eab2a0de81acc1b0fec57eb59c44773518440d  excp/__init__.py
50743aa884d474a4c7bc45c66d3fae56af99f9c01e1a0a3650e04ab7351a8054  excp/cluster.py
16df061ec57c5ec493068c913879b854688c5ac3bf3f3d7f4d4c07a5325db02e  excp/designs.py
518ea2562473f37af3a74bb8e532680b813af58142f4a532ccb08626a5ae616d  excp/engines.py
fdb34049b69afd40dbe357815bc9cd4b3561362900f01b9091ecc284a94c92b7  excp/family.py
3b03d89a5daf7b64049e5c864210693ff73809ae9ab00ffd03cd103aa97c53bd  excp/localizers.py
41923e8f6065278384edcb8f8c92982293cdc4ccf567ab3364f2fa0502b614a8  excp/robust.py
e4561f9ca7b704ab0c504784ea814ff8c8b084182a8afac782addc638bf231f0  excp/scores.py
702bb2a5ed0fdb62630eb4d509e4674f1709a1bff18f11a96a3d2838467e64c3  excp/select.py
828b7ab68c580a6cdfd5975fb5ff2135b1edfca0dba5dff3b8f77a0b719944f0  excp/weights.py
```
