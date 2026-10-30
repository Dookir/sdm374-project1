import numpy as np,pandas as pd
FEATURES=['b1_cost', 'b1_carbon', 'b1_peak', 'b1_discomfort', 'b1_degradation', 'b1_ramping', 'b1_resilience', 'b2_cost', 'b2_carbon', 'b2_peak', 'b2_discomfort', 'b2_degradation', 'b2_ramping', 'b2_resilience', 'b3_cost', 'b3_carbon', 'b3_peak', 'b3_discomfort', 'b3_degradation', 'b3_ramping', 'b3_resilience']

def load_table(path):
    """读取均值表：X 是 N×21 权重，y 是 N 个跨周平均 ramping。"""
    table = pd.read_csv(path, float_precision="round_trip", dtype={"candidate_id": str})
    if set(table.columns) != {"candidate_id", *FEATURES, "ramping"}:
        raise ValueError("需要均值表：candidate_id、21个权重列、ramping")
    if len(table) < 2 or table.candidate_id.isna().any() or table.candidate_id.duplicated().any():
        raise ValueError("至少需要两组数据，且 candidate_id 不能缺失或重复")
    table = table.sort_values("candidate_id", kind="stable").reset_index(drop=True)
    X = table[FEATURES].to_numpy(dtype=float)  # 编号不作为输入特征。
    y = table.ramping.to_numpy(dtype=float)
    sums = X.reshape(-1, 3, 7).sum(axis=2)
    if not np.isfinite(X).all() or (X < 0).any() or not np.allclose(sums, 1, atol=1e-12, rtol=0):
        raise ValueError("每栋楼的7个权重必须有限、非负且和为1")
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("ramping 必须是有限的非负数")
    return table.candidate_id.tolist(), X, y
