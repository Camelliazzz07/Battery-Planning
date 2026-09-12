"""问题一：重新求解并独立复核，输出检验指标，不改写正式结果表。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from openpyxl import load_workbook
import solve_question1 as model


def main():
    out = Path(__file__).resolve().parent / '辅助输出' / '模型检验'
    out.mkdir(parents=True, exist_ok=True)
    frame = model.read_data(model.ATTACHMENT_DIR / '附件1.xlsx')
    solver = model.milp
    captured = []
    def record(*args, **kwargs):
        result = solver(*args, **kwargs)
        captured.append(result)
        return result
    model.milp = record
    try:
        solution = model.solve_model(frame)
    finally:
        model.milp = solver
    raw = captured[0]
    x, c, d, e = [np.asarray(solution[k]) for k in ('purchase', 'charge', 'discharge', 'soc')]
    rows = []
    def check(name, value, tolerance):
        rows.append({'检查项目': name, '误差或违规量': float(value), '容差': tolerance,
                     '通过': bool(np.isfinite(value) and value <= tolerance)})
    check('供给不足量（千瓦时）', max(0, -(x + frame['光伏电量_kWh'].to_numpy() + d - frame['负载电量_kWh'].to_numpy() - c).min()), 1e-5)
    check('储电量递推误差（千瓦时）', np.max(abs(np.diff(e) - .9*c + d/.9)), 1e-5)
    check('储电量越界（千瓦时）', max(0, 1200-e.min(), e.max()-10800), 1e-5)
    check('充放电量越界（千瓦时）', max(0, -c.min(), -d.min(), c.max()-5000/6, d.max()-5000/6), 1e-5)
    check('购电量负值（千瓦时）', max(0, -x.min()), 1e-5)
    check('同时充放电时段数', np.count_nonzero((c>1e-5)&(d>1e-5)), 0)
    check('首尾电量误差（千瓦时）', max(abs(e[0]-6000), abs(e[-1]-6000)), 1e-5)
    cost = float(np.dot(frame['电价'], x))
    check('费用复算与求解目标差（元）', abs(cost-raw.fun), 1e-4)
    check('求解相对最优性间隙', raw.mip_gap, 1e-9+1e-12)
    # 官方表只保存分块充放电；购电按模板顺序还原后独立复算账单。
    book = load_workbook(model.PROJECT_ROOT / '提交结果' / 'result1.xlsx', data_only=True)
    stored = np.array([book['计划购电量'].cell(i, 2).value for i in range(2, 146)], float)
    stored = np.r_[stored[-1], stored[:-1]]
    stored_cost = float(np.dot(frame['电价'], stored))
    check('正式表费用与重算最优费用差（元）', abs(stored_cost-cost), 1e-3)
    book.close()
    report = pd.DataFrame(rows)
    report.to_csv(out / '问题一检验指标.csv', index=False, encoding='utf-8-sig')
    summary = {'最优购电费_元': cost, '求解下界_元': float(raw.mip_dual_bound),
               '相对间隙': float(raw.mip_gap), '最大递推误差': rows[1]['误差或违规量'],
               '检查数': len(rows), '全部通过': bool(report['通过'].all())}
    (out / '问题一检验摘要.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))
    if not summary['全部通过']:
        raise RuntimeError('问题一检验未全部通过，请查看指标表。')


if __name__ == '__main__':
    main()
