"""问题二：复用四方案独立核验，补充分月误差与费用稳定性诊断。"""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')


def main():
    here = Path(__file__).resolve().parent
    aux = here / '辅助输出'
    out = aux / '模型检验'
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(here/'q2_compare.py'), '--skip-run'], check=True, stdout=subprocess.DEVNULL)
    frames = {}
    for name in ('baseline', 'revised'):
        df = pd.read_csv(aux/'方案对照'/name/'q2_timeseries.csv')
        df['月份'] = pd.to_datetime(df['日期']).dt.month
        df['费用'] = df['电价_元每kWh'] * (df['计划购电量_kWh'] + 5*df['紧急购电量_kWh'])
        frames[name] = df
    base, df = frames['baseline'], frames['revised']
    if not df[['日期', '区间编号']].equals(base[['日期', '区间编号']]):
        raise ValueError('对照日期与时段不一致')
    rows = []
    for month, g in df.groupby('月份'):
        row = {'月份': int(month)}
        for label, actual, forecast in (
            ('负荷', '实际负载_kW', '预测负载_kW'),
            ('光伏', '实际光伏_kW', '预测光伏_kW'),
            ('净负荷', '实际净负荷_kWh', '预测净负荷_kWh'),
        ):
            error = (g[actual]-g[forecast]).to_numpy() * (6 if label=='净负荷' else 1)
            row.update({label+'平均绝对误差_千瓦': np.mean(abs(error)),
                        label+'均方根误差_千瓦': np.sqrt(np.mean(error**2)),
                        label+'平均偏差_千瓦': np.mean(error)})
        row['区间覆盖率'] = g['实际净负荷_kWh'].between(g['净负荷预测下界_kWh']-1e-6, g['净负荷预测上界_kWh']+1e-6).mean()
        row['节约费用_元'] = base.loc[base['月份']==month,'费用'].sum()-g['费用'].sum()
        row['紧急购电量_千瓦时'] = g['紧急购电量_kWh'].sum()
        rows.append(row)
    monthly = pd.DataFrame(rows)
    monthly.to_csv(out/'问题二分月检验.csv', index=False, encoding='utf-8-sig')
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
                         'axes.unicode_minus':False, 'svg.fonttype':'path', 'font.size':10})
    fig, axes = plt.subplots(1,2,figsize=(10.6,3.7),layout='constrained')
    months = monthly['月份']
    for label, color in [('负荷','#51999F'),('光伏','#ED8D5A')]:
        axes[0].plot(months, monthly[label+'平均绝对误差_千瓦'], marker='o', ms=4, color=color, label=label)
    axes[0].set(title='（甲）分月预测误差', ylabel='平均绝对误差（千瓦）', xlabel='月份')
    axes[0].legend(loc='lower center',bbox_to_anchor=(.5,1.12),ncol=2,frameon=False)
    savings = monthly['节约费用_元']/1e4
    axes[1].bar(months, savings, color=['#51999F' if v>=0 else '#ED8D5A' for v in savings],width=.62)
    axes[1].axhline(0,color='#666666',lw=.8)
    axes[1].set(title='（乙）相对原模型的分月节约',ylabel='节约费用（万元）',xlabel='月份')
    for ax in axes:
        ax.set_xticks(months)
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'问题二模型检验.svg',bbox_inches='tight')
    plt.close(fig)
    checks = pd.read_csv(aux/'对照报告'/'q2_validation.csv')
    summary = {'独立检查数':len(checks), '全部通过':bool(checks['通过'].all()),
               '节约月份数':int((savings>0).sum()),'评价月份数':len(monthly),
               '最差月份':int(monthly.loc[savings.idxmin(),'月份']),
               '最差月节约_元':float(monthly['节约费用_元'].min()),
               '全年节约_元':float(monthly['节约费用_元'].sum()),
               '名义覆盖率':.9, '实际覆盖率':float(np.average(monthly['区间覆盖率'],weights=df.groupby('月份').size()))}
    (out/'问题二检验摘要.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    main()
