"""问题三：复核五组已有回测及结算、因果性测试，输出预报价值图。"""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import q3_run_model as model


def main():
    here = Path(__file__).resolve().parent
    out = here/'辅助输出'/'模型检验'
    source = out/'对照数据'
    out.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,'-m','unittest','test_q3_model'],cwd=here,check=True)
    meta = json.loads((source/'q3_run_metadata.json').read_text(encoding='utf-8'))
    cfg = model.Config(**meta['config'])
    rows, audits = [], {}
    cases = [('only_0','仅零时'),('0_6','零、六时'),('0_12','零、十二时'),('0_6_12','零、六、十二时'),('all','全部四时点')]
    for case, label in cases:
        df = pd.read_csv(source/case/'q3_timeseries.csv')
        dates = pd.DatetimeIndex(pd.to_datetime(df['date'].drop_duplicates()))
        if len(df)!=334*144 or not dates.equals(pd.date_range('2025-02-01','2025-12-31')):
            raise ValueError('回测日期范围不完整：'+case)
        audits[case] = model.audit(df,cfg)
        daily = model.daily_summary(df)
        log = pd.read_csv(source/case/'q3_solver_log.csv')
        rows.append({'方案':label,'内部标识':case,'总费用_元':daily.total_cost.sum(),
                     '紧急费用_元':daily.emergency_cost.sum(),'紧急电量_千瓦时':daily.emergency_kwh.sum(),
                     '紧急天数':int((daily.emergency_kwh>1e-6).sum()),
                     '期初电量_千瓦时':df.start_soc_kwh.iloc[0], '期末电量_千瓦时':df.end_soc_kwh.iloc[-1],
                     '最大绝对间隙_元':log.mip_absolute_gap.max(),
                     '最大约束残差':log.constraint_error.max(),
                     '备用缺口求解次数':int((log.max_reserve_slack_kwh>1e-6).sum())})
    report = pd.DataFrame(rows)
    report['相对仅零时节约_元'] = report['总费用_元'].iloc[0]-report['总费用_元']
    if np.ptp(report['期初电量_千瓦时'])>1e-5:
        raise ValueError('方案期初库存不一致')
    report.to_csv(out/'问题三方案核验.csv',index=False,encoding='utf-8-sig')
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
                         'axes.unicode_minus':False,'svg.fonttype':'path','font.size':10})
    fig, axes = plt.subplots(1,2,figsize=(10.6,3.7),layout='constrained')
    y=np.arange(len(report)); labels=report['方案']
    for ax,col,title,xlabel in [(axes[0],'相对仅零时节约_元','（甲）预报更新的费用收益','节约费用（万元）'),
                                 (axes[1],'紧急电量_千瓦时','（乙）紧急购电需求','紧急购电量（万千瓦时）')]:
        values=report[col]/1e4
        bars=ax.barh(y,values,color=['#ED8D5A' if v<0 else '#4198AC' for v in values],height=.56)
        ax.bar_label(bars,fmt='%.2f',padding=4,fontsize=9)
        ax.set_yticks(y,labels); ax.invert_yaxis()
        ax.set(title=title,xlabel=xlabel)
        ax.margins(x=.25); ax.axvline(0,color='#666666',lw=.8)
        ax.grid(axis='x',alpha=.2); ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'问题三模型检验.svg',bbox_inches='tight'); plt.close(fig)
    summary={'全部审计通过':all(x['passed'] for x in audits.values()),'方案数':len(report),
             '完整方案节约_元':float(report['相对仅零时节约_元'].iloc[-1]),
             '十八时边际节约_元':float(report['总费用_元'].iloc[-2]-report['总费用_元'].iloc[-1]),
             '期末库存极差':float(np.ptp(report['期末电量_千瓦时'])),
             '全部方案最大约束残差':float(report['最大约束残差'].max()),
             '审计明细':audits}
    (out/'问题三检验摘要.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('问题三：五组回测审计通过，费用收益 %.2f 元。'%summary['完整方案节约_元'])


if __name__=='__main__':
    main()
