"""Make a local, readable demonstration page from archived real responses."""
import argparse
import html
import json
from pathlib import Path


def present(run):
    def read(name):
        return json.loads((run/name).read_text(encoding='utf-8'))
    frozen, manifest = read('freeze.json'), read('manifest.json')
    trace = read('chain_trace.json') if (run/'chain_trace.json').exists() else None
    ledger = read('budget.json') if (run/'budget.json').exists() else {'entries':[]}
    spent = sum(row['charged_or_reserved_yuan'] for row in ledger['entries'])
    sections = []
    for stage in ('pilot','confirmation'):
        if not (run/(stage+'.json')).exists():
            continue
        result = read(stage+'.json')
        rows = ''.join('<tr><td>'+html.escape(model)+'</td><td>'+str(v['correct'])+'/'+str(v['planned'])+
            '</td><td>'+f"{v['balanced_accuracy']:.1%}"+'</td><td>'+str(v['uncertain'])+
            '</td><td>'+str(v['invalid_or_missing'])+'</td><td>'+f"{(v['median_ms'] or 0)/1000:.2f} s"+
            '</td><td>'+f"¥{v['estimated_yuan']:.6f}"+'</td></tr>' for model,v in result['summary'].items())
        sections.append('<h2>'+('三图预跑' if stage=='pilot' else '剩余图片确认')+'</h2><table><tr><th>模型</th><th>正确/总数</th><th>三类平均正确率</th><th>不确定</th><th>无效/缺失</th><th>中位耗时</th><th>估算费用</th></tr>'+rows+'</table>')
    if trace:
        candidate = trace['candidate']
        sections += ['<h2>一次完整真实模型运行</h2><p>公开 IMU + 头部高度 → 固定小模型预测骨架 → '+html.escape(trace['visual_model'])+
            ' 读图 → '+html.escape(trace['decision_model'])+' 选择 → R3 检查</p>',
            '<img alt="本次实际发送的六帧骨架图" src="'+trace['image_file']+'">',
            '<p>视觉请求 '+str(trace['visual_requests_started'])+' 次；决策请求 '+str(trace['decision_requests_started'])+
            ' 次；检查状态：'+html.escape(trace['r3_status'])+'。</p>',
            '<pre>'+html.escape(json.dumps(candidate, ensure_ascii=False, indent=2))+'</pre>',
            '<p><a href="chain_trace.json">完整记录</a> · <a href="calls/chain_visual.json">视觉原回答</a> · <a href="calls/chain_decision.json">决策原回答</a></p>']
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>MetaFit 完整链路实测 · 2026-11-10</title><style>body{font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;max-width:1120px;margin:48px auto;padding:0 24px;color:#203047;background:#f7f9fc}h1{font-size:30px}h2{margin-top:40px}table{border-collapse:collapse;width:100%;background:white}th,td{padding:12px;border-bottom:1px solid #dde4ee;text-align:left}img{width:100%;background:white;border:1px solid #dde4ee;border-radius:8px}pre{padding:24px;background:#e9eff7;white-space:pre-wrap;overflow-wrap:anywhere}a{color:#1763b7}</style><h1>MetaFit：公开 IMU 到 Agent 的完整实测</h1><p>2026-11-10 · 费用与预留合计 '+f'¥{spent:.6f}'+' / ¥10</p>'+''.join(sections)+'<h2>如何理解结果</h2><p>本次比较的是模型能否读对预测骨架的髋部升降。共18张图来自两段公开录制，包含倒序与重复帧对照，按同源簇解释；不是18位独立受试者。几何对照与骨架预测准确性是不同指标。</p><p>本演示使用已有五段训练的岭回归基线，保留之前 DynaIP 的独立比较结果。训练事件为模拟，输出保存在本机供评审查看。</p><p><a href="freeze.json">冻结方案</a> · <a href="manifest.json">数据清单</a> · <a href="budget.json">费用记录</a></p></html>'
    (run/'index.html').write_text(document, encoding='utf-8')
    return {'page':str(run/'index.html'), 'estimated_and_reserved_yuan':spent}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(present(args.run), ensure_ascii=False))
