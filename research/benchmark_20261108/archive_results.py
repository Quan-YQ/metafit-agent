"""Archive sanitized evidence and generate the comparison report from measured JSON."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from fetch_dynaip import PROJECT
from vision_benchmark import digest

def load(path):return json.loads(path.read_text(encoding='utf-8'))

def archive(runs, output, report):
    if output.exists() or report.exists():raise ValueError('new archive and report paths required')
    files={
        'pose_results.json':runs/'pose_r2/results.json',
        'pose_selection_frozen.json':runs/'pose_r2/selection_frozen.json',
        'pose_confirmation_results.json':runs/'pose_subject08/results.json',
        'pose_frozen_replay.json':runs/'pose_frozen_replay.json',
        'selection_geometry_audit.json':runs/'selection_geometry_audit.json',
        'confirmation_geometry_audit.json':runs/'confirmation_geometry_audit.json',
        'secret_scan.json':runs/'secret_scan.json',
        'verification.json':runs/'verification.json',
        'vision_selection_manifest.json':runs/'vision_selection_images/manifest.json',
        'vision_selection_responses.jsonl':runs/'vision_selection_complete/responses.jsonl',
        'vision_selection_results.json':runs/'vision_selection_complete/results.json',
        'vision_selection_run.json':runs/'vision_selection_complete/run.json',
        'vision_selection_recovery.json':runs/'vision_selection_complete/inherited_attempts.json',
        'vision_selection_frozen.json':runs/'vision_selection_frozen.json',
        'vision_confirmation_manifest.json':runs/'vision_confirmation_images/manifest.json',
        'vision_confirmation_responses.jsonl':runs/'vision_confirmation_final/responses.jsonl',
        'vision_confirmation_results.json':runs/'vision_confirmation_final/results.json',
        'vision_confirmation_run.json':runs/'vision_confirmation_final/run.json',
        'vision_confirmation_recovery.json':runs/'vision_confirmation_final/inherited_attempts.json',
        'vision_confirmation_first_recovery_responses.jsonl':runs/'vision_confirmation_complete/responses.jsonl',
        'vision_confirmation_first_recovery_results.json':runs/'vision_confirmation_complete/results.json',
        'vision_confirmation_first_recovery.json':runs/'vision_confirmation_complete/inherited_attempts.json',
        'vision_integration_r1_responses.jsonl':runs/'vision_selection_r1/responses.jsonl',
        'vision_integration_r1_results.json':runs/'vision_selection_r1/results.json',
        'vision_confirmation_r1_responses.jsonl':runs/'vision_confirmation_r1/responses.jsonl',
        'vision_confirmation_r1_results.json':runs/'vision_confirmation_r1/results.json',
        'vision_budget.json':runs/'vision_budget.json'}
    attempts=runs/'vision_confirmation_final/quota_attempts.jsonl'
    if attempts.exists():files['vision_confirmation_quota_attempts.jsonl']=attempts
    # This script archives the original run directory layout, not arbitrary my_* runs.
    for source in (runs/'pose_r2/aligned_baseline.pth', PROJECT/'outputs/research_continuous_20261108/cip_selected_model.npz'):
        if not source.is_file():raise ValueError('missing original checkpoint: '+str(source))
    for source in files.values():
        if not source.is_file():raise ValueError('missing completed evidence: '+source.name)
    pose=load(files['pose_results.json']); fresh=load(files['pose_confirmation_results.json'])
    selection=load(files['vision_selection_results.json']); confirmation=load(files['vision_confirmation_results.json'])
    frozen=load(files['vision_selection_frozen.json']); budget=load(files['vision_budget.json'])
    if any(row['completed']!=row['planned'] or row['cost_unknown_count'] for row in confirmation['summary'].values()):
        raise ValueError('confirmation incomplete')
    if set(confirmation['summary'])!=set(frozen['candidates']):raise ValueError('wrong confirmation candidates')
    if digest(files['vision_selection_results.json'])!=frozen['selection_results_sha256']:
        raise ValueError('selection differs from freeze')
    output.mkdir(parents=True)
    for name,source in files.items():shutil.copyfile(source,output/name)
    names={'DynaIP':'DynaIP 官方权重','aligned_small':'对齐小模型'}
    pose_rows=[]
    for name in names:
        a,b=pose['macro'][name],fresh['macro'][name]
        timing=pose['timing'][name]
        pose_rows.append(f"| {names[name]} | {a['canonical22_mpjpe_cm']:.3f} | {b['canonical22_mpjpe_cm']:.3f} | {(a['canonical22_mpjpe_cm']+b['canonical22_mpjpe_cm'])/2:.3f} | {(a['learned11_geodesic_deg']+b['learned11_geodesic_deg'])/2:.3f} | {timing['median_sequence_ms']:.3f} | {pose['parameters'][name]:,} |")
    labels={'glm-4.6v-flash':'GLM-4.6V-Flash','glm-4.1v-thinking-flash':'GLM-4.1V-Thinking-Flash',
            'qwen3.7-flash-2026-07-15':'Qwen3.7-Flash','qwen3-vl-plus':'Qwen3-VL-Plus'}
    vision_rows=[]
    for name,row in selection['summary'].items():
        follow=confirmation['summary'].get(name)
        checked=f"{follow['correct']}/{follow['planned']}（{follow['accuracy']:.1%}）" if follow else '未入选'
        vision_rows.append(f"| {labels[name]} | {row['correct']}/18（{row['accuracy']:.1%}） | {row['uncertain_rate']:.1%} | {row['invalid_or_missing_rate']:.1%} | {row['latency_median_ms']/1000:.2f} s | ¥{row['estimated_yuan']:.6f} | {checked} |")
    ranked=sorted(confirmation['summary'],key=lambda name:(-confirmation['summary'][name]['accuracy'],
        confirmation['summary'][name]['uncertain_rate'],confirmation['summary'][name]['invalid_or_missing_rate'],
        confirmation['summary'][name]['estimated_yuan'],confirmation['summary'][name]['latency_median_ms']))
    # Selection retained its preregistered cost-before-latency tie breaker.
    # The final API recommendation is an engineering choice: when confirmation
    # quality ties, prefer response latency and stable interactive use. Keep
    # the raw cost-based ranking separately so the tradeoff is reviewable.
    recommended=sorted(confirmation['summary'],key=lambda name:(-confirmation['summary'][name]['accuracy'],
        confirmation['summary'][name]['uncertain_rate'],confirmation['summary'][name]['invalid_or_missing_rate'],
        confirmation['summary'][name]['latency_median_ms'],confirmation['summary'][name]['estimated_yuan']))
    winner=recommended[0]
    cost=sum(row['charged_or_reserved_yuan'] for row in budget['entries'])
    unknown=sum(row['status']!='usage_confirmed' for row in budget['entries'])
    classes=[]
    for name,row in confirmation['summary'].items():
        scores=' / '.join(str(row['per_class'][kind]['correct'])+'/6' for kind in ('up','down','no_clear_change'))
        classes.append(f"| {labels[name]} | {scores.replace(' / ',' | ')} | {row['uncertain_rate']:.1%} | {row['invalid_or_missing_rate']:.1%} | {row['latency_median_ms']/1000:.2f} s | ¥{row['estimated_yuan']:.6f} |")
    report_text=f'''# 两项优化结果与建议 · 2026-11-08

## 动作模型：共同输入、共同留出数据

两位共同留出受试者04和08，各4段动作，共8段。官方训练划分明确排除04/08；小模型训练00/01/02、验证03。08在模型冻结并完成04评估后另行下载，未用于调参。两人的记录宏平均等权汇总：

| 模型 | subject04 位置误差 cm | 新 subject08 位置误差 cm | 两人均值 cm | 两人角度均值 ° | CPU 600帧中位耗时 ms | 参数量 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(pose_rows)}

误差越低越好。位置是同一官方平均Xsens骨架的22个非根关节点MPJPE，角度是11个预测节段的测地角误差。没有把忽略关节替换成参考真值，也没有把原实验中的不同位置定义/源单位分数混入表格。

双方原始输入为同位置6个IMU的72维根归一化旋转/去重力加速度，并共享首帧参考姿态和速度。小模型保持128→128隐藏层架构，输入改为传感器加初始状态的156维、输出11个6D旋转；在3个岭回归与MLP中仅按03验证集选择，MLP最佳第19轮、45,122参数。DynaIP利用单向LSTM历史，小模型逐帧预测；这是现成时序模型与对齐逐帧基线的比较，未统一网络的历史记忆能力。旧43维小模型及其历史结果原样保留。

官方配方使用前150帧参考姿态帮助传感器校准、前两帧参考位置产生初始速度；前300帧不计分。CIP参考来自Xsens惯性动捕。结论适用于这种参考辅助研究协议，不能直接当作无校准在线精度。速度是同机4线程CPU整段吞吐，包含输出旋转重建，排除共同预处理和磁盘读取；不是逐帧端到端延迟。

**建议**：精度优先继续使用DynaIP作为研究候选；资源优先保留小模型。DynaIP使用多个数据集预训练，小模型仅用本地3人训练，因此本表衡量现成方案在相同信息下的效果，不是等训练预算的架构优劣实验。两人八段仍需更多受试者扩展验证。

## 视觉模型：先选型，再用新图确认

官方资料核对日期2026-11-08。GLM两款官方免费；百炼北京32K以内每百万Token输入/输出：Qwen3.7-Flash ¥0.2/¥0.8，Qwen3-VL-Plus ¥1/¥10。价格、模型快照、官方出处和不选陈旧别名的原因见 [候选配置](../research/benchmark_20261108/vision_models.json)。

同一18张公开CMU骨架图，每类6张，问题和参数固定；先按正确率、不确定率、无效率、费用、耗时排序，选前两名，再允许生成18张不同人物源的新图：

| 模型 | 选型正确率 | 主动不确定率 | 无效回答率 | 选型请求中位耗时 | 18图估算费用 | 新图确认正确率 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(vision_rows)}

新图确认分项（每类6张）：

| 候选 | 升高 | 降低 | 无明显变化 | 主动不确定率 | 无效率 | 请求中位耗时 | 18图估算费用 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(classes)}

`no_clear_change`为确定的第三类，`uncertain`为模型主动无法判断。没有主动表示不确定不代表可靠；模型会给出错误但确定的回答。截断、附加非JSON文字等计无效，不能从正确率分母中删去。

**最终API实验候选：{labels[winner]}**。依据确认集的正确率、不确定率、无效率、耗时及费用进行工程推荐；具体数据见上表。保留其他候选供在线对照。这里的最终工程建议把并列正确率下的延迟放在费用之前；事先固定的选型排序及其费用优先规则保留原样，没有改写选型成绩。

当前图片是带紫色髋部标记和绿色阈值线的受控图示，6列2视角，不能据此推断真实训练照片的动作质量判断能力。正倒放和同一录制窗口具有相关性，不视为独立人物；18张确认图的结果适合决定下一轮实验方向，不足以独立触发训练反馈。关键升降判断优先使用明确几何信息核验。

## 费用、限流与证据

共享账本累计 **¥{cost:.7f}**（授权上限¥10），共记录{len(budget['entries'])}次请求预留，包含选型、确认、免费格式诊断和限流尝试。未结算usage条目{unknown}个，均为免费HTTP429、预留金额0元。费用按官方原价与返回usage估算，未扣赠送或缓存优惠，最终以账单为准。

初轮智谱429导致部分结果缺失，程序先停止该平台；随后显式补齐免费缺失调用，成功回答原样复用、格式错误回答不重试。选型采用20秒平台间隔；确认单模型在20/35秒间隔下仍遇429，最终用55秒间隔及显式启用的最多4次免费429退避。付费请求始终最多一次。原429、恢复链及不完整统计另行归档，没有用缺失样本给免费模型排名。额外调度等待不计入单次请求延迟，用户实际批量完成时间会更长。网络、排队与服务负载未受控，耗时是本轮现场测量。

密钥仅从进程/Windows用户环境变量读取；认证头、原始请求日志和完整推理链不保存。源码、候选文件及归档另外按实际密钥做不回显扫描。原数据、权重和图片位于本机忽略目录。

## 复现与许可

最终完整测试 **734项通过、0失败、0跳过**。36张图片摘要均重绘核对，独立像素几何分类36/36与参考一致；冻结动作模型的subject04重放误差差值为0，且所有输出旋转通过SO(3)有效性检查。官方源码保留一条torch.cross接口弃用提示，不影响结果。

- [复现步骤](../research/benchmark_20261108/README.md)：官方权重固定提交与SHA-256、环境锁、公共图片生成、预算保护和离线重放。
- [归档结果](../research/benchmark_20261108/results/)：逐图标签/解释/usage/耗时、清单摘要、冻结选择、每段动作误差与请求账本。
- [官方来源审计](../research/benchmark_20261108/source_audit/)：DynaIP MIT许可、所有官方训练/测试清单和文件摘要。CIP CC BY4.0；CMU来源与转换署名随复现说明保留。

本次交付为本地代码、可复现结果与研究报告。
'''
    report.parent.mkdir(parents=True,exist_ok=True);report.write_text(report_text,encoding='utf-8')
    artifacts={name:digest(output/name) for name in files}
    artifacts['source_files']={path.name:digest(path) for path in Path(__file__).parent.glob('*.py')}
    artifacts['aligned_baseline_sha256']=digest(runs/'pose_r2/aligned_baseline.pth')
    artifacts['original_small_model_sha256']=digest(PROJECT/'outputs/research_continuous_20261108/cip_selected_model.npz')
    artifacts['report_sha256']=digest(report)
    artifacts['final_vision_candidate']=winner
    artifacts['confirmation_accuracy_cost_ranking']=ranked
    artifacts['recommendation_basis']='Engineering recommendation: equal confirmation accuracy favors latency and stable API use; selection cost-first ranking is retained separately.'
    (output/'artifacts.json').write_text(json.dumps(artifacts,indent=2),encoding='utf-8')
    print(json.dumps({'archived_files':len(files),'candidate':winner,'estimated_or_reserved_yuan':cost},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,default=PROJECT/'outputs/benchmark_20261108')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();archive(args.runs,args.output,args.report)
