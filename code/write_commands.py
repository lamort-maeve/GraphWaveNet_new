"""Audit V4 evidence and render only missing standalone Bash commands."""
import argparse
import json
import shlex
from pathlib import Path
from experiment_suite import (ROOT, INDEX, METHODS, ROWS, command, description,
                              scan_results, completed_match, selected_rows)

PREPARED = '/root/autodl-tmp/PCMCI/data/set'
CODE = '/root/autodl-tmp/newcode_ssl_general'
OUTPUT = '/root/autodl-tmp/results/ssl_general_20261026'


def render(args):
    lines = [shlex.join(args[:2])]
    i = 2
    while i < len(args):
        count = 2 if i+1 < len(args) and not args[i+1].startswith('--') else 1
        lines.append('  '+shlex.join(args[i:i+count]))
        i += count
    return (' '+chr(92)+'\n').join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT.parent/'results/3monthsTest/V4')
    a = parser.parse_args()
    if not a.results.is_dir():
        parser.error('Results directory does not exist')
    entries = scan_results(a.results)
    if not entries:
        parser.error('No complete validation evidence found; preserve existing command/index files')
    # Check actual input and supervision evidence across the archived comparisons.
    reference = next(x for x in entries if x['config'].get('run_name') == 'stlag_H_full_mt_s42')
    ref = reference['config']
    for entry in entries:
        config = entry['config']
        for key in ('raw', 'splits', 'graph_files', 'dataset_manifest'):
            if config['input_provenance'][key] != ref['input_provenance'][key]:
                parser.error('Input provenance differs: '+entry['run']+' '+key)
        if (config['supervision_mask_sha256'] != ref['supervision_mask_sha256'] or
                entry['evidence_sha256']['labeled_origins.npy'] != reference['evidence_sha256']['labeled_origins.npy']):
            parser.error('Archived supervision differs: '+entry['run'])
        if config.get('static_matrix'):
            for key in ('static_matrix', 'static_provenance'):
                if config['input_provenance'][key] != ref['input_provenance'][key]:
                    parser.error('Static provenance differs: '+entry['run'])
    index = {'audited_date': '2026-10-26', 'source': str(a.results.resolve()),
             'scope': 'Validated saved metadata; best.pt presence/size checked, not deserialized',
             'completed': entries}
    INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding='utf-8')
    reused, pending = [], []
    for row in selected_rows('main'):
        match = completed_match(row, 42, PREPARED, entries)
        (reused if match else pending).append((row, match))
    header = '''半监督框架普适性：仅列尚未完成的主实验（seed=42）
核对范围：results/3monthsTest/V4 的实际 config、validation_selection、history、标签子集和 best.pt。
上传本地整个 newcode 为 /root/autodl-tmp/newcode_ssl_general；服务器旧运行包和结果保留。
结果写入 /root/autodl-tmp/results/ssl_general_20261026；所有命令均 --skip-test，不 resume 旧实验。
保留原版 mean_teacher；新增改造版 Graph-MT 已取消。三种其余方法使用各自已有实现。
共同条件：label_ratio=0.25、context=25、horizon=12、train_start=4032、frozen_train、K=3、shrunk=0.5。
J=无静态/动态因果消息；A=仅动态；K=仅静态；G=静态＋动态；所有组保留自适应骨干。
先运行静态文件准备/校验（已有完整文件校验复用，不覆盖）：

python /root/autodl-tmp/newcode_ssl_general/prepare_static.py --prepared /root/autodl-tmp/PCMCI/data/set

已完成主实验，不重跑：
'''
    header += '\n'.join(row+' -> '+Path(match['run']).name for row, match in reused)
    header += '''
另复用 B_uniform_mean_teacher_s42 为均匀历史参考；U 属更早实现，仅作辅助参考。
I_full_gmt_s42 是已取消的结构 MT，历史结果保留，不计作原版 MT 或其他三种方法。
旧报告中的 C/E 当前无对应目录；它们属于已取消方法，不补跑。
以下共 %d 条完整 Bash 训练命令：其他三种方法各 4 条，原版 MT 补 2 条。
每条可单独复制执行。补种子不可只修改 --seed，还需修改 --run-name；见文末套件命令。
''' % len(pending)
    blocks = [header]
    for row, _ in pending:
        args = command(row, 42, PREPARED, OUTPUT, code_root=CODE, python='python')
        blocks.append('\n'+'='*60+'\n'+row+'：'+description(row)+'\n\n'+render(args)+'\n')
    blocks.append('''
============================================================
多种子补充：以下为可直接运行的套件预览命令（打印后追加 --execute 才执行）。
仅补齐主实验 seeds=43,44,45,46，共 80 次（5 种训练方式 × 4 个结构 × 4 seeds）。
python /root/autodl-tmp/newcode_ssl_general/experiment_suite.py --prepared /root/autodl-tmp/PCMCI/data/set --output /root/autodl-tmp/results/ssl_general_20261026 --phase main --seeds 43 44 45 46

一次预览全部 5 个种子的未完成主实验：当前共 94 次，已完成 seed=42 的 6 组自动跳过。
python /root/autodl-tmp/newcode_ssl_general/experiment_suite.py --prepared /root/autodl-tmp/PCMCI/data/set --output /root/autodl-tmp/results/ssl_general_20261026 --phase main --seeds 42 43 44 45 46

仅运行原版 MT 缺项：
python /root/autodl-tmp/newcode_ssl_general/experiment_suite.py --prepared /root/autodl-tmp/PCMCI/data/set --output /root/autodl-tmp/results/ssl_general_20261026 --phase mt --seeds 42 --execute

仅运行另外三种半监督方法：
python /root/autodl-tmp/newcode_ssl_general/experiment_suite.py --prepared /root/autodl-tmp/PCMCI/data/set --output /root/autodl-tmp/results/ssl_general_20261026 --phase ssl --seeds 42 --execute

套件默认使用随包附带的“已完成实验索引.json”，并检查新 output/runs 的已完成结果。
如果服务器还有本地未带回的结果，可加 --results <服务器实际结果目录> 重新扫描；指定后替代随包索引。
本清单只针对上述固定数据与预处理；更换数据/标签划分后不能复用旧索引。
同一 run-name 已存在但未完成时，套件会停止并提示检查，不覆盖、不自动续训。
正式测试应在方案冻结后复用最佳检查点评估，不以重训获取已有实验的测试指标。
''')
    (ROOT/'新增实验指令.txt').write_text('\n'.join(blocks), encoding='utf-8')
    mechanism = ['因果机制可选补实验：与半监督普适性主实验分开。\nM 已完成复用；N/O/P/Q/T 当前缺失。\n沿用当前因果构图和融合实现，只切换已有消融参数，seed=42，验证集选轮次。\n静态准备和服务器代码/输出路径同 新增实验指令.txt。\n']
    for row in selected_rows('mechanism'):
        if not completed_match(row, 42, PREPARED, entries):
            mechanism.append('\n'+'='*60+'\n'+row+'：'+description(row)+'\n\n'+render(
                command(row, 42, PREPARED, OUTPUT, code_root=CODE, python='python'))+'\n')
    (ROOT/'因果机制待补指令.txt').write_text('\n'.join(mechanism), encoding='utf-8')
    print(f'Archived complete runs={len(entries)}; main pending={len(pending)} reused={len(reused)}; mechanism pending={len(mechanism)-1}')


if __name__ == '__main__':
    main()
