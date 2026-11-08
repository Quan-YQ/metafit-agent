"""Offline checks for Git candidate files, source syntax and Markdown file links."""
import argparse
import ast
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote


def candidates(root):
    result = subprocess.run(['git', '-C', str(root), 'rev-parse', '--show-toplevel'],
                            capture_output=True, text=True)
    if result.returncode == 0 and Path(result.stdout.strip()).resolve() == root:
        raw = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z',
                                       '--cached', '--others', '--exclude-standard'])
        return sorted({root / p for p in raw.decode('utf-8').split('\0') if p})
    excluded = {'.git', '.venv', '.python312', '.tmp', 'outputs', '__pycache__',
                '.pytest_cache', 'build', 'dist', '.planning', '.gstack'}
    return sorted(p for p in root.rglob('*') if p.is_file() and
                  not any(part in excluded or part.endswith('.egg-info') for part in p.relative_to(root).parts))


def check(root):
    files = [p for p in candidates(root) if p.is_file()]
    issues = []
    docs = links = 0
    secret = re.compile(r'sk-[A-Za-z0-9]{32,}')
    forbidden = {'.npz', '.npy', '.zip', '.tar', '.gz', '.xz', '.bz2', '.7z',
                 '.pt', '.pth', '.ckpt', '.onnx', '.safetensors',
                 '.db', '.sqlite', '.sqlite3', '.partial', '.part'}
    for path in files:
        relative = str(path.relative_to(root))
        if path.is_symlink():
            issues.append(f'{relative}: symlink requires manual review'); continue
        if path.suffix.lower() in forbidden or path.stat().st_size > 5 * 1024 * 1024:
            issues.append(f'{relative}: data, output or oversized file in submission candidates'); continue
        if path.name.startswith('.env') and path.name != '.env.example':
            issues.append(f'{relative}: environment file in submission candidates'); continue
        try:
            text = path.read_text(encoding='utf-8-sig')
        except UnicodeError:
            continue
        if secret.search(text):
            issues.append(f'{relative}: possible API credential (value omitted)')
        if re.search(r'[CD]:\\Users\\LENOVO|D:\\GithubProjects\\metafit-agent', text):
            issues.append(f'{relative}: machine-specific project path')
        if path.suffix == '.py':
            try: ast.parse(text, filename=relative)
            except SyntaxError: issues.append(f'{relative}: Python syntax error')
        if path.suffix == '.json':
            try: json.loads(text)
            except ValueError: issues.append(f'{relative}: invalid JSON')
        if path.suffix == '.jsonl':
            try:
                for line in text.splitlines():
                    if line.strip(): json.loads(line)
            except ValueError: issues.append(f'{relative}: invalid JSONL')
        if path.suffix != '.md': continue
        docs += 1
        prose = re.sub(r'```[^\n]*\n.*?```', '', text, flags=re.S)
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', prose):
            target = target.strip().strip('<>')
            if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', target) or target.startswith('#'): continue
            local = unquote(target.split('#', 1)[0])
            if not local: continue
            links += 1
            resolved = (path.parent / local).resolve()
            if not resolved.is_relative_to(root) or not resolved.exists():
                issues.append(f'{relative}: broken or out-of-repository link: {target}')
    return {'candidate_files': len(files), 'markdown_files': docs, 'local_links': links, 'issues': issues}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    result = check(parser.parse_args().root.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return bool(result['issues'])


if __name__ == '__main__': raise SystemExit(main())
