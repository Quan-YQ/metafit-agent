# 开发与提交指南

## 本地开发

在项目根目录使用 Python 3.12：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe tools\check_repository.py
```

研究计算依赖通过 `pip install -e ".[dev,research]"` 安装；PyTorch 训练使用 `.[dev,train]`。所有命令中的 `pip` 均应由同一个虚拟环境的 `python -m pip` 执行。基础安装不会安装训练框架。

## 修改与验证

1. 从最新代码创建分支，例如 `git switch -c improvement/short-description`。
2. 修改代码和对应说明；修复行为问题时补充能够复现问题的测试。
3. 运行相关测试，再运行完整离线测试及仓库检查。CI 分别检查基础安装和 NumPy/Pillow 研究环境。
4. 实验应保留数据来源、固定划分、选模规则和结果。已经看过的测试片段应记录为已观察，不再称为全新留出集。
5. 在 [改进记录](CHANGELOG.md) 的“未发布”部分添加本次变化及验证方式；接口字段变化另见 [契约记录](docs/CONTRACT_CHANGELOG.md)。

普通测试和仓库检查均离线运行。真实 API 测试应显式启用真实调用、配置自己的密钥并设置请求预算；研究结果注明模型、提示、样本和运行时间。

## 提交前检查

```powershell
git status --short
git diff --check
git diff --stat
git diff
```

确认候选文件后再暂存，并检查 `git diff --cached`。训练数据、受限人体模型、权重、数据库、环境文件及真实密钥保留本地。提交前检查工具扫描当前候选文件；它不扫描整个 Git 历史，也不能代替检查暂存内容。

推荐将提交拆成能独立说明的主题，例如“规范研究脚本入口”“同步实验结果文档”。提交说明写清改了什么、为什么改、怎样验证。推送后查看 GitHub Actions 的检查结果。

## 许可

仓库目前没有指定项目级开源许可证。复用第三方数据或实现时保留来源、适用许可及修改说明；不要将代码许可证自动套用于外部数据。
