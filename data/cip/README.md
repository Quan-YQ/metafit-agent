# CIP 本地数据

来源：[Complete Inertial Pose (CIP) Dataset](https://zenodo.org/records/6480543)，DOI `10.5281/zenodo.6480543`，CC BY 4.0。论文：*From raw measurements to human pose — a dataset with low-cost and high-end inertial-magnetic sensor data*，作者署名、版本和引用信息以 [论文](https://www.nature.com/articles/s41597-022-01690-y) 与 Zenodo 记录为准。使用时保留署名、来源、许可和处理说明。

本次实验使用 MTwAwinda 分支的 subject00～04：00/01/02 训练、03 验证、04 测试。作者发布的位置参考来自 Xsens 惯性动捕系统。代码从原始 ZIP 读取数据、匹配数据包、选取未标异常的录制并构造相对骨盆的节段位置；头高及时间匹配使用参考信息。

在发布页获取 `MTwAwinda_trials_subject00.zip` 至 `MTwAwinda_trials_subject04.zip`，本地分别重命名为：

```text
CIP_MTwAwinda_subject00.zip
CIP_MTwAwinda_subject01.zip
CIP_MTwAwinda_subject02.zip
CIP_MTwAwinda_subject03.zip
CIP_MTwAwinda_subject04.zip
```

可放在任意本地目录，用 `CIP_DATA_DIR` 环境变量指定。每个 ZIP 约 1 GB，训练脚本在读取前核对固定长度与官方 MD5；摘要常量见 [数据处理脚本](../../research/continuous_20261108/cip_prepare.py)。文件无需解压。

本目录默认忽略，只有本说明进入 Git。下载文件放在 `data/cip/` 或仓库外部，使用 `git check-ignore 路径` 确认。完整命令见 [复现指南](../../research/continuous_20261108/README.md)。
