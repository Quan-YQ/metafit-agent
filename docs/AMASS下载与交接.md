# AMASS 下载与交接

本项目目前仅做非商业研究。AMASS 是否可以用于具体研究及后续公开，请以官网当前协议和组成数据集的条款为准；使用者须自行决定是否注册并接受许可。

1. 打开 [AMASS 注册页面](https://amass.is.tue.mpg.de/register.php)，按页面要求填写资料、阅读隐私声明与使用条款并完成账号注册和邮箱验证（实际字段和流程以官网为准）。
2. 登录 [AMASS 官网](https://amass.is.tue.mpg.de/)，在下载页面选择 **ACCAD 的 SMPL+H G**。下载前自行核对并接受相应许可；AMASS 本身不是 IMU 实测数据。
3. 把原始压缩包保存为 `data/amass/ACCAD.tar.bz2`。当前示例直接读取这个压缩包，不能使用任意子集名称或解压目录替代。
4. 在 [MANO 官方站点](https://mano.is.tue.mpg.de/)自行阅读许可，下载用于 AMASS 的 **Extended SMPL+H（16 betas）**，将原始压缩包保存为 `data/body_models/smplh.tar.xz`。
5. AMASS 许可为个人单用户许可，并限制向第三方提供数据。通过第三方服务处理可能涉及访问或传输，须先核对许可及服务设置；不要发送账号密码或 API 密钥。

本地示例需要安装 NumPy 的 Python 环境，例如项目虚拟环境另行安装 NumPy 后执行 `python examples/amass_local_skeleton.py`。默认不联网，在 `data/amass/local_output/` 中生成图像；为避免覆盖，已有同名输出时须通过 `--output` 指定新路径。示例仅为研究，坐标系及模型结果仍需与官方实现核对。

本地骨架实验的细节和局限见《研究原型评审交付说明》。默认离线、没有向外部模型发送 AMASS 数据。

`data/amass/` 和 `data/body_models/` 已在项目 `.gitignore` 中列出，减少意外提交到 Git 的风险，但这不等于任何形式的访问控制。不要擅自重新分发数据或将受限数据及衍生内容上传第三方服务；具体以适用许可为准。
