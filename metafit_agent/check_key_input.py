"""离线检查终端是否读取到类似 API Key 的输入；不连接网络、不保存值。"""

import getpass
import sys
import warnings


def main() -> int:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            key = getpass.getpass("请输入 DeepSeek API Key（输入不会显示）：")
    except getpass.GetPassWarning:
        print("当前终端无法隐藏输入，检查已取消。", file=sys.stderr)
        return 2
    except (EOFError, KeyboardInterrupt):
        print("\n检查已取消。", file=sys.stderr)
        return 2

    plausible = (key.startswith("sk-") and len(key) >= 20 and key == key.strip()
                 and all(33 <= ord(char) <= 126 for char in key))
    del key
    if plausible:
        print("终端输入格式检查：通过。此结果不证明密钥有效，也未发起网络请求。")
        return 0
    print("终端输入格式检查：未通过。请检查是否完整复制，避免引号、空白或其他字符。")
    print("此检查只看输入形式，不证明密钥是否有效，也未发起网络请求。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
