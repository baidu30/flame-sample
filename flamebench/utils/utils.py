import re

from pathlib import Path

def is_numeric_string(input_string):
    pattern = r'^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$'

    if re.match(pattern, input_string):
        return True
    else:
        return False


def get_path_from_root(*parts):
    """
    返回以项目根目录为基准的路径。

    - 如果在 Python 脚本中，会以 `__file__` 的目录为起点。
    - 如果在 Jupyter Notebook 中，会以当前工作目录 `Path.cwd()` 为起点，向上找根目录。

    参数:
        *parts: 依次传入的路径部分，例如 "config", "1d_config.yaml"

    返回:
        Path 对象
    """
    if "__file__" in globals():
        current_path = Path(__file__).resolve()
    else:
        current_path = Path.cwd().resolve()

    # 假设你的根目录就是 ODEBench (包含 config、mechanisms 等文件夹)
    # 向上找一个包含 config 和 mechanisms 的目录
    for parent in [current_path] + list(current_path.parents):
        if (parent / "config").exists() and (parent / "mechanisms").exists():
            return parent.joinpath(*parts)

    raise FileNotFoundError("❌ Error: Could not locate project root containing 'config' and 'mechanisms'.")
