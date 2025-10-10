# Share Viz Kit

生成论文可视化与表格的最小包（仅数据衍生可视化，不含私有推理代码）。默认使用**你当前的 Python/conda 环境**。

## 运行
```bash
# 直接跑（不改环境）
bash run_all.sh

# 如需在当前环境安装依赖
INSTALL=1 bash run_all.sh
配置

config.json 的字段：

frames_dir: 原始帧目录

instances_csv: CSV（列包含 image,mask,conf；mask 仅文件名）

mask_root: 掩码根目录（可选；用于 coverage 和故事板）

以及输出与评估参数
