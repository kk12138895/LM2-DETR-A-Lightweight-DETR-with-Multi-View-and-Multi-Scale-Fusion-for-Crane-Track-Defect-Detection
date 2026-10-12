# 论文配套代码上传说明

## 1. 上传多少代码

判断标准不是代码占整个工程的百分比，而是：读者能否找到论文方法的实现，并按说明运行支撑论文主要结论的实验。期刊具体要求仍应以其代码共享原文为准。

建议首次发布保留整个 `src/`（排除缓存），保留最终模型与消融实验配置及它们递归引用的 YAML，保留 `train.py`、依赖说明、评估/推理脚本、`README.md` 和现有 `LICENSE`。源码体积很小，过早裁剪容易破坏注册器、导入关系和配置继承。确认不使用的历史实现可以在独立发布副本中清理，并重新验证。

只公开几个创新模块也可以形成“方法实现包”，但必须同时说明接口、输入输出、依赖的上游版本及集成方式；不能据此声称完整复现。是否满足期刊要求，需要核对期刊原文。

推荐包含：

- 论文最终模型及论文涉及的消融配置，且与真实训练记录一致。
- 训练、评估入口及必要依赖；如论文报告 FPS，则附对应测速程序和测量协议。
- 数据来源、类别映射、划分清单/生成规则与必要预处理脚本。
- 最终权重链接（若可公开），每个结果对应的配置、权重、命令和软件环境。

通常不需要放入 Git 源码仓库：原始数据集、所有 epoch 权重、训练输出、缓存、临时文件、个人实验笔记、论文草稿和无关图片。数据与权重是否应另行提供，取决于复现需要和共享条件；“不进 Git”不等于“不提供”。

当前 `.gitignore` 已排除常见数据目录、权重、日志、缓存和本地环境文件。它不会自动审查源码内容，也不会排除已经提交的文件。

## 2. 本工程发布前需要确认的事项

### 根据 LM²-DETR 论文确定的发布范围

论文题目为 **LM²-DETR: A Lightweight DETR with Multi-View and Multi-Scale Fusion for Crane Track Defect Detection**。建议 GitHub 仓库命名为 `LM2-DETR`。上传范围应覆盖下表所列实现和复现材料：

| 论文内容 | 本地对应位置/应补充材料 |
| --- | --- |
| L-HGNetv2 / GhostConv | `src/nn/backbone/hgnetv2.py`，明确最终 G3 替换位置 |
| 共享骨干、180° 辅助视图 | `src/zoo/dfine/dfine.py`，保留启用辅助分支及共享权重的配置 |
| CGF | `src/nn/backbone/NewCWAM.py` 的实际执行类，以及 `dfine.py` 外层融合封装 |
| TSA + C3K2-FPN | `src/zoo/dfine/hybrid_encoder_C3K2.py`，记录稀疏率及 C3K2 比例 |
| 表 1 三个数据集主结果 | CRDDs、RFDDs、RSDDs 各自最终配置、类别映射、划分、权重链接和评估命令 |
| 表 2 模块消融 | D-FINE 基线、LCR-1、LCR-2、LCR-3、完整模型的配置或明确的可执行切换方式 |
| 表 3–4 结构消融 | G1–G4、BiFPN、C3K2 比例 0.50/0.75/1.00 的配置或可执行实现差异 |
| 参数量和 GFLOPs | 实际使用的统计脚本、输入尺寸、统计工具版本与口径 |

不必把 YOLO、其他 DETR 对比方法的整个工程复制进这个仓库；应记录其上游版本、实验设置和必要修改。论文没有报告 FPS 表格，因此不需要仅为了文件数量上传所有测速脚本。

如果期刊确实只要求核心实现，可以发布四个模块及运行示例、依赖接口、上游基准版本和集成说明。若希望读者验证表 1–4，则应保留上述训练/评估与消融材料。论文 PDF 不是期刊代码共享政策，仍不能据此确定期刊的最低要求。

### 论文与当前工程的对应检查

- `dfine_hgnetv2_n_custom_new.yml` 启用了 `use_rot180_aux: True` 和 `share_rot180_backbone: True`，可以作为查找最终训练配置的线索，但尚不能认定它就是论文版。
- 论文最终 C3K2 比例是 **0.75**，该 YAML 没有设置 `c3k2_expansion`，当前类默认是 **0.5**；`expansion: 0.34` 是另一个参数。
- `CrossWinBasedAttention` 虽然保留旧名称，但当前执行体确实包含论文所述的差分、通道门控、空间门控、DWConv 和 BN。不要因为文件名或旧注释把它描述成窗口注意力。内部残差缩放与外部相似度门控仍应和最终实验版本核对。
- 论文算法 1 使用三个特征层，上述 YAML 使用 stride 16、32 两层；应根据实际训练代码统一论文表述与发布配置。
- `block_conv: conv` 不足以证明是原始骨干：当前 HG block 中直接实例化了 GhostConv，应明确基线以及 G1–G4 各自代码差异。
- 论文环境是 Python 3.8 / PyTorch 2.3.1 / CUDA 12.1 / cuDNN 8.9.2；本地依赖快照是 PyTorch 2.5.1，旧训练指南建议 Python 3.11.9。需要找回真实实验环境记录。
- 论文写明训练:测试:验证 = **7:2:1**，而 `--test-only` 实际读取 `val_dataloader`；最终测试命令应指向真实测试集。每个数据集的具体数量、原图及扩增图划分还需补充。

以上是静态核对结果，并非运行失败的判定，也没有据此修改模型或实验设置。应优先找回训练时保存的配置、代码和权重，而不是为了匹配文字直接改参数后沿用旧结果。

### 通用发布检查

1. 确认论文最终配置。目录中有多个实验版本，文件名不能证明哪个是最终方法；修改后的配置也不能直接标为官方基线。
2. 将发布副本中数据配置的本机绝对路径改成相对路径或明确的用户配置项。
3. 检查类别 ID：当前 `railmix_detection.yml` 使用 `remap_mscoco_category: True`，它调用 MS COCO 固定映射。自定义的连续零起始类别应使用 `False`；其他编号应先建立一致映射。必须与论文实验一致。
4. 整理并验证安装环境。`requirements_DFINE.txt` 是较大的环境快照，含平台相关包及自定义算子，并且缺少源码直接导入的 `timm`；不要把它描述为已验证的一键安装方案。
5. README 已补入论文题目、作者、方法和论文报告的结果；还需补最终配置、权重、正式发表信息和版本引用。
6. 在干净目录中按 README 安装并至少完成一次推理和评估，核对论文数字。当前仅完成文档和代码接口核对，没有重新训练或复测结果。

## 3. 在 GitHub 创建仓库

登录 GitHub，点击右上角 `+` → `New repository`，填写仓库名（例如 `railway-defect-detection`）。整理阶段可先设为 Private；正式共享时按期刊要求设置可访问性。

创建空仓库时，不勾选自动生成 README、`.gitignore` 或 License，因为本地已有这些文件。不要覆盖工程原有的许可和来源声明。

## 4. 从 PowerShell 上传

当前工程尚未初始化 Git。以下命令由你在确认发布内容后执行；本次文档准备没有初始化仓库或向 GitHub 上传。

```powershell
Set-Location 'E:\project\D-FINE-master'
git init -b main
git add README.md .gitignore LICENSE requirements_DFINE.txt train.py src configs tools/inference/torch_inf.py assets/figures
git status --short
git diff --cached --stat
git diff --cached
```

这里显式添加源码目录，避免第一次就把整个工作目录全部加入。该清单是“保留源码依赖的起步版本”，仍包含多个配置和历史源码；在提交前按论文最终范围审阅。若论文依赖转换脚本、测速程序或划分文件，需要另外 `git add` 对应文件。

如果误暂存某个文件，在首次提交前可运行 `git rm --cached -- 文件路径`；目录加 `-r`。这只移除暂存条目，保留本地文件。不要使用删除本地文件的命令来整理发布清单。

确认暂存内容后提交：

```powershell
git commit -m "Add research code and documentation"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

把 `YOUR_USERNAME` 和 `YOUR_REPOSITORY` 替换为真实名称。若 Git 提示未配置作者身份，按提示设置 `git config user.name "你的名字"` 和 `git config user.email "你的邮箱"`，然后重新提交。推送时完成 Git 的身份验证流程；不要把访问令牌写入代码或 remote URL。

如果当前仍在整理，先推送到私有仓库；完成环境验证和文档补充后，再提供期刊要求的访问方式。

## 5. 固定论文版本

论文结果验证通过后，记录当前提交并创建版本标签：

```powershell
git tag -a v1.0.0 -m "Paper code release"
git push origin v1.0.0
```

在 GitHub 的 Releases 页面基于该标签创建 release，说明论文对应版本，并添加权重链接。大文件不应直接放进普通 Git 历史；GitHub 会阻止超过 100 MiB 的普通 Git 文件，可使用 Releases 或适合的数据/模型托管服务。

论文中的 Code availability 可使用以下句式，替换真实链接，并仅在仓库确实可访问后使用：

> The code supporting the findings of this study is available at [repository URL], release [version/tag]. Instructions for data preparation, training, and evaluation are provided in the repository.

若仅公开模块，应把描述改为实现这些模块的代码，不能暗示已经提供全部训练/评估复现材料。

## 参考

- [GitHub：上传本地代码](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)
- [GitHub：大文件限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
