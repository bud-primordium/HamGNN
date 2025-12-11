# HamGNN Lightning 2.6 迁移记录

## 环境
- 运行命令：`python -c "import torch, lightning, pytorch_lightning, torchmetrics; print('torch=',torch.__version__,' cuda=',torch.version.cuda,' lightning=',getattr(lightning,'__version__',None),' pl=',getattr(pytorch_lightning,'__version__',None),' tm=',torchmetrics.__version__)"`（使用环境 `/ssd/work/yqyang/.conda_envs/hamgnn_v_2_1_test`）
- 输出：`torch=2.5.1+cu121  cuda=12.1  lightning=2.5.3  pl=1.5.10  tm=1.8.2`

## 弃用 API / 旧命名空间扫描
使用命令：
```
rg -n --no-ignore-vcs -S "pl\.utilities\.seed\.seed_everything|from pytorch_lightning|import pytorch_lightning as pl|\bpl\.metrics\b|pytorch_lightning\.metrics|resume_from_checkpoint|progress_bar_refresh_rate|DDPPlugin|DistributedDataParallel|distributed_backend|\bgpus\s*=|\btpu_cores\b|\bnum_processes\b|amp_backend|amp_level|\bprecision\s*=\s*16\b|\buse_amp\b|reload_dataloaders_every_epoch|terminate_on_nan|pl\.callbacks\.|pytorch_lightning\.callbacks|pl\.loggers|pytorch_lightning\.loggers|profiler(=| )|pytorch_lightning\.profiler" -g "!**/venv/**" -g "!**/.venv/**" -g "!**/.mamba/**" -g "!**/.conda/**"
```

命中汇总（排除 `build/` 的历史产物）：
- `HamGNN_v_2_1/main.py`：`import pytorch_lightning as pl`、`from pytorch_lightning.loggers import ...`、`progress_bar_refresh_rate`、`resume_from_checkpoint`、`pl.callbacks.*`、`pl.utilities.seed.seed_everything`
- `HamGNN_v_2_1/models/Model.py`：`import pytorch_lightning as pl`
- `HamGNN_v_2_1/data/graph_data.py`：`import pytorch_lightning as pl`
- `Uni-HamGNN/Uni-HamiltonianPredictor.py`：`import pytorch_lightning as pl`
- 文档 `README.md / README_zh.md / docs/source/user_guide/installation.rst / docs/source/conf.py`：依赖列表或 intersphinx 仍引用 `pytorch_lightning`

## 代码修改摘要
- **Seeding 与命名空间更新**（`HamGNN_v_2_1/main.py:17-41`, `HamGNN_v_2_1/data/graph_data.py:9-12`, `HamGNN_v_2_1/models/Model.py:16-23`, `Uni-HamGNN/Uni-HamiltonianPredictor.py:5`）  
  - 统一改用 `import lightning as L` / `from lightning.pytorch import ...`，替换旧的 `pytorch_lightning` 命名空间。  
  - 新增 `log_seed_verification` 辅助函数，执行两次 `torch.randint` / `numpy.random.randint` 并打印结果，再调用 `L.seed_everything(seed, workers=True)`，确保 DataLoader worker 也可复现。  
  - `graph_data_module` 继承 `LightningDataModule`。

- **Trainer / Callback 参数迁移**（`HamGNN_v_2_1/main.py:214-366`）  
  - `gpus`→`accelerator/devices`，自动根据 `setup.num_gpus` 推断；保留 CPU 路径。  
  - `precision=16` → `'16-mixed'`，同时支持 64-bit。  
  - `resume_from_checkpoint` 改为 `fit(..., ckpt_path=...)`。  
  - `progress_bar_refresh_rate` 替换为 `TQDMProgressBar` 并允许读取 `profiler_params.progress_bar_refresh_rat`。  
  - 新增 `setup.fast_dev_run / limit_*_batches / log_every_n_steps` 的配置透传，便于调试和冒烟验证。  
  - `TensorBoardLogger` 继续使用 lightning 2.6 API，fast-dev-run 场景下自动降级到 `default_root_dir/fast_dev_run_logs`。

- **Metrics / Hook 调用**（`HamGNN_v_2_1/models/Model.py:41-308`）  
  - 移除已废弃的 `validation_epoch_end` / `test_epoch_end`，改用 `on_validation_epoch_end` / `on_test_epoch_end` 并在模块内部缓存 step 输出，保持原有可视化和持久化逻辑。  
  - `on_test_epoch_end` 在 logger 被 fast-dev-run 禁用时自动回退到 `default_root_dir`.

- **配置与示例**（`HamGNN_v_2_1/config/config_parsing.py:33-45`, `config_examples/V2.x/config.yaml`, `config_examples/V2.x/config_fast_dev.yaml`）  
  - 默认配置新增 fast-dev-run 相关键。  
  - 新增 `config_fast_dev.yaml`，指向现有 Si 数据并开启 `fast_dev_run=True`（CPU），作为最小可复现示例。  
  - README / 文档依赖说明改为 `lightning == 2.6.0`，Intersphinx 指向 Lightning 新命名。

- **文档与 seeding 说明**（`MIGRATE_PL_2.6.md`）  
  - 记录环境、弃用点、修改说明、冒烟验证命令和后续建议。

## FastDevRun 冒烟验证
- 命令：  
  ```
  source /ssd/app/anaconda3/etc/profile.d/conda.sh
  conda run -p /ssd/work/yqyang/.conda_envs/hamgnn_v_2_1_test \
    python -m HamGNN_v_2_1.main \
    --config config_examples/V2.x/config_fast_dev.yaml
  ```
- 关键配置：`fast_dev_run=True`、`limit_train/val/test_batches=2`（Lightning 内部 fast-dev-run 仍强制 1 个 batch）、`log_every_n_steps=1`、`num_gpus=0`（CPU 模式）  
- 数据：`../20251209_cw_dataset_abacus/2_djs_graph/graph/graph_data.npz`  
- 日志要点：  
  - `Seed set to 666` 与新增 `[Seed check]` 输出显示两次采样一致。  
  - Lightning 报告 `Running in fast_dev_run mode...`，并在 1 个 step 后停止（`max_steps=1 reached`）。  
  - `test/L1Loss_hamiltonian ≈ 2.27e-03`，过程无错误，TQDM 进度条正常。  
  - 生成的 fast dev 结果保存在 `config_examples/V2.x/train_logs_fastdev`。

## 后续建议
1. **多卡策略**：若未来需要分布式训练，建议在配置中新增 `strategy` 字段并对接 `lightning.pytorch.strategies.DDPStrategy`，便于设置 `find_unused_parameters` 等选项。
2. **AMP / BF16**：当前 precision 逻辑仅映射 16/64，后续可根据硬件条件开放 `'bf16-mixed'`、`'32-true'` 等设置。
3. **数据路径管理**：目前示例 fast-dev 配置引用了上层目录的数据。若要在 CI 或单独仓库中复现，可考虑提供一个最小 synthetic 数据生成脚本。
4. **Logger 友好提示**：fast-dev-run 会禁用 logger，如需在该模式下也收集少量标量，可考虑在 `setup.fast_dev_run` 为真时强制启用 `CSVLogger` 并将其输出路径写入配置。
5. **Lightning 2.6 API 新增功能**：可评估 `Trainer(max_steps)`、`GradientClipAlgorithmType` 等新参数是否需要开放到配置中，方便更细粒度控制训练过程。

---
本文件旨在记录此次升级过程的关键信息，便于团队成员复现与继续演进。若后续对 Lightning 版本进行再次升级，可在本文件基础上追加版本差异。 
