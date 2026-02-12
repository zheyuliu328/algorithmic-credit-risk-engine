# Open-source Model Local Deployment Checklist

> 一页纸落地模板 | 从 GitHub 到生产环境

---

## 1. License Check

- [ ] 开源协议审查（MIT/Apache/GPL）
- [ ] 商业使用授权确认
- [ ] 依赖库许可证扫描
- [ ] 合规风险登记

**证据**: `LICENSE` 文件 + `requirements.txt` 依赖清单

---

## 2. Data Adapter (SAS → Python)

- [ ] 字段映射表（SAS 变量名 ↔ Python 列名）
- [ ] 数据类型转换（SAS format → Python dtype）
- [ ] 缺失值策略对齐
- [ ] 样本数据对比验证

**证据**: `data_adapter.py` + 对比测试报告

---

## 3. Reconciliation Tolerance

- [ ] 基准结果获取（SAS 输出）
- [ ] Python 复现结果
- [ ] 差异分析（绝对误差 / 相对误差）
- [ ] 容忍度设定（如 1e-6）

**证据**: `reconciliation_report.md` + 差异日志

| Metric | SAS | Python | Diff | Tolerance | Status |
|:-------|:----|:-------|:-----|:----------|:-------|
| AUC | 0.8712 | 0.8715 | 0.0003 | ±0.001 | ✅ |
| K-S | 0.5234 | 0.5231 | 0.0003 | ±0.001 | ✅ |

---

## 4. Documentation Pack

- [ ] 模型开发文档 (MDD)
- [ ] 模型验证报告 (MVR)
- [ ] 部署手册
- [ ] 监控与回滚方案

**证据**: `docs/` 目录 + `README.md`

---

## 交付标准

| 检查项 | 标准 | 验证方式 |
|:-------|:-----|:---------|
| 可复现 | `make reproduce` 一键运行 | CI 通过 |
| 可验证 | `make verify` 对账测试 | 差异 < tolerance |
| 可部署 | `docker build` 容器化 | 镜像构建成功 |
| 可监控 | PSI 漂移检测 | 阈值告警正常 |

---

*模板版本: v1.0*  
*适用: 开源模型本地化部署*
