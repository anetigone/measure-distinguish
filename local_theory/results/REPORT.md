# 理论验证汇总

本目录只包含理论模型、反例与有限样本模拟；不将实现集成检查称为本地验证。

多对象方差恒等式、one-hot 等价边界、有界噪声漂移证书与上下文选择反例已完成数值验证。

固定 p=0.35、基础噪声标准差 0.12、假定噪声方差漂移预算 0.02；每次 500 次 bootstrap。

| N | 生成机制 | 有条件拒绝选择 | 均值模型失败 | 假设未验证 | gap 区间覆盖率 |
|---:|---|---:|---:|---:|---:|
| 128 | fixed_pool | 0% | 0% | 100% | 93% |
| 128 | random_pool | 0% | 0% | 100% | 93% |
| 128 | selection | 0% | 0% | 100% | 96% |
| 128 | near_one_hot_pool | 0% | 0% | 100% | 94% |
| 128 | heteroscedastic_pool | 0% | 3% | 97% | 95% |
| 128 | routing_drift | 0% | 100% | 0% | — |
| 128 | nonlinear_readout | 0% | 4% | 96% | — |
| 128 | context_contaminated_selection | 0% | 0% | 100% | 96% |
| 512 | fixed_pool | 69% | 0% | 31% | 94% |
| 512 | random_pool | 78% | 0% | 22% | 95% |
| 512 | selection | 0% | 0% | 17% | 96% |
| 512 | near_one_hot_pool | 0% | 0% | 26% | 97% |
| 512 | heteroscedastic_pool | 0% | 0% | 100% | 97% |
| 512 | routing_drift | 0% | 100% | 0% | — |
| 512 | nonlinear_readout | 0% | 18% | 82% | — |
| 512 | context_contaminated_selection | 67% | 0% | 33% | 94% |
| 2048 | fixed_pool | 100% | 0% | 0% | 93% |
| 2048 | random_pool | 100% | 0% | 0% | 97% |
| 2048 | selection | 0% | 0% | 0% | 96% |
| 2048 | near_one_hot_pool | 0% | 0% | 0% | 95% |
| 2048 | heteroscedastic_pool | 0% | 0% | 47% | 95% |
| 2048 | routing_drift | 0% | 100% | 0% | — |
| 2048 | nonlinear_readout | 0% | 42% | 58% | — |
| 2048 | context_contaminated_selection | 100% | 0% | 0% | 93% |

这里的拒绝均以噪声、路由和读取假设成立为前提。近 one-hot 与异方差反例可以无法区分；上下文污染后的选择可与直接聚合完全等价。

未检测到失配不算验证假设：漂移置信区间必须完整落入容许范围，否则返回假设未验证。

100 个重复中 0 次误拒绝的 Wilson 95% 上界约为 3.7%；观察到 100% 检出也不代表普遍保证。该扫描只覆盖一个 p/噪声/预算组合，不能替代真实网络校准或完整识别边界。
