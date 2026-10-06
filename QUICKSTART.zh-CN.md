# 中文快速上手

这个项目用可运行代码讲清 **JJAP MNIST HCST** 方法：模拟电路的有限增益与固定偏置如何改变电压推理，以及如何将这些参数放进软件训练。它是新写的参考框架，不是原实验脚本的完整公开，也不声称已复现论文准确率。

ReRAM Crossbar Array 也称 Synapse Array；其后 IV-Converter + Subtractor、Activation Function、Voltage Follower 合称 Neuron Circuit。ReLU 是 JJAP 的激活实例；顶层使用通用 Activation Function 名称。

无需私有资料、Torch、HSPICE、数据或权重即可运行：

```powershell
git clone https://github.com/Scophield/HCST.git
cd HCST
python -m pip install '.[test]'
reram run --config configs/smoke.json --out runs/smoke-001
python -m pytest -q
```

每次指定新输出目录。该 demo 用合成数据检查同一数学模型的完整训练→双电导→网表→电压推理→结果链路，不是分类性能基准。

默认 `experimental-analytic` 是实验性计算优化器，不等价于原 HCST 更新。原类训练用 `legacy-source`，需要另行获得有权使用的 `training_tensor` 源目录；项目不分发该历史源码，也不执行其原main程序。核查历史推理时必须显式匹配权重族、offset和gain/load参数。

获取 MNIST 后的运行方式，以及可选 HSPICE 和 Activation Function 接口，见英文 README。HSPICE及设备模型外置，当前没有实际晶体管级验证；新实验的44.73%等结果不得写成论文复现。历史权重、PDK、密码和未发表材料均未包含在仓库中。
