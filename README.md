# DynexSDK
DynexSDK（Dynex软件开发工具包）
Dynex是基于DynexSolve芯片算法的世界上第一个神经形态超级计算区块链，采用了“有用工作证明”（PoUW）方法来解决现实世界的问题。Dynex SDK 用于与Dynex平台进行交互和计算。所有示例都需要使用Python的DynexSDK以及有效的API密钥。我们的存储库会持续更新，请定期查看以获取更新。

## 入门指导

使用以下命令下载并安装Dynex SDK：

```
pip install dynex
```

然后按照[安装Dynex SDK](https://github.com/dynexcoin/DynexSDK/wiki/Installing-the-Dynex-SDK)中的说明配置SDK. 建议首先下载Dynex SDK Hello World示例 [Dynex SDK Hello World Example](https://github.com/dynexcoin/DynexSDK/blob/main/Dynex_hello_world.ipynb) 以了解如何使用Dynex神经形态平台的基本步骤。

Dynex SDK文档:
- [Dynex SDK Wiki](https://github.com/dynexcoin/DynexSDK/wiki)
- [Dynex SDK 文档](https://docs.dynexcoin.org/)

Dynex SDK 专业社区:
- [Dynex Slack工作区](https://join.slack.com/t/dynex-workspace/shared_invite/zt-22eb1n4mo-aXS5zsUBoPs613Dofi8Q4A)
- [Dynex 中文开发社区](http://qm.qq.com/cgi-bin/qm/qr?_wv=1027&k=CKneJaNvrwqvamLjIV0LYkUm4njcTvzq&authKey=0IRFp1FN%2BVrpixA2sOMwEPqVhd5geP6P7qRxhvpCylHbWutm6hSKYBqlznfBybh5&noverify=0&group_code=705867412)

## 指南
- [案例：在Dynex神经形态平台上进行计算：图像分类](https://github.com/DynexCN/DynexCaseCollection/blob/main/Medium_Image_Classification.ipynb)
- [知乎：在Dynex神经形态平台上进行计算：IBM Qiskit 4比特全加器电路](https://zhuanlan.zhihu.com/p/660928628)
- [知乎：使用Q-Score对Dynex神经形态平台进行基准测试](https://zhuanlan.zhihu.com/p/660928628)


## 初学者指南

为了让新手快速熟悉如何在Dynex平台上进行计算，我们准备了一些Python Jupyter笔记本。以下是一些初学者指南，演示了如何使用Dynex SDK。

- [示例：使用Python在Dynex平台上进行计算 - 二进制二次优化问题](https://github.com/DynexCN/DynexCaseCollection/blob/main/beginners_guide_example_bqm.ipynb)
- [使用Python在Dynex平台上进行计算 - 二进制二次优化问题 K4完全图](https://github.com/DynexCN/DynexCaseCollection/blob/main/beginners_guide_example_bqm_k4_complete_graph.ipynb)
- [示例：使用Python在Dynex平台上进行计算 - 逻辑门](https://github.com/DynexCN/DynexCaseCollection/blob/main/beginners_guide_example_logic_gates.ipynb)
- [示例：使用Python在Dynex平台上进行计算 - QUBO](https://github.com/DynexCN/DynexCaseCollection/blob/main/beginners_guide_example_QUBO.ipynb)
- [示例：使用Python在Dynex平台上进行计算 - 反交叉问题](https://github.com/DynexCN/DynexCaseCollection/blob/main/beginners_guide_example_anti_crossing_clique.ipynb)
- [示例：使用Python在Dynex平台上进行计算 - 最大独立集](https://github.com/DynexCN/DynexCaseCollection/blob/main/beginners_guide_example_MIS.ipynb)
- [示例：使用Python在Dynex平台上进行计算 - SAT](https://github.com/DynexCN/DynexCaseCollection/blob/main/beginners_guide_example_SAT.ipynb)
- [示例：使用Python在Dynex平台上进行计算 - NAE3SAT](https://github.com/DynexCN/DynexCaseCollection/blob/main/beginners_guide_example_random_nae3sat.ipynb)

## 高级示例

以下是一些高级的代码示例和笔记本，可用于在Dynex神经形态计算平台上进行计算：

- [示例：烟草花叶病毒RNA折叠](https://github.com/DynexCN/DynexCaseCollection/blob/main/example_rna_folding.ipynb) | 科学背景：Fox DM，MacDermaid CM，Schreij AMA，Zwierzyna M，Walker RC。使用量子计算机进行RNA折叠，PLoS Comput Biol。2022年4月11日；18(4)：e1010032。doi：10.1371/journal.pcbi.1010032。PMID：35404931；PMCID：PMC9022793

- [示例：量子单图像超分辨率](https://github.com/dynexcoin/DynexSDK/tree/main/Quantum-SISR) |  科学背景：Choong HY，Kumar S，Van Gool L。单图像超分辨的量子退火。在计算机视觉和模式识别2023 IEEE / CVF会议文集中的论文（第1150-1159页）。

- [示例：充电站的布置](https://github.com/DynexCN/DynexCaseCollection/blob/main/example_placement_of_charging_stations.ipynb) | 科学背景：Pagany，Raphaela＆Marquardt，Anna＆Zink，Roland。 （2019）。电动充电需求位置模型-一种基于用户和目的地的电动汽车充电站的布置方法。可持续性。11。2301。10.3390/su11082301

- [示例：使用Dynex scikit-learn插件进行乳腺癌预测](https://github.com/DynexCN/DynexCaseCollection/blob/main/Dynex%20Scikit-Learn%20Plugin.ipynb) |  科学背景：Bhatia，H.S.，Phillipson，F.（2021）。在D-Wave量子退火器上实现支持向量机的性能分析。在：Paszynski，M.，Kranzlmüller，D.，Krzhizhanovskaya，V.V.，Dongarra，J.J.，Sloot，P.M.A.（eds）计算科学-ICCS 2021。ICCS 2021。计算机科学讲义（），第12747卷。斯普林格，香槟
  
- [示例：量子整数分解 ](https://github.com/dynexcoin/DynexSDK/blob/main/example_integer_factorisation.ipynb) | 科学背景：江，S.，Britt，K.A.，McCaskey，A.J.等。 量子退火用于素数分解。科学报告8，17667（2018）

- [示例：酶靶标预测](www.github.com/samgr55/Enzyme-TargetPrediction_QUBO-Ising) | 科学背景：Hoang M Ngo，My T Thai，Tamer Kahveci，QuTIE：酶靶标的量子优化，生物信息学进展，2023；，vbad112
  
- [最佳WiFi热点定位预测](https://github.com/samgr55/OptimalWiFi-HotspotPositioning_QUBO-Ising)

## 机器学习示例

用于机器学习的量子计算算法利用了量子力学的力量来增强机器学习任务的各个方面。由于量子计算和神经形态计算都具有相似的特征，因此这些算法也可以在Dynex平台上有效地计算，但不受有限量子比特、错误校正或可用性的限制。

**量子支持向量机（QSVM）：** QSVM是一种受量子启发的算法，旨在使用量子核函数对数据进行分类。它利用了量子叠加和量子特征映射的概念，在某些情况下可能在某些场景中比经典SVM算法提供更高的计算优势。

**量子主成分分析（QPCA）：** QPCA是经典主成分分析（PCA）算法的量子版本。它利用量子线性代数技术从高维数据中提取主成分，从而在量子机器学习中更有效地降维。QPCA是经典主成分分析（PCA）算法的量子版本。它利用量子线性代数技术从高维数据中提取主成分，从而在量子机器学习中更有效地降维。
  
**量子神经网络（QNN）：** QNN是经典神经网络的量子对应物。它利用了量子原理，如量子叠加和纠缠，来处理和操作数据。QNN有潜力学习复杂的模式并执行分类和回归等任务，从量子并行性中受益。

**量子K均值聚类：** 量子K均值是经典K均值聚类算法的量子启发变体。它使用量子算法同时探索多个解决方案，以加速大规模数据集的聚类任务。

**量子Boltzmann机器（QBMs）：** QBMs是用于无监督学习的量子类似于经典Boltzmann机器的算法。QBMs利用量子退火从概率分布中采样并学习数据中的模式和结构。

**量子支持向量回归（QSVR）：** QSVR扩展了QSVM的概念，用于回归任务。它使用量子计算技术执行回归分析，可能在效率和准确性方面比经典回归算法具有优势。

以下是在Dynex平台上实施这些算法的一些示例：

- [示例：在Dynex上实施量子支持向量机](https://github.com/DynexCN/DynexCaseCollection/blob/main/example_support_vector_machine.ipynb) | 科学背景：Rounds, Max and Phil Goddard. “Optimal feature selection in credit scoring and classification using a quantum annealer.” (2017)

- [示例：在Dynex上的量子支持向量机（PyTorch）](https://github.com/DynexCN/DynexCaseCollection/blob/main/Example_SVM_pytorch.ipynb) | 科学背景：Rounds, Max and Phil Goddard. “Optimal feature selection in credit scoring and classification using a quantum annealer.” (2017)

- [示例：在Dynex上的量子Boltzmann机器（PyTorch）](https://github.com/DynexCN/DynexCaseCollection/blob/main/example_neuromorphic_torch_layers%20(1).ipynb) |  科学背景：Dixit V, Selvarajan R, Alam MA, Humble TS and Kais S (2021) Training Restricted Boltzmann Machines With a D-Wave Quantum Annealer. Front. Phys. 9:589626. doi: 10.3389/fphy.2021.589626; Sleeman, Jennifer, John E. Dorband and Milton Halem. “A Hybrid Quantum enabled RBM Advantage: Convolutional Autoencoders For Quantum Image Compression and Generative Learning.” Defense + Commercial Sensing (2020)

- [示例：在Dynex上的量子Boltzmann机器实施（3步QUBO）](https://github.com/DynexCN/DynexCaseCollection/blob/main/Dynex-Full-QRBM.ipynb) | 科学背景：Dixit V, Selvarajan R, Alam MA, Humble TS and Kais S (2021) Training Restricted Boltzmann Machines With a D-Wave Quantum Annealer. Front. Phys. 9:589626. doi: 10.3389/fphy.2021.589626; Sleeman, Jennifer, John E. Dorband and Milton Halem. “A Hybrid Quantum enabled RBM Advantage: Convolutional Autoencoders For Quantum Image Compression and Generative Learning.” Defense + Commercial Sensing (2020)

- [示例：在Dynex上的量子Boltzmann机器（协同过滤）](https://github.com/DynexCN/DynexCaseCollection/blob/main/example_collaborative_filtering_CFQIRBM.ipynb) | 科学背景：Dixit V, Selvarajan R, Alam MA, Humble TS and Kais S (2021) Training Restricted Boltzmann Machines With a D-Wave Quantum Annealer. Front. Phys. 9:589626. doi: 10.3389/fphy.2021.589626; Sleeman, Jennifer, John E. Dorband and Milton Halem. “A Hybrid Quantum enabled RBM Advantage: Convolutional Autoencoders For Quantum Image Compression and Generative Learning.” Defense + Commercial Sensing (2020)

- [示例：在Dynex上实施量子Boltzmann机器](https://github.com/DynexCN/DynexCaseCollection/blob/main/example_quantum_boltzmann_machine_QBM.ipynb) | 科学背景：Dixit V, Selvarajan R, Alam MA, Humble TS and Kais S (2021) Training Restricted Boltzmann Machines With a D-Wave Quantum Annealer. Front. Phys. 9:589626. doi: 10.3389/fphy.2021.589626; Sleeman, Jennifer, John E. Dorband and Milton Halem. “A Hybrid Quantum enabled RBM Advantage: Convolutional Autoencoders For Quantum Image Compression and Generative Learning.” Defense + Commercial Sensing (2020)

- [示例：模式辅助的无监督学习受限制Boltzmann机器（MA-QRBM for Pytorch）](https://github.com/dynexcoin/DynexSDK/tree/main/MAQRBM) | 科学背景：模式辅助的无监督学习受限制Boltzmann机器，通信物理学卷3，文章编号：105（2020）

- [示例：特征选择 - 泰坦尼克号幸存者](https://github.com/DynexCN/DynexCaseCollection/blob/main/example_feature_selection_titanic_survivals.ipynb) | 科学背景：Xuan Vinh Nguyen, Jeffrey Chan, Simone Romano, and James Bailey. 2014. Effective global approaches for mutual information based feature selection. In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining (KDD '14). Association for Computing Machinery, New York, NY, USA, 512–521

- [示例：使用Dynex scikit-learn插件进行乳腺癌预测](https://github.com/DynexCN/DynexCaseCollection/blob/main/Dynex%20Scikit-Learn%20Plugin.ipynb) | 科学背景：Bhatia, H.S., Phillipson, F. (2021). Performance Analysis of Support Vector Machine Implementations on the D-Wave Quantum Annealer. In: Paszynski, M., Kranzlmüller, D., Krzhizhanovskaya, V.V., Dongarra, J.J., Sloot, P.M.A.（eds）Computational Science – ICCS 2021. ICCS 2021. Lecture Notes in Computer Science(), vol 12747. Springer, Cham

## Dynex神经形态Torch层

Dynex神经形态Torch层可以用于任何神经网络模型。欢迎使用混合模型、神经形态、迁移学习和联邦学习与PyTorch。
[PyTorch](https://pytorch.org/)
 
- [示例：在Dynex上的量子Boltzmann机器（PyTorch）](https://github.com/DynexCN/DynexCaseCollection/blob/main/example_neuromorphic_torch_layers%20(1).ipynb) | Scientific background: Dixit V, Selvarajan R, Alam MA, Humble TS and Kais S (2021) Training Restricted Boltzmann Machines With a D-Wave Quantum Annealer. Front. Phys. 9:589626. doi: 10.3389/fphy.2021.589626; Sleeman, Jennifer, John E. Dorband and Milton Halem. “A Hybrid Quantum enabled RBM Advantage: Convolutional Autoencoders For Quantum Image Compression and Generative Learning.” Defense + Commercial Sensing (2020)

- [示例：在Dynex上的量子支持向量机（PyTorch）](https://github.com/DynexCN/DynexCaseCollection/blob/main/Example_SVM_pytorch.ipynb) | Scientific background: Rounds, Max and Phil Goddard. “Optimal feature selection in credit scoring and classification using a quantum annealer.” (2017)

## Dynex Qiskit包

感谢[Richard H. Warren](https://arxiv.org/pdf/1405.2354.pdf)的开创性研究,现在可以直接将Qiskit量子电路翻译成Dynex神经形态芯片。这个概念的背后是将Qiskit对象直接翻译，但是不是在IBM Q上运行，而是在Dynex神经形态平台上执行电路。以下是使用这种方法的一个一比特加法器电路示例：

```
from dynexsdk.qiskit import QuantumRegister, ClassicalRegister
from dynexsdk.qiskit import QuantumCircuit, execute

# 输入寄存器：a = qi[0]；b = qi[1]；ci = qi[2]
qi = QuantumRegister(3)
ci = ClassicalRegister(3)

# 输出寄存器：s = qo[0]；co = qo[1]
qo = QuantumRegister(2)
co = ClassicalRegister(2)
circuit = QuantumCircuit(qi, qo, ci, co)

# 定义加法器电路
for idx in range(3):
    circuit.ccx(qi[idx], qi[(idx+1) % 3], qo[1])
for idx in range(3):
    circuit.cx(qi[idx], qo[0])

circuit.measure(qo, co)

# 运行
execute(circuit)

# 输出
print(circuit)
```

[Dynex Qiskit包（Github）](https://github.com/dynexcoin/Dynex-Qiskit)

## Dynex scikit-learn插件

此包提供了一个用于使用Dynex神经形态计算平台进行特征选择的scikit-learn转换器。它专为与scikit-learn集成而构建，这是一种行业标准的、最先进的Python机器学习库。

 [Dynex scikit-learn 插件](https://github.com/dynexcoin/DynexSDK/tree/main/dynex_scikit_plugin)使在ML工作流程的特征选择方面更容易使用Dynex平台。特征选择是机器学习的关键构建块，它是确定一小组最具代表性特征以改善模型训练和性能的问题。有了这个新的插件，ML开发人员不需要成为优化或混合求解方面的专家，就能获得商业和技术上的双重好处。创建特征选择应用程序的开发人员可以构建与scikit-learn集成的管道，然后更轻松高效地将Dynex平台嵌入到这个工作流程中。


## Dynex QBoost Implementation

D-Wave量子计算机作为一个接受任何问题的离散优化引擎已经广泛研究，该问题被制定为二次不受约束的二进制优化（QUBO）。2008年，Google和D-Wave发表了一篇论文, [Training a Binary Classifier with the Quantum Adiabatic Algorithm](https://arxiv.org/pdf/0811.0416.pdf),介绍了Qboost集成方法是如何使二进制分类适于量子计算的：该问题被制定为一组弱分类器的阈值线性叠加，D-Wave量子计算机用于在学习过程中优化权重，力求最小化训练错误和弱分类器数量。

[Dynex QBoost Implementation](https://github.com/dynexcoin/DynexSDK/tree/main/dynex_qboost) 提供了一个QBoost算法插件，可用于使用Dynex神经形态平台。

## DIMOD：QUBO/ISING采样器的共享API
Dimod是一种共享的采样器API。它提供了二次模型的类，例如包含Dynex神经形态平台或D-Wave系统采样器使用的Ising和QUBO模型的二进制二次模型（BQM）类，以及高阶（非二次）模型、采样器和复合采样器的参考示例，以及构建新采样器和复合采样器的抽象基类。

[Dimod文档](https://docs.ocean.dwavesys.com/en/stable/docs_dimod/)

## PyQUBO：从灵活的数学表达式创建QUBO或Ising模型
PyQUBO允许您轻松地从灵活的数学表达式创建QUBO或Ising模型。它基于Python（C++后端），与Ocean SDK完全集成，支持约束的自动验证，并提供参数调整的占位符。

[PyQUBO 文档](https://pyqubo.readthedocs.io/)

## 进一步阅读

- [Dynex 英文](https://dynexcoin.org/)
- [Dynex 中文](https://dynexcoin.top/)
- [Dynex 企业](https://dynexcoin.org/learn/dynex-for-enterprises)
- [Dynex SDK](https://dynexcoin.org/learn/dynex-sdk)
- [Dynex SDK初学指南](https://dynexcoin.org/learn/beginner-guides)
- [Dynex SDK高级示例](https://dynexcoin.org/learn/advanced-examples)

## 许可证

根据GNU通用公共许可证第3版授权。请参阅Dynex包中的许可证文件。
