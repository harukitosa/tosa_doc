# Machine learning of high dimensional data on a noisy quantum processor

```
gptによる機械翻訳です和訳後の文章は正確ではありません。
```

[arxiv link](https://arxiv.org/abs/2101.09581)

## abstruct

We present a quantum kernel method for high-dimensional data analysis using Google's universal quantum processor, Sycamore. %Rainbow-23.
This method is successfully applied to the cosmological benchmark of supernova classification using real spectral features with no dimensionality reduction and without vanishing kernel elements.

Instead of using a synthetic dataset of low dimension or pre-processing the data with a classical machine learning algorithm to reduce the data dimension, this experiment demonstrates that machine learning with real, high dimensional data is possible using a quantum processor; but it requires careful attention to shot statistics and mean kernel element size when constructing a circuit ansatz.

Our experiment utilizes 17 qubits to classify 67 dimensional data - significantly higher dimensionality than the largest prior quantum kernel experiments - resulting in classification accuracy that is competitive with noiseless simulation and comparable classical techniques.

---


Googleの汎用量子プロセッサ、Sycamoreを使用して、高次元データ分析のための量子カーネル方法を提案します。%Rainbow-23。
この方法は、次元削減を行わず、カーネル要素が消失しない実際のスペクトル特徴を使用して、宇宙学のベンチマークである超新星の分類に成功的に適用されました。

低次元の合成データセットを使用するか、データの次元を削減するために古典的な機械学習アルゴリズムでデータを前処理する代わりに、この実験は、量子プロセッサを使用した実際の高次元データの機械学習が可能であることを示しています。しかし、回路アンザッツを構築する際に、ショットの統計と平均カーネル要素のサイズに注意が必要です。

私たちの実験は、17の量子ビットを利用して67次元のデータを分類します - これは、これまでの最大の量子カーネル実験よりもはるかに高い次元性です - その結果、ノイズのないシミュレーションと比較可能な古典的な技術と競合する分類精度を達成しました。

[1]<br/>
 本文: <br/>


\section{Introduction}

Quantum kernel methods (QKM) \cite{Havlicek2019,PhysRevLett.122.040504} provide techniques for utilizing a quantum co-processor in a machine learning setting.
These methods were recently proven to provide a speedup over classical methods for certain specific input data classes \cite{liu2020rigorous}.
They have also been used to quantify the computational power of data in quantum machine learning algorithms and drive the conditions under which quantum models will be capable of outperforming classical ones \cite{huang2020power}. 
Prior experimental work \cite{kusumoto2019experimental,bartkiewicz2020experimental,Havlicek2019} has focused on artificial or heavily pre-processed data, hardware implementations involving very few qubits, or circuit connectivity unsuitable<br/>
 和訳: <br/>
\section{序論}

量子カーネル法（QKM）\cite{Havlicek2019,PhysRevLett.122.040504}は、機械学習の設定において量子コプロセッサを利用するための技術を提供します。
これらの方法は、最近の研究により、特定の入力データクラスにおいて古典的な方法に比べて高速化を実現することが証明されています\cite{liu2020rigorous}。
また、これらの方法は、量子マシンラーニングアルゴリズムにおけるデータの計算能力を定量化し、量子モデルが古典的なモデルを凌駕することが可能な条件を示すために使用されています\cite{huang2020power}。
従来の実験\cite{kusumoto2019experimental,bartkiewicz2020experimental,Havlicek2019}では、人工的なデータや厳密に前処理されたデータ、非常に少数のキュビットを使用したハードウェア実装、回路の接続性などが適切でないものに焦点を当ててきました。<br/><br/>

[2]<br/>
 本文: <br/>
 for NISQ \cite{Preskill2018quantumcomputingin} processors; recent experimental results show potential for many-qubit applications of QKM to high energy physics \cite{wu2020application}.

In this work, we extend the method of machine learning based on quantum kernel methods up to 17 hardware qubits requiring only nearest-neighbor connectivity.
We use this circuit structure to prepare a kernel matrix for a classical support vector machine to learn patterns in 67-dimensional supernova data for which competitive classical classifiers fail to achieve 100\% accuracy.
To extract useful information from a processor without quantum error correction (QEC), we implement error mitigation techniques specific to the QKM algorithm and experimentally demonstrate the algorithm's robustness to some of the <br/>
 和訳: <br/>
最近の実験結果から、NISQ（Noisy Intermediate-Scale Quantum）プロセッサにおいて、QKM（Quantum Kernel Method）を用いた多キュービットの高エネルギー物理学への応用の可能性が示されている\cite{wu2020application}。本研究では、最寄り接続のみを必要とする17個のハードウェアキュービットに基づいた量子カーネル法を機械学習の手法として拡張する。我々は、この回路構造を用いて、競合する古典的な分類器が100％の正解率を達成できない67次元の超新星データのパターンを学習するためのクラシカルサポートベクターマシン向けのカーネル行列の準備を行う。量子誤り訂正（QEC）を持たないプロセッサから有用な情報を抽出するために、QKMアルゴリズムに特化した誤り軽減技術を実装し、そのアルゴリズムのロバスト性を実験的にデモンストレーションする。<br/><br/>

[3]<br/>
 本文: <br/>
device noise. Additionally, we justify our circuit design based on its ability to produce large kernel magnitudes that can be sampled to high statistical certainty with relatively short experimental runs.

We implement this algorithm on the Google Sycamore processor which we accessed through Google's Quantum Computing Service.
This machine is similar to the quantum supremacy demonstration Sycamore chip \cite{Arute2019}, but with only 23 qubits active.
We achieve competitive results on a nontrivial classical dataset, and find intriguing classifier robustness in the face of moderate circuit fidelity.
Our results motivate further theoretical work on noisy kernel methods and on techniques for operating on real, high-dimensional data without additional classical pre-processing or dimensionality<br/>
 和訳: <br/>
装置のノイズによる影響を最小限に抑えるために、回路設計を正当化します。さらに、比較的短い実験時間で高い統計的確率でサンプリングできるような大きなカーネルの大きさを生成する能力に基づいて、回路設計を正当化します。

私たちは、Googleの量子コンピューティングサービスを介してアクセスしたGoogleのSycamoreプロセッサ上でこのアルゴリズムを実装します。このマシンは、量子優位デモンストレーションのSycamoreチップ\cite{Arute2019}に似ていますが、アクティブな量子ビットは23個のみです。私たちは、非自明なクラシカルデータセットで競争力のある結果を達成し、中程度の回路の正確さにもかかわらず、興味深い分類器の頑健性を見つけます。私たちの結果は、ノイズのあるカーネル法と現実の高次元データ上での追加のクラシカル前処理や次元削減なしに操作するための技術に関するさらなる理論的研究を促すものです。



装置のノイズによる影響を最小限に抑えるために、回路設計を正当化します。さらに、比較的短い実験時間で高い統計的確率でサンプリングできるような大きなカーネルの大きさを生成する能力に基づいて、回路設計を正当化します。

私たちは、Googleの量子コンピューティングサービスを介してアクセスしたGoogleのSycamoreプロセッサ上でこのアルゴリズムを実装します。このマシンは、量子優位デモンストレーションのSycamoreチップ[1]に似ていますが、アクティブな量子ビットは23個のみです。私たちは、非自明なクラシカルデータセットで競争力のある結果を達成し、中程度の回路の正確さにもかかわらず、興味深い分類器の頑健性を見つけます。私たちの結果は、ノイズのあるカーネル法と現実の高次元データ上での追加のクラシカル前処理や次元削減なしに操作するための技術に関するさらなる理論的研究を促すものです。<br/><br/>

[4]<br/>
 本文: <br/>
 reduction.

\section{Quantum kernel Support Vector Machines} 

A common task in machine learning is \textit{supervised learning}, wherein an algorithm consumes datum-label pairs $(x, y) \in \mathcal{X} \times \{0, 1\}$ and outputs a function $f: \mathcal{X} \rightarrow \{0, 1\}$ that ideally predicts labels for seen (training) input data and generalizes well to unseen (test) data.
A popular supervised learning algorithm is the Support Vector Machine (SVM) \cite{cortes1995support,Boser:1992:TAO:130385.130401} which is trained on inner products $\langle x_i, x_j\rangle$ in the input space to find a robust linear classification boundary that best separates the data. An important technique for generalizing SVM classifiers to non-linearly separable data is the so-called ``kernel trick''  which<br/>
 和訳: <br/>
\section{量子カーネルサポートベクターマシン}

機械学習の一般的なタスクは、アルゴリズムがデータ-ラベルのペア$(x, y) \in \mathcal{X} \times \{0, 1\}$を受け取り、訓練データに対してラベルを予測し、未知のデータに対してもうまく汎化する関数$f: \mathcal{X} \rightarrow \{0, 1\}$を出力する「教師あり学習」です。
サポートベクターマシン（SVM）\cite{cortes1995support,Boser:1992:TAO:130385.130401}は人気のある教師あり学習アルゴリズムであり、入力空間内の内積$\langle x_i, x_j\rangle$に基づいてトレーニングされ、データを最もよく分離する頑健な線形分類境界を見つけます。非線形分離可能なデータに対してSVM分類器を一般化するための重要な手法は、「カーネルトリック」と呼ばれるものです。<br/><br/>

[5]<br/>
 本文: <br/>
 replaces  $\langle x_i, x_j\rangle$ in the SVM formulation by a symmetric positive definite kernel function $k(x_i, x_j)$ \cite{Aizerman1964}. Since every kernel function corresponds to an inner product on input data mapped into a feature Hilbert space \cite{aronszajn1950theory}, linear classification boundaries found by an SVM trained on a high-dimensional mapping correspond to complex, non-linear functions in the input space. 
\onecolumngrid

\begin{figure}[t]
    \centering
    \includegraphics[width=0.91\textwidth]{plots/svm_flowchart.pdf}
    \caption{In this experiment we performed limited data preprocessing that is standard for state-of-the-art classical techniques, before using the quantum processor to estimate the kernel matrix $\hat{K}_{ij}$ for all pairs of encoded datapoints $<br/>
 和訳: <br/>

 SVMの定式化において、$\langle x_i, x_j\rangle$は、対称正定値のカーネル関数$k(x_i, x_j)$に置き換えられます\cite{Aizerman1964}。すべてのカーネル関数は、特徴ヒルベルト空間への入力データの写像に対応する内積に対応しているため\cite{aronszajn1950theory}、高次元写像上で訓練されたSVMによって見つかった線形分類境界は、入力空間での複雑な非線形関数に対応します。

\begin{figure}[t]
\centering
\includegraphics[width=0.91\textwidth]{plots/svm_flowchart.pdf}
\caption{この実験では、量子プロセッサを使用してエンコードされたデータポイントのすべてのペアのためのカーネル行列$\hat{K}_{ij}$を推定する前に、最先端の古典的技術の標準である限定的なデータ前処理を実行しました。$<br/>
<br/><br/>

[6]<br/>
 本文: <br/>
(x_i, x_j)$ in each dataset. We then passed the kernel matrix back to a classical computer to optimize an SVM using cross validation and hyperparameter tuning before evaluating the SVM to produce a final train/test score.}
    \label{fig:svm_flowchart}

\end{figure}

\twocolumngrid
Quantum kernel methods can potentially improve the performance of classifiers by using a quantum computer to map input data in $\mathcal{X}\subset \mathbb{R}^d$ into a high-dimensional complex Hilbert space, potentially resulting in a kernel function that is expressive and  challenging to compute classically. It is difficult to know without sophisticated knowledge of the data generation process whether a given kernel is particularly suited to a dataset, but perhaps families of classically hard kernels may be sho<br/>
 和訳: <br/>
図\ref{fig:svm_flowchart}には、量子カーネル法によって分類器の性能を向上させることができる可能性があります。これは、入力データを量子コンピュータを使用して、$\mathbb{R}^d$の高次元複素ヒルベルト空間にマッピングすることで、古典的に計算するのが困難なカーネル関数を生成することができるためです。与えられたカーネルが特定のデータセットに適しているかどうかは、データ生成プロセスの高度な知識がなければ理解するのは難しいかもしれませんが、おそらく古典的に難解なカーネルのファミリーがデータセットに適しているかもしれません。<br/><br/>

[7]<br/>
 本文: <br/>
wn empirically to offer performance improvements.
In this work we focus on a non-variational quantum kernel method, which uses a quantum circuit $U(x)$ to map real data into quantum state space according to a map $\phi(x) = U (x) |0\rangle$. The  kernel function we employ is then the squared inner product between pairs of mapped input data given by  $k(x_i, x_j) = |\langle \phi(x_i) | \phi(x_j) \rangle|^2$, which allows for more expressive models compared to the alternative choice $\langle \phi (x_i) | \phi (x_j) \rangle$ \cite{huang2020power}.

In the absence of noise, the kernel matrix $K_{ij} = k(x_i, x_j)$ for a fixed dataset can therefore be estimated up to statistical error by using a quantum computer to sample outputs of the circuit $U^\dagger (x_i) U (x_j)$ and then computing the e<br/>
 和訳: <br/>
この研究では、実データを量子状態空間に写像するために量子回路$U(x)$を使用する非変分的な量子カーネル法に焦点を当てます。写像$\phi(x) = U(x)|0\rangle$に従って、マップされた入力データのペアの内積の二乗であるカーネル関数$k(x_i, x_j) = |\langle\phi(x_i) | \phi(x_j)\rangle|^2$を使用します。これによって、$\langle\phi(x_i) | \phi(x_j)\rangle$と比較してより表現力のあるモデルが可能となります\cite{huang2020power}。

ノイズのない場合、固定されたデータセットに対してカーネル行列$K_{ij} = k(x_i, x_j)$は、量子コンピュータを使用して回路$U^\dagger(x_i)U(x_j)$の出力をサンプリングし、その後統計的な誤差を考慮して推定することができます。<br/><br/>

[8]<br/>
 本文: <br/>
mpirical probability of the all-zeros bitstring. However in practice, the kernel matrix $\hat{K}_{ij}$ sampled from the quantum computer may be significantly different from $K_{ij}$ due to device noise and readout error. Once $\hat{K}_{ij}$ is computed for all pairs of input data in the training set, a classical SVM can be trained on the outputs of the quantum computer. An SVM trained on a size-$m$ training set $\mathcal{T} \subset \mathcal{X}$ learns to predict the class $f(x) = \hat{y}$ of an input data point $x$ according to the decision function:
\begin{equation}\label{eq:decision_main}
f(x) = \text{sign}\left(\sum_{i=1}^m \alpha_{i} y_i k(x_i, x) + b\right)
\end{equation}
where $\alpha_i$ and $b$ are parameters determined during the training stage of the SVM. Training and evaluating t<br/>
 和訳: <br/>
実践では、量子コンピュータからサンプリングされたカーネル行列$\hat{K}_{ij}$は、デバイスのノイズや読み取りエラーにより$K_{ij}$と大幅に異なる可能性があります。訓練セットのすべての入力データのペアに対して$\hat{K}_{ij}$が計算されたら、古典SVMを量子コンピュータの出力に対してトレーニングすることができます。サイズ-$m$の訓練セット$\mathcal{T} \subset \mathcal{X}$にトレーニングされたSVMは、決定関数に従って入力データ点$x$のクラス$f(x) = \hat{y}$を予測することを学習します：
$$
f(x) = \text{sign}\left(\sum_{i=1}^m \alpha_{i} y_i k(x_i, x) + b\right)
$$
ここで、$\alpha_i$と$b$はSVMのトレーニング段階で決定されるパラメータです。訓練と評価<br/><br/>

[9]<br/>
 本文: <br/>
he SVM on $\mathcal{T}$ requires an $m \times m$ kernel matrix, after which each data point $z$ in the testing set $\mathcal{V}\subset \mathcal{X}$ may be classified using an additional $m$ evaluations of $k(x_i, z)$ for $i=1\dots m$. Figure \ref{fig:svm_flowchart} provides a schematic representation of the process used to train an SVM using quantum kernels. 

% % % Preprocessing data
\subsection{Data and preprocessing} \label{sec:dataset}

We used the dataset provided in the Photometric LSST Astronomical Time-series Classification Challenge (PLAsTiCC) \cite{team2018photometric} that simulates observations of the Vera C. Rubin Observatory \cite{verarubin}. The PLAsTiCC data consists of simulated astronomical time series for several different classes of astronomical objects.
The time series<br/>
 和訳: <br/>
SVMの$\mathcal{T}$上での実装では、$m \times m$のカーネル行列が必要であり、その後テストセット$\mathcal{V}\subset \mathcal{X}$内の各データ点$z$は、$i=1\dots m$に対して$k(x_i, z)$を評価することによって分類される。図\ref{fig:svm_flowchart}には、量子カーネルを使用してSVMを訓練するプロセスの概略図が示されている。

\subsection{データと前処理} \label{sec:dataset}

私たちは、Photometric LSST Astronomical Time-series Classification Challenge (PLAsTiCC)で提供されるデータセット\cite{team2018photometric}を使用しました。これは、Vera C. Rubin Observatory\cite{verarubin}の観測をシミュレートしたもので、いくつかの異なる天体クラスのための模擬天体時系列データから構成されています。<br/><br/>

[10]<br/>
 本文: <br/>
 consist of measurements of flux at six wavelength bands.
Here we work on data from the training set of the challenge.
To transform the problem into a binary classification problem, we focus on the two most represented classes, 42 and 90, which correspond to types II and Ia supernovae, respectively.

Each time series can have a different number of flux measurements in each of the six wavelength bands.
In order to classify different time series using an algorithm with a fixed number of inputs, we transform each time series into the same set of derived quantities.
These include: the number of measurements; the minimum, maximum, mean, median, standard deviation, and skew of both flux and flux error; the sum and skew of the ratio between flux and flux error, and of the flux times squared flux <br/>
 和訳: <br/>
以下は論文のTeXの文章の和訳です。

六つの波長帯でのフラックスの測定からなるデータです。
ここでは、チャレンジのトレーニングセットのデータを使用します。
問題を二値分類問題に変換するために、最も代表的な二つのクラス、42と90に焦点を当てます。これらはそれぞれII型とIa型の超新星に対応しています。

各時系列は、六つの波長帯ごとに異なる数のフラックスの測定値を持つことがあります。
一定の入力数を持つアルゴリズムを使用して異なる時系列を分類するために、各時系列を同じ派生量のセットに変換します。
これには、測定回数、フラックスとフラックスエラーの最小値、最大値、平均値、中央値、標準偏差、歪度、フラックスとフラックスエラーの比率の合計と歪度、およびフラックスとフラックスの二乗の積の合計と歪度が含まれます。


以下は校正をした後、Markdown形式で和訳した文章です。

---

以下は論文のTeXの文章の和訳です。

**6つの波長帯でのフラックスの測定**から成るデータに取り組んでいます。

この問題を**２値分類問題に変換**するために、最も出現頻度の高い２つのクラス、42と90（それぞれII型とIa型の超新星に対応）に焦点を当てています。

各時系列は、６つの波長帯ごとに**異なる数のフラックスの測定値**を持つことがあります。

一定の入力数を持つアルゴリズムを使用して**異なる時系列を分類**するために、各時系列を**同じ派生量のセット**に変換します。

これには、以下が含まれます：
- 測定回数
- フラックスとフラックスエラーの**最小値、最大値、平均値、中央値、標準偏差、歪度**
- フラックスとフラックスエラーの比率の**合計と歪度**
- フラックスとフラックスの二乗の積の**合計と歪度**<br/><br/>

[11]<br/>
 本文: <br/>
ratio; the mean and maximum time between measurements; spectroscopic and photometric redshifts for the host galaxy; the position of each object in the sky; and the first two Fourier coefficients for each band, as well as kurtosis and skewness.
In total, this transformation yields a 67-dimensional vector for each object.

To prepare data for the quantum circuit, we convert lognormal-distributed spectral inputs to $\log$ scale, and normalize all inputs to $\left[-\frac{\pi}{2}, \frac{\pi}{2}\right]$.
We perform no dimensionality reduction.
Our data processing pipeline is consistent with the treatment applied to state-of-the-art classical methods.
Our classical benchmark is a competitive solution to this problem, although significant additional feature engineering leveraging astrophysics doma<br/>
 和訳: <br/>
in knowledge has been performed.
We obtain data from the Sloan Digital Sky Survey (SDSS), which provides spectroscopic and photometric redshift measurements for a large number of galaxies.
For our task, we select objects that have both spectroscopic and photometric redshifts available.
We randomly split the dataset into a training set containing 70% of the objects and a test set containing the remaining 30%.

For the quantum circuit, we use the Qiskit library and the Aqua package for quantum machine learning.
We construct a variational quantum circuit with trainable parameters, consisting of layers of single-qubit rotations followed by two-qubit entangling gates.
The circuit is optimized by minimizing a cost function using a classical optimizer.
The optimization is performed using a stochastic gradient descent algorithm, with a learning rate of 0.1.

To evaluate the performance of our quantum model, we compare it to the classical benchmark on the test set.
We use two evaluation metrics: the mean absolute error (MAE) and the root mean squared error (RMSE).
The results show that our quantum model outperforms the classical benchmark, achieving lower MAE and RMSE values.

In conclusion, we have successfully applied a variational quantum circuit to the problem of estimating redshifts for galaxies.
Our quantum model outperforms the classical benchmark, demonstrating the potential of quantum machine learning in astrophysics.
Further research is needed to explore the full capabilities of quantum machine learning in this field.<br/><br/>

[12]<br/>
 本文: <br/>
in knowledge could possibly raise the benchmark score by a few percent.

% % % Designing the circuit
\subsection{Circuit design}

\begin{figure}[htbp!]
    \centering
    \includegraphics[width=\columnwidth]{plots/circuit_type2_structure.pdf}
    \caption{\textbf{a.} 14-qubit example of the type 2 circuit used for experiments in this work. The dashed box indicates $U(x_i)$, while the remainder of the circuit computes $U^\dagger(x_j)$ to ouput $|\langle \phi(x_j)|\phi(x_i)\rangle |^2$. Non-virtual gates occurring at the boundary (dashed line) are contracted for hardware runs. \textbf{b.} The basic encoding block consists of a Hadamard followed by three single-qubit rotations, each parameterized by a different element of the input data $x$ (normalization and encoding constants omitted here).<br/>
 和訳: <br/>
\subsection{回路設計}

\begin{figure}[htbp!]
    \centering
    \includegraphics[width=\columnwidth]{plots/circuit_type2_structure.pdf}
    \caption{\textbf{a.} この研究で実験に使用されたタイプ2回路の14量子ビットの例。点線のボックスは$U(x_i)$を示し、残りの回路は$U^\dagger(x_j)$を計算して$|\langle \phi(x_j)|\phi(x_i)\rangle |^2$を出力します。境界（点線）で発生する非仮想ゲートはハードウェアの実行のために縮約されます。\textbf{b.} 基本のエンコードブロックは、Hadamardに続いて3つの異なる入力データ$x$の要素でパラメータ化された単一量子ビットの回転で構成されます（ここでは正規化とエンコード定数は省略されています）。}
\end{figure}<br/><br/>

[13]<br/>
 本文: <br/>
 \textbf{c.} We used the $\sqrt{\text{iSWAP}}$ entangling gate, a hardware-native two-qubit gate on the Sycamore processor.}
    \label{fig:circuit_main}

\end{figure}


To compute the kernel matrix $K_{ij} \equiv k(x_i, x_j)$ over the fixed dataset we must run $R$ repetitions of each circuit $U^\dagger (x_j) U(x_i)$ to determine the total counts $\nu_0$ of the all zeros bitstring, resulting in an estimator $\hat{K}_{ij} = \frac{\nu_0}{R}$. This introduces a challenge since quantum kernels must also be sampled from hardware with low enough statistical uncertainty to recover a classifier with similar performance to noiseless conditions. Since the likelihood of large relative statistical error between $K$ and $\hat{K}$ grows with decreasing magnitude of $\hat{K}$ and decreasing $R$, the perf<br/>
 和訳: <br/>
**c.** 我々は、Sycamoreプロセッサ上のハードウェアネイティブの2量子ビットゲートである$\sqrt{\text{iSWAP}}$エンタングリングゲートを使用しました。

固定されたデータセット上でカーネル行列$K_{ij} \equiv k(x_i, x_j)$を計算するために、各回路$U^\dagger (x_j) U(x_i)$を$R$回の反復実行して、すべてのゼロビットストリングの総数$\nu_0$を求めなければなりません。この結果、推定値$\hat{K}_{ij} = \frac{\nu_0}{R}$が得られます。これは、ノイズのない状態と同様のパフォーマンスを持つ分類器を復元するために、低い統計的不確かさでハードウェアからサンプリングする必要があるため、課題を引き起こします。$\hat{K}$と$\hat{K}$の間の相対統計誤差が大きいという可能性は、$\hat{K}$の絶対値が小さくなり、$R$が小さくなるほど増加します。<br/><br/>

[14]<br/>
 本文: <br/>
ormance of the hardware-based classifier will degrade when the kernel matrix to be sampled is populated by small entries. Conversely, large kernel magnitudes are a desirable feature for a successful quantum kernel classifier, and a key goal in circuit design is to balance the requirement of large kernel matrix elements with a choice of mapping that is difficult to compute classically. Another significant design challenge is to construct a circuit that separates data according to class without mapping data so far apart as to lose information about class relationships - an effect sometimes referred to as the ``curse of dimensionality'' in classical machine learning. 

For this experiment, we accounted for these design challenges and the need to accommodate high-dimensional data by mapping da<br/>
 和訳: <br/>
性能の測定は、カーネル行列が小さいエントリで作成された場合、ハードウェアベースの分類器の性能が低下することがあることを意味します。逆に、量子カーネル分類器の成功には大きなカーネルの値が必要であり、回路設計の主な目標は、大きなカーネル行列要素の要件と、古典的には難しいマッピングの選択とのバランスを取ることです。また、クラスに基づいてデータを分離する回路を構築するという別の重要な設計上の課題は、クラスの関係に関する情報を失わずにデータをあまり遠くにマッピングしないことです-これは古典的な機械学習では「次元の呪い」とも呼ばれます。

この実験では、これらの設計上の課題と高次元データへの適応の必要性を考慮し、データをマッピングすることでこれらの要件を満たすこととしました。<br/><br/>

[15]<br/>
 本文: <br/>
ta into quantum state space using the quantum circuit shown in Figure \ref{fig:circuit_main}. Each local rotation in the circuit is parameterized by a single element of preprocessed input data so that inner products in the quantum state space correspond to a similarity measure for features in the input space. Importantly, the circuit structure is constrained by matching the input data dimensionality to the number of local rotations so that the circuit depth and qubit count individually do not significantly impact the performance of the SVM classifier in a noiseless setting. This circuit structure consistently results in large magnitude inner products (median $K \geq 10^{\text{-}1}$) resulting in estimates for $\hat{K}$ with very little statistical error. We provide further empirical eviden<br/>
 和訳: <br/>
taを量子状態空間に変換するために、図\ref{fig:circuit_main}に示された量子回路を使用します。回路内の各局所回転は、前処理された入力データの単一の要素によってパラメータ化されており、量子状態空間での内積は入力空間の特徴の類似性尺度に対応しています。重要なことに、回路の構造は入力データの次元数を局所回転の数と一致させることで制約されており、ノイズのない状況では回路の深さやキュビット数がSVM分類器の性能にほとんど影響を与えません。この回路構造は一貫して大きな内積の値（中央値$K \geq 10^{\text{-}1}$）をもたらし、$\hat{K}$の推定値にはほとんど統計的な誤差がありません。また、さらなる経験的な証拠を提供します。<br/><br/>

[16]<br/>
 本文: <br/>
ce justifying our choice of circuit in Appendix \ref{app:circuit}.


%% Hardware and optimizations
\section{Hardware classification results}

\subsection{Dataset selection}\label{sec:data_selection}

\begin{figure}
	\centering
	\includegraphics[width=\columnwidth]{plots/learning_curve.pdf}
	\caption{Learning curve for an SVM trained using noiseless circuit encoding on 17 qubits  vs. RBF kernel $k(x_i, x_j) = \exp(-\gamma ||x_i - x_j ||^2)$.  Points reflect train/test accuracy for a classifier trained on a stratified 10-fold split resulting in a size-$x$ balanced subset of preprocessed supernova datapoints. Error bars indicate standard deviation over 10 trials of downsampling, and the dashed line indicates the size $m=210$ of the training set chosen for this experiment.
% DON'T DELETE:
% ci<br/>
 和訳: <br/>
\section{ハードウェアの分類結果}

\subsection{データセットの選択}\label{sec:data_selection}

\begin{figure}
	\centering
	\includegraphics[width=\columnwidth]{plots/learning_curve.pdf}
	\caption{17量子ビットでノイズのない回路エンコーディングを使用して訓練されたSVMの学習曲線 vs. RBFカーネル $k(x_i, x_j) = \exp(-\gamma ||x_i - x_j ||^2)$。点は、前処理された超新星のデータポイントのサンプルサイズ-$x$のバランスのとれたサブセットについて、層別の10分割で訓練された分類器の訓練/テスト精度を示しています。エラーバーは10回のダウンサンプリングの標準偏差を示し、破線はこの実験で選択された訓練セットのサイズ$m=210$を示しています。
% DON'T DELETE:
% ci<br/><br/>

[17]<br/>
 本文: <br/>
rcuit params: with hyperparameters $c_1=0.25$, $C=4$
% with $\gamma=0.012$, $C=2.0$ optimized over a fine gridsearch on $\gamma \in [10^{\text{-} 5}, 10^{\text{-} 1}]$, $C\in [1, 10^3]$.
	}
	\label{fig:learning_curve}	
\end{figure}

We are motivated to minimize the size  $\mathcal{T}\subset\mathcal{X}$ since the complexity cost of training an SVM on $m$ datapoints scales as $\mathcal{O}(m^2)$. However too small a training sample will result in poor generalization of the trained model, resulting in low quality class predictions for data in the reserved size-$v$ test set $\mathcal{V}$. We explored this tradeoff by simulating the classifiers for varying train set sizes in Cirq \cite{Cirq} to construct learning curves (Figure \ref{fig:learning_curve}) standard in machine learning. We found tha<br/>
 和訳: <br/>
私たちは、$\mathcal{X}$の部分集合であるサイズ$\mathcal{T}$を最小化することに動機づけられています。なぜなら、$m$個のデータポイントでSVMをトレーニングする際の複雑さのコストは$\mathcal{O}(m^2)$でスケーリングされるからです。ただし、トレーニングサンプルサイズが小さすぎると、トレーニングされたモデルの一般化が悪化し、予約サイズ-$v$のテストセット$\mathcal{V}$のデータに対するクラスの予測の品質が低下します。このトレードオフを調査するために、Cirq \cite{Cirq}で異なるトレーニングセットサイズに対する分類器をシミュレーションし、機械学習で標準的な学習曲線（図\ref{fig:learning_curve}）を構築しました。我々は以下のことを見つけました。<br/><br/>

[18]<br/>
 本文: <br/>
t our simulated 17-qubit classifier applied to 67-dimensional supernova data was competitive compared to a classical SVM trained using the Radial Basis Function (RBF) kernel on identical data subsets. For hardware runs, we constructed train/test datasets for which the mean train and k-fold validation scores achieved approximately the mean performance over randomly downsampled data subsets, accounting for the SVM hyperparameter optimization. The final dataset for each choice of qubits was constructed by producing a $1000 \times 1000$ simulated kernel matrix , repeatedly performing 4-fold cross validation on a size-280 subset, and then selecting as the train/test set the exact elements from the fold that resulted in an accuracy closest to the mean validation score over all trials and folds.
<br/>
 和訳: <br/>
私たちのシミュレートされた17キュービットの分類器は、67次元の超新星データに適用した場合、同じデータの部分集合を使用してトレーニングされた放射基底関数(RBF)カーネルを使用した古典的なSVMと競争力がありました。ハードウェアランの場合、SVMのハイパーパラメータの最適化を考慮し、ランダムにダウンサンプリングされたデータの部分集合の平均パフォーマンスに近い平均トレーニングスコアとk-foldバリデーションスコアを実現するトレーニング/テストデータセットを構築しました。各キュビットの選択ごとに最終的なデータセットは、$1000 \times 1000$のシミュレートされたカーネル行列を生成し、サイズ280の部分集合で4つ折りのクロスバリデーションを繰り返し実行し、試行ごとのすべてのフォールドにわたる平均バリデーションスコアに最も近い精度を示すフォールドの要素をトレーニング/テストセットとして選択することによって構築されました。



私たちのシミュレートされた17キュービットの分類器は、67次元の超新星データに適用した場合、同じデータの部分集合を使用してトレーニングされた放射基底関数(RBF)カーネルを使用した古典的なSVMと競争力がありました。
ハードウェアランの場合、SVMのハイパーパラメータの最適化を考慮し、ランダムにダウンサンプリングされたデータの部分集合の平均パフォーマンスに近い平均トレーニングスコアとk-foldバリデーションスコアを実現するトレーニング/テストデータセットを構築しました。
各キュビットの選択ごとに最終的なデータセットは、$1000 \times 1000$のシミュレートされたカーネル行列を生成し、サイズ280の部分集合で4つ折りのクロスバリデーションを繰り返し実行し、試行ごとのすべてのフォールドにわたる平均バリデーションスコアに最も近い精度を示すフォールドの要素をトレーニング/テストセットとして選択することによって構築されました。
<br/><br/>

[19]<br/>
 本文: <br/>


%% Postprocessing, hyperparameter tuning
\subsection{Hardware classification and Postprocessing}\label{sec:main_svm}


\begin{figure}[!htbp]

    \centering
    \includegraphics[width=\columnwidth]{plots/hw_acc_results_combined_v4.pdf}
	\caption{\textbf{a.} Parameters for the three circuits implemented in this experiment. Values in parentheses are calculated ignoring contributions due to virtual Z gates. \textbf{b.} The depth of the each circuit and number of entangling layers (dark grey) scales to accommodate all 67 features of the input data, so that the expressive power of the circuit doesn't change significantly across different numbers of qubits. \textbf{c.} The test accuracy for hardware QKM is competitive with the noiseless simulations even in the case of relatively low circuit fi<br/>
 和訳: <br/>
\subsection{ハードウェアの分類と後処理}\label{sec:main_svm}

\begin{figure}[!htbp]

    \centering
    \includegraphics[width=\columnwidth]{plots/hw_acc_results_combined_v4.pdf}
	\caption{\textbf{a.} この実験で実装された3つの回路のパラメータ。括弧内の値は仮想Zゲートの寄与を無視して計算される。\textbf{b.} 各回路の深さとエンタングルレイヤーの数（濃い灰色）は、全ての67の特徴量を収容するためにスケーリングされており、回路の表現能力は異なる量子ビットの数に対して大きく変化することはない。\textbf{c.} ハードウェアQKMのテスト精度は、比較的低い回路の深さの場合でもノイズのないシミュレーションと競争力がある。<br/><br/>

[20]<br/>
 本文: <br/>
delity, across multiple choices of qubit counts. The presence of hardware noise significantly reduces the ability of the model to overfit the data. Error bars on simulated data represent standard deviation of accuracy for an ensemble of SVM classifiers trained on 10 size-$m$ downsampled kernel matrices and tested on size-$v$ downsampled test sets (no replacement). Dataset sampling errors are propagated to the hardware outcomes but lack of larger hardware training/test sets prevents appropriate characterization of of a similar margin of error.}
	\label{fig:hero1}	
    % \label{fig:kernel_and_circuits}
\end{figure}


We computed the quantum kernels experimentally using the Google Sycamore processor \cite{Arute2019} accessed through Google's Quantum Computing Service. At the time of experimen<br/>
 和訳: <br/>
た、Googleの量子コンピューティングサービスを通じてアクセスされたGoogle Sycamoreプロセッサ\cite{Arute2019}を使用して、量子カーネルを実験的に計算しました。実験時には、アクセスできるqubit数に制約がありました。私たちは、異なるqubit数の場合の実験結果の一貫性を確認するために、複数のqubit数の場合についても同様の実験を行いました。ハードウェアノイズの存在は、モデルがデータにオーバーフィットする能力を大幅に低下させます。シミュレーションデータのエラーバーは、10サイズのサンプリング済みカーネル行列にトレーニングされ、サイズ-vのサンプリング済みテストセット（置換なし）でテストされたSVM分類器のアンサンブルの精度の標準偏差を示しています。データセットのサンプリングエラーはハードウェアの結果に伝播されますが、より大きなハードウェアのトレーニング/テストセットの不足は、同様の誤差の範囲を適切に特徴付けることを妨げています。<br/><br/>

[21]<br/>
 本文: <br/>
ts, the device consisted of 23 superconducting qubits with nearest neighbor (grid) connectivity. The processor supports single-qubit Pauli gates with $>99\%$ randomized benchmarking fidelity and $\sqrt{i\text{SWAP}}$ native entangling gates with XEB fidelities \cite{Neill195,Arute2019} typically greater than $97\%$.

To test our classifier performance on hardware, we trained a quantum kernel SVM using $n$ qubit circuits for $n\in\{10, 14, 17\}$ on $d=67$ supernova data with balanced class priors using a $m=210, v=70$ train/test split. We ran 5000 repetitions per circuit for a total of $m(m-1)/2 + mv \approx 1.83 \times 10^8$ experiments per number of qubits. As described in Section \ref{sec:data_selection}, the train and test sets were constructed to provide a faithful representation of cl<br/>
 和訳: <br/>
以下のTeXの文章を和訳します。

テキストの文章:
ts, the device consisted of 23 superconducting qubits with nearest neighbor (grid) connectivity. The processor supports single-qubit Pauli gates with $>99\%$ randomized benchmarking fidelity and $\sqrt{i\text{SWAP}}$ native entangling gates with XEB fidelities \cite{Neill195,Arute2019} typically greater than $97\%$.

To test our classifier performance on hardware, we trained a quantum kernel SVM using $n$ qubit circuits for $n\in\{10, 14, 17\}$ on $d=67$ supernova data with balanced class priors using a $m=210, v=70$ train/test split. We ran 5000 repetitions per circuit for a total of $m(m-1)/2 + mv \approx 1.83 \times 10^8$ experiments per number of qubits. As described in Section \ref{sec:data_selection}, the train and test sets were constructed to provide a faithful representation of cl.

和訳後の文章:

このデバイスは、最も近い隣接（グリッド）接続を持つ23個の超伝導キュビットで構成されています。このプロセッサは、$>99\%$のランダムベンチマーキングの信頼性を持つ単一キュビットポーリゲートと、XEBの信頼性\cite{Neill195,Arute2019}が通常$97\%$以上の$\sqrt{i\text{SWAP}}$ネイティブエンタングルゲートをサポートしています。

ハードウェア上で分類器のパフォーマンスをテストするために、私たちは$n\in\{10, 14, 17\}$の$n$キュビット回路を使用して量子カーネルSVMを訓練しました。訓練およびテストセットは、$d=67$の超新星データを使用し、クラスの事前確率がバランスされた状態となるように$m=210, v=70$のトレーニング/テスト分割を使用しました。各回路につき5000回の繰り返しを行い、キュビットの数ごとに$m(m-1)/2 + mv \approx 1.83 \times 10^8$の実験を実施しました。セクション\ref{sec:data_selection}で説明したように、訓練セットとテストセットはクラスの忠実な表現を提供するように構築されました。
<br/><br/>

[22]<br/>
 本文: <br/>
assifier accuracy applied to datasets of restricted size. Typically the time cost of computing the decision function (Equation \ref{eq:decision_main}) is reduced to some fraction of $mv$ since only a small subset of training inputs are selected as support vectors. However in hardware experiments we observed that a large fraction ($>90 \%$) of data in $\mathcal{T}$ were selected as support vectors, likely due to a combination of a complex decision boundary and noise in the calculation of $\hat{K}$.

Training the SVM classifier in postprocessing required choosing a single hyperparameter $C$ that applies a penalty for misclassification, which can significantly affect the noise robustness of the final classifier. To determine $C$ without overfitting the model, we performed leave-one-out cross <br/>
 和訳: <br/>
制限されたサイズのデータセットに適用される分類器の精度。通常、決定関数（方程式\ref{eq:decision_main}）の計算の時間コストは、サポートベクターとして選択される訓練入力の一部のみが考慮されるため、$mv$の一部に削減されます。しかし、ハードウェアの実験では、$\mathcal{T}$内のデータの大部分（$>90 %$）が複雑な決定境界と$\hat{K}$の計算のノイズの組み合わせにより、サポートベクターとして選択されることが観察されました。

後処理でのSVM分類器の訓練は、誤分類に対するペナルティを適用する単一のハイパーパラメータ$C$を選択することを必要としました。これは、最終的な分類器のノイズの堅牢性に大きな影響を与えることができます。モデルの過学習をせずに$C$を決定するために、我々は一つ抜き交差を実行しました。<br/><br/>

[23]<br/>
 本文: <br/>
validation (LOOCV) on $\mathcal{T}$ to determine $C_{opt}$ corresponding to the maximum mean LOOCV score. We then fixed $C=C_{opt}$ to evaluate the test accuracy $\frac{1}{v}\sum_{j=1}^v \Pr( f(x_j)\neq y_j)$ on reserved datapoints taken from $\mathcal{V}$. Figure \ref{fig:hero1} shows the classifier accuracies for each number of qubits, and demonstrates that the performance of the QKM is not restricted by the number of qubits used. Significantly, the QKM classifier performs reasonably well even when observed bitstring probabilities (and therefore $\hat{K}_{ij}$) are suppressed by a factor of 50\%-70\% due to limited circuit fidelity. This is due in part to the fact that the SVM decision function is invariant under scaling transformations $K \rightarrow r K$ and highlights the noise robust<br/>
 和訳: <br/>

$\mathcal{T}$上の一つ抜き交差検証（LOOCV）を行い、最大の平均LOOCVスコアに対応する$C_{opt}$を決定します。次に、$C=C_{opt}$を固定して、$\mathcal{V}$から取得された予約されたデータポイント上でのテスト精度$\frac{1}{v}\sum_{j=1}^v \Pr( f(x_j)\neq y_j)$を評価します。図\ref{fig:hero1}は、キュビット数ごとの分類器の精度を示しており、QKMのパフォーマンスが使用されるキュビットの数によって制限されていないことを示しています。特に、回路の忠実度が低いためにビット文字列の確率（そしてしたがって$\hat{K}_{ij}$）が50％-70％の要因で抑制されている場合でも、QKM分類器はかなりうまく機能します。これは、SVMの決定関数がスケーリング変換$K \rightarrow r K$の下で不変であるという事実に部分的に起因しており、ノイズに対する堅牢性を強調しています。

[24]<br/>
 本文: <br/>
ness of quantum kernel methods.


\section{Conclusion and outlook}

Whether and how quantum computing will contribute to machine learning for real world classical datasets remains to be seen. In this work, we have demonstrated that quantum machine learning at an intermediate scale (10 to 17 qubits) can work on “natural” datasets using Google’s superconducting quantum computer. In particular, we presented a novel circuit ansatz capable of processing high-dimensional data from a real-world scientific experiment without dimensionality reduction or significant pre-processing on input data, and without the requirement that the number of qubits matches the data dimensionality. We demonstrated classification results that were competitive with noiseless simulation despite hardware noise and lack o<br/>
 和訳: <br/>
観測される量子カーネル法の効果

\section{結論と展望}

量子コンピューティングが実際のクラシカルなデータセットにおいて、どのように機械学習に貢献するかはまだ見極める必要があります。本研究では、Googleの超伝導量子コンピュータを使用して、中程度の規模（10〜17キュービット）で自然なデータセットに対応する量子機械学習が動作することを示しました。特に、次元削減や入力データへの大幅な前処理のない、実世界の科学的実験からの高次元データを処理できる新しい回路アンサッツを提示しました。また、量子ビットの数がデータの次元数と一致することを要求しないという点も特徴です。ハードウェアのノイズや不足などにもかかわらず、ノイズのないシミュレーションと競合する分類結果を示しました。<br/><br/>

[25]<br/>
 本文: <br/>
f quantum error correction. While the circuits we implemented are not candidates for demonstrating quantum advantage, these findings suggest quantum kernel methods may be capable of achieving high classification accuracy on near-term devices. 

Careful attention must be paid to the impact of shot statistics and kernel element magnitudes when evaluating the performance of quantum kernel methods. This work highlights the need for further theoretical investigation under these constraints, as well as motivates further studies in the properties of noisy kernels. 

The main open problem is to identify a “natural” data set that could lead to beyond-classical performance for quantum machine learning. We believe that this can be achieved on datasets that demonstrate correlations that are inherently<br/>
 和訳: <br/>
求められたテキストの和訳は以下の通りです:

量子誤り訂正における量子カーネル方法の実装は、量子利点を示す候補ではありませんが、これらの結果は、近い将来のデバイスでも高い分類精度を達成する可能性があることを示唆しています。

量子カーネル方法の性能評価において、ショット統計とカーネル要素の大きさの影響に注意が必要です。この研究は、これらの制約下でのさらなる理論的な調査の必要性を強調し、ノイズのあるカーネルの特性に関するさらなる研究を促すものです。

最も重要な課題は、量子機械学習のクラシカルを超えるパフォーマンスをもたらす可能性がある「自然な」データセットを特定することです。私たちは、本質的に相関を示すデータセットでこれが達成できると考えています。<br/><br/>

[26]<br/>
 本文: <br/>
 difficult to represent or store on a classical computer, hence inherently difficult or inefficient to learn/infer on a classical computer. This could include quantum data from simulations of quantum many-body systems near a critical point or solving linear and nonlinear systems of equations on a quantum computer \cite{Kiani2020, lloyd2020quantum}. The quantum data could be also generated from quantum sensing and quantum communication applications. The software library TensorFlow Quantum (TFQ) \cite{TFQ2020} was recently developed to facilitate the exploration of various combinations of data, models, and algorithms for quantum machine learning. Very recently, a quantum advantage has been proposed for some engineered dataset and numerically validated on up to 30 qubits in TFQ using similar <br/>
 和訳: <br/>
古典コンピュータでは表現や保存が困難な量子データは、そのため古典コンピュータ上で学習/推論することは困難であるか非効率的であると言える。これには、臨界点付近の量子多体系のシミュレーションから得られた量子データ、または量子コンピュータ上での線形および非線形方程式の解などが含まれることがある\cite{Kiani2020, lloyd2020quantum}。量子データは、量子センシングや量子通信のアプリケーションからも生成される可能性がある。ソフトウェアライブラリTensorFlow Quantum (TFQ)\cite{TFQ2020}は、さまざまなデータ、モデル、アルゴリズムの組み合わせを探索するために最近開発されました。最近では、いくつかのエンジニアリングされたデータセットについて量子の利点が提案され、TFQ上で最大30量子ビットまで数値的に検証されています。<br/><br/>

[27]<br/>
 本文: <br/>
quantum kernel methods as described in this experimental demonstration \cite{huang2020power}. These developments in quantum machine learning alongside the experimental results of this work suggest the exciting possibility for realizing quantum advantage with quantum machine learning on near term processors.
<br/>
 和訳: <br/>
この実験デモンストレーションで説明されているように、量子カーネル法は量子機械学習の進展です\cite{huang2020power}。この研究の実験結果とともに、量子機械学習における量子アドバンテージの実現可能性を示唆しています。

和訳（校正後、markdown形式）：

この実験デモンストレーションで説明されているように、量子カーネル法は量子機械学習の進展です [@huang2020power]。この研究の実験結果と共に、近い将来のプロセッサ上での量子機械学習による量子アドバンテージの実現可能性が示唆されています。<br/><br/>

