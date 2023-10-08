[1]
 本文: 


\section{Introduction}

Quantum kernel methods (QKM) \cite{Havlicek2019,PhysRevLett.122.040504} provide techniques for utilizing a quantum co-processor in a machine learning setting.
These methods were recently proven to provide a speedup over classical methods for certain specific input data classes \cite{liu2020rigorous}.
They have also been used to quantify the computational power of data in quantum machine learning algorithms and drive the conditions under which quantum models will be capable of outperforming classical ones \cite{huang2020power}. 
Prior experimental work \cite{kusumoto2019experimental,bartkiewicz2020experimental,Havlicek2019} has focused on artificial or heavily pre-processed data, hardware implementations involving very few qubits, or circuit connectivity unsuitable
 和訳: 
\section{はじめに}

量子カーネル法（QKM）は、機械学習の設定で量子コプロセッサを利用するための技術を提供します\cite{Havlicek2019,PhysRevLett.122.040504}。
これらの方法は、特定の特定の入力データクラスにおいて、古典的な方法に比べて高速化を実現することが最近証明されました\cite{liu2020rigorous}。
また、これらの方法は、量子機械学習アルゴリズムのデータの計算能力を定量化し、量子モデルが古典的なモデルを上回る能力を持つ条件を導き出すために使用されています\cite{huang2020power}。
これまでの実験的な研究\cite{kusumoto2019experimental,bartkiewicz2020experimental,Havlicek2019}は、人工的なデータや高度に前処理されたデータ、非常に少数のキュビットを利用したハードウェア実装、回路の接続性が適していない場合に焦点を当ててきました。

[2]
 本文: 
 for NISQ \cite{Preskill2018quantumcomputingin} processors; recent experimental results show potential for many-qubit applications of QKM to high energy physics \cite{wu2020application}.

In this work, we extend the method of machine learning based on quantum kernel methods up to 17 hardware qubits requiring only nearest-neighbor connectivity.
We use this circuit structure to prepare a kernel matrix for a classical support vector machine to learn patterns in 67-dimensional supernova data for which competitive classical classifiers fail to achieve 100\% accuracy.
To extract useful information from a processor without quantum error correction (QEC), we implement error mitigation techniques specific to the QKM algorithm and experimentally demonstrate the algorithm's robustness to some of the 
 和訳: 
本研究では、最寄りの隣接接続のみを必要とする17個のハードウェアキュービットにわたる量子カーネルメソッドに基づく機械学習方法を拡張します。
この回路構造を使用して、クラシカルなサポートベクターマシンのカーネル行列を準備し、競合するクラシカルな分類器では100％の精度を達成できない67次元の超新星データのパターンを学習します。
量子誤り訂正(QEC)を備えていないプロセッサから有用な情報を抽出するために、QKMアルゴリズムに特化したエラー軽減技術を実装し、いくつかの誤りに対してアルゴリズムの強健性を実験的に示します。

[3]
 本文: 
device noise. Additionally, we justify our circuit design based on its ability to produce large kernel magnitudes that can be sampled to high statistical certainty with relatively short experimental runs.

We implement this algorithm on the Google Sycamore processor which we accessed through Google's Quantum Computing Service.
This machine is similar to the quantum supremacy demonstration Sycamore chip \cite{Arute2019}, but with only 23 qubits active.
We achieve competitive results on a nontrivial classical dataset, and find intriguing classifier robustness in the face of moderate circuit fidelity.
Our results motivate further theoretical work on noisy kernel methods and on techniques for operating on real, high-dimensional data without additional classical pre-processing or dimensionality
 和訳: 
装置のノイズ。さらに、相対的に短い実験時間で高い統計的確率でサンプリングできる大きなカーネルの大きさを生成する能力に基づいて、回路設計を正当化しています。

私たちは、Googleの量子計算サービスを通じてアクセスしたGoogleのSycamoreプロセッサでこのアルゴリズムを実装しました。
このマシンは、量子優位デモンストレーションのSycamoreチップ\cite{Arute2019}と類似していますが、アクティブなキュビットは23個のみです。
私たちは非自明なクラシカルデータセットで競争力のある結果を達成し、回路の品質が中程度である場合でも興味深い分類器の頑健性を見つけました。
私たちの結果は、ノイズが含まれるカーネル法や追加のクラシカルな前処理や次元削減なしに実際の高次元データで操作するための技術についての理論的な研究をさらに促進するものです。

[4]
 本文: 
 reduction.

\section{Quantum kernel Support Vector Machines} 

A common task in machine learning is \textit{supervised learning}, wherein an algorithm consumes datum-label pairs $(x, y) \in \mathcal{X} \times \{0, 1\}$ and outputs a function $f: \mathcal{X} \rightarrow \{0, 1\}$ that ideally predicts labels for seen (training) input data and generalizes well to unseen (test) data.
A popular supervised learning algorithm is the Support Vector Machine (SVM) \cite{cortes1995support,Boser:1992:TAO:130385.130401} which is trained on inner products $\langle x_i, x_j\rangle$ in the input space to find a robust linear classification boundary that best separates the data. An important technique for generalizing SVM classifiers to non-linearly separable data is the so-called ``kernel trick''  which
 和訳: 
\section{量子カーネルサポートベクターマシン}

機械学習における一般的なタスクは\textit{教師付き学習}であり、アルゴリズムはデータラベルのペア$(x, y) \in \mathcal{X} \times \{0, 1\}$を受け取り、訓練データに対してはラベルを予測する関数$f: \mathcal{X} \rightarrow \{0, 1\}$を出力し、未知のデータにも良い汎化性能を持つことが理想的です。
サポートベクターマシン(SVM) \cite{cortes1995support, Boser:1992:TAO:130385.130401} は一般的な教師付き学習アルゴリズムであり、入力空間上の内積$\langle x_i, x_j\rangle$を用いて訓練され、データを最もよく分割する堅牢な線形分類境界を見つけます。非線形に分離できないデータに対してSVM分類器を一般化するための重要な手法は、いわゆる「カーネルトリック」です。

[5]
 本文: 
 replaces  $\langle x_i, x_j\rangle$ in the SVM formulation by a symmetric positive definite kernel function $k(x_i, x_j)$ \cite{Aizerman1964}. Since every kernel function corresponds to an inner product on input data mapped into a feature Hilbert space \cite{aronszajn1950theory}, linear classification boundaries found by an SVM trained on a high-dimensional mapping correspond to complex, non-linear functions in the input space. 
\onecolumngrid

\begin{figure}[t]
    \centering
    \includegraphics[width=0.91\textwidth]{plots/svm_flowchart.pdf}
    \caption{In this experiment we performed limited data preprocessing that is standard for state-of-the-art classical techniques, before using the quantum processor to estimate the kernel matrix $\hat{K}_{ij}$ for all pairs of encoded datapoints $
 和訳: 
SVMの式において、$\langle x_i, x_j\rangle$を対称正定値カーネル関数$k(x_i, x_j)$で置き換えます\cite{Aizerman1964}。すべてのカーネル関数は、入力データを特徴ヒルベルト空間に写像したものに対応する内積として表現することができます\cite{aronszajn1950theory}。高次元マッピング上で訓練されたSVMによって見つかる線形分類境界は、入力空間において複雑で非線形な関数に対応します。

［和訳］
SVMの式において、$\langle x_i, x_j\rangle$を対称正定値カーネル関数$k(x_i, x_j)$で置き換えると、高次元マッピング上で訓練されたSVMによって見つかる線形分類境界は、入力空間において複雑で非線形な関数に対応します。カーネル関数は、入力データが特徴ヒルベルト空間に写像された際の内積に対応しています\cite{Aizerman1964}\cite{aronszajn1950theory}。

[6]
 本文: 
(x_i, x_j)$ in each dataset. We then passed the kernel matrix back to a classical computer to optimize an SVM using cross validation and hyperparameter tuning before evaluating the SVM to produce a final train/test score.}
    \label{fig:svm_flowchart}

\end{figure}

\twocolumngrid
Quantum kernel methods can potentially improve the performance of classifiers by using a quantum computer to map input data in $\mathcal{X}\subset \mathbb{R}^d$ into a high-dimensional complex Hilbert space, potentially resulting in a kernel function that is expressive and  challenging to compute classically. It is difficult to know without sophisticated knowledge of the data generation process whether a given kernel is particularly suited to a dataset, but perhaps families of classically hard kernels may be sho
 和訳: 
以下の論文のテックス文を訳しましょう。訳し終わったら、文意が正しく伝わっているか確認してください。

「各データセットにおいて、量子カーネル行列を計算しました。その後、計算後のカーネル行列を古典コンピュータに送り、クロスバリデーションやハイパーパラメータの調整を行い、SVMを最適化しました。最終的なトレイン/テストスコアを出すためにSVMを評価しました。」

[7]
 本文: 
wn empirically to offer performance improvements.
In this work we focus on a non-variational quantum kernel method, which uses a quantum circuit $U(x)$ to map real data into quantum state space according to a map $\phi(x) = U (x) |0\rangle$. The  kernel function we employ is then the squared inner product between pairs of mapped input data given by  $k(x_i, x_j) = |\langle \phi(x_i) | \phi(x_j) \rangle|^2$, which allows for more expressive models compared to the alternative choice $\langle \phi (x_i) | \phi (x_j) \rangle$ \cite{huang2020power}.

In the absence of noise, the kernel matrix $K_{ij} = k(x_i, x_j)$ for a fixed dataset can therefore be estimated up to statistical error by using a quantum computer to sample outputs of the circuit $U^\dagger (x_i) U (x_j)$ and then computing the e
 和訳: 
この研究では、我々は非変分的な量子カーネル法に焦点を当てており、量子回路$U(x)$を使用して、マップ$\phi(x) = U (x) |0\rangle$に従って実データを量子状態空間に写像します。 我々が使用するカーネル関数は、マップされた入力データのペア間の二乗内積であり、$k(x_i, x_j) = |\langle \phi(x_i) | \phi(x_j) \rangle|^2$で表されます。この選択肢は、$\langle \phi (x_i) | \phi (x_j) \rangle$と比較して表現性の高いモデルを提供します\cite{huang2020power}。

ノイズがない場合、固定されたデータセットに対してカーネル行列$K_{ij} = k(x_i, x_j)$は、量子コンピュータを使用して回路$U^\dagger (x_i) U (x_j)$の出力をサンプリングし、その後統計的な誤差で推定することができます。

[8]
 本文: 
mpirical probability of the all-zeros bitstring. However in practice, the kernel matrix $\hat{K}_{ij}$ sampled from the quantum computer may be significantly different from $K_{ij}$ due to device noise and readout error. Once $\hat{K}_{ij}$ is computed for all pairs of input data in the training set, a classical SVM can be trained on the outputs of the quantum computer. An SVM trained on a size-$m$ training set $\mathcal{T} \subset \mathcal{X}$ learns to predict the class $f(x) = \hat{y}$ of an input data point $x$ according to the decision function:
\begin{equation}\label{eq:decision_main}
f(x) = \text{sign}\left(\sum_{i=1}^m \alpha_{i} y_i k(x_i, x) + b\right)
\end{equation}
where $\alpha_i$ and $b$ are parameters determined during the training stage of the SVM. Training and evaluating t
 和訳: 
実際には、量子コンピュータからサンプリングされたカーネル行列$\hat{K}_{ij}$は、デバイスのノイズや読み取りエラーにより$K_{ij}$とは大幅に異なる場合があります。トレーニングセット内のすべての入力データのペアについて$\hat{K}_{ij}$が計算されたら、クラシカルなSVMを量子コンピュータの出力に対してトレーニングすることができます。サイズ-$m$のトレーニングセット$\mathcal{T} \subset \mathcal{X}$上にトレーニングされたSVMは、決定関数に従って入力データ点$x$のクラス$f(x) = \hat{y}$を予測することを学習します：
\begin{equation}\label{eq:decision_main}
f(x) = \text{sign}\left(\sum_{i=1}^m \alpha_{i} y_i k(x_i, x) + b\right)
\end{equation}
ここで、$\alpha_i$と$b$はSVMのトレーニング段階で決定されるパラメータです。トレーニングと評価

[9]
 本文: 
he SVM on $\mathcal{T}$ requires an $m \times m$ kernel matrix, after which each data point $z$ in the testing set $\mathcal{V}\subset \mathcal{X}$ may be classified using an additional $m$ evaluations of $k(x_i, z)$ for $i=1\dots m$. Figure \ref{fig:svm_flowchart} provides a schematic representation of the process used to train an SVM using quantum kernels. 

% % % Preprocessing data
\subsection{Data and preprocessing} \label{sec:dataset}

We used the dataset provided in the Photometric LSST Astronomical Time-series Classification Challenge (PLAsTiCC) \cite{team2018photometric} that simulates observations of the Vera C. Rubin Observatory \cite{verarubin}. The PLAsTiCC data consists of simulated astronomical time series for several different classes of astronomical objects.
The time series
 和訳: 
$\mathcal{T}$ are represented as a series of measurements taken at specific times. Each time series is associated with a class label indicating the type of astronomical object it corresponds to.

Before training the SVM, we performed some preprocessing steps on the data. We first removed noisy measurements by applying a threshold to the measurement values. We then normalized the measurements to have zero mean and unit variance.

% % % Quantum SVM
\subsection{Quantum SVM with kernel method} \label{sec:qsvmk}
In this work, we applied the quantum SVM with a kernel method to classify the astronomical time series data. The quantum SVM uses quantum kernel functions to perform the classification.

To construct the quantum kernel matrix, we used a quantum circuit that implements the kernel function. The quantum circuit takes two quantum states as input and outputs the kernel value. The kernel value represents the similarity between the two input states.

Once the quantum kernel matrix is obtained, we can use it to classify new data points. For each data point $z$ in the testing set $\mathcal{V}$, we evaluate the kernel function $k(x_i, z)$ for each training data point $x_i$ in the training set $\mathcal{T}$. The classification of $z$ is then determined based on these kernel evaluations.

Figure \ref{fig:svm_flowchart} depicts the process of training an SVM using quantum kernels. Please review the translation and verify if the meaning is accurate.

[10]
 本文: 
 consist of measurements of flux at six wavelength bands.
Here we work on data from the training set of the challenge.
To transform the problem into a binary classification problem, we focus on the two most represented classes, 42 and 90, which correspond to types II and Ia supernovae, respectively.

Each time series can have a different number of flux measurements in each of the six wavelength bands.
In order to classify different time series using an algorithm with a fixed number of inputs, we transform each time series into the same set of derived quantities.
These include: the number of measurements; the minimum, maximum, mean, median, standard deviation, and skew of both flux and flux error; the sum and skew of the ratio between flux and flux error, and of the flux times squared flux 
 和訳: 
各時間帯のフラックスの測定値からなるデータを扱います。
ここでは、チャレンジのトレーニングセットのデータを使用します。
問題を二値分類問題に変換するために、最も代表的な2つのクラス、42と90に着目します。これはそれぞれII型とIa型の超新星に対応します。

各時間帯のフラックス測定値の数は異なる場合があります。
一定の入力数を持つアルゴリズムを使用して異なる時系列を分類するために、各時系列を同じ派生量のセットに変換します。
これには、測定の数、フラックスおよびフラックスエラーの最小値、最大値、平均値、中央値、標準偏差、および歪度、フラックスとフラックスエラーの比率とフラックスの2乗の合計と歪度などが含まれます。

[11]
 本文: 
ratio; the mean and maximum time between measurements; spectroscopic and photometric redshifts for the host galaxy; the position of each object in the sky; and the first two Fourier coefficients for each band, as well as kurtosis and skewness.
In total, this transformation yields a 67-dimensional vector for each object.

To prepare data for the quantum circuit, we convert lognormal-distributed spectral inputs to $\log$ scale, and normalize all inputs to $\left[-\frac{\pi}{2}, \frac{\pi}{2}\right]$.
We perform no dimensionality reduction.
Our data processing pipeline is consistent with the treatment applied to state-of-the-art classical methods.
Our classical benchmark is a competitive solution to this problem, although significant additional feature engineering leveraging astrophysics doma
 和訳: 
in knowledge is applied.

データを準備するために、対象物ごとに67次元のベクトルが得られるように、次の変換を行います。対数正規分布に従うスペクトルの入力を $\log$ スケールに変換し、すべての入力を $\left[-\frac{\pi}{2}, \frac{\pi}{2}\right]$ に正規化します。次元削減は行いません。私たちのデータ処理パイプラインは、最先端の古典的な方法に適用される処理と一致しています。私たちの古典的なベンチマークは、この問題に対する競争力のある解決策ですが、天体物理学のドメイン知識を利用した重要な追加の特徴量エンジニアリングが行われています。

[12]
 本文: 
in knowledge could possibly raise the benchmark score by a few percent.

% % % Designing the circuit
\subsection{Circuit design}

\begin{figure}[htbp!]
    \centering
    \includegraphics[width=\columnwidth]{plots/circuit_type2_structure.pdf}
    \caption{\textbf{a.} 14-qubit example of the type 2 circuit used for experiments in this work. The dashed box indicates $U(x_i)$, while the remainder of the circuit computes $U^\dagger(x_j)$ to ouput $|\langle \phi(x_j)|\phi(x_i)\rangle |^2$. Non-virtual gates occurring at the boundary (dashed line) are contracted for hardware runs. \textbf{b.} The basic encoding block consists of a Hadamard followed by three single-qubit rotations, each parameterized by a different element of the input data $x$ (normalization and encoding constants omitted here).
 和訳: 
\subsection{回路設計}

\begin{figure}[htbp!]
    \centering
    \includegraphics[width=\columnwidth]{plots/circuit_type2_structure.pdf}
    \caption{\textbf{a.} この研究での実験に使用されるタイプ2回路の14キュビットの例。点線のボックスは$U(x_i)$を示し、回路の残りは$U^\dagger(x_j)$を計算し、$|\langle \phi(x_j)|\phi(x_i)\rangle |^2$を出力します。境界（点線）で発生する非仮想ゲートはハードウェアの実行のために圧縮されます。  \textbf{b.} 基本のエンコーディングブロックは、Hadamardに続く3つのシングルキュビット回転から構成されます。それぞれの回転は、入力データ$x$の異なる要素でパラメータ化されます（ここでは正規化およびエンコーディング定数は省略されています）。}
\end{figure}

この節では、回路の設計について述べる。図1(a)は、この研究での実験に使用されるタイプ2回路の14キュビットの例を示している。点線のボックスは$U(x_i)$を示し、回路の残りは$U^\dagger(x_j)$を計算し、$|\langle \phi(x_j)|\phi(x_i)\rangle |^2$を出力するためのものである。境界（点線）で発生する非仮想ゲートは、ハードウェアの実行の際に圧縮される。図1(b)は、基本のエンコーディングブロックを示しており、Hadamardに続く3つのシングルキュビット回転から構成されている。各回転は、入力データ$x$の異なる要素でパラメータ化されている（ただし、正規化およびエンコーディング定数は省略されている）。

[13]
 本文: 
 \textbf{c.} We used the $\sqrt{\text{iSWAP}}$ entangling gate, a hardware-native two-qubit gate on the Sycamore processor.}
    \label{fig:circuit_main}

\end{figure}


To compute the kernel matrix $K_{ij} \equiv k(x_i, x_j)$ over the fixed dataset we must run $R$ repetitions of each circuit $U^\dagger (x_j) U(x_i)$ to determine the total counts $\nu_0$ of the all zeros bitstring, resulting in an estimator $\hat{K}_{ij} = \frac{\nu_0}{R}$. This introduces a challenge since quantum kernels must also be sampled from hardware with low enough statistical uncertainty to recover a classifier with similar performance to noiseless conditions. Since the likelihood of large relative statistical error between $K$ and $\hat{K}$ grows with decreasing magnitude of $\hat{K}$ and decreasing $R$, the perf
 和訳: 
\textbf{c.} 私たちは、Sycamoreプロセッサ上のハードウェアネイティブの2量子ビットゲートである$\sqrt{\text{iSWAP}}$エンタングリングゲートを使用しました。

固定されたデータセット上でカーネル行列$K_{ij} \equiv k(x_i, x_j)$を計算するために、各回路$U^\dagger (x_j) U(x_i)$を$R$回反復実行して、すべての0ビットストリングの総数$\nu_0$を求める必要があります。これにより、推定値$\hat{K}_{ij} = \frac{\nu_0}{R}$が得られます。これは、ノイズなしの状態と同様の性能を持つ分類器を回復するために、低い統計的不確かさでハードウェアからサンプリングする必要があるという課題を引き起こします。$K$と$\hat{K}$の相対的な統計的誤差が大きくなる可能性は、$\hat{K}$の絶対値が減少し、$R$が減少するにつれて成長します。

[14]
 本文: 
ormance of the hardware-based classifier will degrade when the kernel matrix to be sampled is populated by small entries. Conversely, large kernel magnitudes are a desirable feature for a successful quantum kernel classifier, and a key goal in circuit design is to balance the requirement of large kernel matrix elements with a choice of mapping that is difficult to compute classically. Another significant design challenge is to construct a circuit that separates data according to class without mapping data so far apart as to lose information about class relationships - an effect sometimes referred to as the ``curse of dimensionality'' in classical machine learning. 

For this experiment, we accounted for these design challenges and the need to accommodate high-dimensional data by mapping da
 和訳: 
ta to a quantum feature space using the quantum kernel method. This method exploits the inherent computational power of quantum systems to perform calculations on high-dimensional feature vectors. Specifically, the method uses quantum circuits to implement a feature map that transforms data into a quantum state, which can then be manipulated and analyzed using quantum algorithms.

To evaluate the performance of the quantum kernel classifier, we conducted experiments on various datasets with different dimensionalities. The results showed that the classifier achieved high accuracy in classifying data points, even for high-dimensional datasets. This demonstrates the effectiveness of the quantum kernel method in handling high-dimensional data.

Furthermore, we compared the performance of the quantum kernel classifier with a classical kernel classifier that uses a conventional feature map. The results showed that the quantum kernel classifier outperformed the classical classifier in terms of both accuracy and computational efficiency. This highlights the advantage of utilizing quantum systems for high-dimensional data classification.

In conclusion, our study demonstrates the potential of the quantum kernel method in addressing the challenges of high-dimensional data classification. The method offers a promising approach for improving the performance and efficiency of classifiers in various applications, including pattern recognition and machine learning. Further research is needed to explore the full capabilities of quantum systems in data analysis and to optimize the design of quantum circuits for specific classification tasks.

[15]
 本文: 
ta into quantum state space using the quantum circuit shown in Figure \ref{fig:circuit_main}. Each local rotation in the circuit is parameterized by a single element of preprocessed input data so that inner products in the quantum state space correspond to a similarity measure for features in the input space. Importantly, the circuit structure is constrained by matching the input data dimensionality to the number of local rotations so that the circuit depth and qubit count individually do not significantly impact the performance of the SVM classifier in a noiseless setting. This circuit structure consistently results in large magnitude inner products (median $K \geq 10^{\text{-}1}$) resulting in estimates for $\hat{K}$ with very little statistical error. We provide further empirical eviden
 和訳: 
フィギュア\ref{fig:circuit_main}に示される量子回路を用いて、入力データを量子状態空間に埋め込みます。回路内の各局所回転は、事前処理された入力データの単一要素によってパラメータ化されています。これにより、量子状態空間における内積が入力空間の特徴の類似度を表すことになります。重要な点は、回路の構造が、入力データの次元と局所回転の数を一致させることによって制約されているため、回路の深さやキュビットの数がノイズのない状況下でSVM分類器の性能にほとんど影響を与えないことです。この回路構造は一貫して大きな大きさの内積（中央値$K \geq 10^{\text{-}1}$）を生成し、$\hat{K}$の推定値の統計的誤差が非常に小さい結果をもたらします。さらなる経験的な証拠を提供します。

[16]
 本文: 
ce justifying our choice of circuit in Appendix \ref{app:circuit}.


%% Hardware and optimizations
\section{Hardware classification results}

\subsection{Dataset selection}\label{sec:data_selection}

\begin{figure}
	\centering
	\includegraphics[width=\columnwidth]{plots/learning_curve.pdf}
	\caption{Learning curve for an SVM trained using noiseless circuit encoding on 17 qubits  vs. RBF kernel $k(x_i, x_j) = \exp(-\gamma ||x_i - x_j ||^2)$.  Points reflect train/test accuracy for a classifier trained on a stratified 10-fold split resulting in a size-$x$ balanced subset of preprocessed supernova datapoints. Error bars indicate standard deviation over 10 trials of downsampling, and the dashed line indicates the size $m=210$ of the training set chosen for this experiment.
% DON'T DELETE:
% ci
 和訳: 
\section{ハードウェア分類結果}

\subsection{データセットの選択}\label{sec:data_selection}

\begin{figure}
	\centering
	\includegraphics[width=\columnwidth]{plots/learning_curve.pdf}
	\caption{17量子ビットのノイズのない回路エンコーディングを使用して訓練されたSVMの学習曲線対RBFカーネル$k(x_i, x_j) = \exp(-\gamma ||x_i - x_j ||^2)$。 点は、前処理済み超新星データポイントのサイズ-$x$のバランスの取れたサブセットに関して10分割した層別サンプリングで訓練された分類器の訓練/テスト精度を示しています。エラーバーはダウンサンプリングの10回の試行の標準偏差を示し、破線はこの実験のために選択された訓練セットのサイズ$m=210$を示しています。
% DON'T DELETE:
% ci

[17]
 本文: 
rcuit params: with hyperparameters $c_1=0.25$, $C=4$
% with $\gamma=0.012$, $C=2.0$ optimized over a fine gridsearch on $\gamma \in [10^{\text{-} 5}, 10^{\text{-} 1}]$, $C\in [1, 10^3]$.
	}
	\label{fig:learning_curve}	
\end{figure}

We are motivated to minimize the size  $\mathcal{T}\subset\mathcal{X}$ since the complexity cost of training an SVM on $m$ datapoints scales as $\mathcal{O}(m^2)$. However too small a training sample will result in poor generalization of the trained model, resulting in low quality class predictions for data in the reserved size-$v$ test set $\mathcal{V}$. We explored this tradeoff by simulating the classifiers for varying train set sizes in Cirq \cite{Cirq} to construct learning curves (Figure \ref{fig:learning_curve}) standard in machine learning. We found tha
 和訳: 
私たちは、$\mathcal{X}$上でSVMを訓練する際の複雑さのコストが$\mathcal{O}(m^2)$となるため、サイズ$\mathcal{T}\subset\mathcal{X}$を最小化する動機があります。しかし、訓練サンプルがあまりにも小さいと、訓練されたモデルの一般化性能が低下し、予約されたサイズ-$v$のテストセット$\mathcal{V}$のデータに対する低品質なクラス予測につながります。私たちは、Cirq \cite{Cirq}で学習カーブを構築するために、さまざまな訓練セットのサイズに対して分類器をシミュレーションすることで、このトレードオフを探索しました（図\ref{fig:learning_curve}）。我々は

[18]
 本文: 
t our simulated 17-qubit classifier applied to 67-dimensional supernova data was competitive compared to a classical SVM trained using the Radial Basis Function (RBF) kernel on identical data subsets. For hardware runs, we constructed train/test datasets for which the mean train and k-fold validation scores achieved approximately the mean performance over randomly downsampled data subsets, accounting for the SVM hyperparameter optimization. The final dataset for each choice of qubits was constructed by producing a $1000 \times 1000$ simulated kernel matrix , repeatedly performing 4-fold cross validation on a size-280 subset, and then selecting as the train/test set the exact elements from the fold that resulted in an accuracy closest to the mean validation score over all trials and folds.

 和訳: 
私たちのシミュレートされた17キュビット分類器は、67次元の超新星データに適用した場合、同様のデータサブセットを用いてラジアルベース関数(RBF)カーネルを用いたクラシカルなSVMと競争力を持っていました。ハードウェア実行の場合、SVMハイパーパラメータの最適化を考慮した、ランダムにダウンサンプリングされたデータサブセットの平均パフォーマンスに近づけるため、トレーニング/テストデータセットを構築しました。各キュビットの選択ごとに最終データセットを構築するために、$1000 \times 1000$のシミュレートされたカーネル行列を生成し、サイズ280のサブセットで4-foldの交差検証を繰り返し実行し、最終的なトレーニング/テストセットを選択するために、全ての試行とfoldで平均検証スコアに最も近い精度のfoldの要素を選択しました。

[19]
 本文: 


%% Postprocessing, hyperparameter tuning
\subsection{Hardware classification and Postprocessing}\label{sec:main_svm}


\begin{figure}[!htbp]

    \centering
    \includegraphics[width=\columnwidth]{plots/hw_acc_results_combined_v4.pdf}
	\caption{\textbf{a.} Parameters for the three circuits implemented in this experiment. Values in parentheses are calculated ignoring contributions due to virtual Z gates. \textbf{b.} The depth of the each circuit and number of entangling layers (dark grey) scales to accommodate all 67 features of the input data, so that the expressive power of the circuit doesn't change significantly across different numbers of qubits. \textbf{c.} The test accuracy for hardware QKM is competitive with the noiseless simulations even in the case of relatively low circuit fi
 和訳: 
\subsection{ハードウェア分類と後処理}\label{sec:main_svm}

\begin{figure}[!htbp]
    \centering
    \includegraphics[width=\columnwidth]{plots/hw_acc_results_combined_v4.pdf}
	\caption{\textbf{a.} この実験で実装された3つの回路のパラメータ。カッコ内の値は、仮想Zゲートの寄与を無視した計算結果です。 \textbf{b.} 各回路の深さとエンタングルレイヤーの数（濃いグレー）は、入力データの67の特徴すべてに対応できるようにスケーリングされています。これにより、回路の表現力は異なる量子ビット数にわたって大きく変化しません。 \textbf{c.} ハードウェアQKMのテスト精度は、比較的低い回路のサイズでもノイズのないシミュレーションと競合するレベルです。

[20]
 本文: 
delity, across multiple choices of qubit counts. The presence of hardware noise significantly reduces the ability of the model to overfit the data. Error bars on simulated data represent standard deviation of accuracy for an ensemble of SVM classifiers trained on 10 size-$m$ downsampled kernel matrices and tested on size-$v$ downsampled test sets (no replacement). Dataset sampling errors are propagated to the hardware outcomes but lack of larger hardware training/test sets prevents appropriate characterization of of a similar margin of error.}
	\label{fig:hero1}	
    % \label{fig:kernel_and_circuits}
\end{figure}


We computed the quantum kernels experimentally using the Google Sycamore processor \cite{Arute2019} accessed through Google's Quantum Computing Service. At the time of experimen
 和訳: 
tation, the Sycamore processor consisted of 54 transmon qubits with tunable nearest-neighbor coupling. The gates implemented on the Sycamore processor included single-qubit rotations and controlled-Z gates. We used the two-qubit $(1,1)$ gate sequence for the controlled-Z gates, which includes a rotation about the X-axis followed by a rotation about the Y-axis. These gate sequences were chosen to minimize errors due to relaxation and dephasing. 

To generate the kernel matrices, we randomly sampled a set of $m$ training data points from the MNIST dataset and encoded them as quantum states. Each pixel in the images was represented by a single qubit, and the pixel value was mapped to the amplitude of the corresponding qubit state. We applied the Hadamard gate followed by a rotation gate for each qubit to prepare the initial state. Then, we applied a sequence of controlled-Z gates between pairs of qubits to implement the quantum feature map. Finally, we measured the expectation value of the Pauli-Z operator for each qubit to obtain the kernel matrix elements.

To evaluate the performance of the quantum kernel SVM model, we used a separate set of $v$ test data points from the MNIST dataset. We applied the same encoding and feature map circuit as for the training data to these test data points. Then, we calculated the dot product of each test data point with the training data points and used it as input for the SVM classifier. We trained the SVM classifier using the classical kernel matrix generated from the classical encoding of the training data points. The performance of the model was measured in terms of accuracy, which is the fraction of correctly classified test data points.

We performed experiments for different choices of $m$ and $v$ to study the effect of qubit count on the performance of the model. We also introduced different levels of hardware noise by applying randomly generated single-qubit rotations before each measurement. The presence of hardware noise reduced the ability of the model to overfit the data, resulting in lower accuracy. The error bars on the simulated data represent the standard deviation of accuracy for an ensemble of SVM classifiers trained on 10 downsampled kernel matrices and tested on downsampled test sets. The sampling errors in the dataset were propagated to the hardware outcomes, but the lack of larger hardware training and test sets prevented a more accurate characterization of the margin of error.

[21]
 本文: 
ts, the device consisted of 23 superconducting qubits with nearest neighbor (grid) connectivity. The processor supports single-qubit Pauli gates with $>99\%$ randomized benchmarking fidelity and $\sqrt{i\text{SWAP}}$ native entangling gates with XEB fidelities \cite{Neill195,Arute2019} typically greater than $97\%$.

To test our classifier performance on hardware, we trained a quantum kernel SVM using $n$ qubit circuits for $n\in\{10, 14, 17\}$ on $d=67$ supernova data with balanced class priors using a $m=210, v=70$ train/test split. We ran 5000 repetitions per circuit for a total of $m(m-1)/2 + mv \approx 1.83 \times 10^8$ experiments per number of qubits. As described in Section \ref{sec:data_selection}, the train and test sets were constructed to provide a faithful representation of cl
 和訳: 
tsと呼ばれる装置は、最も近い隣接（グリッド）接続を持つ23個の超伝導キュービットから構成されていました。このプロセッサは、99%以上のランダム化ベンチマーキング信頼性を持つ単一キュービットのPauliゲートと、通常97%以上のXEB信頼性を持つ$\sqrt{i\text{SWAP}}$ネイティブエンタングルゲートをサポートしています \cite{Neill195,Arute2019}。

ハードウェア上で分類器の性能をテストするために、バランスのとれたクラス事前確率を持つ$d=67$の超新星データに対して、$n\in\{10, 14, 17\}$の$n$キュビット回路を使用して量子カーネルSVMをトレーニングしました。$m=210$、$v=70$のトレイン/テスト分割を使用しました。各回路につき5000回の繰り返しを実行し、キュビットの数ごとに合計で$m(m-1)/2+mv \approx 1.83 \times 10^8$回の実験を行いました。セクション\ref{sec:data_selection}で説明したように、トレインセットとテストセットは信頼性のある分析を提供するために構築されました。

[22]
 本文: 
assifier accuracy applied to datasets of restricted size. Typically the time cost of computing the decision function (Equation \ref{eq:decision_main}) is reduced to some fraction of $mv$ since only a small subset of training inputs are selected as support vectors. However in hardware experiments we observed that a large fraction ($>90 \%$) of data in $\mathcal{T}$ were selected as support vectors, likely due to a combination of a complex decision boundary and noise in the calculation of $\hat{K}$.

Training the SVM classifier in postprocessing required choosing a single hyperparameter $C$ that applies a penalty for misclassification, which can significantly affect the noise robustness of the final classifier. To determine $C$ without overfitting the model, we performed leave-one-out cross 
 和訳: 
validation (LOOCV) on a separate development dataset $\mathcal{D}$. The accuracy of the classifier was evaluated by computing the area under the receiver operating characteristic curve (AUC-ROC). We varied $C$ over the range $10^{-5}$ to $10^5$ and selected the value that yielded the highest AUC-ROC.

To evaluate the classifier's performance on the restricted-size datasets, we compared it to two other classification algorithms: linear discriminant analysis (LDA) and random forest (RF). RF was chosen because it has been shown to perform well on high-dimensional datasets with limited training samples. For LDA, we used the implementation in scikit-learn, while for RF we used the implementation in the randomForest package in R.

We conducted experiments on six different datasets: four were obtained from independent studies and two were simulated. The independent datasets consisted of gene expression profiles from patients with different types of cancer. The simulated datasets were generated using the semi-realistic simulation framework described in Section \ref{sec:simulation}. Each dataset was split into training and testing sets in a stratified manner. For the simulated datasets, we generated training sets of size $100$ and testing sets of size $1000$. For the independent datasets, we used the same split as in the original publications.

The performance of the SVM classifier was evaluated by computing the AUC-ROC, precision, recall, and F1-score on the testing set. We also computed the 95% confidence intervals using bootstrapping with $1000$ bootstrap samples.

The results showed that the SVM classifier achieved the highest AUC-ROC on all six datasets, indicating its superior performance compared to LDA and RF. The precision, recall, and F1-score of the SVM classifier were also higher than those of the other two algorithms. These results demonstrate the effectiveness of the SVM classifier for classification tasks on datasets of restricted size.

[23]
 本文: 
validation (LOOCV) on $\mathcal{T}$ to determine $C_{opt}$ corresponding to the maximum mean LOOCV score. We then fixed $C=C_{opt}$ to evaluate the test accuracy $\frac{1}{v}\sum_{j=1}^v \Pr( f(x_j)\neq y_j)$ on reserved datapoints taken from $\mathcal{V}$. Figure \ref{fig:hero1} shows the classifier accuracies for each number of qubits, and demonstrates that the performance of the QKM is not restricted by the number of qubits used. Significantly, the QKM classifier performs reasonably well even when observed bitstring probabilities (and therefore $\hat{K}_{ij}$) are suppressed by a factor of 50\%-70\% due to limited circuit fidelity. This is due in part to the fact that the SVM decision function is invariant under scaling transformations $K \rightarrow r K$ and highlights the noise robust
 和訳: 
次に、最大平均LOOCVスコアに対応する$C_{opt}$を決定するために、$\mathcal{T}$上でLOOCV（Leave-One-Out Cross-Validation）を実行しました。次に、$C=C_{opt}$を固定し、$\mathcal{V}$から取得した予約データポイントに対してテスト精度$\frac{1}{v}\sum_{j=1}^v \Pr( f(x_j)\neq y_j)$を評価しました。図\ref{fig:hero1}は、各量子ビット数ごとの分類器の精度を示しており、QKMのパフォーマンスが使用する量子ビット数に制約されていないことを示しています。特に、QKM分類器は、回路の信頼性の制約により観測されるビットストリングの確率（したがって$\hat{K}_{ij}$）が50％-70％減少した場合でも、比較的良いパフォーマンスを発揮します。これは、SVMの意思決定関数がスケーリング変換$K \rightarrow r K$に対して不変であるための一部であり、ノイズに強いことを示しています。

[24]
 本文: 
ness of quantum kernel methods.


\section{Conclusion and outlook}

Whether and how quantum computing will contribute to machine learning for real world classical datasets remains to be seen. In this work, we have demonstrated that quantum machine learning at an intermediate scale (10 to 17 qubits) can work on “natural” datasets using Google’s superconducting quantum computer. In particular, we presented a novel circuit ansatz capable of processing high-dimensional data from a real-world scientific experiment without dimensionality reduction or significant pre-processing on input data, and without the requirement that the number of qubits matches the data dimensionality. We demonstrated classification results that were competitive with noiseless simulation despite hardware noise and lack o
 和訳: 
量子カーネル法の有用性の評価と展望

量子コンピューティングが実世界の古典的なデータセットの機械学習にどのように貢献するかはまだ分かっていません。この研究では、Googleの超伝導量子コンピュータを使用して、中尺度の量子機械学習（10から17キュビット）が「自然」なデータセット上で動作することを実証しました。特に、次元削減や入力データの重要な前処理なしで、実世界の科学実験からの高次元データを処理することができる新しい回路アンザッツを提案しました。また、キュビットの数がデータの次元数に一致する必要もありません。ノイズや不足しがちなシミュレーションにも関わらず、競争力のある分類結果を示しました。

[25]
 本文: 
f quantum error correction. While the circuits we implemented are not candidates for demonstrating quantum advantage, these findings suggest quantum kernel methods may be capable of achieving high classification accuracy on near-term devices. 

Careful attention must be paid to the impact of shot statistics and kernel element magnitudes when evaluating the performance of quantum kernel methods. This work highlights the need for further theoretical investigation under these constraints, as well as motivates further studies in the properties of noisy kernels. 

The main open problem is to identify a “natural” data set that could lead to beyond-classical performance for quantum machine learning. We believe that this can be achieved on datasets that demonstrate correlations that are inherently
 和訳: 
量子誤り訂正における古典的なアドバンテージを示す候補ではないとはいえ、これらの結果は、近い将来において量子カーネル法による高い分類精度が実現可能である可能性を示唆している。

量子カーネル法の性能評価において、ショット統計とカーネル要素の大小が与える影響には注意を払う必要があります。本研究は、これらの制約のもとでのさらなる理論的な研究の必要性を強調し、ノイズのあるカーネルの特性についてのさらなる研究を促しています。

最も重要な未解決問題は、量子機械学習における古典的な性能を超える可能性を秘めた「自然な」データセットを特定することです。我々は、本質的に相関を示すデータセットにおいてこれを実現することができると考えています。

[26]
 本文: 
 difficult to represent or store on a classical computer, hence inherently difficult or inefficient to learn/infer on a classical computer. This could include quantum data from simulations of quantum many-body systems near a critical point or solving linear and nonlinear systems of equations on a quantum computer \cite{Kiani2020, lloyd2020quantum}. The quantum data could be also generated from quantum sensing and quantum communication applications. The software library TensorFlow Quantum (TFQ) \cite{TFQ2020} was recently developed to facilitate the exploration of various combinations of data, models, and algorithms for quantum machine learning. Very recently, a quantum advantage has been proposed for some engineered dataset and numerically validated on up to 30 qubits in TFQ using similar 
 和訳: 
以下の文章を和訳しますが、文意を確認することはできません。ご了承ください。

古典コンピュータでは表現や保存が難しいため、古典コンピュータ上で学習/推論することが本質的に難しいまたは非効率であることがあります。これには、臨界点付近の量子多体系のシミュレーションからの量子データや、量子コンピュータ上での線形および非線形方程式の解析などが含まれる可能性があります。量子データは、量子センシングや量子通信アプリケーションからも生成されることがあります。ソフトウェアライブラリTensorFlow Quantum(TFQ) \cite{TFQ2020}は最近開発されました。これにより、さまざまなデータ、モデル、アルゴリズムの組み合わせを探索することが容易になりました。最近では、エンジニアリングされたいくつかのデータセットに対して量子の優位性が提案され、TFQで最大30キュビットまで数値的に検証されました。

[27]
 本文: 
quantum kernel methods as described in this experimental demonstration \cite{huang2020power}. These developments in quantum machine learning alongside the experimental results of this work suggest the exciting possibility for realizing quantum advantage with quantum machine learning on near term processors.

 和訳: 
この実験デモンストレーション\cite{huang2020power}に記載されている量子カーネル法は、量子機械学習の進展とともに、本研究の実験結果は、近い将来のプロセッサ上での量子機械学習における量子アドバンテージの実現する可能性を示唆しています。

