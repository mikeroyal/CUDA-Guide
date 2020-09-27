<h1 align="center">
 <img src="https://user-images.githubusercontent.com/45159366/94306481-e17b8f00-ff27-11ea-832f-c85374acb3b1.png">
  <br />
  CUDA Guide
</h1>

#### A guide covering CUDA including the applications and tools that will make you a better and more efficient CUDA developer.

## Resources

[CUDA](https://developer.nvidia.com/cuda-zone) is a parallel computing platform and programming model developed by NVIDIA for general computing on graphical processing units (GPUs). With CUDA, developers are able to dramatically speed up computing applications by harnessing the power of GPUs. In GPU-accelerated applications, the sequential part of the workload runs on the CPU, which is optimized for single-threaded. The compute intensive portion of the application runs on thousands of GPU cores in parallel. When using CUDA, developers can program in popular languages such as C, C++, Fortran, Python and MATLAB.

[CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/index.html)

[CUDA Quick Start Guide](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)

[CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

[CUDA GPU support for TensorFlow](https://www.tensorflow.org/install/gpu)

[NVIDIA Deep Learning cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/api/index.html)

[NVIDIA GPU Cloud Documentation](https://docs.nvidia.com/ngc/ngc-introduction/index.html)

[NVIDIA NGC](https://ngc.nvidia.com/) is a hub for GPU-optimized software for deep learning, machine learning, and high-performance computing (HPC) workloads.

[NVIDIA NGC Containers](https://www.nvidia.com/en-us/gpu-cloud/containers/) is a registry that provides researchers, data scientists, and developers with simple access to a comprehensive catalog of GPU-accelerated software for AI, machine learning and HPC. These containers take full advantage of NVIDIA GPUs on-premises and in the cloud.

## Tools

[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) is a collection of tools & libraries that provide a development environment for creating high performance GPU-accelerated applications. The CUDA Toolkit allows you can develop, optimize, and deploy your applications on GPU-accelerated embedded systems, desktop workstations, enterprise data centers, cloud-based platforms and HPC supercomputers. The toolkit includes GPU-accelerated libraries, debugging and optimization tools, a C/C++ compiler, and a runtime library to build and deploy your application on major architectures including x86, Arm and POWER.

[NVIDIA cuDNN](https://developer.nvidia.com/cudnn) is a GPU-accelerated library of primitives for [deep neural networks](https://developer.nvidia.com/deep-learning). cuDNN provides highly tuned implementations for standard routines such as forward and backward convolution, pooling, normalization, and activation layers. cuDNN accelerates widely used deep learning frameworks, including [Caffe2](https://caffe2.ai/), [Chainer](https://chainer.org/), [Keras](https://keras.io/), [MATLAB](https://www.mathworks.com/solutions/deep-learning.html), [MxNet](https://mxnet.incubator.apache.org/), PyTorch](https://pytorch.org/), and [TensorFlow](https://www.tensorflow.org/).

[CUDA-X HPC](https://www.nvidia.com/en-us/technologies/cuda-x/) is a collection of libraries, tools, compilers and APIs that help developers solve the world's most challenging problems. CUDA-X HPC includes highly tuned kernels essential for high-performance computing (HPC). 

[NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) is a collection of tools & libraries that allows users to build and run GPU accelerated Docker containers. The toolkit includes a container runtime [library](https://github.com/NVIDIA/libnvidia-container) and utilities to automatically configure containers to leverage NVIDIA GPUs.

[Minkowski Engine](https://nvidia.github.io/MinkowskiEngine) is an auto-differentiation library for sparse tensors. It supports all standard neural network layers such as convolution, pooling, unpooling, and broadcasting operations for sparse tensors. 

[CUTLASS](https://github.com/NVIDIA/cutlass) is a collection of CUDA C++ template abstractions for implementing high-performance matrix-multiplication (GEMM) at all levels and scales within CUDA. It incorporates strategies for hierarchical decomposition and data movement similar to those used to implement cuBLAS. 

[CUB](https://github.com/NVIDIA/cub) is a cooperative primitives for CUDA C++ kernel authors.

[Tensorman](https://github.com/pop-os/tensorman) is a utility for easy management of Tensorflow containers by developed by [System76]( https://system76.com).Tensorman allows Tensorflow to operate in an isolated environment that is contained from the rest of the system. This virtual environment can operate independent of the base system, allowing you to use any version of Tensorflow on any version of a Linux distribution that supports the Docker runtime.

[Numba](https://github.com/numba/numba) is an open source, NumPy-aware optimizing compiler for Python sponsored by Anaconda, Inc. It uses the LLVM compiler project to generate machine code from Python syntax. Numba can compile a large subset of numerically-focused Python, including many NumPy functions. Additionally, Numba has support for automatic parallelization of loops, generation of GPU-accelerated code, and creation of ufuncs and C callbacks.

[Chainer](https://chainer.org/) is a Python-based deep learning framework aiming at flexibility. It provides automatic differentiation APIs based on the define-by-run approach (dynamic computational graphs) as well as object-oriented high-level APIs to build and train neural networks. It also supports CUDA/cuDNN using [CuPy](https://github.com/cupy/cupy) for high performance training and inference.

[CuPy](https://cupy.dev/) is an implementation of NumPy-compatible multi-dimensional array on CUDA. CuPy consists of the core multi-dimensional array class, cupy.ndarray, and many functions on it. It supports a subset of numpy.ndarray interface.

[CatBoost](https://catboost.ai/) is a fast, scalable, high performance [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting) on Decision Trees library, used for ranking, classification, regression and other machine learning tasks for Python, R, Java, C++. Supports computation on CPU and GPU. 

[cuDF](https://rapids.ai/) is a GPU DataFrame library for loading, joining, aggregating, filtering, and otherwise manipulating data. cuDF provides a pandas-like API that will be familiar to data engineers & data scientists, so they can use it to easily accelerate their workflows without going into the details of CUDA programming.

[cuML](https://github.com/rapidsai/cuml) is a suite of libraries that implement machine learning algorithms and mathematical primitives functions that share compatible APIs with other RAPIDS projects. cuML enables data scientists, researchers, and software engineers to run traditional tabular ML tasks on GPUs without going into the details of CUDA programming. In most cases, cuML's Python API matches the API from scikit-learn.

[ArrayFire](https://arrayfire.com/) is a general-purpose library that simplifies the process of developing software that targets parallel and massively-parallel architectures including CPUs, GPUs, and other hardware acceleration devices.

[Thrust](https://github.com/NVIDIA/thrust) is a C++ parallel programming library which resembles the C++ Standard Library. Thrust's high-level interface greatly enhances programmer productivity while enabling performance portability between GPUs and multicore CPUs.

[AresDB](https://eng.uber.com/aresdb/) is a GPU-powered real-time analytics storage and query engine. It features low query latency, high data freshness and highly efficient in-memory and on disk storage management. 

[Arraymancer](https://mratsim.github.io/Arraymancer/) is a tensor (N-dimensional array) project in Nim. The main focus is providing a fast and ergonomic CPU, Cuda and OpenCL ndarray library on which to build a scientific computing ecosystem.

[Kintinuous](https://github.com/mp3guy/Kintinuous) is a real-time dense visual SLAM system capable of producing high quality globally consistent point and mesh reconstructions over hundreds of metres in real-time with only a low-cost commodity RGB-D sensor.

[GraphVite](https://graphvite.io/) is a general graph embedding engine, dedicated to high-speed and large-scale embedding learning in various applications.
