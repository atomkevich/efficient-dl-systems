# Efficient Deep Learning Systems
This repository contains materials for the Efficient Deep Learning Systems course taught at the [Faculty of Computer Science](https://cs.hse.ru/en/) of [HSE University](https://www.hse.ru/en/) and [Yandex School of Data Analysis](https://academy.yandex.com/dataschool/).

__This is an ongoing 2023 course, for previous version, see [2022 branch](https://github.com/mryab/efficient-dl-systems/tree/2022).__

# Syllabus
- __Week 1:__ __Introduction__
  - Lecture: Course overview and organizational details. Core concepts of the GPU architecture and CUDA API.
  - Seminar: CUDA operations in PyTorch. Introduction to benchmarking.
- __Week 2:__ __Experiment tracking, model and data versioning, testing DL code in Python__
  - Lecture: Experiment management basics and pipeline versioning. Configuring Python applications. Intro to regular and property-based testing.
  - Seminar: Example DVC+W&B project walkthrough. Intro to testing with pytest.
- __Week 3:__ __Training optimizations, profiling DL code__
  - Lecture: Mixed-precision training. Data storage and loading optimizations. Tools for profiling deep learning workloads.
  - Seminar: Automatic Mixed Precision in PyTorch. Dynamic padding for sequence data and JPEG decoding benchmarks. Basics of PyTorch Profiler and cProfile.
- __Week 4:__ __Basics of distributed ML__
  - Lecture: Introduction to distributed training. Process-based communication. Parameter Server architecture.
  - Seminar: Multiprocessing basics. Parallel GloVe training.
- __Week 5:__ __Data-parallel training and All-Reduce__
  - Lecture: Data-parallel training of neural networks. All-Reduce and its efficient implementations.
  - Seminar: Introduction to PyTorch Distributed. Data-parallel training primitives.
- __Week 6:__ __Memory-efficient and model-parallel training__
  - Lecture: Model-parallel training, gradient checkpointing, offloading.
  - Seminar: Gradient checkpointing in practice.
- __Week 7:__ __Python web application deployment__
  - Lecture/Seminar: Building and deployment of production-ready web services. App & web servers, Docker, Prometheus, API via HTTP and gRPC.
- __Week 8:__ __Software for serving neural networks__
  - Lecture/Seminar: Different formats for packing NN: ONNX, TorchScript, IR. Inference servers: OpenVINO, Triton. ML on client devices: TfJS, ML Kit, Core ML.
- __Week 9:__ __Optimizing models for faster inference__
  - Lecture: Knowledge distillation, Pruning, Quantization, NAS, Efficient Architectures.
  - Seminar: Quantization and distillation in practice.
- __Week 10:__ __Invited talks__ (speakers TBA)

## Grading
There will be several home assignments (spread over multiple weeks) on the following topics:
- Training pipelines and code profiling
- Distributed and memory-efficient training
- Deploying and optimizing models for production

The final grade is a weighted sum of per-assignment grades.
Please refer to the course page of your institution for details.

# Staff
- [Max Ryabinin](https://github.com/mryab)
- [Just Heuristic](https://github.com/justheuristic)
- [Alexander Markovich](https://github.com/markovka17)
- [Alexey Kosmachev](https://github.com/ADKosm)
- [Anton Semenkin](https://github.com/topshik/)

# Past versions
- [2022](https://github.com/mryab/efficient-dl-systems/tree/2022)
- [2021](https://github.com/yandexdataschool/dlatscale_draft)
