# Laius
The source code of the paper"Laius: Towards Latency Awareness and Improved Utilization of Spatial Multitasking Accelerators in Datacenters" in ICS 2019.

The link of the paper ：https://dl.acm.org/doi/10.1145/3330345.3330351

The Homepage of Wei Zhang (First author): https://olivia-zhang.github.io/

Laius, a runtime system that carefully allocates the computation resource to co-located applications for maximizing the throughput of batch applications while guaranteeing the required QoS of user-facing services.

# Build from source:
Prerequisites: CUDA, CUDNN, Caffe1.0, Tonic suite, Rodinia.

/build:
cmake ..    make j8

# run:

/mps:
./cudnn_server.sh  /schedule.sh

# Acknowledgement
Abacus is collaboratively developed by the EPCC Lab, led by Quan Chen at Shanghai Jiao Tong University, and Lenovo.
We sincerely thank Liming Xiao and other contributors from Lenovo for their valuable efforts.
Since February 2019, they have provided substantial support in integrating CUDA MPS into a process pool with distinct resource quotas.
Their insights, code contributions, and thoughtful discussions have significantly enhanced the quality and robustness of the project.