#include "scatter_cuda.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

#include "reducer.cuh"
#include "utils.cuh"

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

template <typename scalar_t, ReductionType REDUCE>
__global__ void
scatter_kernel(const scalar_t *src_data,
               const at::cuda::detail::TensorInfo<int64_t, int> index_info,
               scalar_t *out_data, int E, int K, int N, int numel) {

  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  int b = thread_idx / (E * K);  //
  int k = thread_idx % K;

  if (thread_idx < numel) {
    int offset = at::cuda::detail::IndexToOffset<int64_t, int, -1>::get(
        thread_idx, index_info);
    int64_t idx = index_info.data[offset];

    Reducer<scalar_t, REDUCE>::atomic_write(out_data + b * N * K + idx * K + k,
                                            src_data[thread_idx]);
  }
}

template <typename scalar_t>
__global__ void
scatter_arg_kernel(const scalar_t *src_data,
                   const at::cuda::detail::TensorInfo<int64_t, int> index_info,
                   const scalar_t *out_data, int64_t *arg_out_data, int E,
                   int K, int N, int numel) {

  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  int b = thread_idx / (E * K);
  int e = (thread_idx / K) % E;
  int k = thread_idx % K;

  if (thread_idx < numel) {
    int offset = at::cuda::detail::IndexToOffset<int64_t, int, -1>::get(
        thread_idx, index_info);
    int64_t idx = index_info.data[offset];

    if (src_data[thread_idx] == out_data[b * N * K + idx * K + k]) {
      arg_out_data[b * N * K + idx * K + k] = e;
    }
  }
}


// scatter_cuda是python端torch_scatter.scatter函数的cuda接口
// 要注意这个函数是运行在host端的
std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
scatter_cuda(torch::Tensor src, torch::Tensor index, int64_t dim,
             torch::optional<torch::Tensor> optional_out,
             torch::optional<int64_t> dim_size, std::string reduce) {

  CHECK_CUDA(src);
  CHECK_CUDA(index);
  if (optional_out.has_value())
    CHECK_CUDA(optional_out.value());
  cudaSetDevice(src.get_device());  // 设置要用于GPU执行的设备,随后cudamalloc都是在该设备上进行

  CHECK_INPUT(src.dim() == index.dim());
  for (auto i = 0; i < index.dim() - 1; i++)
    CHECK_INPUT(src.size(i) >= index.size(i));

  src = src.contiguous();  // 使得src数据连续存储

  torch::Tensor out;  // 声明返回tensor
  if (optional_out.has_value()) {  // 如果传入的optional_out参数不为空
    out = optional_out.value().contiguous();  // 将其内存存储连续化
    for (auto i = 0; i < out.dim(); i++)
        // 除了被scatter的dim轴以外，检查源tensor和destination tensor(如果有)的纬度是否一致
      if (i != dim)
        CHECK_INPUT(src.size(i) == out.size(i));
  } else {
    // Return callstack as a vector of [Function, SourceRange] pair
    // 返回调用栈，其形式为[Function, SourceRange]的向量
    // sizes()返回对此张量的大小的引用。 只要张量处于活动状态且未调整大小，此参考就一直有效。
    // vec()表示根据src.sizes()这个vector拷贝构造一个新的vector
    // 从源代码来看vec返回的是return std::vector<T>(Data, Data + Length);所以sizes应该是一个host端的数据
    auto sizes = src.sizes().vec();
    if (dim_size.has_value())  // 如果传入的dim_size参数不为空
      sizes[dim] = dim_size.value();
    // numel()返回该tensor中的元素个数.
    else if (index.numel() == 0)
      sizes[dim] = 0;
    else {
        // 如果dim_size为None，一个最小大小的输出out tensor会被返回，
        // 其被scatter的维度大小为index.max()+1
        // index.max()（应该是,对应在python端是这样的)获得一个tensor中的最大元素，返回形式为只有一个元素的tensor如：tensor(12.2131)
        // data_ptr返回张量的第一个元素的地址。
      auto d_size = index.max().data_ptr<int64_t>();
      auto h_size = (int64_t *)malloc(sizeof(int64_t));
      // 注意到index本身是存放在GPU端的，所以这里要将数据从从GPU端拷贝到CPU端
      cudaMemcpy(h_size, d_size, sizeof(int64_t), cudaMemcpyDeviceToHost);
      // 1 + index.max()
      sizes[dim] = 1 + *h_size;  // *h_size的值即为index中的最大值
    }
    out = torch::empty(sizes, src.options());  // 创建大小为sizes的host端空矩阵,options()是创建tensor的一些属性，相见官方文档介绍
  }

  torch::optional<torch::Tensor> arg_out = torch::nullopt;
  int64_t *arg_out_data = nullptr;
  // 在reducer.cuh中定义的操作字典reduce2REDUCE中去确定传入的std::string reduce参数具体指的是哪一种操作
  if (reduce2REDUCE.at(reduce) == MIN || reduce2REDUCE.at(reduce) == MAX) {
    // full_like函数根据out的纬度来填充一个新的tensor,填充值为src.size(dim)
    arg_out = torch::full_like(out, src.size(dim), index.options());
    // data_ptr()函数获得该tensor首元素的指针
    arg_out_data = arg_out.value().data_ptr<int64_t>();
  }

  if (index.numel() == 0)
    return std::make_tuple(out, arg_out); // C++元组类型

  auto B = 1;
  // 我大概猜测一下为什么B就可以直接和存放在GPU上的src tensor进行交互
  // src.size(i)返回的是一个常数，像src的大小这样的元数据应该是存放在host端，或者至少有一份拷贝
  // 这样获得这些元数据就无需与GPU端交互
  for (auto i = 0; i < dim; i++)
    B *= src.size(i); // B为在dim维之前所有维度的乘积
  auto E = src.size(dim);  // E是src在axis=dim上的纬度大小
  auto K = src.numel() / (B * E);
  auto N = out.size(dim);

  // getTensorInfo函数定义在/home/wgt/pytorch-master/aten/src/ATen/cuda/detail/IndexUtils.cuh中
  // 作用是返回一个tensor给个纬度的数据头指针，维度，size和stride（步长）

  auto index_info = at::cuda::detail::getTensorInfo<int64_t, int>(index);  // 注意这个东西是定义在gpu端上的
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_ALL_TYPES(src.scalar_type(), "scatter", [&] {
    auto src_data = src.data_ptr<scalar_t>();  // 指向src tensor数据的指针
    auto out_data = out.data_ptr<scalar_t>();  // 指向out tensor数据的指针

    AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
      if (!optional_out.has_value())
        out.fill_(Reducer<scalar_t, REDUCE>::init());

      scatter_kernel<scalar_t, REDUCE>
          <<<BLOCKS(src.numel()), THREADS, 0, stream>>>(
              src_data, index_info, out_data, E, K, N, src.numel());

      if (!optional_out.has_value() && (REDUCE == MIN || REDUCE == MAX))
        out.masked_fill_(out == Reducer<scalar_t, REDUCE>::init(), (scalar_t)0);

      if (REDUCE == MIN || REDUCE == MAX)
        scatter_arg_kernel<scalar_t>
            <<<BLOCKS(src.numel()), THREADS, 0, stream>>>(
                src_data, index_info, out_data, arg_out_data, E, K, N,
                src.numel());
    });
  });

  return std::make_tuple(out, arg_out);
}
