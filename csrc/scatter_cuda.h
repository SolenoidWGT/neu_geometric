#pragma once

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
scatter_cuda(torch::Tensor src, torch::Tensor index, int64_t dim,
             torch::optional<torch::Tensor> optional_out,
             torch::optional<int64_t> dim_size, std::string reduce);
