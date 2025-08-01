#include <pybind11/pybind11.h>
#include "torch/python.h"

#include "flag_gems/operators.h"

// TODO: use pytorch's argparse utilities to generate CPython bindings, since it is more efficient than
// bindings provided by torch library, since it is in a boxed fashion
PYBIND11_MODULE(c_operators, m) {
  m.def("sum_dim", &flag_gems::sum_dim);
  m.def("add_tensor", &flag_gems::add_tensor);
  m.def("rms_norm", &flag_gems::rms_norm);
  m.def("fused_add_rms_norm", &flag_gems::fused_add_rms_norm);
  m.def("nonzero", &flag_gems::nonzero);
  // Rotary embedding
  m.def("rotary_embedding", &flag_gems::rotary_embedding);
  m.def("rotary_embedding_inplace", &flag_gems::rotary_embedding_inplace);
  m.def("bmm", &flag_gems::bmm);
  m.def("addmm", &flag_gems::addmm);

  m.impl("fill_scalar", &flag_gems::fill_scalar);
  m.impl("fill_tensor", &flag_gems::fill_tensor);
  m.impl("fill_scalar_", &flag_gems::fill_scalar_);
  m.impl("fill_tensor_", &flag_gems::fill_tensor_);
}

namespace flag_gems {
TORCH_LIBRARY(flag_gems, m) {
  m.def(
      "zeros(SymInt[] size, ScalarType? dtype=None,Layout? layout=None, Device? device=None, bool? "
      "pin_memory=None) -> Tensor");
  m.def("sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor");
  m.def("add_tensor(Tensor self, Tensor other) -> Tensor", {at::Tag::pt2_compliant_tag});
  // Norm
  m.def("rms_norm(Tensor input, Tensor weight, float epsilon) -> Tensor");
  m.def("fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, float epsilon) -> ()");
  m.def("addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor");
  m.def("nonzero(Tensor self) -> Tensor");
  // rotary_embedding
  m.def(
      "rotary_embedding_inplace(Tensor! q, Tensor! k, Tensor cos, Tensor sin, Tensor? position_ids=None, "
      "bool rotary_interleaved=False) -> ()");
  m.def(
      "rotary_embedding(Tensor q, Tensor k, Tensor cos, Tensor sin, Tensor? position_ids=None, "
      "bool rotary_interleaved=False) -> (Tensor, Tensor)");  // q and k may be view to other size
  m.def("topk(Tensor x, SymInt k, int dim, bool largest, bool sorted) -> (Tensor, Tensor)");
  m.def("contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format) -> Tensor(a)");
  m.def("cat(Tensor[] tensors, int dim=0) -> Tensor");
  m.def("bmm(Tensor self, Tensor mat2) -> Tensor");
  m.def(
      "embedding(Tensor weight, Tensor indices, SymInt padding_idx=-1, bool scale_grad_by_freq=False, bool "
      "sparse=False) -> Tensor");
  m.def(
      "embedding_backward(Tensor grad_outputs, Tensor indices, SymInt num_weights, SymInt padding_idx, bool "
      "scale_grad_by_freq, bool sparse) -> Tensor");
  m.def("argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor");

  m.def("fill.Scalar(Tensor self, Scalar value) -> Tensor");
  m.def("fill.Tensor(Tensor self, Tensor value) -> Tensor");
  m.def("fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)");
  m.def("fill_.Tensor(Tensor(a!) self, Tensor value) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(flag_gems, CUDA, m) {
  m.impl("zeros", TORCH_FN(zeros));
  m.impl("sum.dim_IntList", TORCH_FN(sum_dim));
  m.impl("add_tensor", TORCH_FN(add_tensor));
  // Norm
  m.impl("rms_norm", TORCH_FN(rms_norm));
  m.impl("fused_add_rms_norm", TORCH_FN(fused_add_rms_norm));
  m.impl("addmm", TORCH_FN(addmm));
  m.impl("nonzero", TORCH_FN(nonzero));
  // Rotary embedding
  m.impl("rotary_embedding", TORCH_FN(rotary_embedding));
  m.impl("rotary_embedding_inplace", TORCH_FN(rotary_embedding_inplace));
  m.impl("topk", TORCH_FN(topk));
  m.impl("contiguous", TORCH_FN(contiguous));
  m.impl("cat", TORCH_FN(cat));
  m.impl("bmm", TORCH_FN(bmm));
  m.impl("embedding", TORCH_FN(embedding));
  m.impl("embedding_backward", TORCH_FN(embedding_backward));
  m.impl("argmax", TORCH_FN(argmax));

  m.impl("fill.Scalar", TORCH_FN(fill_scalar));
  m.impl("fill.Tensor", TORCH_FN(fill_tensor));
  m.impl("fill_.Scalar", TORCH_FN(fill_scalar_));
  m.impl("fill_.Tensor", TORCH_FN(fill_tensor_));
}
}  // namespace flag_gems
