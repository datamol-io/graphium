// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <poplar/Vertex.hpp>
#include <ipu_builtins.h>

class IsNaNUsingF32ClassVertex : public poplar::Vertex {
public:
  // Fields
  poplar::Input<poplar::Vector<float>> in;
  poplar::Output<poplar::Vector<bool>> out;

  // Compute function
  bool compute() {

    for (int i = 0; i < in.size(); ++i) {

      auto inClass = __builtin_ipu_f32class(in[i]);

      // TFPU_CLASS_UNC = 0
      // TFPU_CLASS_SNAN = 1
      // TFPU_CLASS_QNAN = 2
      // All others are > 3 and not NaN
      out[i] = inClass < 3;

    }
    return true;
  }
};
