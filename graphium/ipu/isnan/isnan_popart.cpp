// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popart/popx/opx.hpp>
#include <popops/ElementWise.hpp>

#include <iostream>

namespace CustomOperators {
const popart::OperatorIdentifier IsNanId = {"custom.ops", "IsNanCustom", 1};
} // namespace CustomOperators

class IsNanOp;
class IsNanOpx;
class IsNanGradOpx;

class IsNanOp : public popart::Op {
public:
  IsNanOp(const popart::OperatorIdentifier &_opid,
              const popart::Op::Settings &settings_)
      : popart::Op(_opid, settings_) {}

  std::unique_ptr<Op> clone() const final {
    return std::make_unique<IsNanOp>(*this);
  }

  void setup() final {
    outInfo(0) = popart::TensorInfo(popart::DataType::BOOL,
				    inInfo(0).shape());
  }

  void appendAttributes(popart::OpSerialiserBase &os) const override {
    Op::appendAttributes(os);
  }

  void appendOutlineAttributes(popart::OpSerialiserBase &os) const override {
    Op::appendOutlineAttributes(os);
  }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  bool requiresRandomSeed() const override { return false; }

};

namespace {
using popart::DataType;
using popart::OpDefinition;

  static OpDefinition isNanOpDef({OpDefinition::Inputs({{"input", {popart::DataType::FLOAT}}}),
				  OpDefinition::Outputs({{"output", {popart::DataType::BOOL}}}),
				  OpDefinition::Attributes({})
    });
  
static popart::OpCreator<IsNanOp> isNanOpCreator(
    popart::OpDefinitions({{CustomOperators::IsNanId, isNanOpDef}}),
    [](const popart::OpCreatorInfo &info) {
      return std::make_unique<IsNanOp>(info.opid, info.settings);
    },
    true);
} // namespace

namespace pe = popops::expr;

class IsNanOpx : public popart::popx::Opx {
public:
  IsNanOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<IsNanOp>(op, {CustomOperators::IsNanId});
  }

  void grow(poplar::program::Sequence &prog) const final {

    auto op = getOp<IsNanOp>();

    poplar::Tensor input = getInTensor(0);
    auto output = graph().addVariable(poplar::BOOL, input.shape(), debugContext("IsNanCustom"));
    auto tileMapping = graph().getTileMapping(input);
    graph().setTileMapping(output, tileMapping);
    
    graph().addCodelets("isnan.gp");
    auto computeSet = graph().addComputeSet("isNanComputeSet");

    for (unsigned i = 0; i < tileMapping.size(); ++i) {
      auto intervals = tileMapping.at(i);
      //std::cerr << "i = " << i << std::endl;
      //std::cerr << "intervals.size() = " << intervals.size() << std::endl;
      for (auto interval : intervals) {
	auto vertex = graph().addVertex(computeSet, "IsNaNUsingF32ClassVertex");
	graph().setTileMapping(vertex, i);
	graph().connect(vertex["in"], input.flatten().slice(interval.begin(), interval.end()));
	graph().connect(vertex["out"], output.flatten().slice(interval.begin(), interval.end()));
      }
    }

    prog.add(poplar::program::Execute(computeSet));
    
    setOutTensor(0, output);
  }
};

static popart::popx::OpxCreator<IsNanOpx>
    IsNanOpxCreator({CustomOperators::IsNanId});
