import torch
import poptorch
import pathlib
import ctypes

myso = list(pathlib.Path(__file__).parent.rglob("build/cus*.so"))
assert myso, "Failed to find custom op .so file - please cd into `graphium/ipu/isnan` and run `make`"
assert len(myso) == 1, f"Too many ({len(myso)}) custom op .so files, there should only be one"
ctypes.cdll.LoadLibrary(myso[0])

def _ipu_isnan(self, x):

    return poptorch.custom_op(
        inputs=(x,),
        name="IsNanCustom",
        domain="custom.ops",
        domain_version=1,
        example_outputs=(x.bool(),),
    )
