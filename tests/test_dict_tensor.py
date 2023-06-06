"""
Unit tests for the metrics and wrappers of graphium/utils/...
"""

from typing import Dict
from graphium.utils.dict_tensor import DictTensor
import torch
from torch import Tensor

import unittest as ut
from copy import deepcopy


class test_dict_tensor(ut.TestCase):
    def _assert_dict_tensor_equal(self, dict_ten1, dict_ten2, msg=""):
        self.assertSetEqual(set(dict_ten1.keys()), set(dict_ten2.keys()), msg=msg)
        for key, tensor1 in dict_ten1.items():
            tensor2 = dict_ten2[key]
            if isinstance(tensor1, Tensor):
                msg2 = msg + f"\ntensor1: \n{tensor1}\n\ntensor2: \n{tensor2}\n\n"
                self.assertTrue(torch.all(tensor1 == tensor2), msg=msg2)
            else:
                for ii in range(len(tensor1)):
                    msg2 = msg + f"\ntensor1: \n{tensor1[ii]}\n\ntensor2: \n{tensor2[ii]}\n\n"
                    self.assertTrue(torch.all(tensor1[ii] == tensor2[ii]), msg=msg2)

    def test_tensor_funcs(self):
        dict_ten = DictTensor(
            {
                "a": torch.randn(5, 6),
                "b": torch.randn(2, 3, 4),
                "c": torch.randn(1, 2, 3, 5),
            }
        )

        # Check `DictTensor.func` returns right value
        self.assertDictEqual(dict_ten.shape, {k: v.shape for k, v in dict_ten.items()})
        self._assert_dict_tensor_equal(
            dict_ten.pinverse(), {k: v.pinverse() for k, v in dict_ten.items()}, msg="dict_ten.pinverse()"
        )
        self._assert_dict_tensor_equal(
            dict_ten.sum(), {k: v.sum() for k, v in dict_ten.items()}, msg="dict_ten.sum()"
        )
        self._assert_dict_tensor_equal(
            dict_ten.sum(axis=0), {k: v.sum(axis=0) for k, v in dict_ten.items()}, msg="dict_ten.sum(axis=0)"
        )
        self._assert_dict_tensor_equal(
            dict_ten.sum(0), {k: v.sum(0) for k, v in dict_ten.items()}, msg="dict_ten.sum(0)"
        )
        self._assert_dict_tensor_equal(
            dict_ten.sum(axis=1), {k: v.sum(axis=1) for k, v in dict_ten.items()}, msg="dict_ten.sum(axis=1)"
        )
        self._assert_dict_tensor_equal(
            dict_ten.to(dtype=int),
            {k: v.to(dtype=int) for k, v in dict_ten.items()},
            msg="dict_ten.to(dtype=int)",
        )
        self._assert_dict_tensor_equal(
            dict_ten.max(axis=0),
            {k: v.max(axis=0) for k, v in dict_ten.items()},
            msg="dict_ten.max(axis=0)[0]",
        )
        self._assert_dict_tensor_equal(
            dict_ten.transpose(0, 1),
            {k: v.transpose(0, 1) for k, v in dict_ten.items()},
            msg="dict_ten.max(axis=0)[0]",
        )

        # Check `DictTensor.func` returns right type
        self.assertIsInstance(dict_ten.shape, dict)
        self.assertIsInstance(dict_ten.shape["a"], torch.Size)
        self.assertIsInstance(dict_ten.pinverse(), DictTensor)
        self.assertIsInstance(dict_ten.sum(), DictTensor)
        self.assertIsInstance(dict_ten.sum(axis=0), DictTensor)
        self.assertIsInstance(dict_ten.sum(0), DictTensor)
        self.assertIsInstance(dict_ten.sum(axis=1), DictTensor)
        self.assertIsInstance(dict_ten.to(dtype=int), DictTensor)
        self.assertIsInstance(dict_ten.max(axis=0), dict)
        self.assertIsInstance(dict_ten.max(axis=0)["a"][0], Tensor)

    def test_torch_funcs(self):
        dict_ten = DictTensor(
            {
                "a": torch.randn(5, 6),
                "b": torch.randn(2, 3, 4),
                "c": torch.randn(1, 2, 3, 5),
            }
        )

        # Check `torch.func(DictTensor, *)` returns right value
        self._assert_dict_tensor_equal(
            torch.pinverse(
                dict_ten,
            ),
            {
                k: torch.pinverse(
                    v,
                )
                for k, v in dict_ten.items()
            },
            msg="torch.pinverse(dict_ten, )",
        )
        self._assert_dict_tensor_equal(
            torch.sum(
                dict_ten,
            ),
            {
                k: torch.sum(
                    v,
                )
                for k, v in dict_ten.items()
            },
            msg="torch.sum(dict_ten, )",
        )
        self._assert_dict_tensor_equal(
            torch.sum(dict_ten, 0),
            {k: torch.sum(v, 0) for k, v in dict_ten.items()},
            msg="torch.sum(dict_ten, 0)",
        )
        self._assert_dict_tensor_equal(
            torch.sum(dict_ten, axis=0),
            {k: torch.sum(v, axis=0) for k, v in dict_ten.items()},
            msg="torch.sum(dict_ten, axis=0)",
        )
        self._assert_dict_tensor_equal(
            torch.sum(dict_ten, axis=1),
            {k: torch.sum(v, axis=1) for k, v in dict_ten.items()},
            msg="torch.sum(dict_ten, axis=1)",
        )
        self._assert_dict_tensor_equal(
            torch.max(dict_ten, axis=0),
            {k: torch.max(v, axis=0) for k, v in dict_ten.items()},
            msg="torch.max(dict_ten, axis=0)[0]",
        )

        # Check `torch.func(DictTensor, *)` returns right type
        self.assertIsInstance(
            torch.pinverse(
                dict_ten,
            ),
            DictTensor,
        )
        self.assertIsInstance(
            torch.sum(
                dict_ten,
            ),
            DictTensor,
        )
        self.assertIsInstance(torch.sum(dict_ten, axis=0), DictTensor)
        self.assertIsInstance(torch.sum(dict_ten, 0), DictTensor)
        self.assertIsInstance(torch.sum(dict_ten, axis=1), DictTensor)
        self.assertIsInstance(torch.max(dict_ten, axis=0), dict)
        self.assertIsInstance(torch.max(dict_ten, axis=0)["a"][0], Tensor)

    def test_apply(self):
        dict_ten = DictTensor(
            {
                "a": torch.randn(5, 6),
                "b": torch.randn(2, 3, 4),
                "c": torch.randn(1, 2, 3, 5),
            }
        )

        # Check `DictTensor.apply` function returns right value
        self._assert_dict_tensor_equal(
            dict_ten.apply(
                torch.pinverse,
            ),
            {
                k: torch.pinverse(
                    v,
                )
                for k, v in dict_ten.items()
            },
            msg="torch.pinverse(dict_ten, )",
        )
        self._assert_dict_tensor_equal(
            dict_ten.apply(
                torch.sum,
            ),
            {
                k: torch.sum(
                    v,
                )
                for k, v in dict_ten.items()
            },
            msg="torch.sum(dict_ten, )",
        )
        self._assert_dict_tensor_equal(
            dict_ten.apply(torch.sum, 0),
            {k: torch.sum(v, 0) for k, v in dict_ten.items()},
            msg="torch.sum(dict_ten, 0)",
        )
        self._assert_dict_tensor_equal(
            dict_ten.apply(torch.sum, axis=0),
            {k: torch.sum(v, axis=0) for k, v in dict_ten.items()},
            msg="torch.sum(dict_ten, axis=0)",
        )
        self._assert_dict_tensor_equal(
            dict_ten.apply(torch.sum, axis=1),
            {k: torch.sum(v, axis=1) for k, v in dict_ten.items()},
            msg="torch.sum(dict_ten, axis=1)",
        )
        self._assert_dict_tensor_equal(
            dict_ten.apply(torch.max, axis=0),
            {k: torch.max(v, axis=0) for k, v in dict_ten.items()},
            msg="torch.max(dict_ten, axis=0)[0]",
        )

        # Check `DictTensor.apply` function returns right type
        self.assertIsInstance(
            dict_ten.apply(
                torch.pinverse,
            ),
            DictTensor,
        )
        self.assertIsInstance(
            dict_ten.apply(
                torch.sum,
            ),
            DictTensor,
        )
        self.assertIsInstance(dict_ten.apply(torch.sum, axis=0), DictTensor)
        self.assertIsInstance(dict_ten.apply(torch.sum, 0), DictTensor)
        self.assertIsInstance(dict_ten.apply(torch.sum, axis=1), DictTensor)
        self.assertIsInstance(dict_ten.apply(torch.max, axis=0), dict)
        self.assertIsInstance(dict_ten.apply(torch.max, axis=0)["a"][0], Tensor)

    def test_tensor_operators(self):
        dict_ten = DictTensor(
            {
                "a": torch.randn(5, 6),
                "b": torch.randn(4, 5, 6),
                "c": torch.randn(1, 2, 5, 6),
            }
        )

        # Check product, division, sum, subtraction by a constant
        FACTOR = 0.5
        self._assert_dict_tensor_equal(
            dict_ten * FACTOR, {k: FACTOR * v for k, v in dict_ten.items()}, msg="DT * factor"
        )
        self._assert_dict_tensor_equal(
            FACTOR * dict_ten, {k: FACTOR * v for k, v in dict_ten.items()}, msg="factor * DT"
        )
        self._assert_dict_tensor_equal(
            dict_ten + FACTOR, {k: FACTOR + v for k, v in dict_ten.items()}, msg="DT + factor"
        )
        self._assert_dict_tensor_equal(
            FACTOR + dict_ten, {k: FACTOR + v for k, v in dict_ten.items()}, msg="factor + DT"
        )
        self._assert_dict_tensor_equal(
            dict_ten - FACTOR, {k: v - FACTOR for k, v in dict_ten.items()}, msg="DT - factor"
        )
        self._assert_dict_tensor_equal(
            FACTOR - dict_ten, {k: FACTOR - v for k, v in dict_ten.items()}, msg="factor - DT"
        )
        self._assert_dict_tensor_equal(
            dict_ten / FACTOR, {k: v / FACTOR for k, v in dict_ten.items()}, msg="DT / factor"
        )
        self._assert_dict_tensor_equal(
            dict_ten // FACTOR, {k: v // FACTOR for k, v in dict_ten.items()}, msg="DT // factor"
        )

        # Check product, division, sum, subtraction by a Tensor of shape [6]
        FACTOR = torch.rand(6)
        self._assert_dict_tensor_equal(
            dict_ten * FACTOR, {k: FACTOR * v for k, v in dict_ten.items()}, msg="DT * tensor"
        )
        self._assert_dict_tensor_equal(
            FACTOR * dict_ten, {k: FACTOR * v for k, v in dict_ten.items()}, msg="tensor * DT"
        )
        self._assert_dict_tensor_equal(
            dict_ten + FACTOR, {k: FACTOR + v for k, v in dict_ten.items()}, msg="DT + tensor"
        )
        self._assert_dict_tensor_equal(
            FACTOR + dict_ten, {k: FACTOR + v for k, v in dict_ten.items()}, msg="tensor + DT"
        )
        self._assert_dict_tensor_equal(
            dict_ten - FACTOR, {k: v - FACTOR for k, v in dict_ten.items()}, msg="DT - tensor"
        )
        self._assert_dict_tensor_equal(
            FACTOR - dict_ten, {k: FACTOR - v for k, v in dict_ten.items()}, msg="tensor - DT"
        )
        self._assert_dict_tensor_equal(
            dict_ten / FACTOR, {k: v / FACTOR for k, v in dict_ten.items()}, msg="DT / tensor"
        )
        self._assert_dict_tensor_equal(
            dict_ten // FACTOR, {k: v // FACTOR for k, v in dict_ten.items()}, msg="DT // tensor"
        )

        # Check product, division, sum, subtraction by a Tensor of shape [5, 6]
        FACTOR = torch.rand(5, 6)
        self._assert_dict_tensor_equal(
            dict_ten * FACTOR, {k: FACTOR * v for k, v in dict_ten.items()}, msg="DT * tensor"
        )
        self._assert_dict_tensor_equal(
            FACTOR * dict_ten, {k: FACTOR * v for k, v in dict_ten.items()}, msg="tensor * DT"
        )
        self._assert_dict_tensor_equal(
            dict_ten + FACTOR, {k: FACTOR + v for k, v in dict_ten.items()}, msg="DT + tensor"
        )
        self._assert_dict_tensor_equal(
            FACTOR + dict_ten, {k: FACTOR + v for k, v in dict_ten.items()}, msg="tensor + DT"
        )
        self._assert_dict_tensor_equal(
            dict_ten - FACTOR, {k: v - FACTOR for k, v in dict_ten.items()}, msg="DT - tensor"
        )
        self._assert_dict_tensor_equal(
            FACTOR - dict_ten, {k: FACTOR - v for k, v in dict_ten.items()}, msg="tensor - DT"
        )
        self._assert_dict_tensor_equal(
            dict_ten / FACTOR, {k: v / FACTOR for k, v in dict_ten.items()}, msg="DT / tensor"
        )
        self._assert_dict_tensor_equal(
            dict_ten // FACTOR, {k: v // FACTOR for k, v in dict_ten.items()}, msg="DT // tensor"
        )

    def test_comparison_operators(self):
        dict_ten = DictTensor(
            {
                "a": torch.randn(5, 6),
                "b": torch.randn(4, 5, 6),
                "c": torch.randn(1, 2, 5, 6),
            }
        )
        dict_ten2 = deepcopy(dict_ten).abs()

        # Comparison operators with float
        self._assert_dict_tensor_equal(
            dict_ten > 0.2, {k: v > 0.2 for k, v in dict_ten.items()}, msg="DT > 0.2"
        )
        self._assert_dict_tensor_equal(
            dict_ten < 0.2, {k: v < 0.2 for k, v in dict_ten.items()}, msg="DT < 0.2"
        )
        self._assert_dict_tensor_equal(
            dict_ten >= 0.2, {k: v >= 0.2 for k, v in dict_ten.items()}, msg="DT >= 0.2"
        )
        self._assert_dict_tensor_equal(
            dict_ten <= 0.2, {k: v <= 0.2 for k, v in dict_ten.items()}, msg="DT <= 0.2"
        )
        self._assert_dict_tensor_equal(
            dict_ten.to(int) == 0, {k: v.to(int) == 0 for k, v in dict_ten.items()}, msg="DT.int == 0"
        )
        self._assert_dict_tensor_equal(
            dict_ten.to(int) != 0, {k: v.to(int) != 0 for k, v in dict_ten.items()}, msg="DT.int != 0"
        )

        # Comparison operators with Tensor
        tensor = torch.rand(5, 6)
        self._assert_dict_tensor_equal(
            dict_ten > tensor, {k: v > tensor for k, v in dict_ten.items()}, msg="DT > DT2"
        )
        self._assert_dict_tensor_equal(
            dict_ten < tensor, {k: v < tensor for k, v in dict_ten.items()}, msg="DT < DT2"
        )
        self._assert_dict_tensor_equal(
            dict_ten >= tensor, {k: v >= tensor for k, v in dict_ten.items()}, msg="DT >= DT2"
        )
        self._assert_dict_tensor_equal(
            dict_ten <= tensor, {k: v <= tensor for k, v in dict_ten.items()}, msg="DT <= DT2"
        )
        self._assert_dict_tensor_equal(
            dict_ten.to(int) == tensor,
            {k: v.to(int) == tensor for k, v in dict_ten.items()},
            msg="DT.int == DT2",
        )
        self._assert_dict_tensor_equal(
            dict_ten.to(int) != tensor,
            {k: v.to(int) != tensor for k, v in dict_ten.items()},
            msg="DT.int != DT2",
        )

        # Comparison operators with DictTensor
        self._assert_dict_tensor_equal(
            dict_ten > dict_ten2, {k: v > dict_ten2[k] for k, v in dict_ten.items()}, msg="DT > DT2"
        )
        self._assert_dict_tensor_equal(
            dict_ten < dict_ten2, {k: v < dict_ten2[k] for k, v in dict_ten.items()}, msg="DT < DT2"
        )
        self._assert_dict_tensor_equal(
            dict_ten >= dict_ten2, {k: v >= dict_ten2[k] for k, v in dict_ten.items()}, msg="DT >= DT2"
        )
        self._assert_dict_tensor_equal(
            dict_ten <= dict_ten2, {k: v <= dict_ten2[k] for k, v in dict_ten.items()}, msg="DT <= DT2"
        )
        self._assert_dict_tensor_equal(
            dict_ten.to(int) == dict_ten2,
            {k: v.to(int) == dict_ten2[k] for k, v in dict_ten.items()},
            msg="DT.int == DT2",
        )
        self._assert_dict_tensor_equal(
            dict_ten.to(int) != dict_ten2,
            {k: v.to(int) != dict_ten2[k] for k, v in dict_ten.items()},
            msg="DT.int != DT2",
        )

    def test_dict_functions(self):
        dict1 = {
            "a": torch.randn(5, 6),
            "b": torch.randn(4, 5, 6),
            "c": torch.randn(1, 2, 5, 6),
        }
        dict_ten1 = DictTensor(deepcopy(dict1))
        dict2 = {
            "c": torch.randn(1, 2, 6),
            "d": torch.randn(1, 2, 3, 4),
        }
        dict_ten2 = DictTensor(deepcopy(dict2))

        # Check update
        dict1_temp = deepcopy(dict1)
        dict_ten1_temp = deepcopy(dict_ten1)
        dict1_temp.update(dict2)
        truth = DictTensor(dict1_temp)
        dict_ten1_temp.update(dict_ten2)
        self._assert_dict_tensor_equal(dict_ten1_temp, truth)

        # Check pop
        dict1_temp = deepcopy(dict1)
        dict_ten1_temp = deepcopy(dict_ten1)
        truth_a = dict1_temp.pop("a")
        truth = DictTensor(dict1_temp)
        val_a = dict_ten1_temp.pop("a")
        self._assert_dict_tensor_equal(dict_ten1_temp, truth)
        self.assertTrue(torch.all(truth_a == val_a))

        # Check indexing
        dict1_temp = deepcopy(dict1)
        dict_ten1_temp = deepcopy(dict_ten1)
        truth_b = dict1_temp["b"]
        val_b = dict_ten1_temp["b"]
        self.assertTrue(torch.all(truth_b == val_b))

        # Check adding element
        dict1_temp = deepcopy(dict1)
        dict_ten1_temp = deepcopy(dict_ten1)
        dict1_temp["new"] = dict2["d"]
        truth = DictTensor(dict1_temp)
        dict_ten1_temp["new"] = dict_ten2["d"]
        self._assert_dict_tensor_equal(dict_ten1_temp, truth)


if __name__ == "__main__":
    ut.main()
