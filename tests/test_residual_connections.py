"""
Unit tests for the file residual_connections.py
"""

import numpy as np
import torch
import unittest as ut

from graphium.nn.residual_connections import (
    ResidualConnectionConcat,
    ResidualConnectionDenseNet,
    ResidualConnectionNone,
    ResidualConnectionSimple,
    ResidualConnectionWeighted,
    ResidualConnectionRandom,
)


class test_ResidualConnectionNone(ut.TestCase):
    def test_get_true_out_dims_none(self):
        full_dims = [4, 6, 8, 10, 12]
        in_dims, out_dims = full_dims[:-1], full_dims[1:]
        rc = ResidualConnectionNone(skip_steps=1)
        true_out_dims = rc.get_true_out_dims(out_dims)
        expected_out_dims = out_dims[:-1]

        self.assertListEqual(expected_out_dims, true_out_dims)

    def test_forward_none(self):
        rc = ResidualConnectionNone(skip_steps=1)
        num_loops = 10
        shape = (3, 11)
        h_original = [torch.rand(shape) for _ in range(num_loops)]

        h_prev = None
        for ii in range(num_loops):
            feat, h_prev = rc.forward(h_original[ii], h_prev, step_idx=ii)
            np.testing.assert_array_equal(feat.numpy(), h_original[ii].numpy(), err_msg=f"ii={ii}")
            self.assertIsNone(h_prev)


class test_ResidualConnectionSimple(ut.TestCase):
    def test_get_true_out_dims_simple(self):
        full_dims = [4, 6, 8, 10, 12]
        in_dims, out_dims = full_dims[:-1], full_dims[1:]
        rc = ResidualConnectionSimple(skip_steps=1)
        true_out_dims = rc.get_true_out_dims(out_dims)
        expected_out_dims = out_dims[:-1]

        self.assertListEqual(expected_out_dims, true_out_dims)

    def test_forward_simple(self):
        for skip_steps in [1, 2, 3]:
            rc = ResidualConnectionSimple(skip_steps=skip_steps)
            num_loops = 10
            shape = (3, 11)
            h_original = [torch.ones(shape) * (ii + 1) for ii in range(num_loops)]

            h_prev = None
            for ii in range(num_loops):
                feat, h_prev = rc.forward(h_original[ii], h_prev, step_idx=ii)

                if ((ii % skip_steps) == 0) and (ii > 0):
                    h_expected = (
                        torch.sum(torch.stack(h_original[0 : ii + 1 : skip_steps], dim=0), dim=0)
                    ).numpy()
                    h_expected_prev = h_expected
                else:
                    h_expected = h_original[ii].numpy()
                if ii == 0:
                    h_expected_prev = h_expected

                np.testing.assert_array_equal(
                    feat.numpy(), h_expected, err_msg=f"Error at: skip_steps={skip_steps}, ii={ii}"
                )
                np.testing.assert_array_equal(
                    h_prev.numpy(), h_expected_prev, err_msg=f"Error at: skip_steps={skip_steps}, ii={ii}"
                )


class test_ResidualConnectionRandom(ut.TestCase):
    def test_get_true_out_dims_random(self):
        full_dims = [4, 6, 8, 10, 12]
        in_dims, out_dims = full_dims[:-1], full_dims[1:]
        rc = ResidualConnectionRandom(skip_steps=1, out_dims=out_dims)
        true_out_dims = rc.get_true_out_dims(out_dims)
        expected_out_dims = out_dims[:-1]

        self.assertListEqual(expected_out_dims, true_out_dims)

        rc = ResidualConnectionRandom(skip_steps=1, num_layers=len(out_dims))
        true_out_dims = rc.get_true_out_dims(out_dims)
        expected_out_dims = out_dims[:-1]

        self.assertListEqual(expected_out_dims, true_out_dims)

        with self.assertRaises(ValueError):
            rc = ResidualConnectionRandom(skip_steps=1, out_dims=None, num_layers=None)

    def test_forward_random(self):
        for skip_steps in [1, 2, 3]:
            num_loops = 10

            if skip_steps > 1:
                with self.assertRaises(ValueError):
                    rc = ResidualConnectionRandom(skip_steps=skip_steps, num_layers=num_loops)
                continue

            rc = ResidualConnectionRandom(skip_steps=skip_steps, num_layers=num_loops + 1)
            shape = (3, 11)
            h_original = [torch.ones(shape) * (ii + 1) for ii in range(num_loops)]

            h_prev = None
            # Not really testing the expected values due to randomness, just testing if it runs
            for ii in range(num_loops):
                print(ii)
                feat, h_prev = rc.forward(h_original[ii], h_prev, step_idx=ii)


class test_ResidualConnectionWeighted(ut.TestCase):
    def test_get_true_out_dims_weighted(self):
        full_dims = [4, 6, 8, 10, 12]
        in_dims, out_dims = full_dims[:-1], full_dims[1:]
        rc = ResidualConnectionWeighted(skip_steps=1, out_dims=full_dims[1:])
        true_out_dims = rc.get_true_out_dims(out_dims)
        expected_out_dims = out_dims[:-1]

        self.assertListEqual(expected_out_dims, true_out_dims)

    def test_forward_weighted(self):
        for skip_steps in [1, 2, 3]:
            num_loops = 10
            shape = (3, 11)
            full_dims = [shape[1]] * (num_loops + 1)
            rc = ResidualConnectionWeighted(
                skip_steps=skip_steps,
                out_dims=full_dims[1:],
                activation="none",
                normalization="none",
                bias=False,
            )

            h_original = [torch.ones(shape) * (ii + 1) for ii in range(num_loops)]
            h_forward = []

            h_prev = None
            step_counter = 0
            for ii in range(num_loops):
                h_prev_backup = h_prev
                feat, h_prev = rc.forward(h_original[ii], h_prev, step_idx=ii)

                if ((ii % skip_steps) == 0) and (ii > 0):
                    h_forward.append(rc.residual_list[step_counter].forward(h_prev_backup))
                    h_expected = (h_forward[-1] + h_original[ii]).detach().numpy()
                    h_expected_prev = h_expected
                    step_counter += 1
                else:
                    h_expected = h_original[ii].detach().numpy()
                if ii == 0:
                    h_expected_prev = h_expected

                np.testing.assert_array_equal(
                    feat.detach().numpy(), h_expected, err_msg=f"Error at: skip_steps={skip_steps}, ii={ii}"
                )
                np.testing.assert_array_equal(
                    h_prev.detach().numpy(),
                    h_expected_prev,
                    err_msg=f"Error at: skip_steps={skip_steps}, ii={ii}",
                )


class test_ResidualConnectionConcat(ut.TestCase):
    def test_get_true_out_dims_concat(self):
        full_dims = [4, 6, 8, 10, 12, 14, 16, 18, 20]
        in_dims, out_dims = full_dims[:-1], full_dims[1:]

        # skip_steps=1
        rc = ResidualConnectionConcat(skip_steps=1)
        true_out_dims = rc.get_true_out_dims(out_dims)
        expected_out_dims = [6, 14, 18, 22, 26, 30, 34]
        self.assertListEqual(expected_out_dims, true_out_dims)

        # skip_steps=2
        rc = ResidualConnectionConcat(skip_steps=2)
        true_out_dims = rc.get_true_out_dims(out_dims)
        expected_out_dims = [6, 8, 16, 12, 24, 16, 32]
        self.assertListEqual(expected_out_dims, true_out_dims)

        # skip_steps=3
        rc = ResidualConnectionConcat(skip_steps=3)
        true_out_dims = rc.get_true_out_dims(out_dims)
        expected_out_dims = [6, 8, 10, 18, 14, 16, 30]
        self.assertListEqual(expected_out_dims, true_out_dims)

    def test_forward_concat(self):
        for skip_steps in [1, 2, 3]:
            rc = ResidualConnectionConcat(skip_steps=skip_steps)
            num_loops = 10
            shape = (3, 11)
            h_original = [torch.ones(shape) * (ii + 1) for ii in range(num_loops)]

            h_prev = None
            for ii in range(num_loops):
                feat, h_prev = rc.forward(h_original[ii], h_prev, step_idx=ii)

                if ((ii % skip_steps) == 0) and (ii > 0):
                    h_expected = (
                        torch.cat(h_original[ii - skip_steps : ii + 1 : skip_steps][::-1], dim=-1)
                    ).numpy()
                    h_expected_prev = h_original[ii].numpy()
                else:
                    h_expected = h_original[ii].numpy()
                if ii == 0:
                    h_expected_prev = h_expected

                np.testing.assert_array_equal(
                    feat.numpy(), h_expected, err_msg=f"Error at: skip_steps={skip_steps}, ii={ii}"
                )
                np.testing.assert_array_equal(
                    h_prev.numpy(), h_expected_prev, err_msg=f"Error at: skip_steps={skip_steps}, ii={ii}"
                )


class test_ResidualConnectionDenseNet(ut.TestCase):
    def test_get_true_out_dims_densenet(self):
        full_dims = [4, 6, 8, 10, 12, 14, 16, 18, 20]
        in_dims, out_dims = full_dims[:-1], full_dims[1:]

        # skip_steps=1
        rc = ResidualConnectionDenseNet(skip_steps=1)
        true_out_dims = rc.get_true_out_dims(out_dims)
        expected_out_dims = np.cumsum(out_dims).tolist()[:-1]
        self.assertListEqual(expected_out_dims, true_out_dims)

        # skip_steps=2
        rc = ResidualConnectionDenseNet(skip_steps=2)
        true_out_dims = rc.get_true_out_dims(out_dims)
        expected_out_dims = [6, 8, 10 + 6, 12, 14 + 10 + 6, 16, 18 + 14 + 10 + 6]
        self.assertListEqual(expected_out_dims, true_out_dims)

        # skip_steps=3
        rc = ResidualConnectionDenseNet(skip_steps=3)
        true_out_dims = rc.get_true_out_dims(out_dims)
        expected_out_dims = [6, 8, 10, 12 + 6, 14, 16, 18 + 12 + 6]
        self.assertListEqual(expected_out_dims, true_out_dims)

    def test_forward_densenet(self):
        for skip_steps in [1, 2, 3]:
            rc = ResidualConnectionDenseNet(skip_steps=skip_steps)
            num_loops = 10
            shape = (3, 11)
            h_original = [torch.ones(shape) * (ii + 1) for ii in range(num_loops)]

            h_prev = None
            for ii in range(num_loops):
                feat, h_prev = rc.forward(h_original[ii], h_prev, step_idx=ii)

                if ((ii % skip_steps) == 0) and (ii > 0):
                    h_expected = (torch.cat(h_original[0 : ii + 1 : skip_steps][::-1], dim=-1)).numpy()
                    h_expected_prev = h_expected
                else:
                    h_expected = h_original[ii].numpy()
                if ii == 0:
                    h_expected_prev = h_expected

                np.testing.assert_array_equal(
                    feat.numpy(), h_expected, err_msg=f"Error at: skip_steps={skip_steps}, ii={ii}"
                )
                np.testing.assert_array_equal(
                    h_prev.numpy(), h_expected_prev, err_msg=f"Error at: skip_steps={skip_steps}, ii={ii}"
                )


if __name__ == "__main__":
    ut.main()
