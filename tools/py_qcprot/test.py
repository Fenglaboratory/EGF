__author__ = "Bernhard Thiel"
__email__ = "thiel@tbi.univie.ac.at"
__license__ = "BSD 3-clause"

from py_qcprot import rmsd as rmsd_qc
import unittest
import math
import numpy as np
from timeit import timeit

RAND_TEST_ITERATIONS = 20


def center_coords(coords):
    """
    Return the coordinate array centered on its centroid

    :param coords: A numpy Nx3-array
    """
    centroid = np.sum(coords, axis=0)
    centroid /= float(len(coords))
    # Center on centroids
    return coords - centroid


def rmsd_kabsch(coords1, coords2):
    """
    A numpy based implementation of the kabsch algorithm
    used as a reference.

    Our implementation makes proper use of numpy vectorization,
    so it should not be too slow.
    However, numpy.linalg.det and np.linalg.svd might be
    suboptimal in terms of speed for 3x3 matrices, as they
    are optimized for large matrices.

    In conclusion, our implemenation uses all trivial optimizations
    but is not highly optimized.
    """
    # Center on centroids
    coords1 = center_coords(coords1)
    coords2 = center_coords(coords2)
    # Get the rotation matrix matrix
    correlation_matrix = np.dot(np.transpose(coords1), coords2)
    v, s, w_tr = np.linalg.svd(correlation_matrix)
    is_reflection = (np.linalg.det(v) * np.linalg.det(w_tr)) < 0.0
    if is_reflection:
        v[:, -1] = -v[:, -1]
    rotation_matrix = np.dot(v, w_tr)
    coords1_aligned = np.dot(coords1, rotation_matrix)
    diff_vecs = coords2 - coords1_aligned
    vec_lengths = np.sum(diff_vecs * diff_vecs, axis=1)
    return math.sqrt(sum(vec_lengths) / len(vec_lengths))


def get_random_coord_arrays(num_coords=500):
    """
    Get two random coordinate arrays with shape (num_coords x 3)

    The second array should have a rmsd in the order of magnitude of 10
    to the first array.

    :param num_coords: The number of points that should
                       be contained in each output array
    """
    b1 = (np.random.rand(num_coords, 3) - 0.5) * 300
    b2 = b1 + (np.random.rand(num_coords, 3) - 0.5) * 20
    return b1, b2


class TestRMSD(unittest.TestCase):
    def setUp(self):
        self.a1 = np.array(
            [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]], order="C"
        )
        self.a2 = np.array(
            [[2.0, 2.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -2.0, -2.0]], order="C"
        )  # RMSD of sqrt(2) to a1
        self.a3 = np.array(
            [[2.0, 2.0, 1.0], [1.0, 1.0, 0.0], [0.0, 0.0, -1.0]], order="F"
        )  # a1 only shifted (no rotation)
        self.a4 = np.array(
            [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        )  # a1 rotated

    def test_rmsd_kabsch(self):
        """
        Test our Kabsch reference implementation on trivial examples.
        """
        self.assertAlmostEqual(rmsd_kabsch(self.a1, self.a1), 0)
        self.assertAlmostEqual(rmsd_kabsch(self.a1, self.a2), math.sqrt(2))
        self.assertAlmostEqual(rmsd_kabsch(self.a1, self.a3), 0)
        self.assertAlmostEqual(rmsd_kabsch(self.a1, self.a4), 0)

    def test_rmsd_qc(self):
        """
        Test the py_qcprot rmsd version on trivial examples.
        """
        self.assertAlmostEqual(
            rmsd_qc(self.a1, self.a1), 0
        )  # Special case, treated seperately in python wrapper
        self.assertAlmostEqual(rmsd_qc(self.a1, self.a2), math.sqrt(2))

    @unittest.expectedFailure  # "Currently qcprot cannot handle the case RMSD==0"
    def test_rmsd_qc_2(self):
        """
        The qcprot algorithm currently does not handle the case of 0 RMSD properly.
        """
        self.assertAlmostEqual(rmsd_qc(self.a1, self.a3), 0)  # Only translation
        self.assertAlmostEqual(rmsd_qc(self.a1, self.a4), 0)  # Only rotation

    def test_rmsd_bigger_dataset(self):
        """
        For large point-clouds, kabsch and qcprot yield the same RMSD
        """
        for i in range(RAND_TEST_ITERATIONS):
            b1, b2 = get_random_coord_arrays()
            self.assertAlmostEqual(rmsd_kabsch(b1, b2), rmsd_qc(b1, b2))

    def test_rmsd_skipping_entries(self):
        """
        If the original input array has no contiguouse memory layout, we still don't crash
        """
        for i in range(RAND_TEST_ITERATIONS):
            b1, b2 = get_random_coord_arrays()
            self.assertAlmostEqual(
                rmsd_kabsch(b1[::2], b2[::2]), rmsd_qc(b1[::2], b2[::2])
            )

    def test_qcrmsd_centered(self):
        """
        Test the is_centered flag of qcprot
        """
        for i in range(RAND_TEST_ITERATIONS):
            b1, b2 = get_random_coord_arrays()
            b1 = center_coords(b1)
            b2 = center_coords(b2)
            self.assertAlmostEqual(rmsd_kabsch(b1, b2), rmsd_qc(b1, b2, True))

    def test_handles_wrong_shapes_well(self):
        """
        The cython wrapper raises an error if the shapes are not compatible.
        """
        b1, b2 = get_random_coord_arrays(20)
        c1, c2 = get_random_coord_arrays(21)
        with self.assertRaises(ValueError):
            rmsd_qc(b1, c1)


class ProfileRMSD(unittest.TestCase):
    """
    This is not run by standard test-discovery with nose2.
    It can be run manually
    """

    def profile_rmsd(self):
        """
        The qc-prot implementation is 10 times faster than our Kabsch
        reference implementation, despite the fact that the python
        wrapper copies the data into a new memory layout for use with qcprot.c

        Further optimization could be achieved by making the cython wrapper
        accept only arrays with correct memory layout.
        """
        # Note: The numbers in the arrays are not the same between the two approaches,
        # but it should make no difference after 10000 iterations (each time differeent coordinates are used)
        for num_points in range(100, 1001, 100):
            qc = timeit(
                "rmsd_qc(b1, b2)",
                number=10000,
                setup="from test import get_random_coord_arrays; from py_qcprot import rmsd as rmsd_qc; b1, b2 = get_random_coord_arrays({})".format(
                    num_points
                ),
            )
            kabsch = timeit(
                "rmsd_kabsch(b1, b2)",
                number=10000,
                setup="from test import get_random_coord_arrays, rmsd_kabsch; b1, b2 = get_random_coord_arrays({})".format(
                    num_points
                ),
            )
            print(
                "{} datapoints: Kabsch {}, QCprot {}, ratio {}".format(
                    num_points, kabsch, qc, qc / kabsch
                )
            )
            self.assertLess(qc, kabsch)  # Almost 10 times faster on my system

    def profile_centering_causes_speedup(self):
        """
        Providing the is_centered flag actually speeds up calculation for centered data.
        """
        for num_points in range(100, 1001, 100):
            centered = timeit(
                "rmsd_qc(b1, b2, True)",
                number=10000,
                setup="from test import get_random_coord_arrays, center_coords; from py_qcprot import rmsd as rmsd_qc; b1, b2 = get_random_coord_arrays({}); b1=center_coords(b1); b2=center_coords(b2)".format(
                    num_points
                ),
            )
            complete = timeit(
                "rmsd_qc(b1, b2, False)",
                number=10000,
                setup="from test import get_random_coord_arrays, center_coords; from py_qcprot import rmsd as rmsd_qc; b1, b2 = get_random_coord_arrays({}); b1=center_coords(b1); b2=center_coords(b2)".format(
                    num_points
                ),
            )
            print(
                "{} datapoints: is_centered=True {}, no-op centering {}, ratio {}".format(
                    num_points, centered, complete, centered / complete
                )
            )
            self.assertLess(centered, complete)  # Small speedup is measurable
