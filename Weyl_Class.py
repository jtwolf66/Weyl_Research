from __future__ import division, unicode_literals

#import matplotlib as mp
import numpy as np

from sklearn.utils.extmath import cartesian

from six.moves import map, zip
from numpy.linalg import inv
from numpy import pi, dot, transpose, radians
import math
import itertools

###########################
# This was a WIP to create classes for
# the different types of domains being
# studied. It was not finished before
# I left to work with another research
# group. 
###########################

def abs_cap(val, max_abs_val=1):
    """
    Returns the value with its absolute value capped at max_abs_val.
    Particularly useful in passing values to trignometric functions where
    numerical errors may result in an argument > 1 being passed in.

    Args:
        val (float): Input value.
        max_abs_val (float): The maximum absolute value for val. Defaults to 1.

    Returns:
        val if abs(val) < 1 else sign of val * max_abs_val.
    """
    return max(min(val, max_abs_val), -max_abs_val)

class Domain:

    def __init__(self, matrix,turns):
        m = np.array(matrix, dtype=np.float64).reshape((3, 3))
        lengths = np.sqrt(np.sum(m ** 2, axis=1))
        angles = np.zeros(3)

        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            angles[i] = abs_cap(dot(m[j], m[k]) / (lengths[j] * lengths[k]))

        self._turn_num = turns
        self._angles = np.arccos(angles) * 180. / pi
        self._lengths = lengths
        self._matrix = m
        # The inverse matrix is lazily generated for efficiency.
        self._inv_matrix = None
        self._metric_tensor = None

    @property
    def matrix(self):
        """Copy of matrix representing the Lattice"""
        return np.copy(self._matrix)

    @property
    def inv_matrix(self):
        """
        Inverse of lattice matrix.
        """
        if self._inv_matrix is None:
            self._inv_matrix = inv(self._matrix)
        return self._inv_matrix

    @property
    def metric_tensor(self):
        """
        The metric tensor of the lattice.
        """
        if self._metric_tensor is None:
            self._metric_tensor = np.dot(self._matrix, self._matrix.T)
        return self._metric_tensor

    def get_cartesian_coords(self, fractional_coords):
        """
        Returns the cartesian coordinates given fractional coordinates.

        Args:
            fractional_coords (3x1 array): Fractional coords.

        Returns:
            Cartesian coordinates
        """
        return dot(fractional_coords, self._matrix)


    def get_fractional_coords(self, cart_coords):
        """
        Returns the fractional coordinates given cartesian coordinates.

        Args:
            cart_coords (3x1 array): Cartesian coords.

        Returns:
            Fractional coordinates.
        """
        return dot(cart_coords, self.inv_matrix)



    @staticmethod
    def from_lengths_and_angles(abc, ang):
        """
        Create a Lattice using unit cell lengths and angles (in radians).

        Args:
            abc (3x1 array): Lattice parameters, e.g. (4, 4, 5).
            ang (3x1 array): Lattice angles in degrees, e.g., (90,90,120).

        Returns:
            A Lattice with the specified lattice parameters.
        """
        return Domain.from_parameters(abc[0], abc[1], abc[2],
                                       ang[0], ang[1], ang[2])

    @staticmethod
    def from_parameters(a, b, c, alpha_r, beta_r, gamma_r, turns):
        """
        Create a Lattice using unit cell lengths and angles (in radians).

        Args:
            a (float): *a* lattice parameter.
            b (float): *b* lattice parameter.
            c (float): *c* lattice parameter.
            alpha (float): *alpha* angle in degrees.
            beta (float): *beta* angle in degrees.
            gamma (float): *gamma* angle in degrees.

        Returns:
            Lattice with the specified lattice parameters.
        """

        val = (np.cos(alpha_r) * np.cos(beta_r) - np.cos(gamma_r)) \
              / (np.sin(alpha_r) * np.sin(beta_r))
        # Sometimes rounding errors result in values slightly > 1.
        val = abs_cap(val)
        gamma_star = np.arccos(val)
        vector_a = [a * np.sin(beta_r), 0.0, a * np.cos(beta_r)]
        vector_b = [-b * np.sin(alpha_r) * np.cos(gamma_star),
                    b * np.sin(alpha_r) * np.sin(gamma_star),
                    b * np.cos(alpha_r)]
        vector_c = [0.0, 0.0, float(c)]
        return Domain([vector_a, vector_b, vector_c],turns)

    @staticmethod
    def hexagonal(a, c):
        """
        Convenience constructor for a hexagonal lattice.

        Args:
            a (float): *a* lattice parameter of the hexagonal cell.
            c (float): *c* lattice parameter of the hexagonal cell.

        Returns:
            Hexagonal lattice of dimensions a x a x c.
        """
        return Domain.from_parameters(a, a, c, 90, 90, 120)

    @property
    def turn_num(self):
        """
        Returns number of turns in 3-torus
        """
        return self._turn_num

    @property
    def angles(self):
        """
        Returns the angles (alpha, beta, gamma) of the lattice.
        """
        return tuple(self._angles)

    @property
    def a(self):
        """
        *a* lattice parameter.
        """
        return self._lengths[0]

    @property
    def b(self):
        """
        *b* lattice parameter.
        """
        return self._lengths[1]

    @property
    def c(self):
        """
        *c* lattice parameter.
        """
        return self._lengths[2]

    @property
    def abc(self):
        """
        Lengths of the lattice vectors, i.e. (a, b, c)
        """
        return tuple(self._lengths)

    @property
    def alpha(self):
        """
        Angle alpha of lattice in degrees.
        """
        return self._angles[0]

    @property
    def beta(self):
        """
        Angle beta of lattice in degrees.
        """
        return self._angles[1]

    @property
    def gamma(self):
        """
        Angle gamma of lattice in degrees.
        """
        return self._angles[2]

    @property
    def volume(self):
        """
        Volume of the unit cell.
        """
        m = self._matrix
        return abs(np.dot(np.cross(m[0], m[1]), m[2]))

    @property
    def surface_area(self):
        """
        Surface area of the unit cell.
        """

        size = self._abc
        angles = self._angles
        return 2 * size[2] * size[0] * np.sin(angles[1]) + 2 * size[2] * size[1] * np.sin(angles[0]) + 2 * size[1] * size[
            0] * np.sin(angles[2])

    @property
    def lengths_and_angles(self):
        """
        Returns (lattice lengths, lattice angles).
        """
        return tuple(self._lengths), tuple(self._angles)

    @property
    def reciprocal_lattice(self,turns):
        """
        Return the reciprocal lattice. Note that this is the standard
        reciprocal lattice used for solid state physics with a factor of 2 *
        pi. If you are looking for the crystallographic reciprocal lattice,
        use the reciprocal_lattice_crystallographic property.
        The property is lazily generated for efficiency.
        """
        try:
            return self._reciprocal_lattice
        except AttributeError:
            v = np.linalg.inv(self._matrix).T
            self._reciprocal_lattice = Domain(v * 2 * np.pi,turns)
            return self._reciprocal_lattice

    @property
    def reciprocal_lattice_crystallographic(self,turns):
        """
        Returns the *crystallographic* reciprocal lattice, i.e., no factor of
        2 * pi.
        """
        return Domain(self.reciprocal_lattice.matrix / (2 * np.pi),turns)

    def __repr__(self):
        outs = ["3-Domain : " + " ".join(map(repr, self._turn_num)),
                "    abc : " + " ".join(map(repr, self._lengths)),
                " angles : " + " ".join(map(repr, self._angles)),
                " volume : " + repr(self.volume),
                "      A : " + " ".join(map(repr, self._matrix[0])),
                "      B : " + " ".join(map(repr, self._matrix[1])),
                "      C : " + " ".join(map(repr, self._matrix[2]))]
        return "\n".join(outs)

    def __eq__(self, other):
        """
        A lattice is considered to be equal to another if the internal matrix
        representation satisfies np.allclose(matrix1, matrix2) to be True.
        """
        if other is None:
            return False
        # shortcut the np.allclose if the memory addresses are the same
        # (very common in Structure.from_sites)
        return self is other or np.allclose(self.matrix, other.matrix)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return 7

    def __str__(self):
        return "\n".join([" ".join(["%.6f" % i for i in row])
                          for row in self._matrix])

    def find_all_mappings(self, other_lattice, turns, ltol=1e-5, atol=1, skip_rotation_matrix=False):
        """
        Finds all mappings between current lattice and another lattice.

        Args:
            other_lattice (Lattice): Another lattice that is equivalent to
                this one.
            ltol (float): Tolerance for matching lengths. Defaults to 1e-5.
            atol (float): Tolerance for matching angles. Defaults to 1.
            skip_rotation_matrix (bool): Whether to skip calculation of the
                rotation matrix

        Yields:
            (aligned_lattice, rotation_matrix, scale_matrix) if a mapping is
            found. aligned_lattice is a rotated version of other_lattice that
            has the same lattice parameters, but which is aligned in the
            coordinate system of this lattice so that translational points
            match up in 3D. rotation_matrix is the rotation that has to be
            applied to other_lattice to obtain aligned_lattice, i.e.,
            aligned_matrix = np.inner(other_lattice, rotation_matrix) and
            op = SymmOp.from_rotation_and_translation(rotation_matrix)
            aligned_matrix = op.operate_multi(latt.matrix)
            Finally, scale_matrix is the integer matrix that expresses
            aligned_matrix as a linear combination of this
            lattice, i.e., aligned_matrix = np.dot(scale_matrix, self.matrix)

            None is returned if no matches are found.
        """
        (lengths, angles) = other_lattice.lengths_and_angles
        (alpha, beta, gamma) = angles

        frac, dist, _ = self.get_points_in_sphere([[0, 0, 0]], [0, 0, 0],
                                                  max(lengths) * (1 + ltol),
                                                  zip_results=False)
        cart = self.get_cartesian_coords(frac)
        # this can't be broadcast because they're different lengths
        inds = [np.logical_and(dist / l < 1 + ltol,
                               dist / l > 1 / (1 + ltol)) for l in lengths]
        c_a, c_b, c_c = (cart[i] for i in inds)
        f_a, f_b, f_c = (frac[i] for i in inds)
        l_a, l_b, l_c = (np.sum(c ** 2, axis=-1) ** 0.5 for c in (c_a, c_b, c_c))


        def get_angles(v1, v2, l1, l2):
            x = np.inner(v1, v2) / l1[:, None] / l2
            x[x > 1] = 1
            x[x < -1] = -1
            angles = np.arccos(x) * 180. / pi
            return angles


        alphab = np.abs(get_angles(c_b, c_c, l_b, l_c) - alpha) < atol
        betab = np.abs(get_angles(c_a, c_c, l_a, l_c) - beta) < atol
        gammab = np.abs(get_angles(c_a, c_b, l_a, l_b) - gamma) < atol

        for i, all_j in enumerate(gammab):
            inds = np.logical_and(all_j[:, None],
                                  np.logical_and(alphab,
                                                 betab[i][None, :]))
            for j, k in np.argwhere(inds):
                scale_m = np.array((f_a[i], f_b[j], f_c[k]), dtype=np.int)
                if abs(np.linalg.det(scale_m)) < 1e-8:
                    continue

                aligned_m = np.array((c_a[i], c_b[j], c_c[k]))

                if skip_rotation_matrix:
                    rotation_m = None
                else:
                    rotation_m = np.linalg.solve(aligned_m,
                                                 other_lattice.matrix)

                yield Domain(aligned_m, turns), rotation_m, scale_m

    def find_mapping(self, other_lattice, turns, ltol=1e-5, atol=1, skip_rotation_matrix=False):
        """
        Finds a mapping between current lattice and another lattice. There
        are an infinite number of choices of basis vectors for two entirely
        equivalent lattices. This method returns a mapping that maps
        other_lattice to this lattice.

        Args:
            other_lattice (Lattice): Another lattice that is equivalent to
                this one.
            ltol (float): Tolerance for matching lengths. Defaults to 1e-5.
            atol (float): Tolerance for matching angles. Defaults to 1.

        Returns:
            (aligned_lattice, rotation_matrix, scale_matrix) if a mapping is
            found. aligned_lattice is a rotated version of other_lattice that
            has the same lattice parameters, but which is aligned in the
            coordinate system of this lattice so that translational points
            match up in 3D. rotation_matrix is the rotation that has to be
            applied to other_lattice to obtain aligned_lattice, i.e.,
            aligned_matrix = np.inner(other_lattice, rotation_matrix) and
            op = SymmOp.from_rotation_and_translation(rotation_matrix)
            aligned_matrix = op.operate_multi(latt.matrix)
            Finally, scale_matrix is the integer matrix that expresses
            aligned_matrix as a linear combination of this
            lattice, i.e., aligned_matrix = np.dot(scale_matrix, self.matrix)

            None is returned if no matches are found.
        """
        for x in self.find_all_mappings(other_lattice, turns, ltol, atol, skip_rotation_matrix=skip_rotation_matrix):
            return x

    def scale(self, new_volume, turns):
        """
        Return a new Lattice with volume new_volume by performing a
        scaling of the lattice vectors so that length proportions and angles
        are preserved.

        Args:
            new_volume:
                New volume to scale to.

        Returns:
            New lattice with desired volume.
        """
        versors = self.matrix / self.abc

        geo_factor = abs(np.dot(np.cross(versors[0], versors[1]), versors[2]))

        ratios = self.abc / self.c

        new_c = (new_volume / (geo_factor * np.prod(ratios))) ** (1 / 3.)

        return Domain(versors * (new_c * ratios), turns)

    def dot(self, coords_a, coords_b, frac_coords=False):
        """
        Compute the scalar product of vector(s).

        Args:
            coords_a, coords_b: Array-like objects with the coordinates.
            frac_coords (bool): Boolean stating whether the vector
                corresponds to fractional or cartesian coordinates.

        Returns:
            one-dimensional `numpy` array.
        """
        coords_a, coords_b = np.reshape(coords_a, (-1, 3)), \
                             np.reshape(coords_b, (-1, 3))

        if len(coords_a) != len(coords_b):
            raise ValueError("")

        if np.iscomplexobj(coords_a) or np.iscomplexobj(coords_b):
            raise TypeError("Complex array!")

        if not frac_coords:
            cart_a, cart_b = coords_a, coords_b
        else:
            cart_a = np.reshape([self.get_cartesian_coords(vec)
                                 for vec in coords_a], (-1, 3))
            cart_b = np.reshape([self.get_cartesian_coords(vec)
                                 for vec in coords_b], (-1, 3))

        return np.array([np.dot(a, b) for a, b in zip(cart_a, cart_b)])

    def norm(self, coords, frac_coords=True):
        """
        Compute the norm of vector(s).

        Args:
            coords:
                Array-like object with the coordinates.
            frac_coords:
                Boolean stating whether the vector corresponds to fractional or
                cartesian coordinates.

        Returns:
            one-dimensional `numpy` array.
        """
        return np.sqrt(self.dot(coords, coords, frac_coords=frac_coords))

    def get_points_in_sphere(self, frac_points, center, r, zip_results=True):
        """
        Find all points within a sphere from the point taking into account
        periodic boundary conditions. This includes sites in other periodic
        images.

        Algorithm:

        1. place sphere of radius r in crystal and determine minimum supercell
           (parallelpiped) which would contain a sphere of radius r. for this
           we need the projection of a_1 on a unit vector perpendicular
           to a_2 & a_3 (i.e. the unit vector in the direction b_1) to
           determine how many a_1"s it will take to contain the sphere.

           Nxmax = r * length_of_b_1 / (2 Pi)

        2. keep points falling within r.

        Args:
            frac_points: All points in the lattice in fractional coordinates.
            center: Cartesian coordinates of center of sphere.
            r: radius of sphere.
            zip_results (bool): Whether to zip the results together to group by
                 point, or return the raw fcoord, dist, index arrays

        Returns:
            if zip_results:
                [(fcoord, dist, index) ...] since most of the time, subsequent
                processing requires the distance.
            else:
                fcoords, dists, inds
        """
        recp_len = np.array(self.reciprocal_lattice_crystallographic.abc)
        nmax = float(r) * recp_len + 0.01

        pcoords = self.get_fractional_coords(center)
        center = np.array(center)

        n = len(frac_points)
        fcoords = np.array(frac_points) % 1
        indices = np.arange(n)

        mins = np.floor(pcoords - nmax)
        maxes = np.ceil(pcoords + nmax)
        arange = np.arange(start=mins[0], stop=maxes[0])
        brange = np.arange(start=mins[1], stop=maxes[1])
        crange = np.arange(start=mins[2], stop=maxes[2])
        arange = arange[:, None] * np.array([1, 0, 0])[None, :]
        brange = brange[:, None] * np.array([0, 1, 0])[None, :]
        crange = crange[:, None] * np.array([0, 0, 1])[None, :]
        images = arange[:, None, None] + brange[None, :, None] + \
                 crange[None, None, :]

        shifted_coords = fcoords[:, None, None, None, :] + \
                         images[None, :, :, :, :]

        cart_coords = self.get_cartesian_coords(fcoords)
        cart_images = self.get_cartesian_coords(images)
        coords = cart_coords[:, None, None, None, :] + \
                 cart_images[None, :, :, :, :]
        coords -= center[None, None, None, None, :]
        coords **= 2
        d_2 = np.sum(coords, axis=4)

        within_r = np.where(d_2 <= r ** 2)
        if zip_results:
            return list(zip(shifted_coords[within_r], np.sqrt(d_2[within_r]),
                            indices[within_r[0]]))
        else:
            return shifted_coords[within_r], np.sqrt(d_2[within_r]), \
                   indices[within_r[0]]

    def get_all_distances(self, fcoords1, fcoords2):
        """
        Returns the distances between two lists of coordinates taking into
        account periodic boundary conditions and the lattice. Note that this
        computes an MxN array of distances (i.e. the distance between each
        point in fcoords1 and every coordinate in fcoords2). This is
        different functionality from pbc_diff.

        Args:
            fcoords1: First set of fractional coordinates. e.g., [0.5, 0.6,
                0.7] or [[1.1, 1.2, 4.3], [0.5, 0.6, 0.7]]. It can be a single
                coord or any array of coords.
            fcoords2: Second set of fractional coordinates.

        Returns:
            2d array of cartesian distances. E.g the distance between
            fcoords1[i] and fcoords2[j] is distances[i,j]
        """
        # ensure correct shape
        fcoords1, fcoords2 = np.atleast_2d(fcoords1, fcoords2)

        # ensure that all points are in the unit cell
        fcoords1 = np.mod(fcoords1, 1)
        fcoords2 = np.mod(fcoords2, 1)

        # create images of f2
        shifted_f2 = fcoords2[:, None, :] + MIC_IMAGES[None, :, :]

        cart_f1 = self.get_cartesian_coords(fcoords1)
        cart_f2 = self.get_cartesian_coords(shifted_f2)

        if cart_f1.size * cart_f2.size < 1e5:
            # all vectors from f1 to f2
            vectors = cart_f2[None, :, :, :] - cart_f1[:, None, None, :]
            d_2 = np.sum(vectors ** 2, axis=3)
            distances = np.min(d_2, axis=2) ** 0.5
            return distances
        else:
            # memory will overflow, so do a loop
            distances = []
            for c1 in cart_f1:
                vectors = cart_f2[:, :, :] - c1[None, None, :]
                d_2 = np.sum(vectors ** 2, axis=2)
                distances.append(np.min(d_2, axis=1) ** 0.5)
            return np.array(distances)


    def is_hexagonal(self, hex_angle_tol=5, hex_length_tol=0.01):
        lengths, angles = self.lengths_and_angles
        right_angles = [i for i in range(3)
                        if abs(angles[i] - 90) < hex_angle_tol]
        hex_angles = [i for i in range(3)
                      if abs(angles[i] - 60) < hex_angle_tol or
                      abs(angles[i] - 120) < hex_angle_tol]

        return (len(right_angles) == 2 and len(hex_angles) == 1
                and abs(lengths[right_angles[0]] -
                        lengths[right_angles[1]]) < hex_length_tol)


    def get_all_distance_and_image(self, frac_coords1, frac_coords2):
        """
        Gets distance between two frac_coords and nearest periodic images.

        Args:
            fcoords1 (3x1 array): Reference fcoords to get distance from.
            fcoords2 (3x1 array): fcoords to get distance from.

        Returns:
            [(distance, jimage)] List of distance and periodic lattice
            translations of the other site for which the distance applies.
            This means that the distance between frac_coords1 and (jimage +
            frac_coords2) is equal to distance.
        """
        # The following code is heavily vectorized to maximize speed.
        # Get the image adjustment necessary to bring coords to unit_cell.
        adj1 = np.floor(frac_coords1)
        adj2 = np.floor(frac_coords2)
        # Shift coords to unitcell
        coord1 = frac_coords1 - adj1
        coord2 = frac_coords2 - adj2
        # Generate set of images required for testing.
        # This is a cheat to create an 8x3 array of all length 3
        # combinations of 0,1

        # Create tiled cartesian coords for computing distances.
        vec = np.tile(coord2 - coord1, (len(MIC_IMAGES), 1)) + MIC_IMAGES
        vec = self.get_cartesian_coords(vec)
        # Compute distances manually.
        dist = np.sqrt(np.sum(vec ** 2, 1)).tolist()
        return list(zip(dist, adj1 - adj2 + MIC_IMAGES))

class Torus(Domain):

    def __init__(self, matrix, turns):
        lengths = np.sqrt(np.sum(matrix ** 2, axis=1))
        angles = np.zeros(3)

        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            angles[i] = abs_cap(dot(matrix[j], matrix[k]) / (lengths[j] * lengths[k]))

        self._turn_num = turns
        self._angles = np.arccos(angles) * 180. / pi
        self._lengths = lengths
        self._matrix = matrix
        # The inverse matrix is lazily generated for efficiency.
        self._inv_matrix = None
        self._metric_tensor = None
