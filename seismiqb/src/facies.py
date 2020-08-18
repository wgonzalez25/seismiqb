""" Contains facies-related functions. """
import os
from copy import copy
from textwrap import dedent

import numpy as np
import pandas as pd

from scipy.ndimage import find_objects
from skimage.measure import label

from ..batchflow import HistoSampler

from .plotters import plot_image
from .utils import groupby_min, groupby_max



class GeoBody:
    """ Container for a 3D object inside seismic volume.
    Described by two matrices: upper and lower boundary.

    Main storages are `matrix_1` and `matrix_2`, which are upper and lower boundary depth maps respectively, and
    `points`, which is an array of (N, 4) shape: iline, crossline, upper and lower points.
    """

    # Custom facies format: spatial point (iline and xline), upper and bottom point of a body (heights 1 and 2)
    FACIES_SPEC = ['iline', 'xline', 'height_1', 'height_2']

    # Columns that are used from the file
    COLUMNS = ['iline', 'xline', 'height_1', 'height_2']

    # Value to place into blank spaces
    FILL_VALUE = -999999


    def __init__(self, storage, geometry, name=None, **kwargs):
        # Meta information
        self.path = None
        self.name = name
        self.format = None

        # Location of the geobody inside cube spatial range
        self.i_min, self.i_max = None, None
        self.x_min, self.x_max = None, None
        self.i_length, self.x_length = None, None
        self.bbox = None
        self._len = None

        # Underlying data storages
        self.matrix_1, self.matrix_2 = None, None
        self.points = None

        # Depth stats
        self.h_min_1, self.h_min_2 = None, None
        self.h_max_1, self.h_max_2 = None, None
        self.h_mean_1, self.h_mean_2 = None, None
        self.h_mean = None

        # Attributes from geometry
        self.geometry = geometry
        self.cube_name = geometry.name
        self.cube_shape = geometry.cube_shape

        self.sampler = None

        # Check format of storage, then use it to populate attributes
        if isinstance(storage, str):
            # path to csv-like file
            self.format = 'file'

        elif isinstance(storage, np.ndarray):
            if storage.ndim == 2 and storage.shape[1] == 4:
                # array with row in (iline, xline, height) format
                self.format = 'points'
            else:
                raise NotImplementedError

        getattr(self, 'from_{}'.format(self.format))(storage, **kwargs)


    def __len__(self):
        return len(self.points)


    # Coordinate transforms
    def lines_to_cubic(self, array):
        """ Convert ilines-xlines to cubic coordinates system. """
        array[:, 0] -= self.geometry.ilines_offset
        array[:, 1] -= self.geometry.xlines_offset
        array[:, 2:] -= self.geometry.delay
        array[:, 2:] /= self.geometry.sample_rate
        return array

    def cubic_to_lines(self, array):
        """ Convert cubic coordinates to ilines-xlines system. """
        array = array.astype(float)
        array[:, 0] += self.geometry.ilines_offset
        array[:, 1] += self.geometry.xlines_offset
        array[:, 2:] *= self.geometry.sample_rate
        array[:, 2:] += self.geometry.delay
        return array


    # Initialization from different containers
    def from_points(self, points, transform=False, verify=True, **kwargs):
        """ Base initialization: from point cloud array of (N, 4) shape.

        Parameters
        ----------
        points : ndarray
            Array of points. Each row describes one point inside the cube: iline, crossline,
            upper and lower depth points.
        transform : bool
            Whether transform from line coordinates (ilines, xlines) to cubic system.
        verify : bool
            Whether to remove points outside of the cube range.
        """
        _ = kwargs

        # Transform to cubic coordinates, if needed
        if transform:
            points = self.lines_to_cubic(points)
        if verify:
            idx = np.where((points[:, 0] >= 0) &
                           (points[:, 1] >= 0) &
                           (points[:, 2] >= 0) &
                           (points[:, 3] >= 0) &
                           (points[:, 0] < self.cube_shape[0]) &
                           (points[:, 1] < self.cube_shape[1]) &
                           (points[:, 2] < self.cube_shape[2]) &
                           (points[:, 3] < self.cube_shape[2]))[0]
            points = points[idx]
        self.points = np.rint(points).astype(np.int32)

        # Collect stats on separate axes. Note that depth stats are properties
        self.i_min, self.x_min, self.h_min_1, self.h_min_2 = np.min(self.points, axis=0).astype(np.int32)
        self.i_max, self.x_max, self.h_max_1, self.h_max_2 = np.max(self.points, axis=0).astype(np.int32)
        self.h_mean_1, self.h_mean_2 = np.mean(self.points[:, 2:], axis=0)
        self.h_mean = np.mean(self.points[:, 2:])

        self.i_length = (self.i_max - self.i_min) + 1
        self.x_length = (self.x_max - self.x_min) + 1
        self.bbox = np.array([[self.i_min, self.i_max],
                              [self.x_min, self.x_max]],
                             dtype=np.int32)

        self.matrix_1, self.matrix_2 = self.points_to_matrix(self.points,
                                                             self.i_min, self.x_min,
                                                             self.i_length, self.x_length)


    def from_file(self, path, transform=True, **kwargs):
        """ Init from path to csv-like file. """
        _ = kwargs

        self.path = path
        self.name = os.path.basename(path)
        points = self.file_to_points(path)
        self.from_points(points, transform)

    def file_to_points(self, path):
        """ Get point cloud array from file values. """
        #pylint: disable=anomalous-backslash-in-string
        with open(path) as file:
            line_len = len(file.readline().split(' '))
        if line_len == 4:
            names = GeoBody.FACIES_SPEC
        else:
            raise ValueError('GeoBody labels must be in FACIES_SPEC format.')

        df = pd.read_csv(path, sep='\s+', names=names, usecols=GeoBody.COLUMNS)
        df.sort_values(GeoBody.COLUMNS, inplace=True)
        return df.values

    @staticmethod
    def points_to_matrix(points, i_min, x_min, i_length, x_length):
        """ Convert array of (N, 4) shape to a pair of depth maps (upper and lower boundaries of geobody). """
        matrix_1 = np.full((i_length, x_length), GeoBody.FILL_VALUE, np.int32)
        matrix_1[points[:, 0] - i_min, points[:, 1] - x_min] = points[:, 2]

        matrix_2 = np.full((i_length, x_length), GeoBody.FILL_VALUE, np.int32)
        matrix_2[points[:, 0] - i_min, points[:, 1] - x_min] = points[:, 3]
        return matrix_1, matrix_2


    @staticmethod
    def from_mask(mask, grid_info=None, geometry=None, shifts=None,
                  threshold=0.5, minsize=0, prefix='predict', **kwargs):
        """ Convert mask to a list of geobodies. Returned list is sorted on length (number of points).

        Parameters
        ----------
        grid_info : dict
            Information about mask creation parameters. Required keys are `geom` and `range`
            to infer geometry and leftmost upper point, or they can be passed directly.
        threshold : float
            Parameter of mask-thresholding.
        minsize : int
            Minimum number of points in a geobody to be saved.
        prefix : str
            Name of geobody to use.
        """
        _ = kwargs
        if grid_info is not None:
            geometry = grid_info['geom']
            shifts = np.array([item[0] for item in grid_info['range']])

        if geometry is None or shifts is None:
            raise TypeError('Pass `grid_info` or `geometry` and `shifts`.')

        # Labeled connected regions with an integer
        labeled = label(mask >= threshold)
        objects = find_objects(labeled)

        # Create an instance of GeoBody for each separate region
        geobodies = []
        for i, sl in enumerate(objects):
            max_possible_length = 1
            for j in range(3):
                max_possible_length *= sl[j].stop - sl[j].start

            if max_possible_length >= minsize:
                indices = np.nonzero(labeled[sl] == i + 1)

                if len(indices[0]) >= minsize:
                    coords = np.vstack([indices[i] + sl[i].start for i in range(3)]).T

                    points_min = groupby_min(coords) + shifts
                    points_max = groupby_max(coords) + shifts
                    points = np.hstack([points_min, points_max[:, -1].reshape(-1, 1)])
                    geobodies.append(GeoBody(points, geometry, name=f'{prefix}_{i}'))

        geobodies.sort(key=len)
        return geobodies



    def filter(self, *args, **kwargs):
        """ Remove points outside of the cube data. Yet to be implemented. """
        _ = args, kwargs



    # GeoBody usage: point/mask generation
    def create_sampler(self, bins=None, **kwargs):
        """ Create sampler based on the upper boundary of a geobody.

        Parameters
        ----------
        bins : sequence
            Size of ticks alongs each respective axis.
        """
        _ = kwargs
        default_bins = self.cube_shape // np.array([5, 20, 20])
        bins = bins if bins is not None else default_bins

        self.sampler = HistoSampler(np.histogramdd(self.points[:, :3]/self.cube_shape, bins=bins))


    def add_to_mask(self, mask, locations=None, alpha=1, **kwargs):
        """ Add geobody to a background.
        Note that background is changed in-place.

        Parameters
        ----------
        mask : ndarray
            Background to add to.
        locations : ndarray
            Where the mask is located.
        """
        _ = kwargs

        mask_bbox = np.array([[slc.start, slc.stop] for slc in locations], dtype=np.int32)

        # Getting coordinates of overlap in cubic system
        (mask_i_min, mask_i_max), (mask_x_min, mask_x_max), (mask_h_min, mask_h_max) = mask_bbox
        i_min, i_max = max(self.i_min, mask_i_min), min(self.i_max + 1, mask_i_max)
        x_min, x_max = max(self.x_min, mask_x_min), min(self.x_max + 1, mask_x_max)


        if i_max >= i_min and x_max >= x_min:
            overlap_1 = self.matrix_1[i_min - self.i_min:i_max - self.i_min,
                                      x_min - self.x_min:x_max - self.x_min]
            overlap_2 = self.matrix_2[i_min - self.i_min:i_max - self.i_min,
                                      x_min - self.x_min:x_max - self.x_min]

            # Coordinates of points to use in overlap local system
            idx_i_1, idx_x_1 = np.asarray((overlap_1 != self.FILL_VALUE) &
                                          (overlap_1 >= mask_h_min) &
                                          (overlap_1 <= mask_h_max)).nonzero()

            idx_i_2, idx_x_2 = np.asarray((overlap_2 != self.FILL_VALUE) &
                                          (overlap_2 >= mask_h_min) &
                                          (overlap_2 <= mask_h_max)).nonzero()

            set_1 = set(zip(idx_i_1, idx_x_1))
            set_2 = set(zip(idx_i_2, idx_x_2))

            set_union = set_1 | set_2
            idx_union = np.array(tuple(set_union))
            if len(idx_union) > 0:
                idx_i, idx_x = idx_union[:, 0], idx_union[:, 1]

                heights_1 = overlap_1[idx_i, idx_x]
                heights_2 = overlap_2[idx_i, idx_x]

                # Convert coordinates to mask local system
                idx_i += i_min - mask_i_min
                idx_x += x_min - mask_x_min
                heights_1 -= mask_h_min
                heights_2 -= mask_h_min

                max_depth = mask.shape[-1] - 1
                heights_1[heights_1 < 0] = 0
                heights_1[heights_1 > max_depth] = max_depth
                heights_2[heights_2 < 0] = 0
                heights_2[heights_2 > max_depth] = max_depth

                n = (heights_2 - heights_1).max()
                for _ in range(n + 1):
                    mask[idx_i, idx_x, heights_1] = alpha
                    heights_1 += 1

                    mask_ = heights_1 <= heights_2
                    idx_i = idx_i[mask_]
                    idx_x = idx_x[mask_]
                    heights_1 = heights_1[mask_]
                    heights_2 = heights_2[mask_]
        return mask


    # Properties
    @property
    def full_matrix_1(self):
        """ Matrix in cubic coordinate system. """
        return self.put_on_full(self.matrix_1)

    @property
    def full_matrix_2(self):
        """ Matrix in cubic coordinate system. """
        return self.put_on_full(self.matrix_2)


    def dump(self, path, transform=None, add_height=True):
        """ Save geobody points on disk.

        Parameters
        ----------
        path : str
            Path to a file to save to.
        transform : None or callable
            If callable, then applied to points after converting to ilines/xlines coordinate system.
        add_height : bool
            Whether to concatenate average height to a file name.
        """
        values = self.cubic_to_lines(copy(self.points))
        values = values if transform is None else transform(values)

        df = pd.DataFrame(values, columns=self.COLUMNS)
        df.sort_values(['iline', 'xline'], inplace=True)

        path = path if not add_height else f'{path}_#{self.h_mean}'
        df.to_csv(path, sep=' ', columns=self.COLUMNS, index=False, header=False)


    # Methods of (visual) representation of a geobody
    def __repr__(self):
        return f"""<geobody {self.name} for {self.cube_name} at {hex(id(self))}>"""

    def __str__(self):
        msg = f"""
        GeoBody {self.name} for {self.cube_name} loaded from {self.format}
        Ilines range:      {self.i_min} to {self.i_max}
        Xlines range:      {self.x_min} to {self.x_max}
        Depth range:       {self.h_min_1} to {self.h_max_2}
        Depth mean:        {self.h_mean:.6}

        Length:            {len(self)}
        """
        return dedent(msg)

    @property
    def centers(self):
        """ Midpoints between upper and lower boundaries. """
        return (self.matrix_1 + self.matrix_2) // 2


    def put_on_full(self, matrix=None, fill_value=None):
        """ Create a matrix in cubic coordinate system. """
        fill_value = fill_value or self.FILL_VALUE

        background = np.full(self.cube_shape[:-1], fill_value, dtype=np.float32)
        background[self.i_min:self.i_max+1, self.x_min:self.x_max+1] = matrix
        return background

    def show(self, src='centers', fill_value=None, on_full=True, **kwargs):
        """ Nice visualization of a depth map matrix. """
        matrix = getattr(self, src) if isinstance(src, str) else src
        fill_value = fill_value or self.FILL_VALUE

        if on_full:
            matrix = self.put_on_full(matrix=matrix, fill_value=fill_value)
        else:
            matrix = copy(matrix).astype(np.float32)

        # defaults for plotting if not supplied in kwargs
        kwargs = {
            'cmap': 'viridis_r',
            'title': '{} {} of `{}` on `{}`'.format(src if isinstance(src, str) else '',
                                                    'on full'*on_full, self.name, self.cube_name),
            'xlabel': self.geometry.index_headers[0],
            'ylabel': self.geometry.index_headers[1],
            **kwargs
            }
        matrix[matrix == fill_value] = np.nan
        plot_image(matrix, mode='single', **kwargs)

    def show_slide(self, loc, width=3, axis='i', order_axes=None, zoom_slice=None, **kwargs):
        """ Show slide with geobody on it.

        Parameters
        ----------
        loc : int
            Number of slide to load.
        axis : int
            Number of axis to load slide along.
        stable : bool
            Whether or not to use the same sorting order as in the segyfile.
        """
        # Make `locations` for slide loading
        axis = self.geometry.parse_axis(axis)
        locations = self.geometry.make_slide_locations(loc, axis=axis)
        shape = np.array([(slc.stop - slc.start) for slc in locations])

        # Load seismic and mask
        seismic_slide = self.geometry.load_slide(loc=loc, axis=axis)
        mask = np.zeros(shape)
        mask = self.add_to_mask(mask, locations=locations, width=width)
        seismic_slide, mask = np.squeeze(seismic_slide), np.squeeze(mask)
        xticks = list(range(seismic_slide.shape[0]))
        yticks = list(range(seismic_slide.shape[1]))

        if zoom_slice:
            seismic_slide = seismic_slide[zoom_slice]
            mask = mask[zoom_slice]
            xticks = xticks[zoom_slice[0]]
            yticks = yticks[zoom_slice[1]]

        # defaults for plotting if not supplied in kwargs
        if axis in [0, 1]:
            header = self.geometry.index_headers[axis]
            xlabel = self.geometry.index_headers[1 - axis]
            ylabel = 'depth'
            total = self.geometry.lens[axis]
        if axis == 2:
            header = 'Depth'
            xlabel = self.geometry.index_headers[0]
            ylabel = self.geometry.index_headers[1]
            total = self.geometry.depth

        kwargs = {
            'mode': 'overlap',
            'opacity': 0.25,
            'title': (f'GeoBody `{self.name}` on `{self.geometry.name}`' +
                      f'\n {header} {loc} out of {total}'),
            'xlabel': xlabel,
            'ylabel': ylabel,
            'xticks': xticks[::max(1, round(len(xticks)//8/100))*100],
            'yticks': yticks[::max(1, round(len(yticks)//10/100))*100][::-1],
            'y': 1.02,
            **kwargs
            }

        plot_image([seismic_slide, mask], order_axes=order_axes, **kwargs)
