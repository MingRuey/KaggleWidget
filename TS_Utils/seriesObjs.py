"""
Created on Jan. 04. 2019
@author: MRChou

Scenario: define the helper classes for (time) series data
"""

from numbers import Integral
from copy import deepcopy

import tensorflow as tf
import pyarrow.parquet as pq


class ParquetArray:
    """ Used to retrieve desired series from parquet dataset

    Example:

        Basic usage and support:
        >>> ps = ParquetArray('OneToTen.parquet') # Data is NOT actually loaded

        >>> ps[0].to_array() # Select single column, with data actually loaded
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        >>> ps[0, 1].to_array() # Select multiple columns
        array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
               [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])

        >>> ps[0:].to_array() # or column slicing
        array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
               [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])

        Processing:
        >>> # Smoothing
        >>> # FFT

        Plotting:
        >>> # ...

    """

    def __init__(self, parquetfile):
        """
        Args:
            parquetfile: string specify the location of parquet file
        """
        self._file = parquetfile
        self._colnames = pq.ParquetFile(self._file).schema.names
        # when parquet has row indices
        if self._colnames[-1] == '__index_level_0__':
            self._colnames = self._colnames[:-1]
        self._rowscale = None
        self._rowunit = None
        self._dataunit = None

    @property
    def colnames(self):
        return self._colnames

    @property
    def num_of_cols(self):
        return len(self._colnames)

    @property
    def row_scale(self):
        return self._rowscale

    @property
    def row_unit(self):
        return self._rowunit

    @property
    def dataunit(self):
        return self._dataunit

    def set_rowscale(self, scale, unit):
        """Set real world measurement unit for row indices

        Args:
            scale: float, the conversion rate between row indices and real world unit, i.e. index * scale = unit
            unit: string, the name of the unit
        """
        try:
            scale = float(scale)
            unit = str(unit)
        except ValueError:
            raise ValueError('Can not convert scale, unit to float, string')
        else:
            if scale <= 0:
                raise ValueError('scale must > 0')
            self._rowscale = scale
            self._rowunit = unit

    def set_dataunit(self, unit):
        self._dataunit = str(unit)

    def __getitem__(self, index):
        newobj = deepcopy(self)
        if isinstance(index, slice) or isinstance(index, Integral):
            newobj._colnames = self._colnames.__getitem__(index)
            return newobj

        if isinstance(index, str):
            index = [index]
        else:
            index = list(index)

        # chekc if all indices are valid column names
        if set(index) <= set(self._colnames):
            newobj._colnames = index
        else:
            try:
                newobj._colnames = [self._colnames[i] for i in index]
            except TypeError:
                msg = "invalid column names, checkout .colnames (get %s)"
                raise TypeError(msg % index)
            except IndexError:
                msg = "invalid column ranges, checkout .num_of_cols (get %s)"
                raise IndexError(msg % index)
        return newobj

    def __iter__(self):
        array = self.to_array()
        for index, col in enumerate(self.colnames):
            yield (col, array[:, index])

    def to_dataframe(self):
        return pq.ParquetFile(self._file).read(columns=self._colnames, use_pandas_metadata=True).to_pandas()

    def to_array(self):
        return self.to_dataframe().values

    def to_dataset(self):
        ds = tf.data.Dataset.from_generator(self.__iter__, (tf.string, tf.float32))
        return ds


if __name__ == "__main__":
    pass
