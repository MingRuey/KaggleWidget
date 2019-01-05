"""
Created on Jan. 04. 2019
@author: MRChou

Scenario: define the helper classes for (time) series data
"""

import pyarrow.parquet as pq


class ParquetArray:
    """ Used to retrieve desired series from parquet dataset

    Example:

        Basic usage and support:
        >>> ps = ParquetArray('OneToTen.parquet') # Data is NOT actually loaded

        >>> ps.[0] # Select single column, with data actually loaded
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        >>> ps[0, 1] # Select multiple columns
        array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
               [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])

        >>> ps[0:5] # direct slice is doing slicing on rows
        array([[0, 1, 2, 3, 4],
               [0, 1, 2, 3, 4]])

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
        self._file = pq.ParquetFile(parquetfile)
        self._colnames = self._file.schema.names

    @property
    def colnames(self):
        return self._colnames

    @property
    def num_of_cols(self):
        return len(self._colnames)

    def __getitem__(self, index):
        """note: index are selecting columns while direct slice is slicing rows"""
        if isinstance(index, slice):
            return self._file.read().to_pandas().__getitem__(index)
        else:
            # chekc if all indices are valid column names
            if set(index) <= set(self._colnames):
                return self._file.read(columns=index).to_pandas()

            # if not, try to read indices as column indices
            else:
                try:
                    return self._file.read(columns=[self._colnames[i] for i in list(index)]).to_pandas()
                except TypeError:
                    msg = "invalid column names, checkout .colnames (get %s)"
                    raise TypeError(msg % index)
                except IndexError:
                    msg = "invalid column ranges, checkout .num_of_cols (get %s)"
                    raise IndexError(msg % index)


if __name__ == "__main__":
    pass
