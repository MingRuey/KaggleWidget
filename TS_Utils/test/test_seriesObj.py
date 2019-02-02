"""
Created on 1/8/19
@author: MRChou

Scenario: test cases for seriesObjs.py
"""

import os

import numpy
import pandas
import pyarrow as pa
import pyarrow.parquet as pq

from TS_Utils.seriesObjs import ParquetArray

DATA = pandas.DataFrame({'1': [i for i in range(10)],
                         '2': [i*0.1 for i in range(10)],
                         '3': [str(i) for i in range(10)]},
                        index=[chr(i+65) for i in range(10)])
FILE = 'TS_Utils_test.parquet'


def init():
    pq.write_table(pa.Table.from_pandas(DATA), FILE)


def clean():
    if os.path.exists(FILE):
        os.remove(FILE)


def test_createobj():
    obj = ParquetArray(FILE)
    assert obj.num_of_cols == 3
    assert obj.colnames == list(DATA.columns)
    assert obj.row_scale is None
    assert obj.row_unit is None


def test_setunit():
    obj = ParquetArray(FILE)
    try:
        obj.set_rowscale('CannotBeString', 'OK')
    except ValueError:
        pass

    try:
        obj.set_rowscale(1, 1)
    except ValueError:
        pass

    try:
        obj.set_rowscale(-1.5, 'OK')
    except ValueError:
        pass

    obj.set_rowscale(1.5, 'OK')
    assert obj.row_scale == 1.5
    assert obj.row_unit == 'OK'


def test_slicing():
    obj = ParquetArray(FILE)
    assert numpy.array_equal(obj['2'].to_array(), DATA[['2']].values)
    assert numpy.array_equal(obj['1', '3'].to_array(), DATA[['1', '3']].values)
    assert numpy.array_equal(obj[1].to_array(), DATA[['2']].values)
    assert numpy.array_equal(obj[0, 2].to_array(), DATA[['1', '3']].values)

    try:
        obj['1', '4']
    except TypeError:
        pass

    try:
        obj[1, 4]
    except IndexError:
        pass


if __name__ == "__main__":

    class test:

        cases = [test_createobj, test_setunit, test_slicing]

        def __enter__(self):
            init()
            return self

        @staticmethod
        def go():
            for case in test.cases:
                case.__call__()

        def __exit__(self, exc_type, exc_val, exc_tb):
            clean()

    with test() as t:
        t.go()