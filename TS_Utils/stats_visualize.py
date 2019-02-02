"""
Created on 1/6/19
@author: MRChou

Scenario: visualization of various statistic & analysis for series data
"""

import scipy
import seaborn as sns
import matplotlib.pyplot as plt


class RawPlot:

    def __init__(self, parquetarray, xscale=1.0, xunit=None, yscale=1.0, yunit=None):
        self.df = parquetarray.to_dataframe()
        self.units = (xscale, xunit, yscale, yunit)

    def show(self):
        y_lims = []
        y_ranges = []
        for col in self.df:
            lims = (self.df[col].min(), self.df[col].max())
            y_lims.append(lims)
            y_ranges.append(int(lims[1]) - int(lims[0]))

        fig, axes = plt.subplots(nrows=self.df.shape[1], ncols=1,
                                 sharex='all', gridspec_kw={'height_ratios': y_ranges},
                                 squeeze=False)

        for index, col in enumerate(self.df):
            axes[index, 0].scatter(x=self.df.index.values*self.units[0],
                                   y=self.df[col].values*self.units[2],
                                   s=0.1)
            axes[index, 0].set_ylim(y_lims[index][0] - y_ranges[index]*0.1,
                                    y_lims[index][1] + y_ranges[index]*0.1)
            axes[index, 0].text(0.1, 0.9,
                                'dataID: ' + col,
                                horizontalalignment='left',
                                transform=axes[index, 0].transAxes)

        if self.units[1]:
            plt.xlabel(self.units[1])
        if self.units[3]:
            plt.ylabel(self.units[3])
        plt.show()


if __name__ == "__main__":

    from TS_Utils.seriesObjs import ParquetArray

    arr = ParquetArray('/rawdata/VSB_Power/train.parquet')
    arr.set_rowscale(0.02, 'sec, 20ms per cycle')
    arr.set_dataunit('voltage')
    data = arr[1, 2]

    plot = RawPlot(data, xscale=arr.row_scale, xunit=arr.row_unit, yscale=1.0, yunit=arr.dataunit)
    plot.show()
