import sys
import skyclusters
from skycompare import NodeComparison
import json
import cPickle as pkl
from multiprocessing import Pool, Process, RawArray, Queue
from skypings import Pings
import numpy as np
import itertools
from experimentdata import DataGetter
import clusterAnalysis
import skyresolvers
import pandas
from matplotlib import pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import time
from matplotlib import cm
from collections import defaultdict


def plot_geo_distribution():
    pass


def plot_cnre_distribution():
    pass


def plot_perf_distribution():
    pass


def plot_perfdfm_vs_geodfm():
    pass


def plot_perfdfm_vs_cnredfm():
    pass


def plot_geodfm_vs_cnredfm():
    pass


if __name__ == '__main__':
    g_ca = ClusterAnalysis()

