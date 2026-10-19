# -*- coding: utf-8 -*-
"""
Main entry point for PSO-TCWavKAN v9.1.
Usage:
    python main.py                  # runs fast demo
    python main.py --mode demo      # runs fast demo
    python main.py --help           # view all options
"""
import runpy

if __name__ == '__main__':
    runpy.run_module('pso_tcwavkan_v91', run_name='__main__')
