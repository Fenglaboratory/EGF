# py_qcprot
A cython-based wrapper of the QCP C-code for RMSD calculation

# About

This repository contains slightly modified C-code from http://theobald.brandeis.edu/QCP/ 
and a python wrapper using cython.

QCP is a very fast way to calculate the minimal RMSD between two point-clouds of the same 
length and order (e.g. Calpha backbone atom positions of proteins).

The cython wrapper copies the input data into a memory layout supported by the original qcprot
code. Despite this overhead, it is 10 times faster than our reference implementation of the Kabsch algorithm (in test.py).
Users who only use numpy arrays with defined memory layout,
might profit from adapting the code to avoid the overhead (only changes in py_qcprot.py are needed for 
F-contiguouse arrays, but changes in the cython and c code are needed for C-contiguouse arrays.)

# Authors

The original C-code was written by Pu Liu and Douglas Theobald (See citations below) 
and is licensed under a BSD 3-clause license.
Bernhard Thiel contributed the cython wrapper and tiny modifications to the C-code.

# Citation

Citations

Liu P, Agrafiotis DK, & Theobald DL (2011)
Reply to comment on: "Fast determination of the optimal rotation matrix for macromolecular superpositions."
Journal of Computational Chemistry 32(1):185-186. [Open Access], doi:10.1002/jcc.21606

Liu P, Agrafiotis DK, & Theobald DL (2010)
"Fast determination of the optimal rotation matrix for macromolecular superpositions."
Journal of Computational Chemistry 31(7):1561-1563. [Open Access] doi:10.1002/jcc.21439

Douglas L Theobald (2005)
"Rapid calculation of RMSDs using a quaternion-based characteristic polynomial."
Acta Crystallogr A 61(4):478-480. [Open Access] doi:10.1107/S0108767305015266 

# See also:
See https://github.com/charnley/rmsd for pure python/ numpy based RMSD calculations
using the Kabsch algorithm or doi:10.1016/1049-9660(91)90036-O

