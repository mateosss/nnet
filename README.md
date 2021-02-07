# Run

python setup.py build_ext --inplace
export OMP_PLACES=cores OMP_PROC_BIND=close CC=gcc-10 OMP_NUM_THREADS=4
python main.py

<!-- TODO: how to run py, np and cy versions -->
<!-- TODO: Add note on parallelism and cython "does the job" -->
<!-- TODO: Add note on how to run tests -->
<!-- TODO: Mention reason for weird backprop -->
<!-- The idea of this project is to have a from the ground up implementation to write plug and play algorithms to test out ideas. -->
