#!/bin/bash
#coding:utf-8

python3 test_simple_avg.py $1 data/w2v_model/traditional_size32_linewindow3_stride1_wordwindow7 32 ans/traditional_size32_linewindow3_stride1_wordwindow7.csv
python3 test_simple_avg.py $1 data/w2v_model/traditional_size64_linewindow3_stride1_wordwindow7 64 ans/traditional_size64_linewindow3_stride1_wordwindow7.csv
python3 test_simple_avg.py $1 data/w2v_model/traditional_size128_linewindow3_stride1_wordwindow7 128 ans/traditional_size128_linewindow3_stride1_wordwindow7.csv
python3 test_simple_avg.py $1 data/w2v_model/w2v_dim32_lw3_ls1.bin 32 ans/w2v32.csv
python3 test_simple_avg.py $1 data/w2v_model/w2v_dim64_lw3_ls1.bin 64 ans/w2v64.csv
python3 test_simple_avg.py $1 data/w2v_model/w2v_dim128_lw3_ls1.bin 128 ans/w2v128.csv

python3 ensemble.py ans/traditional_size32_linewindow3_stride1_wordwindow7.csv ans/traditional_size64_linewindow3_stride1_wordwindow7.csv ans/traditional_size128_linewindow3_stride1_wordwindow7.csv ans/w2v32.csv ans/w2v64.csv ans/w2v128.csv $2
