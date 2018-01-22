
cd src
python3 test_w2v.py -t $1 -w data/w2v_model/w2v_dim128_lw3_ls1.bin  data/w2v_model/w2v_dim32_lw3_ls1.bin data/w2v_model/w2v_dim64_lw3_ls1_n27.bin data/w2v_model/w2v_dim64_lw3_ls1_n47.bin -o $2

