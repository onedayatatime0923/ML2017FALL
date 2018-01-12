
mkdir data_preprocess; mkdir models    
cd data_preprocess
wget 'https://www.dropbox.com/s/t6zdaiw6etc12tb/word2vec_model.bin?dl=1' -O word2vec_model.bin
cd ../models
wget 'https://www.dropbox.com/s/szgqczuygv8xcgz/model1.hdf5?dl=1' -O model1.hdf5
wget 'https://www.dropbox.com/s/6cyn962kdyoopg4/model4.hdf5?dl=1' -O model4.hdf5
cd ..
python3.6 setdata.py --testfile $1
python3.6 sequence2matrix.py
python3.6 continue.py --model models/model1.hdf5 -o test1.npy
python3.6 continue.py --model models/model4.hdf5 -o test2.npy
python3.6 write_out.py --test test1.npy test2.npy -o $2
