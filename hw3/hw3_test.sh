python3.6 set_test_data.py $1
python3.6 continue.py weights/model0.1.hdf5 1.npy
python3.6 continue.py weights/model0.3.hdf5 2.npy
python3.6 continue.py weights/model0.5.hdf5 3.npy
python3.6 continue.py weights/weights-7Cnn-nodropout.hdf5 4.npy
python3.6 continue.py weights/weights-7Cnn-0.1.hdf5 5.npy
python3.6 continue.py weights/weights-7Cnn-0.3.hdf5 6.npy
python3.6 continue.py weights/weights-7Cnn-0.5.hdf5 7.npy
python3.6 continue.py weights/weights-drop_out-0.1.hdf5 8.npy
python3.6 continue.py weights/weights-drop_out-0.3.hdf5 9.npy
python3.6 continue.py weights/weights-drop_out-0.5.hdf5 10.npy
python3.6 sum.py 1.npy 2.npy 3.npy 4.npy 5.npy 6.npy 7.npy 8.npy 9.npy 10.npy
python3.6 write_out.py test_total.npy $2
