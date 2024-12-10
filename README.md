# TSUCAE
# Temperature scaling unmixing framework based on convolutional autoencoder
The code in this toolbox implements the "Temperature scaling unmixing framework based on convolutional autoencoder".
# System-specific notes
The code was tested in the environment of `Python 3.9.16`, `torch 1.13.1` and `CUDA 11.6`.
# Citation
```
Xu, J., Xu, M., Liu, S., Sheng, H., Yang, Z., 2024. Temperature scaling unmixing framework based on convolutional autoencoder. Int. J. Appl. Earth Obs. Geoinf. 129, 103864.
```
# Description of the DC data set
The DC dataset and its similarity matrix, which cannot be uploaded directly because the file size exceeds 25mb, are stored in the `Release` on the right side of the web page.
# Description of the similarity matrix in the code
The closer the cosine distance is to 1, the greater the similarity value. However, in this version of the code, the cosine distance is subtracted by 1 and multiplied by 1000, initially to better observe the difference between similarities. At this point in the code, the smaller the value in the similarity matrix, the more similar it is. You can operate the similarity matrix as follows: similarity_matrix=1-similarity_matrix/1000; And change the mapping interval to: Mapping_ranges = [(2, 1.8),(1.8, 1.6),(1.6, 1.4),(1.4, 1.2),(1.2, 1), (1, 0.8),(0.8, 0.6),(0.6, 0.4),(0.4, 0.2),(0.2, 0.01)]. This corresponds to the explanation in the paper. It's the same in theory. The two are only different in operation, and there is no essential difference.
# Contact
s23160009@s.upc.edu.cn
# Related work
https://github.com/UPCGIT/MsCM-Net  
https://github.com/UPCGIT/Hyperspectral-Unmixing-Models
