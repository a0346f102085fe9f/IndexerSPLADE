## IAS2 to IAS2.1 index conversion

```sh
mkdir ponebin
cd ponebin/
mkdir 1
cd 1
wget https://bafybeibhqchj6vkypxxuy3xwonp2e4f27g5ilupbm2wphs22ycgt2ky6si.ipfs.w3s.link/IndexA/datatape_k.bin https://bafybeibhqchj6vkypxxuy3xwonp2e4f27g5ilupbm2wphs22ycgt2ky6si.ipfs.w3s.link/IndexA/datatape_v.json https://bafybeibhqchj6vkypxxuy3xwonp2e4f27g5ilupbm2wphs22ycgt2ky6si.ipfs.w3s.link/IndexA/idx.json
cd ..
mkdir 2
cd 2
wget https://bafybeibhqchj6vkypxxuy3xwonp2e4f27g5ilupbm2wphs22ycgt2ky6si.ipfs.w3s.link/IndexB/datatape_k.bin https://bafybeibhqchj6vkypxxuy3xwonp2e4f27g5ilupbm2wphs22ycgt2ky6si.ipfs.w3s.link/IndexB/datatape_v.json https://bafybeibhqchj6vkypxxuy3xwonp2e4f27g5ilupbm2wphs22ycgt2ky6si.ipfs.w3s.link/IndexB/idx.json
cd ..
mkdir 3
cd 3
wget https://bafybeibhqchj6vkypxxuy3xwonp2e4f27g5ilupbm2wphs22ycgt2ky6si.ipfs.w3s.link/IndexC/datatape_k.bin https://bafybeibhqchj6vkypxxuy3xwonp2e4f27g5ilupbm2wphs22ycgt2ky6si.ipfs.w3s.link/IndexC/datatape_v.json https://bafybeibhqchj6vkypxxuy3xwonp2e4f27g5ilupbm2wphs22ycgt2ky6si.ipfs.w3s.link/IndexC/idx.json
cd ..
cd ..
py merge_x_sparse.py ponebin/
py sparse2dense.py
py transpose.py
mkdir results
gzip_transposed.py st_datatape.f32
```
