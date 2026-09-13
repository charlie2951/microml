# microml
Machine learning library for Micropython
```bash
git clone https://github.com/micropython/micropython.git
cd micropython
git submodule update --init
make -C mpy-cross
cd ..
```

```bash
cd micropython/ports/unix
make submodules
make MICROPY_ENABLE_DYNRUNTIME=1
```
```bash
pip install pyelftools
```
```shell
sudo apt-get install gcc-mingw-w64
```

```bash
cd /content/micropython/ports/windows
make clean
make CROSS_COMPILE=x86_64-w64-mingw32- USER_C_MODULES=/content/microml
```
```
