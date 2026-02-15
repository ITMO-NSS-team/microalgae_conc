# Instructions for compiling the application
1) In the repository terminal pre-install these dependencies
```sh
pip install matplotlib PySide6 opencv-python numpy pyinstaller
```
2) Application from a file app.py compiled using the command 
```sh
python -m PyInstaller --onefile --windowed --icon=hemohelper.ico --name="MicroalgaeAnalyzer" app.py
```
3) It will take a few minutes and create two folders with names "build" and "dist". You must go to the folder "dist" and there will be an application with the extension ".exe". It's ready to use!
