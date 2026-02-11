# instructions for compiling the application
1) In the repository terminal pre-install these dependencies
```sh
pip install matplotlib PySide6 opencv-python numpy pyinstaller
```
2) Add a file that named "hemohelper.ico" to your repository or update it wirh command:
```sh
git pull origin main
```
3) Application from a file app.py compiled using the command 
```sh
python -m PyInstaller --onefile --windowed --icon=hemohelper.ico --name="MicroalgaeAnalyzer" app.py
```
4) it will take a few minutes and create two folders with names "build" and "dist". You must go to the folder and there will be an application with the extension ".exe". It's ready to use!
