#!/bin/bash

echo Fixing ui_MainWindow.py...
sed -i "s/(object)/(QMainWindow)/g" designs/ui_MainWindow.py
echo Fixing ui_ErrorWindow.py...
sed -i "s/(object)/(QDialog)/g" designs/ui_ErrorWindow.py
echo Fixing ui_VideoPreviewWindow.py...
sed -i "s/(object)/(QWidget)/g" designs/ui_VideoPreviewWindow.py

echo Done. Continue your work.
