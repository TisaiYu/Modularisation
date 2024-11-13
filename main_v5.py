from version.v5.response_v5 import *
import sys
import qdarkstyle
from PyQt5.QtWidgets import QApplication,QMainWindow

if __name__ == "__main__":
    app = QApplication (sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet())
    mainWindow = Modularization()
    mainWindow.show()
    sys.exit(app.exec_())
