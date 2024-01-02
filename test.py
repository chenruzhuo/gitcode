from PyQt5.QtWidgets import QWidget,QApplication
import sys

if __name__=='__main__':
    app=QApplication(sys.argv)
    window=QWidget()
    window.show()
    sys.exit(app.exec_())