import sys
from PyQt5.QtWidgets import QApplication
from Ui_main import ModernUI

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 使用Fusion风格，在所有平台上看起来更现代
    window = ModernUI()
    window.show()
    sys.exit(app.exec_()) 