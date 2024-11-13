import sys
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel, QLineEdit, QComboBox, QDialogButtonBox

class CustomInputDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("输入对话框")

        # 创建布局
        layout = QVBoxLayout()

        # 创建类别输入
        self.category_label = QLabel("类别:")
        self.category_input = QComboBox()
        self.category_input.addItem("单行")
        self.category_input.addItem("多行")
        layout.addWidget(self.category_label)
        layout.addWidget(self.category_input)

        # 创建名称输入
        self.name_elabel = QLabel("属性英文名称（用于属性列）:")
        self.name_einput = QLineEdit()
        layout.addWidget(self.name_elabel)
        layout.addWidget(self.name_einput)
        self.name_label = QLabel("属性中文名称（用于表名称）:")
        self.name_input = QLineEdit()
        layout.addWidget(self.name_label)
        layout.addWidget(self.name_input)
        self.name_label.hide()
        self.name_input.hide()
        self.if_code_label = QLabel("是否有参考编码")
        self.if_code_input = QComboBox()
        self.if_code_input.addItem("是")
        self.if_code_input.addItem("否")
        layout.addWidget(self.if_code_label)
        layout.addWidget(self.if_code_input)
        self.if_code_label.hide()
        self.if_code_input.hide()




        # 创建按钮
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.setLayout(layout)
        self.category_input.currentTextChanged.connect(self.if_multi_chiname)
    def if_multi_chiname(self):
        text = self.category_input.currentText()
        if text == "多行":
            self.name_label.show()
            self.name_input.show()
            self.if_code_label.show()
            self.if_code_input.show()
        else:
            self.name_label.hide()
            self.name_input.hide()
            self.if_code_label.hide()
            self.if_code_input.hide()

    def getInputs(self):
        text = self.category_input.currentText()
        if text == "多行":
            return self.name_einput.text(),self.category_input.currentText(),self.name_input.text(),self.if_code_input.currentText()
        else:
            return self.name_einput.text(),self.category_input.currentText()
