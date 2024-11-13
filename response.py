import pandas as pd

from mainwindow_ui import *
from PyQt5.QtCore import qDebug
from PyQt5.QtWidgets import QMainWindow,QTableWidget,QStackedWidget,QTableWidgetItem,QFileDialog,QTableView
from PyQt5 import QtWidgets
from algorithm.AHP import *
from algorithm.HierarchicalClustering import *
from algorithm.ClusterOptimize import *
from utils.utils import *
import numpy as np
"""
@coding: utf-8
@File Name: response
@Author: TisaiYu
@Creation Date: 2024/6/5
@version: 1.0
------------------------------------------------------------------------------------------------------------------------------------------
@Description: 
界面ui的相关响应操作文件。继承ui类，在继承ui界面的基础上续写界面的响应；同时继承QMainWindow类，可以作为显示的组件。
------------------------------------------------------------------------------------------------------------------------------------------
@Modification Record: 
None
-----------------------------------------------------------------------------------------------------------------------------------------
文件重要修改
None
-------------------------------------------------------------------------------------------------------------------------------------------


----------------------------------------------------------------------------------------------------------------------------------------

"""


correlation_nums = 7 # TisaiYu[2024/5/31] 相关性评估的数目，功能相关性、连接相关性......后续可能改为可选一些相关性，目前固定
reciprocal_count = 0 # TisaiYu[2024/6/5] AHP计算时，填充为对称的。这个参数防止填充对称时进入无限循环
class Modularization(QMainWindow,Ui_MainWindow): # TisaiYu[2024/5/30] 因为要在ui生成的py文件上添加槽函数，为了不在原py文件上修改（因为可能ui还需要修改，再生成新的py文件）。通过继承来添加槽函数，而不影响ui,并且由于要调用setupui，所以还要继承QMainwindow
    def __init__(self,parent=None): # TisaiYu[2024/6/5] parent不晓得有什么用，但别人都这样写
        super(Modularization,self).__init__()
        # TisaiYu[2024/5/31] 一些成员变量，因为槽函数无法返回，所以只能在槽函数调用时对成员变量赋值保存，除此还有一些判断逻辑变量如是否CR=0和执行变量如模块化的类成员
        # TisaiYu[2024/6/3] 一些保存数据的变量
        self.parts_num = 0
        self.correlation_weight = None
        self.correlation_matrices = None  # TisaiYu[2024/5/31] 存储7个相关性矩阵的3维张量
        self.correlation_selected_indexs = []
        # 一些判断逻辑变量
        self.CR_0 = 1  # TisaiYu[2024/5/31] 1则完全按照CR=0严格要求相关性推断，只输入第一行即可。如1对2重要性为4,1对3重要性为8，则2对3重要性可推出为2。一致性检验就是判断输入是否偏离这个准则太多


        self.setupUi(self)
        self.setWindowTitle("Modularization")
        self.initStatus()
        self.buildSlots()




    def buildSlots(self):
        self.ahpButton.clicked.connect(self.ahpButton_clicked)
        self.ahpHomepageButton.clicked.connect(self.ahpHomepageButton_clicked)
        self.ahpInputButton.clicked.connect(self.ahpInputButton_clicked)
        self.ahpClearButton.clicked.connect(self.ahpClearButton_clicked)
        self.judge_matrix_table.cellChanged.connect(self.setRecripocal)
        self.classifyButton.clicked.connect(
        self.classifyButton_clicked)  # TisaiYu[2024/5/31] pyqt的槽函数似乎会连接所有信号，包括带参和不带的导致槽函数多次运行，就别用自动命名那种了。
        self.classifyHomepageButton.clicked.connect(self.classifyHomepageButton_clicked)
        self.partsInputNumEdit.textChanged.connect(self.setCorrelationTableShape)
        self.classifyInputButton.clicked.connect(self.classifyInputButton_clicked)
        self.classifyClearAllButton.clicked.connect(self.classifyClearAllButton_clicked)
        self.classifyClearCurrentButton.clicked.connect(self.classifyClearCurrentButton_clicked)
        self.correlationCombox.currentIndexChanged.connect(self.change_correlation_table)
        self.factorsSelectAll.stateChanged.connect(self.factors_selectAll_changed)
        self.factorsList.itemSelectionChanged.connect(self.factors_select_table_show)
        self.loadFileButton.clicked.connect(self.loadFileButton_clicked)

    def initStatus(self): # TisaiYu[2024/6/6] 相关组件设置初始化状态的都放在这里（designer里无法设置的，或者逻辑上赋值不方便的）
        self.factorsList.selectAll()
        for i in range(self.factorsList.count()):
            self.correlation_selected_indexs.append(self.factorsList.item(i))
    def ahpButton_clicked(self):
        qDebug("ahpButton_clicked!")
        self.stackedWidget.setCurrentWidget(self.relativeImportacePage)
        self.stackedWidget.currentWidget().findChild(QTableWidget).setRowCount(correlation_nums)
        self.stackedWidget.currentWidget().findChild(QTableWidget).setColumnCount(correlation_nums)

        # TisaiYu[2024/5/31] 由于这里没法像C++那样控制组件类的初始顺序了，只能先断连接再连接回来，否则这里初始化对角线为1会调用setRecripocal
        self.judge_matrix_table.cellChanged.disconnect(self.setRecripocal)
        for i in range(correlation_nums):
            self.judge_matrix_table.setItem(i, i, QTableWidgetItem("1")) # TisaiYu[2024/5/31] 必须是string类型，输入其他类型不会自动转的
        self.judge_matrix_table.cellChanged.connect(self.setRecripocal)
        qDebug(f"ahp matrix is set to {correlation_nums}*{correlation_nums}")

    def ahpHomepageButton_clicked(self):
        qDebug("ahpHomepageButton_clicked!")
        self.stackedWidget.setCurrentWidget(self.homepage)

    def ahpInputButton_clicked(self):
        row = self.judge_matrix_table.rowCount()
        column = self.judge_matrix_table.columnCount()
        judge_matrix = np.zeros((row,column))
        qDebug("ahpInputButton_clicked!")

        if self.CR_0: # TisaiYu[2024/5/31] 严格要求相关性推断，只输入第一行即可，否则除对角线都要输入，并且可能一致性检验不通过。
            for i in range(1, correlation_nums): # TisaiYu[2024/5/31] 依次组合 (2,3-7) (3,4-7) (4,5-7)......
                for j in range(i + 1, correlation_nums):
                    CR_0_value = float(self.judge_matrix_table.item(0, j).text())/float(self.judge_matrix_table.item(0, i).text())
                    self.judge_matrix_table.setItem(j, i, QTableWidgetItem(str(1 / CR_0_value)))
        for i in range(row):
            for j in range(column):
                judge_matrix[i][j] = float(self.judge_matrix_table.item(i, j).text())

        print(judge_matrix)

        ahp = AHP(judge_matrix)
        self.correlation_weight = ahp.calculate()

    def ahpClearButton_clicked(self):
        qDebug("ahpClearButton_clicked!")
        self.stackedWidget.currentWidget().findChild(QTableWidget).clearContents()


    def setRecripocal(self, row, column):
        global reciprocal_count
        text = self.judge_matrix_table.item(row, column).text()
        if is_float(text):
            value = float(text)
        else:
            return False
        if reciprocal_count == 0:
            reciprocal_count += 1
            self.judge_matrix_table.setItem(column, row, QTableWidgetItem(str(1 / value)))
        else:
            reciprocal_count = 0
            return False
        self.judge_matrix_table.resizeColumnsToContents()
        self.judge_matrix_table.resizeRowsToContents()

    def classifyButton_clicked(self):
        qDebug("classify_Button_clicked!")
        self.stackedWidget.setCurrentWidget(self.matricesPage)

    def classifyHomepageButton_clicked(self):
        qDebug("classifyHomepageButton_clicked!")
        self.stackedWidget.setCurrentWidget(self.homepage)

    def setCorrelationTableShape(self,n):
        if not n.isdigit(): # TisaiYu[2024/5/31] 防止为空时程序异常终止
            return False
        n = int(n)
        qDebug(f"All correlation input matrixs are set to {n}*{n}")
        for i in range(self.correlationMatrixStackWidget.count()):
            self.correlationMatrixStackWidget.widget(i).findChild(QTableWidget).clearContents()
            self.correlationMatrixStackWidget.widget(i).findChild(QTableWidget).setRowCount(n)
            self.correlationMatrixStackWidget.widget(i).findChild(QTableWidget).setColumnCount(n)
            for j in range(n):
                self.correlationMatrixStackWidget.widget(i).findChild(QTableWidget).setItem(j, j, QTableWidgetItem("1"))
            self.correlationMatrixStackWidget.widget(i).findChild(QTableWidget).resizeColumnsToContents()
            self.correlationMatrixStackWidget.widget(i).findChild(QTableWidget).resizeRowsToContents()
        # TisaiYu[2024/6/7] 测试时自动生成数据，不测试时把这里注释
        # row = self.funcMatrix.rowCount()
        # self.testWithSetValue(row)  # TisaiYu[2024/5/31] 给相关矩阵设置的值，但是还是需要输入AHP权重分析先。用于测试吧

    def classifyInputButton_clicked(self):
        qDebug("classifyInputButton_clicked!")

        """
        下面执行聚类
        """
        row = self.funcMatrix.rowCount()
        column = self.funcMatrix.columnCount()

        for i in range(correlation_nums):
            index = self.factorsList.row(self.correlation_selected_indexs[i])
            now_table = self.correlationMatrixStackWidget.widget(index).findChild(QTableWidget)
            for j in range(row):
                for k in range(column):
                    self.correlation_matrices[i][j][k] = float(now_table.item(j, k).text())

        # TisaiYu[2024/6/5] 先聚类（以模块化或者聚类的指标先得到一个方案）
        clustering = HierarchicalClustering(self.correlation_matrices,self.correlation_weight)
        best_cluster_labels,Z,dist_matrix = clustering.hierarchy_clustering()

        # TisaiYu[2024/6/5] 再优化，暂时编写的遗传模拟退火（以后续经济，周期等等新的目标函数再优化，但是目前优化的结果的类别数只能等于上面聚类结果的类别数，理论上应该对聚类树上的多个结果进行优化才对，后续可能要改逻辑）
        hgsa = GeneticSimulatedAnnealing(best_cluster_labels, dist_matrix, num_clusters=self.parts_num)
        best_individual, best_fitness = hgsa.optimize()
        print("遗传模拟退火优化后的fitness：",best_fitness)
        print("遗传模拟退火优化后的最佳分类方案：",best_individual)
        # best_individual, best_fitness = hgsa.optimize_Q()
        # print("遗传模拟退火优化后的fitness：", best_fitness)
        # print("遗传模拟退火优化后的最佳分类方案：", best_individual)

    def classifyClearAllButton_clicked(self):
        qDebug("classifyClearAllButton_clicked!")
        for i in range(self.correlationMatrixStackWidget.count()):
            now_table = self.correlationMatrixStackWidget.widget(i).findChild(QTableWidget)
            now_table.clearContents()
            n = now_table.rowCount()
            for j in range(n):
                now_table.setItem(j, j, QTableWidgetItem("1"))
            now_table.resizeColumnsToContents()
            now_table.resizeRowsToContents()
    def classifyClearCurrentButton_clicked(self):
        qDebug("classifyClearCurrentButton_clicked!")
        now_table = self.correlationMatrixStackWidget.currentWidget().findChild(QTableWidget)
        now_table.clearContents()
        n = now_table.rowCount()
        for j in range(n):
            now_table.setItem(j, j, QTableWidgetItem("1"))
        now_table.resizeColumnsToContents()
        now_table.resizeRowsToContents()

    def change_correlation_table(self,index):
        qDebug(f"correlation_table is changed to {1}")
        self.stackedWidget.currentWidget().findChild(QStackedWidget).setCurrentIndex(index)

    def factors_selectAll_changed(self):

        if self.factorsSelectAll.isChecked():
            self.factorsList.selectAll()
        else:
            self.factorsList.clearSelection()
    def factors_select_table_show(self):
        global correlation_nums
        self.correlation_selected_indexs.clear()
        for i in range(self.factorsList.count()):
            item = self.factorsList.item(i)
            if item.isSelected():
                self.correlation_selected_indexs.append(item)

        # selected_items = self.factorsList.selectedItems() # TisaiYu[2024/6/6] 这个因为是保持的选择的顺序而不是排列的顺序，所以不用这个，正好用循环可以保证另一个全选逻辑正常
        correlation_nums = len(self.correlation_selected_indexs)
        if correlation_nums == self.factorsList.count():
            self.factorsSelectAll.setChecked(1)
        else:
            self.factorsSelectAll.stateChanged.disconnect(self.factors_selectAll_changed) # TisaiYu[2024/6/6] 防止设置0时触发上面的槽函数
            self.factorsSelectAll.setChecked(0)
            self.factorsSelectAll.stateChanged.connect(self.factors_selectAll_changed)
        self.judge_matrix_table.setRowCount(correlation_nums)
        self.judge_matrix_table.setColumnCount(correlation_nums)

        # TisaiYu[2024/6/7] 设置combox的选项全隐藏，下面再设置listwidget中选中的显示
        view = self.correlationCombox.view() # TisaiYu[2024/6/7] 返回的是一个QListView
        for i in range(self.correlationCombox.count()):
            view.setRowHidden(i,True) # TisaiYu[2024/6/7] 只有QListView，QTableView，QTreeView有setRowHidden方法。是通过模型/视图架构的抽象来实现的，模型管数据，视图管显示

        for i,item in enumerate(self.correlation_selected_indexs):
            self.judge_matrix_table.setHorizontalHeaderItem(i,QTableWidgetItem(item.text()))
            self.judge_matrix_table.setVerticalHeaderItem(i,QTableWidgetItem(item.text()))
            self.judge_matrix_table.setItem(i, i,QTableWidgetItem("1"))

            # TisaiYu[2024/6/7] 只有选中的影响因素才显示
            row = self.factorsList.row(item)
            view.setRowHidden(row,False)

    def loadFileButton_clicked(self):
        file_dialog = QFileDialog.getOpenFileName(self,r"E:\Postgraduate\YY\code\Modularization_to_python\ModularizationPy")[0] # TisaiYu[2024/6/12] 改打开文件的初始路径
        data_file_exc = pd.ExcelFile(file_dialog)
        sheet_names = data_file_exc.sheet_names
        data_dict = {sheet_name:data_file_exc.parse(sheet_name,header=None) for sheet_name in sheet_names}
        data = np.stack([df.values for df in data_dict.values()], axis=0)
        data = np.array(data)
        self.parts_num = data.shape[1]
        self.correlation_matrices = np.zeros((correlation_nums,self.parts_num,self.parts_num))
        for i in range(correlation_nums):
            self.correlation_matrices[i,:,:] = data[i,:,:]
        self.partsInputNumEdit.setText(str(self.parts_num))
        for i in range(correlation_nums):
            index = self.factorsList.row(self.correlation_selected_indexs[i])
            now_table = self.correlationMatrixStackWidget.widget(index).findChild(QTableWidget)
            for j in range(self.parts_num):
                for k in range(self.parts_num):
                    now_table.setItem(j, k, QTableWidgetItem(str(self.correlation_matrices[i][j][k])))
            now_table.resizeColumnsToContents()
            now_table.resizeRowsToContents()

    def testWithSetValue(self,n):
        row = self.funcMatrix.rowCount()
        column = self.funcMatrix.columnCount()
        self.correlation_matrices = np.zeros((correlation_nums, row, column))
        # s1 = generate_symmetric_matrix1(n)
        # s2 = generate_symmetric_matrix1(n)
        # s3 = generate_symmetric_matrix2(n)
        # s4 = generate_symmetric_matrix1(n)
        # s5 = generate_symmetric_matrix2(n)
        # s6 = generate_symmetric_matrix3(n)
        # s7 = generate_symmetric_matrix3(n)
        # s1 = np.array([[1, 0.8, 0.2, 0.5, 0],
        #                [0.8, 1, 0, 0, 0],
        #                [0.2, 0, 1, 0.2, 0],
        #                [0.5, 0, 0.2, 1, 0.5],
        #                [0, 0, 0, 0.5, 1]])
        #
        # s2 = np.array([[1, 0.2, 0.8, 0, 0.5],
        #                [0.2, 1, 0, 0, 0],
        #                [0.8, 0, 1, 0.8, 0],
        #                [0, 0, 0.8, 1, 0.2],
        #                [0.5, 0, 0, 0.2, 1]])
        #
        # s3 = np.array([[1, 0, 0.4, 0, 0.8],
        #                [0, 1, 0, 0, 0],
        #                [0.4, 0, 1, 0.4, 0],
        #                [0, 0, 0.4, 1, 0],
        #                [0.8, 0, 0, 0, 1]])
        #
        # s4 = np.array([[1, 0.8, 0, 0.5, 0.2],
        #                [0.8, 1, 0, 0, 0],
        #                [0, 0, 1, 0.8, 0],
        #                [0.5, 0, 0.8, 1, 0],
        #                [0.2, 0, 0, 0, 1]])
        #
        # s5 = np.array([[1, 0.8, 0, 0.4, 0],
        #                [0.8, 1, 0, 0, 0],
        #                [0, 0, 1, 0, 0],
        #                [0.4, 0, 0, 1, 0.4],
        #                [0, 0, 0, 0.4, 1]])
        #
        # s6 = np.array([[1, 0.8, 0, 0.3, 0.2],
        #                [0.8, 1, 0, 0, 0],
        #                [0, 0, 1, 0.8, 0],
        #                [0.3, 0, 0.8, 1, 0.3],
        #                [0, 0, 0, 0.3, 1]])
        #
        # s7 = np.array([[1, 0.8, 0.3, 0, 0],
        #                [0.8, 1, 0, 0, 0],
        #                [0.3, 0, 1, 0.3, 0],
        #                [0, 0, 0.3, 1, 0.8],
        #                [0, 0, 0, 0.8, 1]])
        # mm = np.stack((s1, s2, s3, s4, s5, s6, s7), axis=0)
        mm = generate_test_value(n,5,correlation_nums)
        for i in range(correlation_nums):
            index = self.factorsList.row(self.correlation_selected_indexs[i])
            now_table = self.correlationMatrixStackWidget.widget(index).findChild(QTableWidget)
            for j in range(row):
                for k in range(column):
                    now_table.setItem(j, k, QTableWidgetItem(str(mm[i][j][k])))
