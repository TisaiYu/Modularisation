import matplotlib.pyplot as plt

from mainwindow_ui import *
from PyQt5.QtCore import qDebug,Qt,QSize
from PyQt5.QtWidgets import QMainWindow,QFileDialog,QTableView,QStyledItemDelegate,QMessageBox,QFrame,QHBoxLayout,QApplication
from PyQt5.QtGui import QStandardItemModel, QStandardItem,QBrush,QTextDocument,QColor
from algorithm.AHP import *
from algorithm.HierarchicalClustering import *
from scipy.spatial.distance import pdist
from utils.utils import *
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
np.set_printoptions(precision=3)

"""
@coding: utf-8
@File Name: response_view
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
改为了模型/视图显示
-------------------------------------------------------------------------------------------------------------------------------------------


----------------------------------------------------------------------------------------------------------------------------------------

"""

correlation_nums = 7 # TisaiYu[2024/5/31] 相关性评估的数目，功能相关性、连接相关性......后续可能改为可选一些相关性，目前固定
reciprocal_count = 0 # TisaiYu[2024/6/5] AHP计算时，填充为对称的。这个参数防止填充对称时进入无限循环
pricision = 3


# TisaiYu[2024/6/13] 设置tableview的小数位数显示，重写父类的data函数
class MyTableModel(QtCore.QAbstractTableModel):
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.data = data


    def columnCount(self, parent=None):
        return len(self.data[0])

    def rowCount(self, parent=None):
        return len(self.data[1])

    def data(self, index, role):
        if role == QtCore.Qt.DisplayRole:
            row = index.row()
            col = index.column()
            return str(self.data[row][col])

class DecimalDelegate(QStyledItemDelegate):
    def __init__(self, decimals, parent=None):
        super(DecimalDelegate, self).__init__(parent)
        self.decimals = decimals

    def displayText(self, value, locale):
        try:
            return locale.toString(float(value), 'f', self.decimals)
        except:
            return value


class WordWrapDelegate(QStyledItemDelegate): # TisaiYu[2024/6/26] 调整列表内容显示（g)
    def paint(self, painter, option, index):
        painter.save()  # 保存当前的画笔状态

        # 设置文本颜色为白色
        painter.setPen(QColor(255, 255, 255))


        # 获取文本内容
        text = index.model().data(index)

        # 绘制文本
        painter.drawText(option.rect.adjusted(5, 5, -5, -5), Qt.AlignLeft | Qt.TextWordWrap, text)

        painter.restore()  # 恢复画笔状态

    def sizeHint(self, option, index):
        text = index.model().data(index)
        document = QTextDocument()
        document.setTextWidth(option.rect.width())
        document.setHtml(text)
        return QSize(document.idealWidth(), document.size().height())

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self,parent=None,width=5,height =4,dpi=150):
        self.fig = Figure(figsize=(width,height),dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas,self).__init__(self.fig)

class Modularization(QMainWindow,Ui_MainWindow): # TisaiYu[2024/5/30] 因为要在ui生成的py文件上添加槽函数，为了不在原py文件上修改（因为可能ui还需要修改，再生成新的py文件）。通过继承来添加槽函数，而不影响ui,并且由于要调用setupui，所以还要继承QMainwindow
    def __init__(self,parent=None): # TisaiYu[2024/6/5] parent不晓得有什么用，但别人都这样写
        super(Modularization,self).__init__()
        # TisaiYu[2024/5/31] 一些成员变量，因为槽函数无法返回，所以只能在槽函数调用时对成员变量赋值保存，除此还有一些判断逻辑变量如是否CR=0和执行变量如模块化的类成员
        self.thread_CE = None
        self.thread_Sil = None
        self.thread_G = None
        # TisaiYu[2024/6/3] 一些保存数据的变量
        self.parts_num = 0
        self.correlation_weight = np.zeros([correlation_nums])
        self.correlation_matrices = None  # TisaiYu[2024/5/31] 存储7个相关性矩阵的3维张量
        self.DSM = None
        self.correlation_selected_indexs = []
        self.clustering = None
        self.curve_draw_list = []
        self.metrics_list = []
        self.best_labels = []
        self.input_sequence = None
        # 一些判断逻辑变量
        self.CR_0 =1  # TisaiYu[2024/5/31] 1则完全按照CR=0严格要求相关性推断，只输入第一行即可。如1对2重要性为4,1对3重要性为8，则2对3重要性可推出为2。一致性检验就是判断输入是否偏离这个准则太多
        self.curve_mutex = QMutex()
        self.read_excel_file = False
        self.setupUi(self)
        self.setWindowTitle("Modularization")
        self.set_model()
        self.buildSlots()
        self.initStatus()





    def buildSlots(self):
        self.ahpButton.clicked.connect(self.ahpButton_clicked)
        self.ahpInputButton.clicked.connect(self.ahpInputButton_clicked)
        self.ahpClearButton.clicked.connect(self.ahpClearButton_clicked)

        self.classifyButton.clicked.connect(self.classifyButton_clicked)  # TisaiYu[2024/5/31] pyqt的槽函数似乎会连接所有信号，包括带参和不带的导致槽函数多次运行，就别用自动命名那种了。
        self.classifyInputButton.clicked.connect(self.classifyInputButton_clicked)
        self.resultButton.clicked.connect(self.resultButton_clicked)
        self.factorsSelectAll.stateChanged.connect(self.factors_selectAll_changed)
        self.factorsList.itemSelectionChanged.connect(self.factors_select_table_show)
        self.loadFileButton.clicked.connect(self.loadFileButton_clicked)

    def set_model(self):
        resmodel = QStandardItemModel(4, 4)
        column_labels = ["编号", "指标", "模块数", "划分结果"]
        resmodel.setHorizontalHeaderLabels(column_labels)
        self.moduleResultInfoTable.setModel(resmodel)
    def initStatus(self): # TisaiYu[2024/6/6] 相关组件设置初始化状态的都放在这里（designer里无法设置的，或者逻辑上赋值不方便的）
        self.factorsList.selectAll()
        for i in range(self.factorsList.count()):
            self.correlation_selected_indexs.append(self.factorsList.item(i))

    def ahpButton_clicked(self):
        qDebug("ahpButton_clicked!")
        self.stackedWidget.setCurrentWidget(self.relativeImportacePage)
        # self.stackedWidget.currentWidget().findChild(QTableView).model().setRowCount(correlation_nums)
        # self.stackedWidget.currentWidget().findChild(QTableView).model().setColumnCount(correlation_nums)

        # TisaiYu[2024/5/31] 由于这里没法像C++那样控制组件类的初始顺序了，只能先断连接再连接回来，否则这里初始化对角线为1会调用setRecripocal
        self.judge_matrix_table.model().dataChanged.disconnect(self.setRecripocal)
        for i in range(correlation_nums):
            self.judge_matrix_table.model().setItem(i, i, QStandardItem("1")) # TisaiYu[2024/5/31] 必须是string类型，输入其他类型不会自动转的
        self.judge_matrix_table.model().dataChanged.connect(self.setRecripocal)
        qDebug(f"ahp matrix is set to {correlation_nums}*{correlation_nums}")


    def ahpInputButton_clicked(self):
        row = self.judge_matrix_table.model().rowCount()
        column = self.judge_matrix_table.model().columnCount()
        judge_matrix = np.zeros((row,column))
        qDebug("ahpInputButton_clicked!")
        if self.CR_0: # TisaiYu[2024/5/31] 严格要求相关性推断，只输入第一行即可，否则除对角线都要输入，并且可能一致性检验不通过。
            for i in range(1, correlation_nums): # TisaiYu[2024/5/31] 依次组合 (2,3-7) (3,4-7) (4,5-7)......
                for j in range(i + 1, correlation_nums):
                    try:
                        cr1 = float(self.judge_matrix_table.model().item(0, j).text())
                        cr2 = float(self.judge_matrix_table.model().item(0, i).text())
                        CR_0_value = cr1 / cr2
                    except:
                        self.judge_error_label.setText("输入数据不对，请完整输入第一行的数据！")
                        self.judge_error_label.setStyleSheet("color: red")
                        self.judge_error_label.adjustSize()
                        return False
                    rec_value = 1 / CR_0_value
                    self.judge_matrix_table.model().setItem(j, i, QStandardItem(str(rec_value)))
                    judge_matrix[j][i] = rec_value
        for i in range(row):
            for j in range(column):
                judge_matrix[i][j] = float(self.judge_matrix_table.model().item(i, j).text())

        print(judge_matrix)
        ahp = AHP(judge_matrix)
        self.correlation_weight = ahp.calculate()
        row = self.ahpResultTable.model().rowCount()
        for i in range(row-1):
            self.ahpResultTable.model().setItem(i,0,QStandardItem(str(self.correlation_weight[i])))

        if ahp.CR <= 0.1:
            item = QStandardItem(str(f"{ahp.CR:.4f}<=0.1"))
            item.setForeground(QBrush(Qt.green))
            self.ahpResultTable.model().setItem(row-1, 0, item)
        else:
            item = QStandardItem(str(f"{ahp.CR:.4f}>0.1"))
            item.setForeground(QBrush(Qt.red))
            self.ahpResultTable.model().setItem(row-1, 0, item)

    def ahpClearButton_clicked(self):
        qDebug("ahpClearButton_clicked!")
        for i in range(correlation_nums):
            for j in range(correlation_nums):
                if i!=j:
                    item = self.judge_matrix_table.model().item(i,j)
                    if item:
                        item.setText("")


    def setRecripocal(self, model_index):
        global reciprocal_count
        row = model_index.row()
        column = model_index.column()
        text = self.judge_matrix_table.model().item(row, column).text()
        if is_float(text):
            value = float(text)
        else:
            return False
        if reciprocal_count == 0:
            reciprocal_count += 1
            self.judge_matrix_table.model().setItem(column, row, QStandardItem(str(1 / value)))
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
            self.correlationMatrixStackWidget.widget(i).findChild(QTableView).setModel(QStandardItemModel())
            self.correlationMatrixStackWidget.widget(i).findChild(QTableView).model().setRowCount(n)
            self.correlationMatrixStackWidget.widget(i).findChild(QTableView).model().setColumnCount(n)
            for j in range(n):
                self.correlationMatrixStackWidget.widget(i).findChild(QTableView).model().setItem(j, j, QStandardItem("1"))
            self.correlationMatrixStackWidget.widget(i).findChild(QTableView).resizeColumnsToContents()
            self.correlationMatrixStackWidget.widget(i).findChild(QTableView).resizeRowsToContents()
        # TisaiYu[2024/6/7] 测试时自动生成数据，不测试时把这里注释
        # row = self.funcMatrix.rowCount()
        # self.testWithSetValue(row)  # TisaiYu[2024/5/31] 给相关矩阵设置的值，但是还是需要输入AHP权重分析先。用于测试吧

    def classifyInputButton_clicked(self):
        qDebug("classifyInputButton_clicked!")
        """
        下面执行聚类
        """
        # TisaiYu[2024/6/5] 先聚类（以模块化或者聚类的指标先得到一个方案）
        if np.any(self.correlation_weight == 0):
            QMessageBox.information(None, "信息", "请先执行AHP权重分析！")
            return False
        if self.read_excel_file == False:
            QMessageBox.information(None, "信息", "请先从文件导入各影响因素下的零件相关性矩阵数据！")
            return False
        dist_matrix = 1 - self.DSM
        dis_mat = pdist(dist_matrix,
                        'euclidean')  # TisaiYu[2024/6/25] 层次聚类输入要么是一维数组（表示距离矩阵的压缩，比如30*30关联度矩阵，距离矩阵有30*30但是对称只取450），或者是二维数组（就是特征矩阵）
        Z = hierarchy.linkage(self.DSM, method='weighted',metric="cosine")
        self.draw_dedrogram(Z)

        # TisaiYu[2024/6/26] 把多线程里花时间的全部放在run里面去，因为只有run是多线程，其他函数都是在主线程上的。
        self.thread_CE = HierarchicalClustering(self.DSM,Z,'CE',1)
        self.thread_Sil = HierarchicalClustering(self.DSM,Z ,'Sil',2)
        self.thread_G = HierarchicalClustering(self.DSM,Z,'G',3)
        self.thread_Q = HierarchicalClustering(self.DSM,Z,'Q',4)
        self.thread_CE.clustering_finished.connect(self.cluster_optimize)
        self.thread_Sil.clustering_finished.connect(self.cluster_optimize)
        self.thread_G.clustering_finished.connect(self.cluster_optimize)
        self.thread_Q.clustering_finished.connect(self.cluster_optimize)
        self.thread_CE.start()
        self.thread_Sil.start()
        self.thread_G.start()
        self.thread_Q.start()


    def draw_dedrogram(self,Z):
        # TisaiYu[2024/6/17] 再画图
        dedrogram_view = MplCanvas()
        dedrogram_view.axes.set_title('Dendrogram')
        dedrogram_view.axes.set_xlabel('Customers')
        dedrogram_view.axes.set_ylabel('distances')
        re = hierarchy.dendrogram(Z, color_threshold=0.2, above_threshold_color='#bcbddc',
                                  ax=dedrogram_view.axes)
        num_array = [str(int(s) + 1) for s in re["ivl"]]
        re["ivl"] = num_array
        self.input_sequence = re['ivl']
        print("输入零件的序号：", re["ivl"])
        dedrogram_view.axes.set_xticklabels(num_array)
        # TisaiYu[2024/6/17] 初始化显示聚类树的类
        self.dendrogramLayout.addWidget(dedrogram_view)

    def cluster_optimize(self,best_cluster_labels,best_value,metric_name,values,curve_id):
        curvesFrameList = self.curvesFrame.findChildren(QFrame)
        curvesFrameList = sorted(curvesFrameList,key = lambda x:int(x.objectName()[-1].split("Frame")[-1]))
        frame = curvesFrameList[curve_id-1]
        curveLayout = frame.findChild(QHBoxLayout)
        self.dipict_cureve(metric_name, "class_nums", "value", curveLayout,
                           values,3,2,80)  # TisaiYu[2024/6/21] 多线程导致还没有记录i时另一个线程就已经判断完了，导致绘制在了一个layout上。

        self.moduleResultInfoTable.setWordWrap(True)
        item_col1 = QStandardItem(f"方案{curve_id}:")
        item_col2 = QStandardItem(f"{metric_name}")
        item_col3 = QStandardItem(f"{len(np.unique(best_cluster_labels))}")
        self.moduleResultInfoTable.model().setItem(curve_id - 1, 0, item_col1)
        self.moduleResultInfoTable.model().setItem(curve_id - 1, 1, item_col2)
        self.moduleResultInfoTable.model().setItem(curve_id - 1, 2, item_col3)
        item_col4_str = ''
        for i,label in enumerate(np.unique(best_cluster_labels)):
            module_parts = np.array(self.input_sequence)[best_cluster_labels == (i+1)]
            if label != np.unique(best_cluster_labels)[-1]:
                item_col4_str += f"M{i+1}:{module_parts},\n"
            else:
                item_col4_str += '.'
        item_col4 = QStandardItem(item_col4_str)
        self.moduleResultInfoTable.model().setItem(curve_id-1,3,item_col4)
        delegate = WordWrapDelegate()
        self.moduleResultInfoTable.setItemDelegate(delegate)
        self.moduleResultInfoTable.resizeRowsToContents()

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
        j_model = QStandardItemModel(correlation_nums,correlation_nums)
        self.judge_matrix_table.setModel(j_model)
        self.judge_matrix_table.model().dataChanged.connect(self.setRecripocal)
        delegate_j = DecimalDelegate(3)
        self.judge_matrix_table.setItemDelegate(delegate_j)


        labels = [label.text() for label in self.correlation_selected_indexs]
        self.judge_matrix_table.model().setHorizontalHeaderLabels(labels)
        self.judge_matrix_table.model().setVerticalHeaderLabels(labels)
        labels.append("一致性检验")
        a_model = QStandardItemModel(len(labels),1)
        self.ahpResultTable.setModel(a_model)
        delegate_a = DecimalDelegate(3)
        self.ahpResultTable.setItemDelegate(delegate_a)
        self.ahpResultTable.model().setHorizontalHeaderLabels(["结果"])
        self.ahpResultTable.model().setVerticalHeaderLabels(labels)



        for i, item in enumerate(self.correlation_selected_indexs):
            self.judge_matrix_table.model().setItem(i, i, QStandardItem("1"))


    def loadFileButton_clicked(self):
        file_dialog = QFileDialog.getOpenFileName(self,r"E:\Postgraduate\YY\code\Modularization_to_python\ModularizationPy")[0] # TisaiYu[2024/6/12] 改打开文件的初始路径
        if file_dialog == '': # TisaiYu[2024/6/20] 处理什么都没选就叉掉的情况
            return False
        data_file_exc = pd.ExcelFile(file_dialog)
        sheet_names = data_file_exc.sheet_names
        time1 = time.time()
        data_dict = {sheet_name: pd.read_excel(file_dialog, sheet_name,header=None) for sheet_name in sheet_names}
        self.parts_num = data_dict["func"].shape[0]
        self.correlation_matrices = np.zeros((correlation_nums, self.parts_num, self.parts_num))
        matricesFrameList = self.matricesFrame.findChildren(QFrame)
        matricesFrameList = sorted(matricesFrameList,key=lambda x:int(x.objectName().split("Frame")[-1]))
        for i in range(correlation_nums):
            index = self.factorsList.row(self.correlation_selected_indexs[i])
            self.correlation_matrices[i, :, :] = np.array(data_dict[sheet_names[i]])
            matrix_layout = matricesFrameList[i].findChild(QHBoxLayout)
            print(matricesFrameList[i].objectName())
            self.dipict_matrix(sheet_names[i],"index_row","index_col",matrix_layout,self.correlation_matrices[i, :, :],2.4,1.8,80)
        time2 = time.time()
        print("加载excel文件时间：",time2-time1)
        self.partsInputNumEdit.setText("数量："+str(self.parts_num))
        self.DSM = np.zeros((self.parts_num,self.parts_num))
        for i in range(self.correlation_weight.shape[0]):
            self.DSM += self.correlation_weight[i] * self.correlation_matrices[i]

        # 将结果保存为Excel文件
        df = pd.DataFrame(self.DSM)
        df.index = df.index+1
        df.to_excel('./data/DSM.xlsx',header=False)
        self.dipict_matrix('Correlation Matrix', 'index_row', 'index_col', self.DSMLayout, self.DSM)
        self.read_excel_file = True
    def resultButton_clicked(self):
        qDebug("resultButton_clicked!")
        self.stackedWidget.setCurrentWidget(self.resultPage)

    def dipict_matrix(self,title,xlabel,ylabel,layout,pic_array,width=4.,height=5.,dpi=100.):
        """
        :param title:
        :param xlabel:
        :param ylabel:
        :return:
        """
        canvas = MplCanvas(width,height,dpi)
        canvas.axes.set_title(title)
        canvas.axes.set_xlabel(xlabel)
        canvas.axes.set_ylabel(ylabel)
        layout.addWidget(canvas)
        canvas.axes.imshow(pic_array, cmap='Blues', interpolation='nearest')

    def dipict_cureve(self,title,xlabel,ylabel,layout,pic_array,width=2.4,height=1.8,dpi=100.):
        """
        :param title:
        :param xlabel:
        :param ylabel:
        :return:
        """
        fontsize= 6
        canvas = MplCanvas(width, height, dpi)
        canvas.axes.set_title(title,fontsize=fontsize,y=0.94)
        canvas.axes.set_xlabel(xlabel,fontsize=fontsize)
        canvas.axes.set_ylabel(ylabel,fontsize=fontsize)

        canvas.axes.set_xticks(range(0,len(pic_array)+3,len(pic_array)//10))

        canvas.axes.xaxis.set_label_coords(0.9, 1.15)
        canvas.axes.yaxis.set_label_coords(1.1, 0.9)
        for label in canvas.axes.get_xticklabels():
            label.set_verticalalignment('top')  # 垂直对齐方式，可选 'center', 'top', 'bottom', 'baseline', 'center_baseline'
            label.set_y(0.025)
        canvas.axes.tick_params(axis='both', labelsize=4)

        layout.addWidget(canvas)
        QApplication.processEvents()

        canvas.axes.plot(pic_array)



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
            now_table = self.correlationMatrixStackWidget.widget(index).findChild(QTableView)
            for j in range(row):
                for k in range(column):
                    now_table.model().setItem(j, k, QStandardItem(str(mm[i][j][k])))
