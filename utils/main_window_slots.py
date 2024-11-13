import sqlite3
from userInterface.CustomInputDialog import *
from userInterface.mainwindow_ui import *
from PyQt5.QtCore import qDebug, QSize
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QTableView, QStyledItemDelegate, QMessageBox, QFrame, QHBoxLayout, \
    QApplication, QDialog, QTableWidgetItem, QWidget, QPushButton, QHeaderView
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QBrush, QTextDocument, QColor
from algorithm.AHP import *
from algorithm.HierarchicalClusteringV5 import *
from algorithm.Reprod_general_fuzz_improveV1 import *
from scipy.spatial.distance import pdist, squareform
from utils.common_utils import *
from userInterface.PlotCanvas import PlotCanvas
np.set_printoptions(precision=3)
"""
@coding: utf-8
@File Name: response_v3sqlimprove
@Author: TisaiYu
@Creation Date: 2024/8/22
@version: 1.0
------------------------------------------------------------------------------------------------------------------------------------------
@Description: 
在V3sqlimprove里加入了安装成本算法，具体是加入到多线程里面去了，因为每个聚类结果要算一个ACP和MT以及最后的SIC
------------------------------------------------------------------------------------------------------------------------------------------
@Modification Record: 

-----------------------------------------------------------------------------------------------------------------------------------------
文件重要修改

-------------------------------------------------------------------------------------------------------------------------------------------


----------------------------------------------------------------------------------------------------------------------------------------

"""

correlation_nums = 7  # TisaiYu[2024/5/31] 相关性评估的数目，功能相关性、连接相关性......后续可能改为可选一些相关性，目前固定
reciprocal_count = 0  # TisaiYu[2024/6/5] AHP计算时，填充为对称的。这个参数防止填充对称时进入无限循环
pricision = 3





class DrawDendrogram(QThread):
    draw_finished = pyqtSignal(np.ndarray, list,
                               PlotCanvas)  # TisaiYu[2024/6/20] 信号定义为类的属性，而不是实例属性，是因为连接要写在实例化调用函数之前，这样实例化的函数发出信号才可以被响应

    def __init__(self, DSM):
        super(DrawDendrogram, self).__init__()
        self.DSM = DSM
        self.dedrogram_view = PlotCanvas()
        self.dedrogram_view.axes.set_title('Dendrogram')
        self.dedrogram_view.axes.set_xlabel('Customers')
        self.dedrogram_view.axes.set_ylabel('distances')

    def run(self):
        dist_m1 = 1 - self.DSM
        # 将距离矩阵转换为压缩形式
        condensed_distance_matrix = squareform(dist_m1, checks=False)
        dis_mat = pdist(self.DSM,
                        'cosine')  # TisaiYu[2024/6/25] 层次聚类输入要么是一维数组（表示距离矩阵的压缩，比如30*30关联度矩阵，距离矩阵有30*30但是对称只取450），或者是二维数组（就是特征矩阵）

        # TisaiYu[2024/9/10] 因为功能对于很多零部件都有关联度的影响，而连接只对一些少数零部件有影响，而ward是类内方差链接法，因此对于功能这种对很多零部件有影响的大片区域数据，能够有所识别，因为功能影响大片，所以是方差稳定的主要因素，所以能先根据功能初始得到模块方案，再迭代求解
        Z = hierarchy.linkage(dis_mat, method='ward',
                              metric="euclidean")  # TisaiYu[2024/9/5] 当输入是一个一维的压缩距离矩阵时（使用pdist的结果），那么metric是没有用，如果是一个二维的表示输入特征的数组，metric实际和pdist做了一样的事情
        distances = Z[:, 2]
        reversed_last = distances[::-1]
        indexes = np.arange(1, len(distances) + 1)
        # 计算距离增长加速度,或者采用距离的最大变化
        accelerations = np.diff(reversed_last, 2)
        # accelerations = np.diff(reversed_last, 1)
        accelerations = np.pad(accelerations, (1, 1), 'constant', constant_values=(0, 0))  # 在首尾补零
        # 确定最佳聚类数
        optimal_cluster_num = np.argmax(accelerations) + 1  # +1 是因为 diff 函数减少了一个元素
        re = hierarchy.dendrogram(Z, color_threshold=Z[-(optimal_cluster_num-1),2], above_threshold_color='#bcbddc',
                                  ax=self.dedrogram_view.axes)
        num_array = [str(int(s)) for s in re["ivl"]]
        re["ivl"] = num_array
        input_sequence = re['ivl']
        print("输入零件的序号：", re["ivl"])
        self.dedrogram_view.axes.set_xticklabels(num_array)

        self.draw_finished.emit(Z, input_sequence, self.dedrogram_view)


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


class WordWrapDelegate(QStyledItemDelegate):  # TisaiYu[2024/6/26] 调整列表内容显示（g)
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


class Modularization(QMainWindow,
                     Ui_MainWindow):  # TisaiYu[2024/5/30] 因为要在ui生成的py文件上添加槽函数，为了不在原py文件上修改（因为可能ui还需要修改，再生成新的py文件）。通过继承来添加槽函数，而不影响ui,并且由于要调用setupui，所以还要继承QMainwindow
    def __init__(self, parent=None):  # TisaiYu[2024/6/5] parent不晓得有什么用，但别人都这样写
        super(Modularization, self).__init__()
        # TisaiYu[2024/5/31] 一些成员变量，因为槽函数无法返回，所以只能在槽函数调用时对成员变量赋值保存，除此还有一些判断逻辑变量如是否CR=0和执行变量如模块化的类成员
        self.thread_CE = None
        self.thread_Sil = None
        self.thread_G = None
        self.thread_Q = None
        self.thread_drawgram = None
        self.deview = None
        self.time1 = 0
        self.time2 = 0
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
        self.db = QtSql.QSqlDatabase.addDatabase("QSQLITE")
        self.sum_query_model = QtSql.QSqlTableModel(self,
                                                    self.db)  # TisaiYu[2024/8/12] 如果不加入第二个参数说明数据库，则必须在db成员赋值后使用，否则不清楚是哪个db，后续操作有误
        self.sum_query_model.setEditStrategy(QtSql.QSqlTableModel.EditStrategy.OnFieldChange)
        self.table_names_dict = {}
        self.attribute_dict = {}  # TisaiYu[2024/8/22] 记录添加的属性属于哪个表的

        # 一些判断逻辑变量
        self.CR_0 = 0  # TisaiYu[2024/5/31] 1则完全按照CR=0严格要求相关性推断，只输入第一行即可。如1对2重要性为4,1对3重要性为8，则2对3重要性可推出为2。一致性检验就是判断输入是否偏离这个准则太多
        self.curve_mutex = QMutex()
        self.cube_create = False
        self.querying = False
        self.count = 0
        # 一些其他窗口的变量
        self.sqlinput_dialog_window = QDialog()


        self.read_excel_file = False
        self.setupUi(self)
        self.setWindowTitle("Modularization")
        self.set_model()
        self.buildInitSlots()
        self.initStatus()

    def buildInitSlots(self):
        self.optionTree.itemClicked.connect(self.change_page)
        self.ahpInputButton.clicked.connect(self.ahpInputButton_clicked)
        self.ahpClearButton.clicked.connect(self.ahpClearButton_clicked)
        self.classifyInputButton.clicked.connect(self.classifyInputButton_clicked)
        self.factorsSelectAll.stateChanged.connect(self.factors_selectAll_changed)
        self.factorsList.itemSelectionChanged.connect(self.factors_select_table_show)
        self.loadFileButton.clicked.connect(self.loadFileButton_clicked)
        self.exportModuleInfoButton.clicked.connect(self.export_module_info)
        self.dbCreateButton.clicked.connect(self.create_db_qt)
        self.dbLoadButton.clicked.connect(self.load_db_qt)
        self.addSQLDataButton.clicked.connect(self.sqlinput_ui_show)
        self.sqlinput_dialog_window.accepted.connect(self.sql_add_one_row)
        self.addAttributeButton.clicked.connect(self.add_attribute)
        self.deleteAttributeButton.clicked.connect(self.delete_attribute)
        self.deleteSQLDataButton.clicked.connect(self.sql_delete_select_row)
        self.readExcelToSQLButton.clicked.connect(self.read_excel_to_sql)
        self.selectPartInfoButton.clicked.connect(self.select_part_info)
        self.SQLTableView.horizontalHeader().sectionMoved.connect(self.synMovedColumn)
        self.ClearSQLTableButton.clicked.connect(self.clear_sql_table)
        self.SqlTable2ExcelButton.clicked.connect(self.export_sql_table)
        self.SqlDSMButton.clicked.connect(self.generate_dsm_from_sql)
        self.CubeCreateButton.clicked.connect(self.update_sql_excel_cube_create)

    def set_model(self):
        resmodel = QStandardItemModel(4, 4)
        column_labels = ["编号", "指标", "模块数", "划分结果"]
        resmodel.setHorizontalHeaderLabels(column_labels)
        self.moduleResultInfoTable.setModel(resmodel)

    def initStatus(self):  # TisaiYu[2024/6/6] 相关组件设置初始化状态的都放在这里（designer里无法设置的，或者逻辑上赋值不方便的）
        self.factorsList.selectAll()
        for i in range(self.factorsList.count()):
            self.correlation_selected_indexs.append(self.factorsList.item(i))
        self.SQLTableView.horizontalHeader().setSectionsMovable(True)
        self.SQLTableView.horizontalHeader().setDragEnabled(True)
        self.SQLTableView.horizontalHeader().setDragDropMode(QHeaderView.InternalMove)
        # 确保列可以自由拖动到任意位置
        self.SQLTableView.horizontalHeader().setDragDropOverwriteMode(False)

    def change_page(self, item, column):
        if item.text(column) == "主页":
            self.stackedWidget.setCurrentWidget(self.homepage)
        elif item.text(column) == "相关性因素选择及权重分析":
            self.stackedWidget.setCurrentWidget(self.relativeImportacePage)
            # self.stackedWidget.currentWidget().findChild(QTableView).model().setRowCount(correlation_nums)
            # self.stackedWidget.currentWidget().findChild(QTableView).model().setColumnCount(correlation_nums)

            # TisaiYu[2024/5/31] 由于这里没法像C++那样控制组件类的初始顺序了，只能先断连接再连接回来，否则这里初始化对角线为1会调用setRecripocal
            self.judge_matrix_table.model().dataChanged.disconnect(self.setRecripocal)
            for i in range(correlation_nums):
                self.judge_matrix_table.model().setItem(i, i, QStandardItem(
                    "1"))  # TisaiYu[2024/5/31] 必须是string类型，输入其他类型不会自动转的
            self.judge_matrix_table.model().dataChanged.connect(self.setRecripocal)
        elif item.text(column) == "关联度表填写":
            self.stackedWidget.setCurrentWidget(self.matricesPage)
        elif item.text(column) == "模块划分结果":
            self.stackedWidget.setCurrentWidget(self.resultPage)
        elif item.text(column) == "零部件信息数据库填写":
            self.stackedWidget.setCurrentWidget(self.SQLpage)

    def ahpInputButton_clicked(self):
        row = self.judge_matrix_table.model().rowCount()
        column = self.judge_matrix_table.model().columnCount()
        if row == 0:
            QMessageBox.information(None, "信息", "请至少选择一个相关因素！")
            return -1
        judge_matrix = np.zeros((row, column))
        if self.CR_0:  # TisaiYu[2024/5/31] 严格要求相关性推断，只输入第一行即可，否则除对角线都要输入，并且可能一致性检验不通过。
            for i in range(1, correlation_nums):  # TisaiYu[2024/5/31] 依次组合 (2,3-7) (3,4-7) (4,5-7)......
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
                item = self.judge_matrix_table.model().item(i, j)
                if item != None:
                    text = item.text()
                    judge_matrix[i][j] = float(text)
                else:
                    QMessageBox.information(None, "信息", "请填写完整相关性因素的相对重要程度矩阵！")
                    return False

        ahp = AHP(judge_matrix)
        self.correlation_weight = ahp.calculate()
        row = self.ahpResultTable.model().rowCount()
        for i in range(row - 1):
            self.ahpResultTable.model().setItem(i, 0, QStandardItem(str(self.correlation_weight[i])))

        if ahp.CR <= 0.1:
            item = QStandardItem(str(f"{ahp.CR:.4f}<=0.1"))
            item.setForeground(QBrush(Qt.green))
            self.ahpResultTable.model().setItem(row - 1, 0, item)
        else:
            item = QStandardItem(str(f"{ahp.CR:.4f}>0.1"))
            item.setForeground(QBrush(Qt.red))
            self.ahpResultTable.model().setItem(row - 1, 0, item)

    def ahpClearButton_clicked(self):
        for i in range(correlation_nums):
            for j in range(correlation_nums):
                if i != j:
                    item = self.judge_matrix_table.model().item(i, j)
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

    def setCorrelationTableShape(self, n):
        if not n.isdigit():  # TisaiYu[2024/5/31] 防止为空时程序异常终止
            return False
        n = int(n)
        qDebug(f"All correlation input matrixs are set to {n}*{n}")
        for i in range(self.correlationMatrixStackWidget.count()):
            self.correlationMatrixStackWidget.widget(i).findChild(QTableView).setModel(QStandardItemModel())
            self.correlationMatrixStackWidget.widget(i).findChild(QTableView).model().setRowCount(n)
            self.correlationMatrixStackWidget.widget(i).findChild(QTableView).model().setColumnCount(n)
            for j in range(n):
                self.correlationMatrixStackWidget.widget(i).findChild(QTableView).model().setItem(j, j,
                                                                                                  QStandardItem("1"))
            self.correlationMatrixStackWidget.widget(i).findChild(QTableView).resizeColumnsToContents()
            self.correlationMatrixStackWidget.widget(i).findChild(QTableView).resizeRowsToContents()
        # TisaiYu[2024/6/7] 测试时自动生成数据，不测试时把这里注释
        # row = self.funcMatrix.rowCount()
        # self.testWithSetValue(row)  # TisaiYu[2024/5/31] 给相关矩阵设置的值，但是还是需要输入AHP权重分析先。用于测试吧

    def progress_bar_change(self):
        self.count += 1
        value = self.progressBar.value() / 100
        value += 1 / (self.parts_num - 1) / 4 * 100
        if (100 - value) < (100 / (self.parts_num - 1) / 4):
            self.progressBar.setValue(10000)
            self.progressBar.setFormat("%.02f %%" % 100)
        else:
            self.progressBar.setValue(int(value * 100))
            self.progressBar.setFormat("%.02f %%" % value)
        if self.count == (self.parts_num - 1) * 4:
            self.progressBar.setValue(10000)
            self.progressBar.setFormat("%.02f %%" % 100)

    def classifyInputButton_clicked(self):
        self.time1 = time.time()
        """
        下面执行聚类
        """
        self.progressBar.setValue(0)
        if isinstance(self.thread_drawgram, QThread):
            self.thread_drawgram.quit()
            print("quit")
        if isinstance(self.thread_Q, QThread):
            self.thread_Q.quit()
            print("quit")
        if isinstance(self.thread_CE, QThread):
            self.thread_CE.quit()
            print("quit")
        if isinstance(self.thread_G, QThread):
            self.thread_G.quit()
            print("quit")
        if isinstance(self.thread_Sil, QThread):
            self.thread_Sil.quit()
            print("quit")
        # TisaiYu[2024/6/5] 先聚类（以模块化或者聚类的指标先得到一个方案）
        if np.any(self.correlation_weight == 0):
            QMessageBox.information(None, "信息", "请先执行AHP权重分析！")
            return False
        if self.read_excel_file == False:
            QMessageBox.information(None, "信息", "请先从文件导入各影响因素下的零件相关性矩阵数据！")
            return False
        self.thread_drawgram = DrawDendrogram(self.DSM)
        self.thread_drawgram.draw_finished.connect(self.cluster_begin)
        self.thread_drawgram.start()

    def reorg_dsm(self,dsm,cluster_labels):
        # 获取唯一的聚类标签并排序
        unique_clusters = np.unique(cluster_labels)
        # 创建新的索引数组
        new_indices = []
        for cluster in unique_clusters:
            new_indices.extend(np.where(cluster_labels == cluster)[0])
        # 重新排列矩阵
        reordered_matrix = self.DSM[np.ix_(new_indices, new_indices)]
        # 绘制矩阵
        fig, ax = plt.subplots()
        cax = ax.imshow(reordered_matrix, cmap='Blues', interpolation='nearest')
        # 绘制矩形框
        start_idx = 0
        for cluster in unique_clusters:
            indices = np.where(cluster_labels == cluster)[0]
            if len(indices) > 1:
                rect = matplotlib.patches.Rectangle((start_idx - 0.5, start_idx - 0.5), len(indices), len(indices),
                                                    linewidth=2,
                                                    edgecolor='r', facecolor='none')
                ax.add_patch(rect)
            start_idx += len(indices)

        plt.colorbar(cax)
        plt.show()

    def elbow_iter(self,Z):
        distances = Z[:, 2]
        if np.all(distances == distances[0]):  # TisaiYu[2024/10/28] 出现所有距离一样，那么就是要么全为一类，要么就每个都分开
            distance = distances[0] - 0.01
            cluster_labels = hierarchy.fcluster(Z, distance, criterion='distance')
        else:
            reversed_last = distances[::-1]
            indexes = np.arange(1, len(distances) + 1)
            # 计算距离增长加速度,或者采用距离的最大变化
            accelerations = np.diff(reversed_last, 2)
            # accelerations = np.diff(reversed_last, 1)
            accelerations = np.pad(accelerations, (1, 1), 'constant', constant_values=(0, 0))  # 在首尾补零
            # 确定最佳聚类数
            reversed_index = abs(np.argmax(
                accelerations[::-1]))  # TisaiYu[2024/10/28] 是因为存在多个相同最大值的情况的话，应该返回下标最大的那一个类数，比如5个相同，要么全为1类要么是5类
            module_optimum_num = len(accelerations) - reversed_index

            # TisaiYu[2024/10/28] 论文里的计算公式\
            d = np.zeros(distances.shape[0] + 1)
            d[0] = 0
            d[1:] = distances
            accelerations = np.array(
                [d[i] - 2 * d[i - 1] + d[i - 2] for i in range(2, len(d))])
            max_index = np.argmax(accelerations) + 3
            module_optimum_num = len(d) + 2 - max_index

            distance = float(Z[-(module_optimum_num - 1), 2]) - 0.01
            cluster_labels = hierarchy.fcluster(Z, distance, criterion='distance')

        best_cluster_labels = cluster_labels
        return best_cluster_labels
    def cluster_begin(self, Z, input_sequence, dendrogram_view):

        #  validate
        dis_mat = pdist(self.DSM,
                        'cosine')  # TisaiYu[2024/6/25] 层次聚类输入要么是一维数组（表示距离矩阵的压缩，比如30*30关联度矩阵，距离矩阵有30*30但是对称只取450），或者是二维数组（就是特征矩阵）

        # TisaiYu[2024/9/10] 因为功能对于很多零部件都有关联度的影响，而连接只对一些少数零部件有影响，而ward是类内方差链接法，因此对于功能这种对很多零部件有影响的大片区域数据，能够有所识别，因为功能影响大片，所以是方差稳定的主要因素，所以能先根据功能初始得到模块方案，再迭代求解
        Z = hierarchy.linkage(dis_mat, method='ward',
                              metric="cosine")  # TisaiYu[2024/9/5] 当输入是一个一维的压缩距离矩阵时（使用pdist的结果），那么metric是没有用，如果是一个二维的表示输入特征的数组，metric实际和pdist做了一样的事情
        distances = Z[:, 2]
        reversed_last = distances[::-1]
        indexes = np.arange(1, len(distances) + 1)
        # 计算距离增长加速度,或者采用距离的最大变化
        accelerations = np.diff(reversed_last, 2)
        # accelerations = np.diff(reversed_last, 1)
        accelerations = np.pad(accelerations, (1, 1), 'constant', constant_values=(0, 0))  # 在首尾补零
        # 确定最佳聚类数
        optimal_cluster_num = np.argmax(accelerations) + 1  # +1 是因为 diff 函数减少了一个元素
        cluster_labels = hierarchy.fcluster(Z, float(Z[-(optimal_cluster_num - 1), 2]) - 0.01,"distance")
        self.reorg_dsm(self.DSM,cluster_labels)


        # TisaiYu[2024/11/13] 每次迭代都显示划分结果后的排列DSM
        if  not self.essay_data:
            last_iter_module_num = 0
            unique_labels = np.unique(cluster_labels)
            while len(unique_labels)!=self.DSM.shape[0]:
                for module_index in unique_labels:
                    disobey_module_comps = np.where(cluster_labels == unique_labels[module_index-1])[0]
                    if len(disobey_module_comps)>1:
                        module_comps_indices = np.ix_(disobey_module_comps, disobey_module_comps)
                        DSM = self.DSM[module_comps_indices]
                        Z_new = hierarchy.linkage(DSM, method='single', metric="cosine")
                        small_best_clustering_labels = self.elbow_iter(Z_new)
                        cluster_labels[
                             disobey_module_comps] = small_best_clustering_labels + cluster_labels.max()  # TisaiYu[2024/8/30] 确保标签位移，就算不是连续也不影响分类
                        unique_labels_new = np.unique(cluster_labels)
                        # 创建一个字典，将原始标签映射到新的连续标签
                        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels_new, start=1)}
                        copy_labels = np.copy(cluster_labels)
                        # 使用字典进行标签转换
                        new_labels = np.array([label_mapping[label] for label in copy_labels])
                        self.reorg_dsm(self.DSM, new_labels)
                        this_iter_module_num = len(np.unique(new_labels))
                        if this_iter_module_num == last_iter_module_num:
                            continue
                    else:
                        continue
                cluster_labels = new_labels
                unique_labels = np.unique(cluster_labels)


        while self.dendrogramLayout.count():
            child = self.dendrogramLayout.takeAt(0)
            print(child)
            if child.widget():
                child.widget().deleteLater()
        # TisaiYu[2024/6/17] 初始化显示聚类树的类
        self.deview = dendrogram_view

        self.input_sequence = input_sequence

        AC_dict, HC_dict = self.get_AC_HC_dict_by_sql()
        # TisaiYu[2024/8/30] 测试查看选择的零部件组成的模块是否和论文里的数据一致
        # indices = [69,70,71,74]
        # minx = np.inf
        # maxx = -np.inf
        # miny = np.inf
        # maxy = -np.inf
        # minz = np.inf
        # maxz = -np.inf
        # part_weight_data = HC_dict.get("Weight")
        # part_size_data = HC_dict.get("Size")
        # module_weight=0
        # for idx in indices:
        #     module_weight += part_weight_data[idx]
        #     lowx, highx, lowy, highy, lowz, highz = part_size_data[idx]
        #     if lowx < minx:
        #         minx = lowx
        #     if highx > maxx:
        #         maxx = highx
        #     if lowy < miny:
        #         miny = lowy
        #     if highy > maxy:
        #         maxy = highy
        #     if lowz < minz:
        #         minz = lowz
        #     if highz > maxz:
        #         maxz = highz
        # module_size = [maxx - minx + 200, maxy - miny + 200, maxz - minz + 200]
        # z = module_weight
        # l = module_size[0]
        # w = module_size[1]
        # h = module_size[2]
        # mtf = FuzzySystemMT()
        # MT = mtf.calculate_MT(z,l,w,h)
        # print("MT:",MT)
        # exit()

        # self.thread_CE = HierarchicalClustering(self.DSM, Z, 'CE', 1,AC_dict,HC_dict)
        # self.thread_Sil = HierarchicalClustering(self.DSM, Z, 'Sil', 2,AC_dict,HC_dict)
        # self.thread_Q = HierarchicalClustering(self.DSM, Z, 'Q', 4,AC_dict,HC_dict)
        # self.thread_G = HierarchicalClustering(self.DSM, Z, 'G', 3, AC_dict, HC_dict)

        self.thread_SIC = HierarchicalClustering(self.DSM, Z,'SIC', 5, AC_dict, HC_dict, self.ACP_dict)

        # self.thread_CE.clustering_finished.connect(self.cluster_optimize)
        # self.thread_Sil.clustering_finished.connect(self.cluster_optimize)
        # self.thread_Q.clustering_finished.connect(self.cluster_optimize)
        # self.thread_G.clustering_finished.connect(self.cluster_optimize)
        self.thread_SIC.clustering_finished.connect(self.cluster_optimize)

        # self.thread_CE.progress_sig.connect(self.progress_bar_change)
        # self.thread_Q.progress_sig.connect(self.progress_bar_change)
        # self.thread_Sil.progress_sig.connect(self.progress_bar_change)
        # self.thread_G.progress_sig.connect(self.progress_bar_change)
        self.thread_SIC.progress_sig.connect(self.progress_bar_change)
        self.thread_SIC.divide_dedrogram.connect(self.dedrogram_new)

        # self.thread_CE.start()
        # self.thread_Sil.start()
        # self.thread_Q.start()
        # self.thread_G.start()
        self.thread_SIC.start()

        self.time1 = time.time()
        self.dendrogramLayout.addWidget(dendrogram_view)


    def cluster_optimize(self, best_cluster_labels, best_value, metric_name, values_dict, curve_id,module_constraints_index,MTs,ATs):

        print(f'最佳聚类方案的 {metric_name} 值：', best_value)
        print('最佳聚类方案的类别标签：', best_cluster_labels)
        print("-------------------------------------------------------")
        module_weight, module_size = module_iterate_constraints(best_cluster_labels, self.HC_dict, iter_phase=False)
        print(module_weight)
        print(module_size)
        curvesFrameList = self.curvesFrame.findChildren(QFrame)
        curvesFrameList = sorted(curvesFrameList, key=lambda x: int(x.objectName()[-1].split("Frame")[-1]))
        frame = curvesFrameList[curve_id - 1]
        curveLayout = frame.findChild(QHBoxLayout)
        self.dipict_cureve(metric_name, "class_nums", "value", curveLayout,
                           values_dict,MTs,ATs, module_constraints_index,3, 2, 80)  # TisaiYu[2024/6/21] 多线程导致还没有记录i时另一个线程就已经判断完了，导致绘制在了一个layout上。

        self.moduleResultInfoTable.setWordWrap(True)
        item_col1 = QStandardItem(f"方案{curve_id}:")
        item_col2 = QStandardItem(f"{metric_name}")
        item_col3 = QStandardItem(f"{len(np.unique(best_cluster_labels))}")
        self.moduleResultInfoTable.model().setItem(curve_id - 1, 0, item_col1)
        self.moduleResultInfoTable.model().setItem(curve_id - 1, 1, item_col2)
        self.moduleResultInfoTable.model().setItem(curve_id - 1, 2, item_col3)
        item_col4_str = ''
        for i, label in enumerate(np.unique(best_cluster_labels)):
            module_parts = np.array(self.input_sequence)[best_cluster_labels == (i + 1)]
            if label != np.unique(best_cluster_labels)[-1]:
                item_col4_str += f"M{i + 1}:{module_parts}, "
            else:
                item_col4_str += '.'
        item_col4 = QStandardItem(item_col4_str)
        self.moduleResultInfoTable.model().setItem(curve_id - 1, 3, item_col4)
        delegate = WordWrapDelegate()
        self.moduleResultInfoTable.setItemDelegate(delegate)
        self.moduleResultInfoTable.resizeRowsToContents()
        self.time2 = time.time()
        print(f"655个输入的模块化计算时间为：{self.time2-self.time1}秒")
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
            self.factorsSelectAll.stateChanged.disconnect(
                self.factors_selectAll_changed)  # TisaiYu[2024/6/6] 防止设置0时触发上面的槽函数
            self.factorsSelectAll.setChecked(0)
            self.factorsSelectAll.stateChanged.connect(self.factors_selectAll_changed)
        j_model = QStandardItemModel(correlation_nums, correlation_nums)
        self.judge_matrix_table.setModel(j_model)
        self.judge_matrix_table.model().dataChanged.connect(self.setRecripocal)
        delegate_j = DecimalDelegate(3)
        self.judge_matrix_table.setItemDelegate(delegate_j)

        labels = [label.text() for label in self.correlation_selected_indexs]
        self.judge_matrix_table.model().setHorizontalHeaderLabels(labels)
        self.judge_matrix_table.model().setVerticalHeaderLabels(labels)
        labels.append("一致性检验")
        a_model = QStandardItemModel(len(labels), 1)
        self.ahpResultTable.setModel(a_model)
        delegate_a = DecimalDelegate(3)
        self.ahpResultTable.setItemDelegate(delegate_a)
        self.ahpResultTable.model().setHorizontalHeaderLabels(["结果"])
        self.ahpResultTable.model().setVerticalHeaderLabels(labels)

        for i, item in enumerate(self.correlation_selected_indexs):
            self.judge_matrix_table.model().setItem(i, i, QStandardItem("1"))

    def loadFileButton_clicked(self):

        file_dialog = \
        QFileDialog.getOpenFileName(self, "打开关联度表excel文件！", '', 'Excel文件 (*.xlsx);;所有文件 (*)')[
            0]  # TisaiYu[2024/6/12] 改打开文件的初始路径
        if file_dialog == '':  # TisaiYu[2024/6/20] 处理什么都没选就叉掉的情况
            return False
        data_file_exc = pd.ExcelFile(file_dialog)
        sheet_names = data_file_exc.sheet_names
        time1 = time.time()
        data_dict = {sheet_name: pd.read_excel(file_dialog, sheet_name, header=None) for sheet_name in sheet_names}
        self.parts_num = data_dict[sheet_names[0]].shape[0]
        self.correlation_matrices = np.zeros((correlation_nums, self.parts_num, self.parts_num))
        matricesFrameList = self.matricesFrame.findChildren(QFrame)
        matricesFrameList = sorted(matricesFrameList, key=lambda x: int(x.objectName().split("Frame")[-1]))
        for i in range(correlation_nums):
            index = self.factorsList.row(self.correlation_selected_indexs[i])
            self.correlation_matrices[i, :, :] = np.array(data_dict[sheet_names[i]])
            matrix_layout = matricesFrameList[i].findChild(QHBoxLayout)
            print(matricesFrameList[i].objectName())
            self.dipict_matrix(sheet_names[i], "index_row", "index_col", matrix_layout,
                               self.correlation_matrices[i, :, :], 2.4, 1.8, 80)
        time2 = time.time()
        print("加载excel文件时间：", time2 - time1)
        self.partsInputNumEdit.setText("数量：" + str(self.parts_num))
        self.DSM = np.zeros((self.parts_num, self.parts_num))
        for i in range(self.correlation_weight.shape[0]):
            self.DSM += self.correlation_weight[i] * self.correlation_matrices[i]

        # 将结果保存为Excel文件
        # df = pd.DataFrame(self.DSM)
        # df.index = df.index+1
        # df.to_excel('./data/DSM.xlsx',header=False)
        self.dipict_matrix('Correlation Matrix', 'index_row', 'index_col', self.DSMLayout, self.DSM)
        self.read_excel_file = True

    def dipict_matrix(self, title, xlabel, ylabel, layout, pic_array, width=4, height=5, dpi=100):
        """
        :param title:
        :param xlabel:
        :param ylabel:
        :return:
        """
        # TisaiYu[2024/7/10]  先清空
        while layout.count():
            child = layout.takeAt(0)
            print(child)
            if child.widget():
                child.widget().deleteLater()

        canvas = PlotCanvas(width, height, dpi)
        canvas.axes.set_title(title)
        canvas.axes.set_xlabel(xlabel)
        canvas.axes.set_ylabel(ylabel)
        layout.addWidget(canvas)
        canvas.axes.imshow(pic_array, cmap='Blues', interpolation='nearest')

    def dipict_cureve(self, title, xlabel, ylabel, layout, pic_array_dict,MTs,ATs, module_constraints_index,width=3, height=2, dpi=100):
        """
        :param title:
        :param xlabel:
        :param ylabel:
        :return:
        """
        while layout.count():
            child = layout.takeAt(0)
            print(child)
            if child.widget():
                child.widget().deleteLater()
        fontsize = 6
        canvas = PlotCanvas(width, height, dpi)
        canvas.axes.set_title(title, fontsize=fontsize, y=0.94)
        canvas.axes.set_xlabel(xlabel, fontsize=fontsize)
        canvas.axes.set_ylabel(ylabel, fontsize=fontsize)
        canvas.axes.set_ylim(0,3000)
        canvas.axes.xaxis.set_label_coords(0.9, 1.15)
        canvas.axes.yaxis.set_label_coords(1.1, 0.9)
        for label in canvas.axes.get_xticklabels():
            label.set_verticalalignment('top')  # 垂直对齐方式，可选 'center', 'top', 'bottom', 'baseline', 'center_baseline'
            label.set_y(0.025)
        canvas.axes.tick_params(axis='both', labelsize=4)
        for values_name in pic_array_dict.keys():
            pic_array = list(pic_array_dict[values_name].values())
            module_num_list = list(pic_array_dict[values_name].keys())
            if len(pic_array) > 10:
                canvas.axes.set_xticks(range(0, len(pic_array) + 3, len(pic_array) // 10))
            else:
                canvas.axes.set_xticks(range(0, len(pic_array) + 3))
            canvas.axes.plot(module_num_list, pic_array,label=values_name)
        canvas.axes.legend()
        layout.addWidget(canvas)
        QApplication.processEvents()

        canvas.axes.axvline(x=module_constraints_index,color='red')

        # validate
        fig, ax = plt.subplots()
        ax.plot(module_num_list, MTs, label="MT")
        ax.plot(module_num_list, ATs, label="AT")
        for values_name in pic_array_dict.keys():
            pic_array = list(pic_array_dict[values_name].values())
            module_num_list = list(pic_array_dict[values_name].keys())

            ax.plot(module_num_list,pic_array,label=values_name)
            ax.axvline(x=module_constraints_index,color='red')
        ax.text(module_constraints_index-2, ax.get_ylim()[0], f'{module_constraints_index}', color='red',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        plt.show()

    def export_module_info(self):
        if self.moduleResultInfoTable.model().item(0, 0) is None:
            QMessageBox.information(None, "信息", "还没有模块划分结果！")
            return False

        exp_filename = QFileDialog.getSaveFileName(self, '保存文件', '', 'Excel文件 (*.xlsx);;所有文件 (*)')[0]
        if exp_filename:
            model = self.moduleResultInfoTable.model()
            data = []
            for row in range(model.rowCount()):
                data.append([])
                for col in range(model.columnCount()):
                    index = model.index(row, col)
                    data[row].append(str(model.data(index)))
            df = pd.DataFrame(data)
            df.to_excel(exp_filename, index=False, header=False)

            # 打开Excel文件并获取数据
            df = pd.read_excel(exp_filename, header=None)

            # 遍历所有列，调整列宽以适应内容
            writer = pd.ExcelWriter(exp_filename, engine='xlsxwriter')
            df.to_excel(writer, index=False, header=False, sheet_name='Sheet1')
            worksheet = writer.sheets['Sheet1']

            for idx, col in enumerate(df):
                max_length = max(df[col].astype(str).map(len).max(), len(str(df.columns[idx]))) + 2
                worksheet.set_column(idx, idx, max_length)

            # 保存更改
            writer.close()
        else:
            return False

    def create_db(self):
        dbfile = QFileDialog.getSaveFileName(None, "创建数据库文件", '', "SQLite Database Files (*.db);;All Files (*)")[
            0]
        self.db_file = dbfile
        if dbfile != '':
            self.conn = sqlite3.connect(dbfile)
            self.cursor = self.conn.cursor()  # TisaiYu[2024/7/22] cursor是数据库的相关概念，表示操作数据库的实体，进行增删改查命令的实体
            # TisaiYu[2024/7/22] 在这下面添加数据库的表创建的相关代码
            self.cursor.execute('''
                CREATE TABLE connection_table (
                    component_id VARCHAR(255),
                    connected_component_id VARCHAR(255)
                );
            ''')
            self.cursor.execute('''
                CREATE TABLE FunctionTable (
                    FunctionID VARCHAR(255) PRIMARY KEY,
                    FunctionName TEXT NOT NULL
                );
            ''')
            self.cursor.execute('''
                CREATE TABLE PartFunctionTable (
                    PartID INT PRIMARY KEY,
                    FunctionID VARCHAR(255),
                    FOREIGN KEY(FunctionID) REFERENCES FunctionTable(FunctionID)
                );
            ''')
            self.conn.commit()
            self.cursor.execute('''
                        SELECT name
                        FROM sqlite_master
                        WHERE type='table'
                        ''')
            tables = self.cursor.fetchall()  # TisaiYu[2024/7/23] 返回查询结果每一列，这里只有一列，所以下面取[0]
            sql_tree_model = QStandardItemModel()
            for table in tables:
                tree_item = QStandardItem(table[0])
                sql_tree_model.appendRow(tree_item)
            self.SQLformsTree.setModel(sql_tree_model)
        else:
            return False

    def create_db_qt(self):
        dbfile = QFileDialog.getSaveFileName(None, "创建数据库文件", '', "SQLite Database Files (*.db);;All Files (*)")[
            0]
        self.db_file = dbfile  # TisaiYu[2024/8/28] 这个是用于传入给后续模块的计算函数的，因为是多线程，必须有单独的连接，不能共享一个连接
        if dbfile != '':
            self.db.setDatabaseName(dbfile)
            self.db.open()
            query = QtSql.QSqlQuery(self.db)
            query.exec_("PRAGMA foreign_keys = ON;")

            query.exec_('''
                            CREATE TABLE FunctionTable (
                                FunctionID VARCHAR(255) PRIMARY KEY,
                                Description TEXT NOT NULL
                            );
                        ''')
            query.exec_('''
                           CREATE TABLE ConnectionTable (
                               ConnectionID VARCHAR(255) PRIMARY KEY,
                               Description TEXT NOT NULL
                           );
                                   ''')
            query.exec_('''
                            CREATE TABLE AddRecordTable (
                            PartID VARCHAR(255)
                            );
                        ''')  # TisaiYu[2024/8/12] 注意这里FunctionID是逗号分割的，所以不能写为外键
            query.exec_('''
                CREATE TABLE PartsTable (
                    PartID VARCHAR(255) PRIMARY KEY
                    );
            ''')  # TisaiYu[2024/8/12] 外键必须要等引用的那个表的属性有记录后才能添加，否则添加不进去，也就是要先编写好功能编码表

            query.exec_('''
                CREATE TABLE PartsConnectionTable (
                    PartID VARCHAR(255)
                    )
            ''')

            query.exec_('''
                CREATE TABLE PartsFunctionTable (
                    PartID VARCHAR(255),
                    FunctionID VARCHAR(255),
                    FOREIGN KEY(PartID) REFERENCES PartsTable(PartID) ON DELETE CASCADE,
                    FOREIGN KEY(FunctionID) REFERENCES FunctionTable(FunctionID) ON DELETE CASCADE
                );
            ''')
            query.exec_('''
                            CREATE TABLE TablesInfo (
                                TableChineseName VARCHAR(255),
                                TableEnglishName VARCHAR(255),
                                TableType VARCHAR(255),
                                PRIMARY KEY (TableChineseName, TableEnglishName)
                            );
                        ''')
            query.exec_('''
                            CREATE TABLE AttributesInfo (
                                AttributeName VARCHAR(255) PRIMARY KEY,
                                AttributeTable VARCHAR(255),
                                AttributeRow INT
                            );
                        ''')
            query.exec_("PRAGMA FOREIGN_KEYS=ON;")
            self.sum_query_model.setTable("AddRecordTable")  # TisaiYu[2024/8/9] 用于一直更新总表的数据，而qt表格显示的数据的模型用另一个变量来
            sqlmodel = QtSql.QSqlTableModel()
            sqlmodel.setTable("AddRecordTable")
            sql_tree_model = QStandardItemModel()
            parent_tree_item1 = QStandardItem("组件多行表")
            parent_tree_item2 = QStandardItem("参考表")
            self.table_names_dict = {"添加记录": ["AddRecordTable", 0], "组件单行表": ["PartsTable", 1],
                                     "零部件连接表": ["PartsConnectionTable", 2],
                                     "零部件功能表": ["PartsFunctionTable", 2], "功能编码表": ["FunctionTable", 3],
                                     "连接编码表": ["ConnectionTable", 3]}
            self.attribute_dict = {"PartID": ["PartsTable", 0], "FunctionID": ["FunctionTable", 0],
                                   "ConnectionID": ["ConnectionTable", 0], "ConnectPartID": ["ConnectionTable", 1],
                                   "ConnectionPartID": ["ConnectionTable", 1]}
            for key, item in self.attribute_dict.items():
                query.exec_(f'''
                                INSERT INTO AttributesInfo (AttributeName,AttributeTable,AttributeRow)
                                VALUES ('{key}','{item[0]}',{item[1]})
                ''')
            for key, item in self.table_names_dict.items():
                query.exec_(f'''
                                INSERT INTO TablesInfo (TableChineseName,TableEnglishName,TableType)
                                VALUES ('{key}','{item[0]}','{item[1]}')
                ''')
                if item[1] not in [2, 3]:
                    tree_item = QStandardItem(key)
                    sql_tree_model.appendRow(tree_item)
                elif item[1] == 2:
                    tree_item = QStandardItem(key)
                    parent_tree_item1.appendRow(tree_item)
                elif item[1] == 3:
                    tree_item = QStandardItem(key)
                    parent_tree_item2.appendRow(tree_item)
            sql_tree_model.appendRow(parent_tree_item1)
            sql_tree_model.appendRow(parent_tree_item2)

            self.SQLformsTree.setModel(sql_tree_model)
            self.SQLformsTree.clicked.connect(self.SQLTableshow)
            self.SQLTableView.setModel(self.sum_query_model)
        else:
            return False

    def load_db(self):
        dbfile = QFileDialog.getOpenFileName(None, "加载数据库文件", '', "SQLite Database Files (*.db);;All Files (*)")[
            0]
        if dbfile != '':
            self.conn = sqlite3.connect(dbfile)
            # TisaiYu[2024/7/22] 在这下面添加读取数据库表并且显示到树结构和表格的代码
            self.cursor = self.conn.cursor()
            self.cursor.execute('''
                                    SELECT name
                                    FROM sqlite_master
                                    WHERE type='table'
                                    ''')
            tables = self.cursor.fetchall()  # TisaiYu[2024/7/23] 返回查询结果每一列，这里只有一列，所以下面取[0]
            sql_tree_model = QStandardItemModel()
            for table in tables:
                tree_item = QStandardItem(table[0])
                sql_tree_model.appendRow(tree_item)
            self.SQLformsTree.setModel(sql_tree_model)
            self.SQLformsTree.clicked.connect(self.SQLTableshow)
            self.sum_query_model.setTable("AddRecordTable")
            self.attribute_dict = {"PartID": "PartsTable", "ConnectPartID": "PartsConnectionTable"}
        else:
            return False

    def load_db_qt(self):
        if self.db.isOpen():
            self.db.close()
        dbfile = QFileDialog.getOpenFileName(None, "加载数据库文件", '', "SQLite Database Files (*.db);;All Files (*)")[
            0]
        self.db_file = dbfile
        if dbfile != '':
            self.db.setDatabaseName(dbfile)
            self.db.open()
            self.attribute_dict = {"PartID": ["PartsTable", 0], "FunctionID": ["FunctionTable", 0],
                                   "ConnectionID": ["ConnectionTable", 0], "ConnectPartID": ["ConnectionTable", 1],
                                   "ConnectionPartID": ["ConnectionTable", 1]}
            self.update_form_tree()
            self.update_attributes()

        else:
            return False

    def update_attributes(self):
        query = QtSql.QSqlQuery()
        query.exec_("PRAGMA FOREIGN_KEYS=ON;")
        query.exec_('''
                                            SELECT *
                                            FROM AttributesInfo;
                                            ''')

        while query.next():
            attribute_name = query.value(0)
            attribute_table = query.value(1)
            attribute_row = query.value(2)
            self.attribute_dict[attribute_name] = [attribute_table, attribute_row]

    def update_form_tree(self):
        query = QtSql.QSqlQuery()
        query.exec_("PRAGMA FOREIGN_KEYS=ON;")
        query.exec_('''
                                    SELECT *
                                    FROM TablesInfo;
                                    ''')
        row_list = []
        while query.next():
            chinese_name = query.value(0)
            english_name = query.value(1)
            type_ = query.value(2)
            row_list.append([chinese_name, english_name, type_])
            self.table_names_dict[chinese_name] = [english_name, type_]
        sql_tree_model = QStandardItemModel()
        parent_tree_item1 = QStandardItem("组件多行表")
        parent_tree_item2 = QStandardItem("参考表")
        for row in row_list:
            if row[2] not in ['2', '3']:
                tree_item = QStandardItem(row[0])
                sql_tree_model.appendRow(tree_item)
            elif row[2] == '2':
                tree_item = QStandardItem(row[0])
                parent_tree_item1.appendRow(tree_item)
            elif row[2] == '3':
                tree_item = QStandardItem(row[0])
                parent_tree_item2.appendRow(tree_item)
        sql_tree_model.appendRow(parent_tree_item1)
        sql_tree_model.appendRow(parent_tree_item2)
        self.SQLformsTree.setModel(sql_tree_model)
        self.SQLformsTree.clicked.connect(self.SQLTableshow)
        self.sum_query_model.setTable("AddRecordTable")
        self.sum_query_model.select()
        self.SQLTableView.setModel(self.sum_query_model)
        first_index = self.SQLformsTree.model().index(0, 0)
        self.SQLformsTree.setCurrentIndex(first_index)

    def SQLTableshow(self, index):

        if self.SQLformsTree.model().itemData(index)[0] in ["组件多行表", "参考表"]:
            return False
        table_name = self.table_names_dict[self.SQLformsTree.model().itemData(index)[0]][0]
        if table_name != "AddRecordTable":
            sql_model = QtSql.QSqlTableModel()
            sql_model.setTable(table_name)
            sql_model.setEditStrategy(
                QtSql.QSqlTableModel.EditStrategy.OnFieldChange)  # TisaiYu[2024/7/23] 设置qt中表格更新时数据库更新的模式，参考https://www.baidu.com/link?url=1LeHpdwr3tETGWPrEVunG-Cq9nLXgXMOx1A5n82TwXU6KH7fTXhY0xLkQMfk8IeX6frt1D-SFke3OOsPaZVAzeaIfJXpxWxZEl3T0vSg93e&wd=&eqid=dded5474005e72fd00000006669f9ec2
            sql_model.select()
            self.SQLTableView.setModel(sql_model)
        else:
            self.sum_query_model.select()
            self.SQLTableView.setModel(self.sum_query_model)
        self.SQLTableView.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.querying = False

    def sql_add_one_row(self):
        table_name = self.table_names_dict[self.SQLformsTree.model().itemData(self.SQLformsTree.currentIndex())[0]][0]
        if table_name != "AddRecordTable":
            sqltable_model = QtSql.QSqlTableModel()
            sqltable_model.setTable(table_name)
            sqltable_model.setEditStrategy(QtSql.QSqlTableModel.EditStrategy.OnFieldChange)
            sqltable_model.select()
            record = sqltable_model.record()
            for i in range(self.sqlinput_dialog_ui.partSQLinputInfotable.columnCount()):
                text = self.sqlinput_dialog_ui.partSQLinputInfotable.item(0, i).text()
                record.setValue(i, text)
            sqltable_model.insertRecord(-1, record)
            sqltable_model.submitAll()
            sqltable_model.select()
            self.SQLTableView.setModel(sqltable_model)
            return True
        self.update_sql_add(table_name)

    def update_sql_add(self, table_name):
        sqltable_model = QtSql.QSqlTableModel()
        query = QtSql.QSqlQuery()
        # TisaiYu[2024/8/12] 更新添加记录表，只是记录，单元格里有逗号分割的多个值
        if table_name == "AddRecordTable":  # TisaiYu[2024/8/14] 更新除AddRecordTable和参考表的其他表
            # TisaiYu[2024/8/12] 更新其他属性表,是按照添加属性的顺序来的。

            for j in range(self.SQLTableView.model().rowCount()):
                part_id = self.SQLTableView.model().data(self.SQLTableView.model().index(j, 0))
                name_type = self.SQLTableView.model().data(self.SQLTableView.model().index(j, 1))
                if name_type != "cube":  # TisaiYu[2024/8/15] 添加记录的和管道的就别管了，在添加记录零部件时就自动添加了管道进入数据库的其他表了
                    attribute_multirow_list = []
                    attribute_singlerow_list = []
                    multi_attr_count = 0
                    single_attr_count = 0
                    for i in range(1, self.SQLTableView.model().columnCount()):
                        text = self.SQLTableView.model().data(self.SQLTableView.model().index(j, i))
                        part_attribute = text.split(',')
                        if len(part_attribute) > 1:  # TisaiYu[2024/8/14] 是多行属性
                            combinations = list(itertools.product(
                                *attribute_singlerow_list))  # TisaiYu[2024/8/12] 记住，多个集合，每个集合取一个出来组合，组合出来的所有可能就是笛卡尔积，使用list(itertools.product(*set_list))来得到
                            sqltable_model.setTable("PartsTable")
                            sqltable_model.setEditStrategy(QtSql.QSqlTableModel.EditStrategy.OnFieldChange)
                            sqltable_model.select()
                            record = sqltable_model.record()
                            for record_list in combinations:
                                for i in range(len(record_list)):
                                    record.setValue(i + 1, record_list[i])
                                    record.setValue(0, part_id)
                                sqltable_model.insertRecord(-1, record)
                                sqltable_model.submitAll()
                                sqltable_model.select()

                            attribute_multirow_list.append(part_attribute)
                            multi_attr_count += 1
                            if multi_attr_count == 2:  # TisaiYu[2024/8/14] 前两个是连接相关的，连接相关的要处理为添加管道记录进去，两个零部件的连接要添加两个记录，分别表示这两个零部件和连接管道的连接和连接类型
                                # TisaiYu[2024/8/15] 先添加有主键的，也就是添加记录，把管道添加进去，尺寸重量先默认为0，ID为“非管道零部件ID-管道ID”，其中管道ID为”连接的第一个组件ID-第二个组件ID“，Name字段为Cube
                                sql_model = QtSql.QSqlTableModel()
                                sql_model.setTable("PartsTable")
                                sql_model.select()
                                record = sql_model.record()
                                for count_index, connect_partid in enumerate(
                                        attribute_multirow_list[0]):  # TisaiYu[2024/8/15] 多少个连接就添加多少个管道组件，也就是添加记录和单行表数据
                                    cube_id = f"{part_id}*{connect_partid}"
                                    cube_id_count = f"{connect_partid}*{part_id}"

                                    query.exec_(f'''
                                                        SELECT COUNT(*) FROM PartsTable
                                                        WHERE PartID='{cube_id_count}'
                                                    ''')
                                    if query.next() and query.value(0) != 0:
                                        continue
                                    record.setValue(0, cube_id)
                                    record.setValue(1, "cube")
                                    for m in range(sql_model.columnCount() - 2):
                                        record.setValue(m + 2, "0")
                                    sql_model.insertRecord(-1, record)
                                    sql_model.submitAll()
                                    sql_model.setTable(
                                        "PartsConnectionTable")  # TisaiYu[2024/8/15] 不想多写一个外层循环，就只有每次循环都设置一下表了。
                                    record = sql_model.record()
                                    connection_id1 = attribute_multirow_list[1][2 * count_index]
                                    connection_id2 = attribute_multirow_list[1][2 * count_index + 1]
                                    record.setValue(0, part_id)
                                    record.setValue(1, cube_id)
                                    record.setValue(2, connection_id1)
                                    sql_model.insertRecord(-1, record)
                                    sql_model.submitAll()
                                    sqltable_model.select()
                                    record.setValue(0, connect_partid)
                                    record.setValue(1, cube_id)
                                    record.setValue(2, connection_id2)
                                    sql_model.insertRecord(-1, record)
                                    sql_model.submitAll()
                                    sqltable_model.select()
                                    sql_model.setTable("PartsFunctionTable")
                                    sqltable_model.select()
                                    record = sql_model.record()
                                    record.setValue(0, cube_id)
                                    record.setValue(1, "0")
                                    sql_model.insertRecord(-1, record)
                                    sql_model.submitAll()
                                    sql_model.select()
                            elif multi_attr_count > 2:
                                tablename = self.table_names_dict[
                                    self.SQLformsTree.model().item(2).child(multi_attr_count - 2).text()][0]
                                sqltable_model.setTable(tablename)
                                sqltable_model.setEditStrategy(QtSql.QSqlTableModel.EditStrategy.OnFieldChange)
                                sqltable_model.select()
                                record = sqltable_model.record()
                                multi_attr_count += 1
                                for attribute in part_attribute:
                                    record.setValue(0, part_id)
                                    record.setValue(1, attribute)
                                sqltable_model.insertRecord(-1, record)
                                sqltable_model.submitAll()
                                sqltable_model.select()
                        else:  # TisaiYu[2024/8/14] 是单行属性
                            single_attr_count += 1
                            attribute_singlerow_list.append(part_attribute)
            sqltable_model.setTable("PartsTable")
            sqltable_model.select()
            query.exec_('''
                        CREATE TABLE NewPartsTable AS
                        SELECT * FROM PartsTable
                        ORDER BY 
                            CASE 
                                WHEN PartID LIKE '%-%' THEN 1 
                                ELSE 0 
                            END, 
                            PartID;
                    ''')
            query.exec_("PRAGMA foreign_keys = OFF;")  # TisaiYu[2024/8/16] 不然其他引用了零部件表的会被删除数据
            query.exec_('DROP TABLE PartsTable')
            query.exec_('ALTER TABLE NewPartsTable RENAME TO PartsTable')
            query.exec_("PRAGMA FOREIGN_KEYS=ON;")

            # TisaiYu[2024/8/12] 如果当前就是汇总表则更新汇总表的显示
            self.sum_query_model.setTable("AddRecordTable")
            self.sum_query_model.select()
            self.SQLTableView.setModel(self.sum_query_model)
        else:  # TisaiYu[2024/8/14] 参考表的数据
            for j in range(self.sqlinput_dialog_ui.partSQLinputInfotable.rowCount()):
                sqltable_model.setTable(table_name)
                sqltable_model.setEditStrategy(QtSql.QSqlTableModel.EditStrategy.OnFieldChange)
                sqltable_model.select()
                record = sqltable_model.record()
                for i in range(0, self.sqlinput_dialog_ui.partSQLinputInfotable.columnCount()):
                    text = self.sqlinput_dialog_ui.partSQLinputInfotable.item(j, i).text()
                    record.setValue(i, text)
                sqltable_model.insertRecord(-1, record)
                sqltable_model.submitAll()
                sqltable_model.select()

    def update_sql_excel_cube_create(self):
        type = "Cube"
        query = QtSql.QSqlQuery(self.db)
        query.exec_(F'''SELECT DISTINCT o1.PartID, o1.ConnectionPartID, o2.PartName, o3.PartName 
                        FROM PartsConnectionTable o1
                        JOIN PartsTable o2 ON o1.PartID=o2.PartID
                        JOIN PartsTable o3 ON o1.ConnectionPartID=o3.PartID
                        WHERE o1.PartID<o1.ConnectionPartID;''')
        sql_model = QtSql.QSqlTableModel()
        while query.next():
            connpartid1 = query.value(0)
            connpartid2 = query.value(1)
            connpartname1 = query.value(2)
            connpartname2 = query.value(3)
            cube_name = f'''{connpartname1}*{connpartname2}'''
            sql_model.setTable("PartsTable")
            sql_model.setEditStrategy(QtSql.QSqlTableModel.EditStrategy.OnFieldChange)
            sql_model.select()
            record = sql_model.record()
            row_count = sql_model.rowCount()
            row_count = str(row_count)
            set_dict = {0: row_count, 1: cube_name, 3: type}
            for i in range(sql_model.columnCount()):
                if i in set_dict.keys():
                    record.setValue(i, set_dict[i])
                else:
                    record.setValue(i, '/')
            sql_model.insertRecord(-1, record)
            sql_model.submitAll()
            sql_model.select()

            sql_model.setTable("PartsConnectionTable")
            sql_model.setEditStrategy(QtSql.QSqlTableModel.EditStrategy.OnFieldChange)
            sql_model.select()
            record = sql_model.record()
            connpair = [[connpartid1, row_count], [connpartid2, row_count], [row_count, connpartid2],
                        [row_count, connpartid1]]
            for i in range(len(connpair)):
                record.setValue(0, connpair[i][0])
                record.setValue(1, connpair[i][1])
                for j in range(2, sql_model.columnCount()):
                    record.setValue(j, '/')
                sql_model.insertRecord(-1, record)
                sql_model.submitAll()
                sql_model.select()

            sql_model.setTable("PartsFunctionTable")
            sql_model.setEditStrategy(QtSql.QSqlTableModel.EditStrategy.OnFieldChange)
            sql_model.select()
            record = sql_model.record()
            query_new = QtSql.QSqlQuery()  # TisaiYu[2024/8/31] 这里不能和循环外的那个变量名相同，否则只执行一行
            query_new.exec_(f'''SELECT o1.Function
                            FROM PartsFunctionTable o1
                            JOIN PartsFunctionTable o2
                            ON o1.Function = o2.Function
                            WHERE o1.PartID={connpartid1} AND o2.PartID={connpartid2};''')
            function_list = []
            if query_new.next():
                function_list.append(query_new.value(0))
                record.setValue(0, row_count)
                record.setValue(1, query_new.value(0))
                sql_model.insertRecord(-1, record)
                while query_new.next():
                    function_list.append(query_new.value(0))
                    record.setValue(0, row_count)
                    record.setValue(1, query_new.value(0))
                    sql_model.insertRecord(-1, record)
            else:
                function_list.append('/')
                record.setValue(0, row_count)
                record.setValue(1, '/')
                sql_model.insertRecord(-1, record)
                sql_model.submitAll()
                sql_model.select()
            functions = ','.join(function_list)
            cube_conn_part = [connpartid1, connpartid2]
            cube_conns = ','.join(cube_conn_part)
            sql_model.setTable("AddRecordTable")
            sql_model.setEditStrategy(QtSql.QSqlTableModel.EditStrategy.OnFieldChange)
            sql_model.select()
            record = sql_model.record()
            set_value_list = [0, 1, 3, 5, 6]
            record.setValue(0, row_count)
            record.setValue(1, cube_name)
            record.setValue(3, type)
            record.setValue(5, functions)
            record.setValue(6, cube_conns)
            for i in range(sql_model.columnCount()):
                if i not in set_value_list:
                    record.setValue(i, '/')
            sql_model.insertRecord(-1, record)
            sql_model.submitAll()
            sql_model.select()

    def update_sql_excel(self, table_name):
        sqltable_model = QtSql.QSqlTableModel()
        attribute_multirow_index = defaultdict(list)  # TisaiYu[2024/8/22] 默认字典的值为列表的字典
        attribute_singlerow_index = []
        if table_name == "AddRecordTable":  # TisaiYu[2024/8/14] 更新除AddRecordTable和参考表的其他表

            for i in range(self.SQLTableView.model().columnCount()):
                attr_name = self.SQLTableView.horizontalHeader().model().headerData(i, Qt.Horizontal)
                attr_row = self.attribute_dict[attr_name][1]
                if attr_row == 1:  # TisaiYu[2024/8/14] 是多行属性
                    attr_table_name = self.attribute_dict[attr_name][0]
                    attribute_multirow_index[attr_table_name].append(i)
                else:  # TisaiYu[2024/8/14] 是单行属性
                    attribute_singlerow_index.append(i)
            while self.SQLTableView.model().canFetchMore():
                self.SQLTableView.model().fetchMore()
            for i in range(self.SQLTableView.model().rowCount()):
                # TisaiYu[2024/8/22] 添加单行的
                sqltable_model.setTable("PartsTable")
                sqltable_model.setEditStrategy(QtSql.QSqlTableModel.EditStrategy.OnFieldChange)
                sqltable_model.select()
                record = sqltable_model.record()
                for index, attr_sigindex in enumerate(attribute_singlerow_index):
                    text = self.SQLTableView.model().data(self.SQLTableView.model().index(i, attr_sigindex))
                    record.setValue(index, text)
                sqltable_model.insertRecord(-1, record)
                sqltable_model.submitAll()
                sqltable_model.select()

                # TisaiYu[2024/8/22] 添加多行的
                for table_name, this_table_multi_attribues_index in attribute_multirow_index.items():
                    max_rows_count = 0
                    for attr_index in this_table_multi_attribues_index:
                        anytext = self.SQLTableView.model().data(
                            self.SQLTableView.model().index(i, attr_index))
                        text_split_list = anytext.split(',')
                        rows_count = len(text_split_list)
                        if rows_count > max_rows_count:
                            max_rows_count = rows_count
                    sqltable_model.setTable(table_name)
                    sqltable_model.setEditStrategy(QtSql.QSqlTableModel.EditStrategy.OnFieldChange)
                    sqltable_model.select()
                    record = sqltable_model.record()

                    for mm in range(max_rows_count):
                        record.setValue(0, self.SQLTableView.model().data(
                            self.SQLTableView.model().index(i, 0)))  # TisaiYu[2024/8/22] partID
                        for index, attr_index in enumerate(this_table_multi_attribues_index, 1):
                            text = self.SQLTableView.model().data(self.SQLTableView.model().index(i, attr_index))
                            part_attribute = text.split(',')
                            if mm >= len(part_attribute):
                                record.setValue(index, '/')
                            else:
                                record.setValue(index, part_attribute[mm])
                        sqltable_model.insertRecord(-1, record)
                        sqltable_model.submitAll()
                        sqltable_model.select()
            self.SQLTableView.reset()
            # TisaiYu[2024/8/12] 如果当前就是汇总表则更新汇总表的显示
            self.sum_query_model.setTable("AddRecordTable")
            self.sum_query_model.select()
            self.SQLTableView.setModel(self.sum_query_model)

        else:  # TisaiYu[2024/8/14] 参考表的数据,由于是读取excel，参考表已经更新了，其他不用更新。
            return True

    def clear_sql_table(self):
        query = QtSql.QSqlQuery()
        table_name = self.table_names_dict[self.SQLformsTree.model().itemData(self.SQLformsTree.currentIndex())[0]][0]
        query.exec_("PRAGMA foreign_keys = OFF;")  # TisaiYu[2024/8/22] 有外键约束删除不了单行表，但是其他比表都空了，不清楚为什么
        query.exec("BEGIN TRANSACTION")  # 开始事务
        query.exec_(f"DELETE FROM {table_name}")
        query.exec("COMMIT")  # 提交事务
        query.exec_("PRAGMA foreign_keys = ON;")
        model = QtSql.QSqlTableModel()
        model.setTable(table_name)
        model.setEditStrategy(QtSql.QSqlTableModel.EditStrategy.OnFieldChange)
        model.select()
        self.SQLTableView.setModel(model)

    def sql_delete_select_row(self):
        indexes = self.SQLTableView.selectionModel().selectedRows()
        selected_row = indexes[0].row()

        reply = QMessageBox.question(self, '确认删除', '你确定要删除这一行吗？', QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.sum_query_model.removeRow(selected_row)
            self.sum_query_model.submitAll()
            self.sum_query_model.select()
            # TisaiYu[2024/8/12] 如果当前就是汇总表则更新汇总表的显示
            index = self.SQLformsTree.currentIndex()
            if index.isValid():
                if not index.parent().isValid() and index.row() == 0:
                    self.SQLTableView.setModel(self.sum_query_model)
        else:
            return False

    def sqlinput_ui_show_dialog(self):  # TisaiYu[2024/8/12] 原本添加和删除属性在新dialog里的函数
        if self.SQLformsTree.model() is None:
            QMessageBox.information(None, "信息", "先创建或者读取数据库文件！")
            return False
        else:
            if self.SQLformsTree.model().rowCount() == 0:
                QMessageBox.information(None, "信息", "先创建或者读取数据库文件！")
                return False
        # TisaiYu[2024/7/22] 不写为成员变量，在点击显示的时候只会闪一下就没了
        for i in range(self.sqlinput_dialog_ui.partSQLinputInfotable.columnCount()):
            self.sqlinput_dialog_ui.partSQLinputInfotable.setCellWidget(1, i,
                                                                        self.set_partSQLinputInfotable_cellwidget(i))
        self.sqlinput_dialog_window.show()

    def sqlinput_ui_show(self):  # TisaiYu[2024/8/12] 改成了在主界面添加属性，然后新dialog的属性对应主界面的
        if self.SQLformsTree.model() is None:
            QMessageBox.information(None, "信息", "先创建或者读取数据库文件！")
            return False
        else:
            if self.SQLformsTree.model().rowCount() == 0:
                QMessageBox.information(None, "信息", "先创建或者读取数据库文件！")
                return False
        # TisaiYu[2024/7/22] 不写为成员变量，在点击显示的时候只会闪一下就没了
        columncount = self.sum_query_model.columnCount()
        if self.table_names_dict[self.SQLformsTree.model().itemData(self.SQLformsTree.currentIndex())[0]][
            0] != "AddRecordTable":
            self.sqlinput_dialog_ui.partSQLinputInfotable.setRowCount(1)
            self.sqlinput_dialog_ui.partSQLinputInfotable.setColumnCount(2)

            self.sqlinput_dialog_ui.partSQLinputInfotable.setHorizontalHeaderItem(0, QTableWidgetItem(
                self.SQLTableView.horizontalHeader().model().headerData(0, Qt.Horizontal)))
            self.sqlinput_dialog_ui.partSQLinputInfotable.setHorizontalHeaderItem(1, QTableWidgetItem(
                self.SQLTableView.horizontalHeader().model().headerData(1, Qt.Horizontal)))
        else:
            self.sqlinput_dialog_ui.partSQLinputInfotable.setColumnCount(columncount)
            self.sqlinput_dialog_ui.partSQLinputInfotable.setRowCount(1)
            for i in range(columncount):
                column_label = self.sum_query_model.headerData(i, Qt.Horizontal, Qt.DisplayRole)
                self.sqlinput_dialog_ui.partSQLinputInfotable.setHorizontalHeaderItem(i, QTableWidgetItem(column_label))
        self.sqlinput_dialog_window.show()

    def set_partSQLinputInfotable_cellwidget(self, column_index):
        widget = QWidget()
        layout = QHBoxLayout()
        widget.setLayout(layout)
        add_button = QPushButton("添加")
        delete_button = QPushButton("删除")
        add_button.column = column_index  # TisaiYu[2024/7/22] 在pyqt中，组件都可以直接对一个没有的属性赋值，相当于直接创建一个属性（不能和预留的属性重名），C++中可能需要手动设置一个动态属性的效果。
        delete_button.column = column_index  # TisaiYu[2024/7/22] 在pyqt中，组件都可以直接对一个没有的属性赋值，相当于直接创建一个属性（不能和预留的属性重名），C++中可能需要手动设置一个动态属性的效果。
        add_button.clicked.connect(self.add_attribute)
        delete_button.clicked.connect(self.delete_attribute)
        layout.addWidget(add_button)
        layout.addWidget(delete_button)
        widget.setMinimumSize(40, 40)
        widget.setAutoFillBackground(
            True)  # TisaiYu[2024/7/22] 设置自动填充背景，设置True表示palette的调色板设置会立即反应到组件上，设置为False就只能通过样式表来改变
        palette = widget.palette()
        palette.setColor(widget.backgroundRole(), QColor(255, 255, 255))  # TisaiYu[2024/7/22] 第一个参数表示背景角色，相应的还有前景，按钮色等等
        return widget

    def add_attribute_dialog(self):  # TisaiYu[2024/8/9] 通过点击添加数据里的“添加”按钮来增加属性，后面改为了通过在窗口ui点击“添加属性”来先决定表格的列。
        sender_object = self.sender()  # TisaiYu[2024/7/22] QWidget类的self.sender()可以获取发送信号的对象
        column_index = sender_object.column + 1

        for i in range(column_index, self.sqlinput_dialog_ui.partSQLinputInfotable.columnCount()):
            self.sqlinput_dialog_ui.partSQLinputInfotable.cellWidget(1, i).findChildren(QPushButton)[0].column += 1
            self.sqlinput_dialog_ui.partSQLinputInfotable.cellWidget(1, i).findChildren(QPushButton)[1].column = \
            self.sqlinput_dialog_ui.partSQLinputInfotable.cellWidget(1, i).findChildren(QPushButton)[0].column
        input_dialog = QtWidgets.QInputDialog()
        add_column_label, whether_ok = input_dialog.getText(None, "添加零部件属性名", "属性名称")
        if whether_ok:
            self.sqlinput_dialog_ui.partSQLinputInfotable.insertColumn(column_index)
            self.sqlinput_dialog_ui.partSQLinputInfotable.setCellWidget(1, column_index,
                                                                        self.set_partSQLinputInfotable_cellwidget(
                                                                            column_index))
            self.sqlinput_dialog_ui.partSQLinputInfotable.setHorizontalHeaderItem(column_index,
                                                                                  QTableWidgetItem(add_column_label))
        else:
            return whether_ok  # TisaiYu[2024/7/22] False

    def delete_attribute_dialog(self):
        if self.sqlinput_dialog_ui.partSQLinputInfotable.columnCount() == 1:
            QMessageBox.information(None, "信息", "至少保留一个属性！")
            return False
        sender_object = self.sender()
        column_index = sender_object.column
        print(column_index)
        self.sqlinput_dialog_ui.partSQLinputInfotable.removeColumn(column_index)
        for i in range(column_index, self.sqlinput_dialog_ui.partSQLinputInfotable.columnCount()):
            print(self.sqlinput_dialog_ui.partSQLinputInfotable.cellWidget(1, i).findChildren(QPushButton)[0].column)
            self.sqlinput_dialog_ui.partSQLinputInfotable.cellWidget(1, i).findChildren(QPushButton)[0].column -= 1
            self.sqlinput_dialog_ui.partSQLinputInfotable.cellWidget(1, i).findChildren(QPushButton)[1].column = \
            self.sqlinput_dialog_ui.partSQLinputInfotable.cellWidget(1, i).findChildren(QPushButton)[0].column

    def add_attribute(self):
        input_dialog = CustomInputDialog()
        if input_dialog.exec_() == QDialog.Accepted:
            input_list = input_dialog.getInputs()
            add_column_label_english = input_list[0]
            query = QtSql.QSqlQuery()
            query.exec_(
                f"ALTER TABLE AddRecordTable ADD COLUMN {add_column_label_english} TEXT")  # TisaiYu[2024/8/9] 列名不能数字开头
            # TisaiYu[2024/8/9] 必须在执行对表的操作后再获取model对象，否则在表操作前获得的model对象没有更新表操作后的一些东西
            self.sum_query_model.setTable("AddRecordTable")
            self.sum_query_model.setEditStrategy(QtSql.QSqlTableModel.EditStrategy.OnFieldChange)
            self.sum_query_model.select()
            # TisaiYu[2024/8/12] 如果当前就是汇总表则更新汇总表的显示
            index = self.SQLformsTree.currentIndex()
            if index.isValid():
                if not index.parent().isValid() and index.row() == 0:
                    self.SQLTableView.setModel(self.sum_query_model)

            if input_list[1] == "多行":
                add_column_label_chinese = input_list[2]
                multi_row_table_name = "Parts" + add_column_label_english + "Table"
                refer_table_name = add_column_label_english + "Table"
                chinese_mtable_name = "零部件" + add_column_label_chinese + "表"

                if chinese_mtable_name not in self.table_names_dict.keys():
                    query.exec_(f'''
                                    INSERT INTO AttributesInfo (AttributeName,AttributeTable,AttributeRow)
                                    VALUES ('{add_column_label_english}','{multi_row_table_name}',1)
                                                                    ''')
                    if input_list[3] == '是':  # TisaiYu[2024/8/21] 是有参考编码表要填写
                        chinese_rtable_name = add_column_label_chinese + "编码表"
                        query.exec_(f'''
                                                        CREATE TABLE {refer_table_name} (
                                                            {add_column_label_english} VARCHAR(255) PRIMARY KEY,
                                                            Description TEXT NOT NULL  
                                                        );
                                                    ''')
                        query.exec_(f'''
                                                            INSERT INTO TablesInfo (TableChineseName,TableEnglishName,TableType)
                                                            VALUES ('{chinese_rtable_name}','{refer_table_name}','3')
                                                                                            ''')
                        self.table_names_dict[chinese_rtable_name] = [refer_table_name, 3]

                        query.exec_(f'''
                                                        CREATE TABLE {multi_row_table_name} (
                                                            PartID VARCHAR(255),
                                                            {add_column_label_english} VARCHAR(255)


                                                        );
                                                    ''')
                    query.exec_(f'''
                                    CREATE TABLE {multi_row_table_name} (
                                        PartID VARCHAR(255),
                                        {add_column_label_english} VARCHAR(255)

                                                    );
                                                ''')
                    self.table_names_dict[chinese_mtable_name] = [multi_row_table_name, 2]
                    query.exec_(f'''
                                    INSERT INTO TablesInfo (TableChineseName,TableEnglishName,TableType)
                                    VALUES ('{chinese_mtable_name}','{multi_row_table_name}','2')
                                                                    ''')

                    self.update_form_tree()
                    self.attribute_dict[add_column_label_english] = [multi_row_table_name, 1]
                else:
                    multi_row_table_name = self.table_names_dict[chinese_mtable_name][0]
                    query.exec_(f'''
                                    INSERT INTO AttributesInfo (AttributeName,AttributeTable,AttributeRow)
                                    VALUES ('{add_column_label_english}','{multi_row_table_name}',1)
                                                                    ''')
                    self.attribute_dict[add_column_label_english] = [multi_row_table_name, 1]
                    query.exec_(
                        f'''ALTER TABLE {multi_row_table_name} ADD COLUMN {add_column_label_english} VARCHAR(255)''')
                    if input_list[3] == '是':  # TisaiYu[2024/8/21] 是有参考编码表要填写
                        chinese_rtable_name = add_column_label_chinese + "编码表"
                        query.exec_(f'''
                                                                                CREATE TABLE {refer_table_name} (
                                                                                    {add_column_label_english} VARCHAR(255) PRIMARY KEY,
                                                                                    Description TEXT NOT NULL  
                                                                                );
                                                                            ''')
                        query.exec_(f'''
                                                                                    INSERT INTO TablesInfo (TableChineseName,TableEnglishName,TableType)
                                                                                    VALUES ('{chinese_rtable_name}','{refer_table_name}','3')
                                                                                                                    ''')
                        self.table_names_dict[chinese_rtable_name] = [refer_table_name, 3]
            else:
                add_column_label_english = input_list[0]
                query.exec_(f'''
                                                                INSERT INTO AttributesInfo (AttributeName,AttributeTable,AttributeRow)
                                                                VALUES ('{add_column_label_english}','PartsTable',0)
                                                ''')
                self.attribute_dict[add_column_label_english] = ["PartsTable", 0]
                query.exec_(f'''ALTER TABLE PartsTable ADD COLUMN {add_column_label_english} VARCHAR(255)''')
            self.update_attributes()
        else:
            return False

    def delete_attribute(self):
        input_dialog = CustomInputDialog()
        if input_dialog.exec_() == QDialog.Accepted:
            input_list = input_dialog.getInputs()
            add_column_label_english = input_list[0]
            query = QtSql.QSqlQuery()
            sql_model = QtSql.QSqlTableModel()
            query.exec_(f'''
                                        DELETE FROM AttributesInfo WHERE AttributeName='{add_column_label_english}'
                                                                                                ''')
            if input_list[1] == "单行":
                sql_model.setTable("PartsTable")
                sql_model.setEditStrategy(QtSql.QSqlTableModel.EditStrategy.OnFieldChange)
                sql_model.select()
                column_names = []
                for i in range(self.sum_query_model.columnCount()):
                    column_label = self.sum_query_model.headerData(i, Qt.Horizontal, Qt.DisplayRole)
                    if column_label != add_column_label_english:
                        column_names.append(column_label)
                columns_str = ", ".join(column_names)
                query.exec_(f"CREATE TABLE new_table AS SELECT {columns_str} FROM AddRecordTable")
                query.exec_("DROP TABLE AddRecordTable")
                query.exec_("ALTER TABLE new_table RENAME TO AddRecordTable")
                self.sum_query_model.setTable("AddRecordTable")
                self.sum_query_model.select()
                column_names.clear()
                for i in range(sql_model.columnCount()):
                    column_label = sql_model.headerData(i, Qt.Horizontal, Qt.DisplayRole)
                    if column_label != add_column_label_english:
                        column_names.append(column_label)
                columns_str = ", ".join(column_names)
                query.exec_(f"CREATE TABLE NewPartsTable AS SELECT {columns_str} FROM PartsTable")
                query.exec_("PRAGMA foreign_keys = OFF;")  # TisaiYu[2024/8/16] 不然其他引用了零部件表的会被删除数据
                query.exec_('DROP TABLE PartsTable')
                query.exec_('ALTER TABLE NewPartsTable RENAME TO PartsTable')
                query.exec_("PRAGMA FOREIGN_KEYS=ON;")

                # TisaiYu[2024/8/12] 如果当前就是汇总表则更新汇总表的显示
                index = self.SQLformsTree.currentIndex()
                if index.isValid():
                    if not index.parent().isValid() and index.row() == 0:
                        self.SQLTableView.setModel(self.sum_query_model)
            else:
                column_names = []
                for i in range(self.sum_query_model.columnCount()):
                    column_label = self.sum_query_model.headerData(i, Qt.Horizontal, Qt.DisplayRole)
                    if column_label != add_column_label_english:
                        column_names.append(column_label)
                columns_str = ", ".join(column_names)
                query.exec_(f"CREATE TABLE new_table AS SELECT {columns_str} FROM AddRecordTable")
                query.exec_("DROP TABLE AddRecordTable")
                query.exec_("ALTER TABLE new_table RENAME TO AddRecordTable")
                self.sum_query_model.setTable("AddRecordTable")
                self.sum_query_model.select()
                add_column_label_chinese = input_list[2]
                chinese_mtable_name = "零部件" + add_column_label_chinese + "表"
                multi_row_table_name = self.table_names_dict[chinese_mtable_name][0]

                sql_model.setTable(multi_row_table_name)

                if sql_model.columnCount() == 2:
                    del self.table_names_dict[chinese_mtable_name]
                    query.exec_(f'''DROP TABLE {multi_row_table_name}''')
                    query.exec_(f'''
                                        DELETE FROM TablesInfo WHERE TableChineseName='{chinese_mtable_name}'
                                                                    ''')
                    if input_list[3] == '是':
                        chinese_rtable_name = add_column_label_chinese + "编码表"
                        refer_table_name = self.table_names_dict[chinese_rtable_name][0]
                        del self.table_names_dict[chinese_rtable_name]
                        query.exec_(f'''DROP TABLE {refer_table_name}''')
                        query.exec_(f'''
                                            DELETE FROM TablesInfo WHERE TableChineseName='{chinese_rtable_name}'
                                                                        ''')
                else:
                    multi_row_table_name = self.table_names_dict[chinese_mtable_name][0]
                    column_names = []
                    sql_model.setTable(multi_row_table_name)
                    sql_model.setEditStrategy(QtSql.QSqlTableModel.EditStrategy.OnFieldChange)
                    sql_model.select()
                    for i in range(sql_model.columnCount()):
                        column_label = sql_model.headerData(i, Qt.Horizontal, Qt.DisplayRole)
                        if column_label != add_column_label_english:
                            column_names.append(column_label)
                    columns_str = ", ".join(column_names)
                    query.exec_(f"CREATE TABLE NewTable AS SELECT {columns_str} FROM {multi_row_table_name}")
                    query.exec_("PRAGMA foreign_keys = OFF;")  # TisaiYu[2024/8/16] 不然其他引用了零部件表的会被删除数据
                    query.exec_(f'DROP TABLE {multi_row_table_name}')
                    query.exec_(f'ALTER TABLE NewTable RENAME TO {multi_row_table_name}')
                    query.exec_("PRAGMA FOREIGN_KEYS=ON;")
                    if input_list[3] == '是':
                        chinese_rtable_name = add_column_label_chinese + "编码表"
                        refer_table_name = self.table_names_dict[chinese_rtable_name][0]
                        del self.table_names_dict[chinese_rtable_name]
                        query.exec_(f'''DROP TABLE {refer_table_name}''')
                        query.exec_(f'''
                                            DELETE FROM TablesInfo WHERE TableChineseName='{chinese_rtable_name}'
                                                                        ''')
            self.update_form_tree()

    def get_all_columns(self, table_name):
        query = QtSql.QSqlQuery(f"PRAGMA table_info({table_name})")
        columns = []
        while query.next():
            columns.append(query.value(1))  # 获取列名
        return columns

    def generate_query(self, main_table, multi_tables, primary_key, primary_key_value=None):
        main_columns = self.get_all_columns(main_table)
        main_columns_str = ", ".join([f"{main_table}.{col}" for col in main_columns])

        join_clauses = []
        multi_columns_str = []

        for multi_table in multi_tables:
            multi_columns = self.get_all_columns(multi_table)
            multi_columns_str.extend([f"{multi_table}.{col}" for col in multi_columns if col != primary_key])
            join_clauses.append(f"LEFT JOIN {multi_table} ON {main_table}.{primary_key} = {multi_table}.{primary_key}")

        multi_columns_str = ", ".join(multi_columns_str)
        join_clauses_str = " ".join(join_clauses)
        if primary_key_value == None:
            query_str = f"""
            SELECT {main_columns_str}, {multi_columns_str}
            FROM {main_table}
            {join_clauses_str}
            """
        else:
            query_str = f"""
            SELECT {main_columns_str}, {multi_columns_str}
            FROM {main_table}
            {join_clauses_str}
            WHERE {main_table}.{primary_key}={primary_key_value}
            """
        return query_str

    def select_part_info(self, generate_dsm=False):
        if generate_dsm == False:
            text = self.searchLineEdit.text()
        else:
            text = None
        main_table = 'PartsTable'  # 替换为你的主表名
        multi_tables = []  # 替换为你的多行属性表名
        for values in self.table_names_dict.values():
            if values[1] == '2':
                multi_tables.append(values[0])
        primary_key = 'PartID'  # 替换为你的主键名
        if text == "all" or text == None:
            query_str = self.generate_query(main_table, multi_tables, primary_key)
        else:
            query_str = self.generate_query(main_table, multi_tables, primary_key, text)
        query = QtSql.QSqlQuery()
        query.exec_(query_str)
        if generate_dsm == False:
            self.SQLTableView.model().setQuery(query)
            self.querying = True
        else:
            return query_str

    def read_excel_to_sql(self):
        current_path = sys.path[0]
        filedialog = QFileDialog.getOpenFileName(self, "选择数据库表excel文件！", current_path)[0]
        if filedialog == '':  # TisaiYu[2024/6/20] 处理什么都没选就叉掉的情况
            return False
        df = pd.read_excel(filedialog, header=0)

        query = QtSql.QSqlQuery()
        table_name = self.table_names_dict[self.SQLformsTree.model().itemData(self.SQLformsTree.currentIndex())[0]][0]
        for row in df.itertuples(index=False):
            values = ', '.join([f"'{str(cell)}'" for cell in row])
            query.exec_(f"INSERT INTO {table_name} VALUES ({values})")
        self.SQLTableView.model().setTable(table_name)
        self.SQLTableView.model().select()
        self.update_sql_excel(table_name)

    def synMovedColumn(self, logicalIndex, oldVisualIndex, newVisualIndex):
        # 在这里更新数据库表的列顺序
        column_order = [self.SQLTableView.horizontalHeader().visualIndex(i) for i in
                        range(self.SQLTableView.model().columnCount())]
        print(f'New column order: {column_order}')
        # 获取当前列名
        oldColumnName = self.SQLTableView.model().headerData(oldVisualIndex, Qt.Horizontal)
        newColumnName = self.SQLTableView.model().headerData(newVisualIndex, Qt.Horizontal)

        # 根据新列顺序重新定义数据库表结构
        # 这里假设你使用 SQLite 数据库
        query = QtSql.QSqlQuery()

        if self.querying == False:
            table_name = self.table_names_dict[self.SQLformsTree.model().itemData(self.SQLformsTree.currentIndex())[0]][
                0]
            # 使用 SQL 查询交换列数据
            query.exec(f'ALTER TABLE {table_name} RENAME COLUMN {oldColumnName} TO temp_column;')
            query.exec(f'ALTER TABLE {table_name} RENAME COLUMN {newColumnName} TO {oldColumnName};')
            query.exec(f'ALTER TABLE {table_name} RENAME COLUMN temp_column TO {newColumnName};')
            self.SQLTableView.model().setTable(table_name)
            self.SQLTableView.model().select()
        else:
            self.SQLTableView.horizontalHeader().sectionMoved.disconnect(self.synMovedColumn)
            self.SQLTableView.horizontalHeader().swapSections(oldVisualIndex, newVisualIndex)
            self.SQLTableView.horizontalHeader().sectionMoved.connect(self.synMovedColumn)

    def export_sql_table(self):
        if self.SQLTableView.model().rowCount() == 0:
            QMessageBox.information(None, "信息", "还没有选中表或者表没有数据！")
            return False

        exp_filename = QFileDialog.getSaveFileName(self, '保存文件', '', 'Excel文件 (*.xlsx);;所有文件 (*)')[0]
        if exp_filename:
            model = self.SQLTableView.model()
            data = []
            headers = []

            # 获取表头
            for col in range(model.columnCount()):
                headers.append(model.headerData(col, Qt.Horizontal))
            for row in range(model.rowCount()):
                data.append([])
                for col in range(model.columnCount()):
                    index = model.index(row, col)
                    data[row].append(str(model.data(index)))
            df = pd.DataFrame(data, columns=headers)
            df.to_excel(exp_filename, index=False)

            # 打开Excel文件并获取数据
            df = pd.read_excel(exp_filename, header=0)

            # 遍历所有列，调整列宽以适应内容
            writer = pd.ExcelWriter(exp_filename, engine='xlsxwriter')
            df.to_excel(writer, index=False, sheet_name='Sheet1')
            worksheet = writer.sheets['Sheet1']

            for idx, col in enumerate(df):
                max_length = max(df[col].astype(str).map(len).max(), len(str(df.columns[idx]))) + 2
                worksheet.set_column(idx, idx, max_length)

            # 保存更改
            writer.close()
        else:
            return False

    def generate_dsm_from_sql(self):
        query_all_str = self.select_part_info(generate_dsm=True)
        self.reprod = Reprod()
        self.reprod.get_conn_relationship(query_all_str)
        self.DSM, self.part_info_arr, self.ACP_dict,self.essay_data = self.reprod.sql_dsm()
        df = pd.DataFrame(self.DSM)
        df.index = df.index + 1
        df.to_excel('DSM.xlsx', header=False)

        self.parts_num = self.DSM.shape[0]
        self.dipict_matrix('Correlation Matrix', 'index_row', 'index_col', self.DSMLayout, self.DSM)
        self.read_excel_file = True

    def get_AC_HC_dict_by_sql(self):
        AC_dict = defaultdict(dict)
        HC_dict = {}
        HC_dict["Weight"] = []
        HC_dict["Size"] = []
        query = QtSql.QSqlQuery(self.db)
        query_all_str = self.select_part_info(generate_dsm=True)
        query_conn_AC = f'''SELECT 
                            o1.PartID,
                            o2.PartID,
                            o2.AssemblyType
                        FROM 
                            ({query_all_str}) o1
                        JOIN 
                            ({query_all_str}) o2
                        ON 
                            o1.PartID = o2.ConnectionPartID
                        WHERE 
                            o1.PartID <= o2.PartID;'''
        query.exec_(query_conn_AC)
        while query.next():
            AC_dict[query.value(0)][query.value(1)] = query.value(2)

        query.exec_(f'''SELECT Weight
                    FROM AddRecordTable
                    GROUP BY PartID, Weight
                    ORDER BY CAST(PartID AS INTEGER) ASC;''')
        while query.next():
            if query.value(0) != '/':
                HC_dict["Weight"].append(float(query.value(0)))
            else:
                HC_dict["Weight"].append(0)

        # TisaiYu[2024/8/30] 下面这个数值类型一定要是浮点数，否则选择最大最小值会出错，出现最小值大于最大值的情况，因为sqlite不支持least和greatest，所以使用case实现相同功能
        query.exec_(f'''          SELECT 
                                  MIN(
                        CASE 
                            WHEN CAST(x AS FLOAT) <= CAST(x1 AS FLOAT) AND CAST(x AS FLOAT) <= CAST(x2 AS FLOAT) THEN CAST(x AS FLOAT)
                            WHEN CAST(x1 AS FLOAT) <= CAST(x AS FLOAT) AND CAST(x1 AS FLOAT) <= CAST(x2 AS FLOAT) THEN CAST(x1 AS FLOAT)
                            ELSE CAST(x2 AS FLOAT)
                        END
                    ) AS min_x,
                    MAX(
                        CASE 
                            WHEN CAST(x AS FLOAT) >= CAST(x1 AS FLOAT) AND CAST(x AS FLOAT) >= CAST(x2 AS FLOAT) THEN CAST(x AS FLOAT)
                            WHEN CAST(x1 AS FLOAT) >= CAST(x AS FLOAT) AND CAST(x1 AS FLOAT) >= CAST(x2 AS FLOAT) THEN CAST(x1 AS FLOAT)
                            ELSE CAST(x2 AS FLOAT)
                        END
                    ) AS max_x,
                    MIN(
                        CASE 
                            WHEN CAST(y AS FLOAT) <= CAST(y1 AS FLOAT) AND CAST(y AS FLOAT) <= CAST(y2 AS FLOAT) THEN CAST(y AS FLOAT)
                            WHEN CAST(y1 AS FLOAT) <= CAST(y AS FLOAT) AND CAST(y1 AS FLOAT) <= CAST(y2 AS FLOAT) THEN CAST(y1 AS FLOAT)
                            ELSE CAST(y2 AS FLOAT)
                        END
                    ) AS min_y,
                    MAX(
                        CASE 
                            WHEN CAST(y AS FLOAT) >= CAST(y1 AS FLOAT) AND CAST(y AS FLOAT) >= CAST(y2 AS FLOAT) THEN CAST(y AS FLOAT)
                            WHEN CAST(y1 AS FLOAT) >= CAST(y AS FLOAT) AND CAST(y1 AS FLOAT) >= CAST(y2 AS FLOAT) THEN CAST(y1 AS FLOAT)
                            ELSE CAST(y2 AS FLOAT)
                        END
                    ) AS max_y,
                    MIN(
                        CASE 
                            WHEN CAST(z AS FLOAT) <= CAST(z1 AS FLOAT) AND CAST(z AS FLOAT) <= CAST(z2 AS FLOAT) THEN CAST(z AS FLOAT)
                            WHEN CAST(z1 AS FLOAT) <= CAST(z AS FLOAT) AND CAST(z1 AS FLOAT) <= CAST(z2 AS FLOAT) THEN CAST(z1 AS FLOAT)
                            ELSE CAST(z2 AS FLOAT)
                        END
                    ) AS min_z,
                    MAX(
                        CASE 
                            WHEN CAST(z AS FLOAT) >= CAST(z1 AS FLOAT) AND CAST(z AS FLOAT) >= CAST(z2 AS FLOAT) THEN CAST(z AS FLOAT)
                            WHEN CAST(z1 AS FLOAT) >= CAST(z AS FLOAT) AND CAST(z1 AS FLOAT) >= CAST(z2 AS FLOAT) THEN CAST(z1 AS FLOAT)
                            ELSE CAST(z2 AS FLOAT)
                        END
                    ) AS max_z
                    FROM 
                        ({query_all_str})
                    GROUP BY 
                        CAST(PartID AS INTEGER)
                    ORDER BY CAST(PartID AS INTEGER) ASC;''')
        while query.next():
            minx = float(query.value(0))
            maxx = float(query.value(1))
            miny = float(query.value(2))
            maxy = float(query.value(3))
            minz = float(query.value(4))
            maxz = float(query.value(5))
            HC_dict["Size"].append([minx, maxx, miny, maxy, minz, maxz])
        query.exec_(f'''SELECT MIN(o.x) AS min_x, MAX(o.x) AS max_x,
       MIN(o.y) AS min_y, MAX(o.y) AS max_y,
       MIN(o.z) AS min_z, MAX(o.z) AS max_z
    FROM 
    PartsConnectionTable o
    JOIN
    PartsConnectionTable o1 ON o1.ConnectionPartID=o.PartID
    WHERE o1.ConnectionPartID IN (0, 1, 2, 3, 4, 5, 6, 7, 131, 132, 133, 134, 135, 136, 137, 138, 262, 263, 264, 265, 266, 267, 268, 269, 393, 394, 395, 396, 397, 398, 399, 400, 524, 525, 526, 527, 528, 529, 530, 531)
    AND o.PartID=o1.ConnectionPartID
    GROUP BY CAST(o.PartID AS INTEGER)
    ORDER BY CAST(o.PartID AS INTEGER) ASC;''')
        e_count = 0
        epoch_count = 0
        while query.next():
            minx = float(query.value(0))
            maxx = float(query.value(1))
            miny = float(query.value(2))
            maxy = float(query.value(3))
            minz = float(query.value(4))
            maxz = float(query.value(5))
            HC_dict["Size"][e_count+131*epoch_count]=(minx, maxx, miny, maxy, minz, maxz)
            e_count+=1
            if e_count==8:
                epoch_count+=1
                e_count=0

        HC_dict["Weight"] = np.array(HC_dict["Weight"])
        HC_dict["Size"] = np.array(HC_dict["Size"])
        self.HC_dict = HC_dict
        return AC_dict, HC_dict

    def dedrogram_new(self,Z):
        pass
        # plt.figure()
        # re = hierarchy.dendrogram(Z, color_threshold=0.2, above_threshold_color='#bcbddc')
        # plt.show()

    def testWithSetValue(self, n):
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
        mm = generate_test_value(n, 5, correlation_nums)
        for i in range(correlation_nums):
            index = self.factorsList.row(self.correlation_selected_indexs[i])
            now_table = self.correlationMatrixStackWidget.widget(index).findChild(QTableView)
            for j in range(row):
                for k in range(column):
                    now_table.model().setItem(j, k, QStandardItem(str(mm[i][j][k])))
