
import pandas as pd


# 读取 Excel 文件
df = pd.read_excel(fr'E:\Postgraduate\YY\code\Modularization_to_python\ModularizationPy\data\essay_example\example.xlsx')
print(df.columns)
df.columns = ['Component Number', 'Z','Type', 'D', 'x', 'y', 'z', "x'", "y'", "z'", "x''", "y''", "z''", 'connection axis', 'Connecting component', 'Assembly type', 'Sub-System']


# 定义一个函数来合并连接对象 ID 和连接坐标
def merge_rows(group):
    def process_column(column,if_conn_comp = False):
        if column.isnull().any():
            return '/'
        else:
            if if_conn_comp:
                return ','.join(column.astype(int).astype(str))

            else:
                return ','.join(column.astype(str))

    compids = process_column(group['Connecting component'],True)
    x_coords = process_column(group['x'])
    y_coords = process_column(group['y'])
    z_coords = process_column(group['z'])
    x_prime_coords = process_column(group["x'"])
    y_prime_coords = process_column(group["y'"])
    z_prime_coords = process_column(group["z'"])
    x_double_prime_coords = process_column(group["x''"])
    y_double_prime_coords = process_column(group["y''"])
    z_double_prime_coords = process_column(group["z''"])
    connection_axes = process_column(group['connection axis'])
    assembly_types = process_column(group['Assembly type'])

    return pd.Series({
        'Z': group['Z'].iloc[0],
        'Type': group['Type'].iloc[0],
        'D': group['D'].iloc[0],
        'Sub-System': group['Sub-System'].iloc[0],
        'Connecting component': compids,
        'x': x_coords,
        'y': y_coords,
        'z': z_coords,
        "x'": x_prime_coords,
        "y'": y_prime_coords,
        "z'": z_prime_coords,
        "x''": x_double_prime_coords,
        "y''": y_double_prime_coords,
        "z''": z_double_prime_coords,
        'connection axis': connection_axes,
        'Assembly type': assembly_types
    })

def raw2sqlload():
    # 按 Component Number 分组并应用合并函数
    result = df.groupby('Component Number').apply(merge_rows).reset_index()
    # 保存结果到新的 Excel 文件
    result.to_excel('essay_merged_excel_file.xlsx', index=False)
    print(result)

def sqlload2duplicate():
    df = pd.read_excel(r"E:\Postgraduate\YY\2024_1\模块化项目\后期我和瀚哥接手\发瀚哥\论文数据\（读到数据库的）零部件信息汇总表.xlsx")
    result_df = pd.DataFrame()
    original_length = len(df)

    # 创建一个空的 DataFrame 来存储结果
    result_df = pd.DataFrame()

    # 定义前缀列表
    prefixes = ['', 'one', 'two', 'three', 'four']

    def conn_partid_increment(cell,part_num):
        conn_partids = cell.split(',')
        if conn_partids[0]=='/':
            conn_partids_new = ['/','/']
        else:
            conn_partids_new = [str(int(conn_partid)+part_num) for conn_partid in conn_partids]
        return ','.join(conn_partids_new)
    # 复制行并修改 ID 和第五列
    for i in range(5):
        temp_df = df.copy()
        temp_df['Component Number'] = temp_df['Component Number'] + i * original_length
        if i > 0:
            temp_df.iloc[:, 4] = prefixes[i] + temp_df.iloc[:, 4].astype(str)
            temp_df.iloc[:,5] = temp_df.iloc[:,5].apply(lambda cell:conn_partid_increment(cell,original_length*i))
        result_df = pd.concat([result_df, temp_df], ignore_index=True)

    # 保存结果到新的 Excel 文件
    result_df.to_excel('modified_file.xlsx', index=False)
if __name__ == "__main__":
    # raw2sqlload() # TisaiYu[2024/9/12] 把论文原始数据转换为可以读取数据库的
    sqlload2duplicate()
