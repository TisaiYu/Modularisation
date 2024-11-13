import pandas as pd

# 读取第一个Excel文件
df2 = pd.read_excel('SqlLoadExcel/connection.xlsx')

# 读取第二个Excel文件
df1 = pd.read_excel('SqlLoadExcel/function.xlsx')

# 通过ID列进行合并
merged_df = pd.merge(df1, df2, on='ID')

# 保存合并后的DataFrame为新的Excel文件
merged_df.to_excel('SqlLoadExcel/system.xlsx', index=False)

