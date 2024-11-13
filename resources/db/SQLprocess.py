from PyQt5 import QtSql

class SqlProcess():
    def __init__(self):
        pass

    def get_all_columns(self,table_name):
        query = QtSql.QSqlQuery(f"PRAGMA table_info({table_name})")
        columns = []
        while query.next():
            columns.append(query.value(1))  # 获取列名
        return columns

    def get_selected_columns(self,table_name,selected_columns_str_list):
        selected_columns_str = ", ".join(selected_columns_str_list)
        query = QtSql.QSqlQuery(f"SELECT {selected_columns_str} FROM {table_name}")

    def generate_query(self, main_table, multi_tables, primary_key=None, primary_key_value=None):
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

        if primary_key!=None and primary_key_value!=None:
            query_str = f"""
            SELECT {main_columns_str}, {multi_columns_str}
            FROM {main_table}
            {join_clauses_str}
            WHERE {main_table}.{primary_key}={primary_key_value}
            """
            return query_str
        elif primary_key!=None and primary_key_value==None:
            query_str = f"""
                        SELECT {main_columns_str}, {multi_columns_str}
                        FROM {main_table}
                        {join_clauses_str}
                        """
            return query_str
        else:
            query_str = f"""
                        SELECT {main_columns_str}, {multi_columns_str}
                        FROM {main_table}
                        {join_clauses_str}
                        """
            return query_str



    def select_part_info(self):
        text = self.searchLineEdit.text()
        main_table = 'PartsTable'  # 替换为你的主表名
        multi_tables = []  # 替换为你的多行属性表名
        for values in self.table_names_dict.values():
            if values[1] == '2':
                multi_tables.append(values[0])
        primary_key = 'PartID'  # 替换为你的主键名
        query_str = self.generate_query(main_table, multi_tables, primary_key, text)
        query = QtSql.QSqlQuery()
        query.exec_(query_str)
        self.SQLTableView.model().setQuery(query)
        self.querying = True