import pandas as pd

def example_system_connection_parse():
    # 输入数据
    data = """
    1:W001,3
    2:W002,3
    3:T001,1,2,4
    4:T002,3,5,6
    5:P001,4,7
    6:P002,4,8
    7:W007,5,9
    8:W008,6,9
    9:T003,7,8,10
    10:D001,9,11
    11:T004,10,12,58
    12:T005,11,13,64
    13:J003,12,14
    14:T006,13,15,61
    15:J004,14,16
    16:E001,15,17,34,35
    17:E002,16,18
    18:T007,17,19,39
    19:J009,18,20
    20:J010,19,21
    21:T008,20,22,24
    22:T009,21,23,25
    23:T010,22,26,48
    24:J011,21,27
    25:J012,22,28
    26:J013,23,29
    27:E003,24,31,40
    28:E004,25,30,41
    29:E005,26,30,42
    30:T011,28,29,31
    31:T012,27,30,32
    32:F001,31,33
    33:J014,32,34
    34:T013,16,33,39
    35:J015,16,63
    36:T014,37,38,60
    37:J025,36
    38:J026,36
    39:J007,18,34
    40:J016,27,43
    41:J017,28,43
    42:J018,29,44
    43:T100,40,41,44
    44:T101,42,43,45
    45:T102,44,46,47
    46:J020,45,65
    47:J022,45,52
    48:T103,23,49,50
    49:J019,48,65
    50:J021,48,51
    51:C001,50,53
    52:C002,47,55
    53:J023,51,54
    54:T104,53,55,56
    55:J024,52,54
    56:J027,54,57
    57:F002,56
    58:J001,11,59
    59:J002,58,60
    60:T200,36,59,63
    61:J005,14,62
    62:J006,61,63
    63:T201,35,60,62
    64:J008,12
    65:T105,48,46
    """

    # 解析数据
    lines = data.strip().split('\n')
    rows = []
    for line in lines:
        parts = line.split(':')
        component_id = int(parts[0]) - 1  # 将ID从1开始转换为从0开始
        component_info = parts[1].split(',')
        component_code = component_info[0]

        # 处理连接的ID，减去1
        connected_ids = ','.join(str(int(id) - 1) for id in component_info[1:])

        rows.append([component_id, component_code, connected_ids])

    # 创建DataFrame
    df = pd.DataFrame(rows, columns=['Component ID', 'Component Code', 'Connected IDs'])

    # 保存为Excel文件
    df.to_excel('part_connection_mapping.xlsx', index=False)

    print("数据已成功保存为 part_connection_mapping.xlsx")


def system_connection_parse(system_id : str = 0):
    part_id_list = []
    part_name = []
    part_connecting_ids = []
    data = {"ID":part_id_list,"Part Name":part_name,"Connecting ID":part_connecting_ids}
    with open("systemtxt/11.11system/connection.txt") as f:
        for line in f:
            if line.strip():
                if line[0].isdigit():
                    line = line.replace('\n', '')
                    part_id,connect_description_line = line.split(':')
                    dot_split = connect_description_line.split(',')
                    part_name = dot_split[0]
                    connecting_ids = dot_split[1:]
                    connecting_ids = ','.join(connecting_ids)
                    data["ID"].append(part_id)
                    data["Part Name"].append(part_name)
                    data["Connecting ID"].append(connecting_ids)
                else:
                    continue
    df = pd.DataFrame(data)
    df["ID"] = df["ID"].astype(int)
    df = df.sort_values(by="ID")
    df.to_excel('SqlLoadExcel/connection.xlsx',index=False)

if __name__ == "__main__":
    system_connection_parse()