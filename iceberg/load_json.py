import json
import reprlib
import psycopg2

def pprint(s):
    print(reprlib.repr(z))

def data_to_insert(data_entry, table, test):
    tag = "'" + data_entry['id'] + "'"
    band_1 = "'{"
    band_1_array = data_entry['band_1']
    for x in band_1_array:
        band_1 += str(x)
        band_1 += ", "
    band_1 = band_1[:-2] + "}'"
    band_2 = "'{"
    band_2_array = data_entry['band_2']
    for x in band_2_array:
        band_2 += str(x)
        band_2 += ", "
    band_2 = band_2[:-2] + "}'"

    inc_angle = str(data_entry['inc_angle'])
    if inc_angle == "na":
        inc_angle = "NULL"

    if test:
        is_iceberg = ""
    else:
        is_iceberg = str(data_entry['is_iceberg'])

    if test:
        insert = "insert into {} values({}, {}, {}, {})".format(table, tag, band_1, band_2, inc_angle)
    else:   
        insert = "insert into {} values({}, {}, {}, {}, {})".format(table, tag, band_1, band_2, inc_angle, is_iceberg)
    
    return insert

with open("train.json") as f:
    x = json.load(f)

with open("test.json") as f:
    y = json.load(f)


try:
    connect_str = "dbname='postgres' user='postgres' host='localhost' " + \
                  "password='{}'".format(input("enter postgres password for postgres user:"))

    conn = psycopg2.connect(connect_str)
    cur = conn.cursor()

    cur.execute("""create table icebergtrain(id varchar(20),
                                                band1 double precision[],
                                                band2 double precision[],
                                                inc_angle double precision,
                                                is_iceberg integer)""")

    cur.execute("""create table icebergtest(id varchar(20),
                                                band1 double precision[],
                                                band2 double precision[],
                                                inc_angle double precision)""")
    print("inserting training data")
    for entry in x:
        ins = data_to_insert(entry, "icebergtrain", test=False)
        cur.execute(ins)
        
    print("inserting testing data")
    for entry in y:
        ins = data_to_insert(entry, "icebergtest", test=True)
        cur.execute(ins)

    conn.commit()
    print("we did it")

except Exception as e:
    print("ERRRORORORORTORO")
    print(e)



