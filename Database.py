import sqlite3
import os
import numpy as np
import pandas as pd


class Database:
    @staticmethod
    def load_tableList(con):
        cursor = con.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        df_slist = []
        for table_name in tables:
            df_slist.append(table_name[0])
        return np.array(sorted(df_slist))

    @staticmethod
    def checkTableExists(con, tablename):
        dbcur = con.cursor()
        dbcur.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='{}'".format(tablename))
        if dbcur.fetchone()[0] == 1:
            dbcur.close()
            return True
        dbcur.close()
        return False
