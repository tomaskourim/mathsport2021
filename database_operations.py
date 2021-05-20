# support file to manipulate with SQLite and Postgresql database
import logging
from configparser import ConfigParser
from typing import Optional

import psycopg2


def config(filename: str = './database.ini', section: str = 'postgresql') -> dict:
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return db


def execute_sql_postgres(query: str, param: Optional, modifying: bool = False, return_multiple: bool = False,
                         execute_multiple: bool = False) -> tuple:
    conn = None
    cur = None
    try:
        params = config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        if param is not None:
            if execute_multiple:
                for index in range(0, len(param[0])):
                    one_param = [item[index] for item in param]
                    cur.execute(query, one_param)
            else:
                cur.execute(query, param)
        else:
            cur.execute(query)
        if cur.description is not None:
            if return_multiple:
                return_value = cur.fetchall()
            else:
                return_value = cur.fetchone()
        else:
            return_value = "success"
        if modifying:
            conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        logging.debug(f"Sql exception: {error}")
        return_value = error
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

    return return_value
