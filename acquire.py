import pandas as pd
import numpy as np
import env

def get_db_url(database):
    from env import host, user, password
    url = f'mysql+pymysql://{user}:{password}@{host}/{database}'
    return url

def get_titanic_data():
    query = """
    SELECT *
    FROM passengers
    """
    df = pd.read_sql(query, get_db_url('titanic_db'))
    return df


def get_iris_data():
    query = """
    SELECT * 
    FROM `measurements`
    JOIN species USING (species_id)
    """
    df = pd.read_sql(query, get_db_url("iris_db"))
    return df