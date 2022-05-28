import pandas as pd
import mysql.connector
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing



# Functions that used to connect with the cloud database and display results.
def create_connection():
    conn = None
    try:
       conn = mysql.connector.connect(
          user="DB00004175",
          password="tTNlXEirHh)EmwXhtp4koTR",
          host="menu-data-db.mdb0002067.db.skysql.net",
          ssl_ca="skysql_chain.pem",
          port=5001)
       conn.autocommit = False
       return conn
    except mysql.connector.Error as e:
       print(f"Error connecting to MariaDB Platform: {e}")
    return conn

def sql_execute(conn, sql):

    try:
        cur = conn.cursor()
        cur.execute(sql, multi=True)
        conn.commit()
    except mysql.connector.Error as e:
        print(e)

def sql_execute_show(conn, sql):

    try:
        df = pd.read_sql(sql, con = conn)
        return df
    except mysql.connector.Error as e:
        print(e)

##call once
conn = create_connection()

## fetch data from database and run clustering method to identify groups of restaurant based on their menu nutrient
## segment restaurant into groups with similar nutrient type
## It will help build our "people also like" recommendation
df_sql = sql_execute_show(conn, sql = '''
SELECT b.r_name, AVG(calories), AVG(total_fat), AVG(Protein), AVG(sodium), AVG(sugar), AVG(dietary_Fiber)
FROM menu_cstore.nutrition_facts a
LEFT JOIN menu_cstore.restaurant b ON a.r_id = b.r_id
GROUP BY b.r_name
;
''')
##data cleasing and normalization
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

df_re = df_sql.set_index('r_name')
clean_dataset(df_re)
names = pd.DataFrame(df_re.index)


scaler = preprocessing.MinMaxScaler()
df_re_normal = scaler.fit_transform(df_re)


##clustering and find out k = 4 where the decrease in distortion starts to level off
inertia = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(df_re_normal)
    kmeanModel.fit(df_re_normal)
    inertia.append(kmeanModel.inertia_)
plt.plot(K, inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')


kmeans = KMeans(n_clusters= 4).fit(df_re_normal)
df_re.insert(0, 'labels', kmeans.labels_, True)

##showing restaurants in the same group that we can recommand resaurants with similiar nutrients pattern to people.
group0 = df_re[df_re['labels'] == 0]
print(group0)
group1 = df_re[df_re['labels'] == 1]
print(group1)
group2 = df_re[df_re['labels'] == 2]
print(group2)
group3 = df_re[df_re['labels'] == 3]
print(group3)

conn.close()




