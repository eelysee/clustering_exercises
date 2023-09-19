import pandas as pd
import numpy as np
import os

from env import get_connection
from sklearn.model_selection import train_test_split


def get_zillow_mvp():
    '''
    MVP query
    
    cache's dataframe in a .csv
    '''
    filename = 'mvp.csv'
    
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    
    else:
        query ='''
               SELECT bathroomcnt, bedroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt
FROM properties_2017
WHERE propertylandusetypeid = (
                                SELECT propertylandusetypeid
                                FROM propertylandusetype
                                WHERE propertylandusedesc = 'Single Family Residential'
                                )
AND parcelid IN (
                SELECT parcelid
                FROM predictions_2017
                WHERE LEFT(transactiondate, 4) = '2017'
                )

;
                '''
        
        url = get_connection('zillow')
        df = pd.read_sql(query,url)
        df.to_csv(filename, index=False)
        
        return df

##############################################
# This section is from Madeline cappers wranlge file.
def get_zillow_cluster():
    '''
    capper query on zillow data
    
    cache's dataframe in a .csv
    '''
    filename = 'zillow_cluster.csv'
    
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    
    else:
        query = '''
        -- here, I want to ensure that I am selecting
        -- properties that have a transaction in 2017,
        -- the most recent version of those properties
        -- from there, I want to get the logerror for the zestimate
        -- and any potential supplementary information 
        -- available in the other tables
        -- SELECT: everything from properties aliased as prop
        SELECT prop.*,
        -- predictions_2017 : logerror and transactiondate
        predictions_2017.logerror,
        predictions_2017.transactiondate,
        -- all the other supplementary stuff
        air.airconditioningdesc,
        arch.architecturalstyledesc,
        build.buildingclassdesc,
        heat.heatingorsystemdesc,
        land.propertylandusedesc,
        story.storydesc,
        type.typeconstructiondesc
        FROM properties_2017 prop
        JOIN (
            SELECT parcelid, MAX(transactiondate) AS max_transactiondate
            FROM predictions_2017
            GROUP BY parcelid
            ) pred USING(parcelid)
        JOIN predictions_2017 ON pred.parcelid = predictions_2017.parcelid
                          AND pred.max_transactiondate = predictions_2017.transactiondate
        LEFT JOIN airconditioningtype air USING(airconditioningtypeid)
        LEFT JOIN architecturalstyletype arch USING(architecturalstyletypeid)
        LEFT JOIN buildingclasstype build USING(buildingclasstypeid)
        LEFT JOIN heatingorsystemtype heat USING(heatingorsystemtypeid)
        LEFT JOIN propertylandusetype land USING(propertylandusetypeid)
        LEFT JOIN storytype story USING(storytypeid)
        LEFT JOIN typeconstructiontype type USING(typeconstructiontypeid)
        WHERE propertylandusedesc = "Single Family Residential"
            AND transactiondate <= '2017-12-31'
            AND prop.longitude IS NOT NULL
            AND prop.latitude IS NOT NULL
        '''
        
        url = get_connection('zillow')
        df = pd.read_sql(query,url)
        df.to_csv(filename, index=False)
        
        return df
    
# Above is from capper wrangle
#####################################################    
  
def drop_zill_mvp(zillow):    
    '''
    Dropping 1 null values
    bed 7 = 6 plus
    bath 6 = 5.5+
    rename columns 
    '''
    zillow.dropna(inplace= True)
    zillow.rename(columns = {'bedroomcnt':'bed', 'bathroomcnt':'bath', 'calculatedfinishedsquarefeet': 'sqft', 'taxvaluedollarcnt': 'value'}, inplace=True)
    zillow['bath'] = np.where(zillow['bath'] >= 5.5, '6', zillow['bath'])
    zillow['bed'] = np.where(zillow['bed'] >= 7, '8', zillow['bed'])
    
    return zillow



def train_val_test(df, seed = 55):
    '''
    splits to train val test
    TAKES 1 df
    '''
    train, val_test = train_test_split(df, train_size = 0.7,
                                       random_state = seed)
    
    val, test = train_test_split(val_test, train_size = 0.5,
                                 random_state = seed)
    
    return train, val, test


def X_y_split(train, val, target):
    '''
    Splits train and val into X and Y splits for target testing.
    
    target is target variable entered as the name of the column only in quotes 
    
    returns X_train, y_train , X_val , y_val
    '''
    t = target
    X_train = train.drop(columns=[t])
    y_train = train[t]
    X_val = val.drop(columns=[t])
    y_val = val[t]
    return X_train, y_train , X_val , y_val
