#!/usr/bin/env python
# coding: utf-8

# In[5]:


#Loading the necessary libraries 
import pandas as pd
import sqlalchemy as db
from sqlalchemy import create_engine, MetaData, Table, Column, Float, Integer, String #Loading the necessary objects to work with SQLALchemy

def insert_data_to_sql():
    
    """Load csv files to SQLALchemy using Pandas."""
    
    #Source: https://docs.sqlalchemy.org/en/20/core/engines.html; https://www.datacamp.com/tutorial/sqlalchemy-tutorial-examples
    #Create a database MySQL using create_engine function
    
    engine = db.create_engine('sqlite:///python_1.db',echo=True)
    
    #Loading a 'Train_data' file into SQL
    df_train=pd.read_csv('/Users/marththe/Desktop/Python/train.csv') #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    df_train.columns = ['X', 'Y1(training func)','Y2(training func)', 'Y3(training func)', 'Y4(training func)'] #https://www.geeksforgeeks.org/add-column-names-to-dataframe-in-pandas/
    df_train.to_sql(con=engine, name='Train_data', if_exists ='replace',index=False)#https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.DataFrame.to_sql.html
    
    #Loading an 'Ideal_data' file into SQL
    df_ideal=pd.read_csv('/Users/marththe/Desktop/Python/ideal.csv') #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    df_ideal.columns = ['X'] + [f'Y{i}(ideal func)' for i in range(1, 51)] #https://www.geeksforgeeks.org/add-column-names-to-dataframe-in-pandas/; https://peps.python.org/pep-0498/
    df_ideal.to_sql(con=engine, name='Ideal_data', if_exists ='replace',index=False) #https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.DataFrame.to_sql.html
    
    #Loading a 'Test_data' file into SQL
    df_test=pd.read_csv('/Users/marththe/Desktop/Python/test.csv') #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    df_test.columns=['X(test func)', 'Y(test func)'] #https://www.geeksforgeeks.org/add-column-names-to-dataframe-in-pandas/
    df_test.to_sql(con=engine, name='Test_data', if_exists='replace', index=False) #https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.DataFrame.to_sql.html
    
    conn = engine.connect() #https://www.datacamp.com/tutorial/sqlalchemy-tutorial-examples
    metadata = db.MetaData()
    stock = db.Table('Train_data', metadata, autoload=True, autoload_with=engine)
    stock = db.Table('Ideal_data', metadata, autoload=True, autoload_with=engine)
    stock = db.Table('Test_data', metadata, autoload=True, autoload_with=engine)
    query = stock.select()
    exe = conn.execute(query)
    result = exe.fetchmany(5)
    for r in result:
        print(r)
insert_data_to_sql()


# In[27]:


import pandas as pd
import sqlalchemy as db
from sqlalchemy import create_engine

def load_data_from_sql():
    """ Load data tables from SQL to Pandas for Data Aggregation and Merging. """
    
    #Create a database engine
    engine = db.create_engine('sqlite:///python_1.db')
    
    #Load data from SQL into Pandas
    #Source: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_sql.html
    Train_data = pd.read_sql('SELECT * FROM Train_data', con=engine)
    Ideal_data = pd.read_sql('SELECT * FROM Ideal_data', con=engine)
    Test_data = pd.read_sql('SELECT * FROM Test_data', con=engine)
   
    return Train_data, Ideal_data, Test_data

Train_data, Ideal_data, Test_data = load_data_from_sql()

load_data_from_sql()


# In[25]:


import numpy as np

def find_ideal_func(train_column, ideal_df):
    
    """Find the ideal function from an 'Ideal_data' file for one of the four functions in a 'Train_data' file.
       Create a new table 'New_data' with the found Ideal functions.
    """
    min_error = float('inf') #https://note.nkmk.me/en/python-inf-usage/
    ideal_column = None
    for column in ideal_df.columns[1:]:  #Start  with the second column since the first one is 'X'
        #Find the best function using R-Squared https://medium.com/@muhammadsohaib3434/r-squared-r²-6582386b8821
        error = np.sum((train_column - ideal_df[column])**2) #https://numpy.org/doc/stable/reference/generated/numpy.sum.html
        if error < min_error:
            min_error = error
            ideal_column = column
    return ideal_column

#Create a new table 'New_data' to represent new data with new ideal functions
New_data = pd.DataFrame() #https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
New_data['X'] = Train_data ['X']   # The first column of 'New_data' is the same as in 'Train_data' and 'Ideal_data'

for col in Train_data.columns[1:]: #Start  with the second column since the first one is 'X'
    ideal_func_column = find_ideal_func(Train_data[col], Ideal_data)
    New_data[col +'(Ideal Func)'] = Ideal_data[ideal_func_column]
    

#Save a 'New_data' file to csv for better analysis 
New_data.to_csv('New_data.csv', index=False) #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html
print(New_data)


# In[57]:


import pandas as pd
import numpy as np

def match_test_with_ideal():    
    """Find from ideal functions a function with the standard deviation less than sqrt(2). 
       Match the chosen fucntion with Y(test func). 
       Save the results in a new column called 'DeltaY' in Test_data.
       Save a number of the chosen ideal function in another new column called 'No. of ideal function' in Test_data.
       Update a SQL table 'Test_data' with new columns 'DeltaY'
    """
#New_data and Test_data DataFrames with np.linspace, np.sin, np.cos, np.tan, np.log1p for better testing, visualization: https://numpy.org/doc/stable/reference/generated/numpy.sin.html
#https://indianaiproduction.com/numpy-trigonometric-functions/
#https://medium.com/@sagnikkundu25/simple-linear-regression-in-python-numpy-ffbdcbf603db
    New_data = pd.DataFrame({
    'X': np.linspace(0, 10, 100),
    'Y1(training func)(Ideal Func)': np.sin(np.linspace(0, 10, 100)),
    'Y2(training func)(Ideal Func)': np.cos(np.linspace(0, 10, 100)),
    'Y3(training func)(Ideal Func)': np.tan(np.linspace(0, 10, 100)),
    'Y4(training func)(Ideal Func)': np.log1p(np.linspace(0, 10, 100))
    })
    Test_data = pd.DataFrame({
    'X(test func)': np.linspace(0, 10, 100),
    'Y(test func)': np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
    })

# Preparing the DataFrame to store the standard deviations and function names
std_devs = pd.DataFrame(index = Test_data.index)

# Calculate the standard deviation of the differences for each function and select the best one
threshold = np.sqrt(2) #https://www.geeksforgeeks.org/numpy-sqrt-in-python/
for i in range(1, 5):
        func_col = f'Y{i}(training func)(Ideal Func)'
    
        # Interpolate New_data Y values at Test_data X points https://sparkbyexamples.com/python/numpy-interpolate-function/
        interpolated_values = np.interp(Test_data['X(test func)'], New_data['X'], New_data[func_col])
    
        # Calculate the differences
        differences = Test_data['Y(test func)'] - interpolated_values
    
        # Calculate the standard deviation of the differences
        std_dev = np.std(differences) #https://numpy.org/doc/stable/reference/generated/numpy.std.html
    
        # Standard deviation should be less than sqrt(2)
        # Update std_devs DataFrame
        if std_dev < threshold:
            std_devs[func_col] = differences

# Select the column with the minimum standard deviation for each row
min_std_func = std_devs.idxmin(axis=1) #https://www.w3schools.com/python/pandas/ref_df_idxmin.asp
    
Test_data['DeltaY(test func)'] = std_devs.min(axis=1)

# Include the name of the chosen function 
Test_data['No. of ideal func'] = min_std_func


print(Test_data.head())
Test_data.to_csv('updated.csv', index=False)


# In[58]:


import pandas as pd
from sqlalchemy import create_engine
engine = create_engine('sqlite:///python_1.db')
data = pd.read_csv('/Users/marththe/Desktop/updated.csv')
table_name = 'Test'
data.to_sql('Test', con=engine, if_exists='replace', index=False)
engine.dispose()


# In[59]:


import bokeh
from bokeh.plotting import figure, show #https://docs.bokeh.org/en/latest/docs/reference/plotting.html
from bokeh.models import ColumnDataSource #https://docs.bokeh.org/en/3.0.1/docs/user_guide/basic/data.html
from bokeh.io import curdoc #https://docs.bokeh.org/en/latest/docs/reference/themes.html

# Preparing data source for Bokeh https://docs.bokeh.org/en/3.0.1/docs/user_guide/basic/data.html
source = ColumnDataSource(data={
    'X': Test_data['X(test func)'],
    'Y': Test_data['Y(test func)'],
    'Delta_y': Test_data['DeltaY(test func)']
})
# Choosing a dark theme #https://docs.bokeh.org/en/latest/docs/reference/themes.html
curdoc().theme = "dark_minimal" 

# Create a new plot with a title and axis labels https://docs.bokeh.org/en/latest/docs/reference/themes.html
p = figure(title="Test Data and Ideal Functions", x_axis_label='X', y_axis_label='Y')


# Add a scatter renderer with circle glyphs to the plot https://docs.bokeh.org/en/2.4.3/docs/user_guide/plotting.html
p.circle('X', 'Y', size=9, color="yellow", alpha=0.5, legend_label="Test Data", source=source)

colors = ['red', 'green', 'blue', 'purple']  # Different colors for each ideal function https://docs.bokeh.org/en/latest/docs/first_steps/first_steps_4.html
func_labels = [f'Y{i}(training func)(Ideal Func)' for i in range(1, 5)]
for i, color in zip(range(1, 5), colors):
    func_label = f'Y{i}(training func)(Ideal Func)'
    p.line(New_data['X'], New_data[func_label], line_width=2, color=color, legend_label=func_label)


# Add interactivity https://docs.bokeh.org/en/latest/docs/first_steps/first_steps_3.html
p.legend.location = "top_left"
p.legend.click_policy="hide"

# Show the results
show(p)


# In[ ]:




