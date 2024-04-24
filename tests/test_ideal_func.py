#!/usr/bin/env python
# coding: utf-8

# In[3]:


import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from test_ideal import insert_data_to_sql  # Ensure your function is importable

class TestDatabaseInsertion(unittest.TestCase):
    @patch('pandas.read_csv')
    @patch('pandas.DataFrame.to_sql')
    @patch('sqlalchemy.create_engine')
    def test_insert_data_to_sql(self, mock_create_engine, mock_to_sql, mock_read_csv):
        # Setup mock objects
        mock_df = pd.DataFrame({
            'X': [1, 2, 3],
            'Y1(training func)': [4, 5, 6]
        })
        mock_read_csv.return_value = mock_df
        mock_engine = mock_create_engine.return_value
        mock_connection = mock_engine.connect.return_value
        mock_metadata = MagicMock()
        mock_table = MagicMock()
        
        # Configure the mock to return a connection and a table
        mock_metadata.Table.return_value = mock_table
        mock_table.select.return_value = 'SELECT * FROM Table'
        mock_connection.execute.return_value.fetchmany.return_value = [('row1', 'row2')]

        # Call the function under test
        insert_data_to_sql()

        # Check that read_csv was called correctly
        mock_read_csv.assert_any_call('/Users/marththe/Desktop/Python/train.csv')
        mock_read_csv.assert_any_call('/Users/marththe/Desktop/Python/ideal.csv')
        mock_read_csv.assert_any_call('/Users/marththe/Desktop/Python/test.csv')

        # Check that to_sql was called correctly
        mock_to_sql.assert_called()

        # Check that create_engine was called with the correct database URL
        mock_create_engine.assert_called_with('sqlite:///python_1.db', echo=True)

        # Print output from mocked fetch
        print(mock_connection.execute.return_value.fetchmany.return_value)

if __name__ == '__main__':
    unittest.main()


# In[ ]:




