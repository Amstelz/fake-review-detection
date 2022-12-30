import sqlite3
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords

y_flagged = pd.read_csv('../Data/flagged_by_all_y.csv')
n_flagged = pd.read_csv('../Data/flagged_by_all_n.csv')

content_y = y_flagged[['reviewContent']]
content_y.values.tofile('file.txt', sep=" ")