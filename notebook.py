
from pyaxis import pyaxis

fp = r"/Users/boasnydegger/Desktop/github/dsfgp/data/px-x-1003020000_101.px" 
#parse contents of *.px file
px = pyaxis.parse(uri = fp , encoding = 'ISO-8859-2')

#store data as pandas dataframe
data_df = px['DATA']
#store metadata as a dictionary (not necessary in this case, but might be helpful)
meta_dict = px['METADATA']

