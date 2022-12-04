import pandas as pd
pd.set_option('display.max_columns', 6)
from sklearn.decomposition import PCA
import seaborn
import matplotlib.pyplot as plt
from nlpia.data.loaders import get_data
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from mlxtend.preprocessing import minmax_scaling


df = get_data('pointcloud').sample(1000)
pca = PCA(n_components=2)
df2d = pd.DataFrame(pca.fit_transform(df), columns=list('xy'))
df2d.plot(kind='scatter', x='x', y='y', title='Horse via PCA')
plt.show()


svd = TruncatedSVD(n_components=2, n_iter=10)
df2d = pd.DataFrame(svd.fit_transform(df), columns=list('xy'))
df2d.plot(kind='scatter', x='x', y='y', title='Horse via SVD')
plt.show()

ldia = LatentDirichletAllocation(n_components=2)
df = minmax_scaling(df, columns=list('xyz'), min_val=-1, max_val=1)
df2d = pd.DataFrame(ldia.fit_transform(df), columns=list('xy'))
df2d.plot(kind='scatter', x='x', y='y', title='Horse via LDiA')
plt.show()
