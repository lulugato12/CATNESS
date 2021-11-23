from CATNESS.catness import lioness_algorithm, plot_networks, compute_properties
import timeit
import pandas as pd

start = timeit.default_timer()

path = ''
mim = pd.read_csv(path + 'output/cases_pam50_mim.csv', index_col = 0)
compute_properties(mim, path)

end = timeit.default_timer()
print('Properties computation time:', end-start, 's')
