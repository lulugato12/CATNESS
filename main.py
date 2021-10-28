import CATNESS.lioness

start = timeit.default_timer()

path = "C:/Users/hp/Desktop/mission_catness/"
data = pd.read_csv(path + "output/15_cases.csv", index_col = 0)

output, size = lioness.lioness_algorithm(data)
lioness.plot(output, size)
pd.DataFrame(output).to_csv(path + 'output/' + str(size) + "_mim.csv")

end = timeit.default_timer()
print("Tiempo de ejecusion:", end-start, "s")
