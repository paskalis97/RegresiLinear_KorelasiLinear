import data
import korelasi


def main():

    print("-------------------------")
    print("| Mikael Prapaskalis G - 152017104|")
    print("| Pemograman Simulasi Kelas B     |")
    print("-------------------------\n")

    independent = []
    dependent = []



    for d in range(1, 4):

        if data.populatedata(independent, dependent, d) == True:

            rho = korelasi.pearson_correlation(independent, dependent)

            print("Dataset %d\n---------" % d)
            print("Independent data: " + (str(independent)))
            print("Dependent data:   " + (str(dependent)))
            print("Pearson Correlation Coefficient rho = %1.2f\n" % rho)
        else:
            print("Cannot populate Dataset %d" % d)


main()