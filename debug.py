p = [1009.003, 1004.113, 998.952, 993.504, 987.758, 981.697, 975.308, 968.575, 961.482, 954.012, 946.148, 937.874, 929.173, 920.027, 910.418, 900.329, 889.686, 878.253, 865.827, 852.338, 837.717, 821.888, 804.78, 786.313, 765.947, 743.078, 717.486, 688.965, 657.333, 622.449, 585.465, 549.021, 514.475, 481.73, 450.706, 421.317, 393.484, 367.137, 342.208, 318.637, 296.37, 275.363, 255.568, 236.951, 219.478, 203.111, 187.807, 173.519, 160.191, 147.772, 136.214, 125.471, 115.511, 106.299, 97.793, 89.969, 82.792, 76.205, 69.978, 63.722, 57.289, 50.767, 44.258, 37.873, 31.735, 25.969, 20.768, 16.458, 13.077, 10.414, 8.315, 6.659, 5.35, 4.316]

# Find the index with the value closest to 500
closest_index = min(range(len(p)), key=lambda i: abs(p[i] - 500))

print("Index closest to 500 in the given sequence:", closest_index)
