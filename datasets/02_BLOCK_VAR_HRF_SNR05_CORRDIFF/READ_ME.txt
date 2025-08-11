1) Матрица ground_truth - содержит различия для сравнения TaskA>TaskB И TaskB>TaskA (вместе, не отдельно)
2) Матрицы CorrDiff для TaskA и TaskB
3) Thresholded матрицы для FDR 0.05 (я не делил 0.05/2, т.е. не делал коррекцию на сравнение в обе стороны), отдельно для TaskA>TaskB и TaskB>TaskA
4) Thresholded матрицы для NBS FWE-extent 0.05 (я не делил 0.05/2, т.е. не делал коррекцию на сравнение в обе стороны), отдельно для TaskA>TaskB и TaskB>TaskA.Кол-во пермутаций = 5000, первичный порог tvalue = 3.1746 (p = 0.001)
