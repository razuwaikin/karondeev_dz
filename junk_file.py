import cupy as cp

kernel = r"""
extern "C" __global__ void test(int* a) { a[0] = 777; }
"""

# Убираем ненужные include - CUDA использует свои стандартные библиотеки
module = cp.RawModule(code=kernel)

func = module.get_function("test")

arr = cp.zeros(1, dtype=cp.int32)
func((1,), (1,), (arr,))
print(arr)  # Должно вывести [777]
