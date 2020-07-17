num = [1, 2, 3, 4, 5]
num = iter(num)
print(num)

# ######################### 迭代器 ##############################
# # 迭代器可以通过 next 进行遍历
# print(next(num))  # 1
# print(next(num))  # 2
# print(next(num))  # 3
# print(next(num))  # 4
# print(next(num))  # 5
# # print(next(num))  # 源列表只有五个元素，当要输入第六个是会报错

# # for 循环亦可以对迭代器进行遍历，输出前需要先把上面注释掉才可以
# print("=" * 80)
# for i in num:
#     print(i)

# 此方法也可以
# while True:
#     try:
#         print(next(num))
#     except StopIteration:
#         break

# ######################### 生成器本身也是迭代器 ##############################
# 接下来我们创建生成器对象：
# ge = (x for x in range(1))
# 这将创建一个生成器对象，类似于列表解析，将外层用括号括起来，同样可以调用next()方法取值。
# 1. 通过 yield 创建生成器对象
# 我们通过斐波拉契数列来定义一个函数说明yield，如果函数包含yield，则这个函数是一个生成器，
# 当生成器对象每次调用next()时，遇到yield关键字会中断执行，当下一次调用next()，会从中断出开始执行。


def fib(max):
    n, a, b = 0, 0, 1
    print("停")
    while n < max:
        yield b
        a, b = b, a + b 
        n += 1


data = fib(10)
for num in data:
    print(num)
