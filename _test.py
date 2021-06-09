
a = ['0_la','1_la','2_la','10_la','5_la']
print(a)
a = sorted(a,key=lambda x: int(x.split('_')[0]) )  # 按照每个元素切分‘_’后前面的数字来排序
print(a)

l = [int(256 / 6 * i) for i in range(6)]
print(l)

'''
sorted() lamda表达式的用法
'''