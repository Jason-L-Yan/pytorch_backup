from pyecharts.charts import Bar  # 导入bar模块
attr = ["音响", "电视", "相机", "Pad", "手机", "电脑"]  # 设置x轴数据
v1 = [5, 20, 36, 10, 75, 90]  # 第一组数据
v2 = [10, 25, 8, 60, 20, 80]  # 第二组数据
bar = Bar()  # 实例一个柱状图#
# bar._use_theme("macarons")  # 指定图表显示的主题风格，后面会讲
bar.add_xaxis(attr)
bar.add_yaxis("京东", v1)  # 用add函数往图里添加数据并设置is_stack为堆叠
bar.add_yaxis("淘宝", v2)  # mark_point标记min,max,average, mark_line标记线
bar.render("1.1.柱状图数据堆叠示例33.html")  # 保存为html类型