# 导入库
from pyecharts.charts import Line
import pyecharts.options as opts
from pyecharts.render import make_snapshot
from snapshot_selenium import snapshot
# 绘制散点数据
x = [i + 1 for i in range(14)]
y1 = [-6.794, -10.644, -14.296, -19.763, -24.937, -31.317, -37.182, -43.498, -48.441, -51.683, -52.682, -52.788, -52.935, -53.004]
y2 = [-6.836, -10.576, -14.190, -19.848, -24.824, -31.104, -36.358, -40.341, -42.128, -42.501, -42.699, -42.815, -42.593, -42.701]


# 定义一个Line_charts函数
def Line_charts() ->Line:
    c = Line()
    c.add_xaxis(xaxis_data=x)
    c.add_yaxis(series_name='', y_axis=y1)
    c.add_yaxis(series_name='', y_axis=y2)
    # c.is_inverse()
    c.set_series_opts(label_opts=opts.LabelOpts(position="right"))
    c.set_global_opts(title_opts=opts.TitleOpts(title="Bar-测试渲染图片"))
    return c


# # 绘制图表
c = Line_charts()
c.render("second_line.html")
# make_snapshot(snapshot, Line_charts().render(), r"Line888.jpeg")
