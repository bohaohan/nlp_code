# coding: utf-8
__author__ = 'bohaohan'
import xlrd
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )
data = xlrd.open_workbook('旅游信息数据.xlsx')
table = data.sheet_by_name(u"0311标注")
nrows = table.nrows
f = file("3.17data.txt", "w+")
for i in range(nrows):
    cont = table.cell_value(i, 8)
    res = table.cell_value(i, 12)
    if u"垃圾" in res or u"无预警价值" in res:
        res = " \t垃圾\n"
    else:
        res = " \t预警\n"
    cont = cont.replace("\b", "")
    cont = cont.replace(" ", "")
    line = cont.strip() + res
    f.write(line)
f.close()
