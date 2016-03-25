# coding: utf-8
__author__ = 'bohaohan'
import xlrd
import datetime
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

extract_file = "weibo_warn0316.xlsx"
table_name = u"Sheet2"
out_file = str(datetime.datetime.now().month) + "." + str(datetime.datetime.now().day) + "-data.txt"
print "Start extract", extract_file, "on", table_name

data = xlrd.open_workbook(extract_file)
table = data.sheet_by_name(table_name)
nrows = table.nrows
f = file(out_file, "w+")
for i in range(nrows):
    if i == 1:
        continue
    cont = table.cell_value(i, 8)
    res = table.cell_value(i, 12)
    if res is None:
        continue
    if u"垃圾" in res or u"无预警价值" in res:
        res = " \t垃圾\n"
    else:
        res = " \t预警\n"
    cont = cont.replace("\b", "")
    cont = cont.replace(" ", "")
    line = cont.strip() + res
    f.write(line)
f.close()
print "success!"
