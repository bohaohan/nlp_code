# coding: utf-8
__author__ = 'bohaohan'
import requests

i = 0
def yes():
    r = requests.get(params={'strSearchType':'title',
'match_flag':'forward',
'historyCount':'1',
'strText':'gay',
'doctype':'ALL',
'displaypg':'20',
'showmode':'list',
'sort':'CATA_DATE',
'orderby':'desc',
'location':'ALL'}, url="http://opac.lib.ustc.edu.cn/opac/openlink.php")
    print r.status_code


while True:
    yes()
    print i + 1
    i = i + 1