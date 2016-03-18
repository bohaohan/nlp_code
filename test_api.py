from urllib3 import HTTPConnectionPool

__author__ = 'bohaohan'
import requests

base_url = "http://104.197.25.253:8080/"
user_key = "a671c04b01c3487b6d1f7dab9dc325db"


'''
Api of polySearch
http://104.197.25.253:8080/polysearch?query=LRP16&target_type=gene&query_type=gene
&userkey=a671c04b01c3487b6d1f7dab9dc325db


http://104.197.25.253:8080/cache?sid=1234556577&userkey=a671c04b01c3487b6d1f7dab9dc325db

'''


def get_search_url(query):
    return "http://104.197.25.253:8080/polysearch?query=" + query + "&target_type=gene&query_type=gene&userkey=" + user_key


def get_cache_url(sid):
    return "http://104.197.25.253:8080/cache?sid="+sid+"&userkey=" + user_key


def test_search():
    url = get_search_url("LRP16")
    r = requests.post("http://104.197.25.253:8080/polysearch",
                      data={'query': 'LRP16', 'target_type': 'gene', 'query_type': 'gene', 'userkey': user_key},
                      timeout=6000)
    # r = requests.get(url, timeout=6000)
    # data1=r.content
    # print data1
    # print {'http': line}
    # sleep(1)
    # pool = HTTPConnectionPool('http://api.polysearch.ca:8080', retries=False)
    # r = pool.request('GET', '/polysearch',
    # fields={'query': 'LRP16', 'target_type': 'gene', 'query_type': 'gene', 'userkey': user_key})
    print r.text
    print url
if __name__ == "__main__":
    test_search()