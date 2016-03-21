from urllib3 import HTTPConnectionPool

__author__ = 'bohaohan'
import requests

base_url = "http://104.197.25.253:8080/"
user_key = "669f3eba890661564ae16ae1b3280033"


'''
Api of polySearch
http://104.197.25.253:8080/polysearch?query=LRP16&target_type=gene&query_type=gene
&userkey=669f3eba890661564ae16ae1b3280033


http://104.197.25.253:8080/cache?sid=1234556577&userkey=669f3eba890661564ae16ae1b3280033

'''


def get_search_url(query):
    return "http://104.197.25.253:8080/polysearch?query=" + query + "&target_type=gene&query_type=gene&userkey=" + user_key


def get_cache_url(sid):
    return "http://104.197.25.253:8080/cache?sid="+sid+"&userkey=" + user_key


def test_search():
    f = open("LRP16.json", "aw+")
    url = get_search_url("LRP16")
    r = requests.get("http://104.197.25.253:8080/polysearch",
                      params={'query': 'LRP16', 'target_type': 'gene', 'query_type': 'gene', 'userkey': user_key},
                      timeout=6000)
    print r.text
    f.write(r.text)
    print url
if __name__ == "__main__":
    test_search()