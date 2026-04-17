import requests

url = 'https://data.gov.tw/api/v1/rest/dataset/14261'
try:
    r = requests.get(url, timeout=30)
    print('status:', r.status_code)
    print('content-type:', r.headers.get('Content-Type'))
    print('len:', len(r.text))
    print('\n--- body head ---\n')
    print(r.text[:2000])
except Exception as e:
    print('ERROR:', type(e).__name__, e)
