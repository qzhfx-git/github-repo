import requests
from bs4 import BeautifulSoup

# 假设这是特定竞赛的URL
contest_url = 'http://124.222.52.164/contest.php?cid=1015'

# 发送请求
response = requests.get(contest_url)

# # 检查请求是否成功
# if response.status_code == 200:
#     # 解析HTML内容
#     soup = BeautifulSoup(response.text, 'html.parser')
    
#     # 假设榜单在一个表格中，找到这个表格
#     rank_table = soup.find('table', {'id': 'ranklist'})
    
#     # 提取表格中的行
#     rows = rank_table.find_all('tr')
#     for row in rows:
#         # 提取每一列的数据
#         cols = row.find_all('td')
#         # 假设第一列是排名，第二列是用户名，第三列是解决的问题数
#         rank = cols[0].text.strip()
#         username = cols[1].text.strip()
#         solved = cols[2].text.strip()
#         print(f'Rank: {rank}, Username: {username}, Solved: {solved}')
# else:
#     print('请求失败，状态码：', response.status_code)
