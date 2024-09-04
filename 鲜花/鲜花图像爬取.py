import requests
import os
import urllib

class GetImage():
    def __init__(self,keyword='鲜花',paginator=1):
        # self.url: 链接头
        self.url = 'http://image.baidu.com/search/acjson?'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT10.0; WOW64) AppleWebKit/537.36(KHTML, like Gecko) Chrome/69.0.3497.81 Safari/537.36'
            }
        self.keyword = keyword      # 定义关键词
        self.paginator = paginator  # 定义要爬取的页数

    def get_param(self):
        # 将中文关键词转换为符合规则的编码
        keyword = urllib.parse.quote(self.keyword)
        params = []
        # 为爬取的每页链接定制参数
        for i in range(1, self.paginator + 1):
            params.append(
                'tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord={}&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=&hd=1&latest=0&copyright=0&word={}&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&fr=&expermode=&force=&cg=star&pn={}&rn=30&gsm=78&1557125391211='.format(keyword, keyword, 30 * i))
        return params   # 返回链接参数
    
    def get_urls(self, params):
        urls = []
        for param in params:
            # 拼接每页的链接
            urls.append(self.url + param)
        return urls   # 返回每页链接

    def get_image_url(self, urls):
        image_url = []
        for url in urls:
            json_data = requests.get(url, headers=self.headers).json()
            json_data = json_data.get('data')
            for i in json_data:
                if i:
                    image_url.append(i.get('thumbURL'))
            
        return image_url

    def get_image(self, image_url):
        """
        根据图片url，在本地目录下新建一个以搜索关键字命名的文件夹，然后将每一个图片存入。
        :param image_url:
        :return:
        """
        file_name = os.path.join('.', self.keyword)
        print(file_name)
        if not os.path.exists(file_name):
            os.mkdir(file_name)
        for index, url in enumerate(image_url, start=1):
            with open(file_name+'/{}.jpg'.format(index), 'wb') as f:
                f.write(requests.get(url,headers=self.headers).content)
            if index != 0 and index % 30 == 0:
                print('第{}页下载完成'.format(index/30))

    def __call__(self, *args, **kwargs):
        params = self.get_param()  # 获取链接参数
        urls = self.get_urls(params)
        image_url = self.get_image_url(urls)
        self.get_image(image_url)

if __name__ == '__main__':
    spider = GetImage('鲜花', 3)
    spider()