import requests
from bs4 import BeautifulSoup
from os import listdir, mkdir
import hashlib
import urllib.request


url = 'https://www.foodpal-app.com/uploads/images/food/4/butter-603911932624d-800.webp'
user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
headers={'User-Agent':user_agent,} 

def get_from_page(url, name):
    page = requests.get(url)
    page = BeautifulSoup(page.text, "html.parser")
    
    img = page.find_all('img', alt=name)[-1].attrs['data-src']
    request=urllib.request.Request(img,None,headers)
    response = urllib.request.urlopen(request)
    out = open(f'{img.split("/")[-1].replace(".webp", ".png")}', 'wb')
    out.write(response.read())
    out.close()
    
    return {'dataset_info': 'foodpal', 'path_default': f'{img.split("/")[-1].replace(".webp", ".png")}', 
            'kcal_100': float(page.find_all('div', class_='calories')[0].contents[0].split(' ')[0]),
            'prot_100': float(page.find_all('div', class_='protein')[0].contents[0].split(' ')[0]),
            'fat_100': float(page.find_all('div', class_='fat')[0].contents[0].split(' ')[0]),
            'carb_100': float(page.find_all('div', class_='carbohydrates')[0].contents[0].split(' ')[0])}
