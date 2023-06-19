import requests
from bs4 import BeautifulSoup
from os import listdir, mkdir
import hashlib
import urllib.request


url = 'https://www.foodpal-app.com/uploads/images/food/4/butter-603911932624d-800.webp'
user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
headers={'User-Agent':user_agent,} 

def get_from_url(page):
    description = ''
    for i in page.find_all('meta'):
        if 'name' in i.attrs:
            if i.attrs['name'] == 'description':
                description += i.attrs['content']

    img = 'https://www.weightloss.com.au' + page.find('img', itemprop="image").attrs['src']
    
    if not page.find('span', itemprop="servingSize").contents:
        mass = np.nan
    else:
        mass = float(page.find('span', itemprop="servingSize").contents[0].replace('g', '')) / 100

    kcal, prot, fat, carb = [float(page.find('td', itemprop="calories").contents[0].split()[0]), 
                             float(page.find('td', itemprop="proteinContent").contents[0].split()[0]), 
                             float(page.find('td', itemprop="fatContent").contents[0].split()[0]), 
                             float(page.find('td', itemprop="carbohydrateContent").contents[0].split()[0])]
    
    request=urllib.request.Request(img,None,headers)
    resource = urllib.request.urlopen(request)
    out = open(f"{img.split('/')[-1].replace('jpg', 'png')}", 'wb')
    out.write(resource.read())
    out.close()

    return {'path_default': f"{img.split('/')[-1].replace('jpg', 'png')}", 
            'kcal_100': kcal / mass, 'prot_100': prot / mass, 'fat_100': fat / mass, 'carb_100': carb / mass, 
            'mass': mass * 100, 'text': description, 'dataset_info': 'weightloss', 
            'kcal': kcal, 'prot': prot, 'fat': fat, 'carb': carb}
