import requests
from bs4 import BeautifulSoup
from os import listdir, mkdir
import hashlib
import urllib.request


def get_info_from_url(url):
    # read page
    page = requests.get(url)
    page = BeautifulSoup(page.text, "html.parser")
    
    # name of dish
    title, label, _ = ', '.join(page.title.string.split(', ')[1:]).replace(' Calories', '').split(' - ')
    
    # link for image
    for i in page.find_all('a'):
        try:
            if 'https://fddb' in i.img.attrs['src']:
                link = i.img.attrs['src']
                break
        except:
            continue
    
    # mass of dish
    for i in page.find_all('p'):
        for j in i.find_all('a', class_="servb"):
            continue
    name, amount = j.contents[0].replace(')', '').split(' (')
    amount, unit = amount.split(' ')
    amount = float(amount)
    
    # kpfc
    nutr_info = page.find('div', id=f'more{j.get("onclick").split("(")[1][:-2]}')
    nutr_info = [[j.string for j in i.contents] for i in nutr_info.children][1:5]
    nutr_info = {i[0][:-1]: float(i[1].split(' ')[0]) for i in nutr_info}
    
    # save pic
    mkdir(f'dish_{hashlib.md5(url.encode("utf-8")).hexdigest()}')
    resource = urllib.request.urlopen(link)
    out = open(f'dish_{hashlib.md5(url.encode("utf-8")).hexdigest()}/rgb.png', 'wb')
    out.write(resource.read())
    out.close()
    
    return {'dataset_info': 'fddb', 'dish_id': f'dish_{hashlib.md5(url.encode("utf-8")).hexdigest()}', 
            'path_default': f'dish_{hashlib.md5(url.encode("utf-8")).hexdigest()}/rgb.png', 
            'path_additional': None, 'mass': amount, 'kcal_total': nutr_info['Calories'], 
            'prot_total': nutr_info['Protein'], 'fat_total': nutr_info['Fat'],
            'carb_total': nutr_info['Carbohydrates'], 'kcal_100': nutr_info['Calories'] * 100 / amount,
            'prot_100': nutr_info['Protein']  * 100 / amount, 'fat_100': nutr_info['Fat']  * 100 / amount, 
            'carb_100': nutr_info['Carbohydrates']  * 100 / amount,
            'ingridients': None, 'text': f'{title}, {label}', 'img_exist': 1, 'link': link, 'url': url}

def get_urls_from_groups(url):
    urls = []
    page = requests.get(url)
    page = BeautifulSoup(page.text, "html.parser")
    for column in page.find_all('td', valign="top", width="50%"):
        for dish in str(column.p).split('<br/>'):
            dish = BeautifulSoup(dish, 'html.parser')
            try:
                if dish.img or 'style' in dish.a.attrs:
                    urls.append('https://fddb.info' + dish.a.attrs['href'])
            except:
                continue
    return urls
