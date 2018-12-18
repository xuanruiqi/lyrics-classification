#! /usr/bin/env python3
# Code mostly adapted from: https://blog.aidemy.net/entry/2018/08/21/104135

import requests
from bs4 import BeautifulSoup
import re
import collections

url_spring = 'http://www.rankingbook.com/category/music/spring/favorite/'
url_summer = 'http://www.rankingbook.com/category/music/summer/favorite/'
url_autumn = 'http://www.rankingbook.com/category/music/autumn/favorite/'
url_winter = 'http://www.rankingbook.com/category/music/winter/favorite/'

def get_music(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "html.parser")
    music = soup.find_all("a", attrs={"target": "_top"})
    music.extend(soup.find_all("td", attrs={"class": "b","align":"LEFT"}))
    
    for i in range(len(music)):
        music[i] = list(music[i].strings)[0][:-1]
        music[i] = re.split("[(（／＜…『</]",music[i])
    
    return music


def del_error(music):
    delindex = []
    for i in range(len(music)):
        if(len(music[i])!=2):
            delindex.append(i)
    delindex.reverse()
    for i in delindex:
        del music[i]


def trim(music):
    for item in music:
        item[0] = str.strip(item[0])

        
def scrape_and_clean(url):
    music_list = get_music(url)
    del_error(music_list)
    trim(music_list)
    return music_list


def write_to_csv(music_list, fp):
    for music in music_list:
        f.write("{}, {}\n".format(music[0], music[1]))


if __name__ == "__main__":
    print("Start scraping...")
    
    music_spring = scrape_and_clean(url_spring)
    music_summer = scrape_and_clean(url_summer)
    music_autumn = scrape_and_clean(url_autumn)
    music_winter = scrape_and_clean(url_winter)

    print("Done with scraping...")
    
    with open('spring.txt', 'w+') as f:
        write_to_csv(music_spring, f)

    with open('summer.txt', 'w+') as f:
        write_to_csv(music_summer, f)

    with open('autumn.txt', 'w+') as f:
        write_to_csv(music_autumn, f)

    with open('winter.txt', 'w+') as f:
        write_to_csv(music_winter, f)
