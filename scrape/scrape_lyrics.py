#! /usr/bin/env python3

import requests
from bs4 import BeautifulSoup
from multiprocessing import Process

from lyrics import *
from scrape_seasons import *

url_spring = 'http://www.rankingbook.com/category/music/spring/favorite/'
url_summer = 'http://www.rankingbook.com/category/music/summer/favorite/'
url_autumn = 'http://www.rankingbook.com/category/music/autumn/favorite/'
url_winter = 'http://www.rankingbook.com/category/music/winter/favorite/'

def write_files(directory, lyrics):
    for lyr_object in lyrics:
        print("Creating {}/{}_{}.txt".format(directory, lyr_object.name, lyr_object.artist))
        with open("{}/{}_{}.txt".format(directory, lyr_object.name, lyr_object.artist), "w+") as f:
            f.write(lyr_object.text)

def handle_spring():
    music_spring = scrape_and_clean(url_spring)
    spring_lyrics = get_lyrics(music_spring)
    write_files("spring", spring_lyrics)

    
def handle_summer():
    music_summer = scrape_and_clean(url_summer)
    summer_lyrics = get_lyrics(music_summer)
    write_files("summer", summer_lyrics)

    
def handle_autumn():
    music_autumn = scrape_and_clean(url_autumn)
    autumn_lyrics = get_lyrics(music_autumn)
    write_files("autumn", autumn_lyrics)


def handle_winter():
    music_winter = scrape_and_clean(url_winter)
    winter_lyrics = get_lyrics(music_winter)
    write_files("winter", winter_lyrics)

    
# This is too slow, I'd have to do this in parallel to speed it up
def main():
    p_spring = Process(target=handle_spring)
    p_summer = Process(target=handle_summer)
    p_autumn = Process(target=handle_autumn)
    p_winter = Process(target=handle_winter)

    procs = [p_spring, p_summer, p_autumn, p_winter]

    for p in procs:
        p.start()

    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
