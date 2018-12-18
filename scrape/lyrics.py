#! /usr/bin/env python3
# Code mostly adapted from: https://blog.aidemy.net/entry/2018/08/21/104135

import requests
from bs4 import BeautifulSoup
from multiprocessing import Process, Lock, Array


class LyricObject():
    def __init__(self, name, artist, text):
        self.name = name
        self.artist = artist
        self.text = text

    def __str__(self):
        return self.name + "　" + self.artist + "\n" + self.text


def get_lyrics(music_list):
    lyric_list = []

    for music in music_list:

        query_url = "https://www.uta-net.com/search/?Aselect=2&Keyword=" + music[0]
    
        r = requests.get(query_url)
        soup = BeautifulSoup(r.content, "html.parser")
        
        if soup.find_all("td", attrs={"class": "side td1"}):
            singers = soup.find_all("td", attrs={"class": "td2"})
            singer = -1
            
            for i in range(len(singers)):
                if list(singers[i].strings)[0] == music[1]:
                    singer = i
                
            if singer != -1:
                print("Found lyrics for: {}, {}".format(music[0], music[1]))
                
                href = soup.find_all("td", attrs={"class": "side td1"})[singer].contents[0].get("href")
            
                r = requests.get("https://www.uta-net.com"+href)
                soup = BeautifulSoup(r.content, "html.parser")
                lyrics = '\n'.join(list(soup.find_all("div", attrs={"id": "kashi_area"})[0].strings))
            
                # print(lyrics)
                
                lyric_obj = LyricObject(music[0], music[1], lyrics)
            
            
                lyric_list.append(lyric_obj)

                # print("{} lyrics collected".format(len(lyric_list)))

    # print("{} songs found".format(len(lyric_list)))
    
    return lyric_list
