'''
    Collect the set of domains that are resolved on each website's landing page
'''


from wanderful import browser
from helpers import format_dirpath, mydir
import csv
from bs4 import BeautifulSoup
import pyautogui as pag
from time import sleep

topdir = format_dirpath(mydir()+"../")
supportdir = format_dirpath(topdir+"support_files/")

doms = list()


with open(supportdir+"top-1m.csv", 'r+') as f:
    reader = csv.reader(f)
    for line in reader:
        doms.append(line[1])
        if len(doms) > 1499:
            break

ff = browser.firefox_manager(headless=False)

def setup_browser(b):
    b.launch_browser()
    print("sleeping...")
    pag.press('f12')  # open firebug
    pag.moveTo(650, 1299)  # click network (waterfall) tab
    pag.click()
    sleep(1)
    pag.click()
    sleep(1)
    pag.moveTo(345, 1325)  # filter to only contain images
    pag.click()            # ^ this is to make right click menu consistent
                           # ^^ this only needs to happen once (it persists)

for dom in doms:
    if ff.browser_pid is None:
        setup_browser(ff)
    print(dom)
    src = ff.get(dom, 1, failkill=False)
    if src is None:
        continue
    if len(src) == 0:
        continue
    pag.moveTo(528, 1403)  # right click item in waterfall
    pag.click(button='right')
    sleep(1)
    pag.moveTo(630, 1277)  # click "save HAR" from right click menu
    pag.click()
    sleep(2)
    pag.moveTo(760, 1051)  # make sure "save file" radio button is checked
    pag.click()
    sleep(1)
    pag.moveTo(1141, 1150)  # click "OK" to save
    pag.click()
    sleep(1)
    if ff.browser_pid is not None:
        ff.kill_browser()
