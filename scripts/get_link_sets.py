'''
    Collect the set of domains that are resolved on each website's landing page

    Uses Umbrella data for set of sites to check
'''


from wanderful import browser
from helpers import format_dirpath, mydir, listfiles
import csv
from bs4 import BeautifulSoup
import pyautogui as pag
from time import sleep
import signal
import unicodedata

class ToutEx(Exception):
    pass

def touthand(signum, frame):
    raise toutex

topdir = format_dirpath(mydir()+"../")
supportdir = format_dirpath(topdir+"support_files/")
hardir = "/home/marc/Downloads/"

doms = list()


with open(supportdir+"top-1m.csv", 'r+') as f:
    reader = csv.reader(f)
    for line in reader:
        doms.append(line[1])
        if len(doms) > 9999:
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
    pag.moveTo(122, 1327)  # filter to only contain HTML
    pag.click()
    sleep(1)
    pag.moveTo(318,1349)  # reverse sort
    pag.click()
    sleep(1)
    pag.click()


def capture_har(t=0):
    pag.moveTo(459, 1373+t*25)  # right click item in waterfall
    pag.click(button='right')
    sleep(1)
    pag.press('down')
    pag.press('down')
    pag.press('enter')  # navigate to and select "Save all as HAR"
    sleep(5)
    pag.press('down')
    sleep(1)
    pag.press('enter')
    sleep(1)


def count_hars():
    files = listfiles(hardir, suffix=".har")
    return len(files)


def normtext(t):
    return unicodedata.normalize("NFKD", t.casefold())


broken = ['404', 'not found', '403', '503', 'error', 'invalid', 'denied']
broken = [normtext(z) for z in broken]


last_count = count_hars()
num = 0
for dom in doms:
    num += 1
    if num <= 6880:
        continue
    while ff.browser_pid is None:
        setup_browser(ff)
    print(dom)
    src = ff.get(dom, 1, failkill=False, waittime=30)
    if src is None:
        print("broken")
        if ff.browser_pid is not None:
            ff.kill_browser()
        continue
    if len(src) < 2000:
        print("broken")
        if ff.browser_pid is not None:
            ff.kill_browser()
        continue
    title = normtext(ff.active_browser.title)
    src = normtext(src[:500])
    if any([z in title or z in src for z in broken]):
        print("broken")
        if ff.browser_pid is not None:
            ff.kill_browser()
        continue
    attempts = 0
    signal.signal(signal.SIGALRM, ToutEx)
    signal.alarm(300)
    try:
        while attempts < 4:  # sometimes it doesn't save the first time
            capture_har(attempts)
            new_count = count_hars()
            if new_count != last_count:
                last_count = new_count
                break
            sleep(5)
            attempts += 1
    except ToutEx:
        pass
    finally:
        signal.alarm(0)
    if ff.browser_pid is not None:
        ff.kill_browser()
