import random
import requests
import tempfile
from bs4 import BeautifulSoup
import re
import os
import json
from wikiextractor.extract import Extractor, acceptedNamespaces

templateNamespace = ''

sys_random = random.SystemRandom()
tagRE = re.compile(r'(.*?)<(/?\w+)[^>]*>(?:([^<]*)(<.*?>)?)?')


def collect_pages(text):
    """
    :param text: the text of a wikipedia file dump.
    """
    # we collect individual lines, since str.join() is significantly faster
    # than concatenation
    page = []
    id = ''
    revid = ''
    last_id = ''
    inText = False
    redirect = False
    for line in text:
        if '<' not in line:     # faster than doing re.search()
            if inText:
                page.append(line)
            continue
        m = tagRE.search(line)
        if not m:
            continue
        tag = m.group(2)
        if tag == 'page':
            page = []
            redirect = False
        elif tag == 'id' and not id:
            id = m.group(3)
        elif tag == 'id' and id: # <revision> <id></id> </revision>
            revid = m.group(3)
        elif tag == 'title':
            title = m.group(3)
        elif tag == 'redirect':
            redirect = True
        elif tag == 'text':
            inText = True
            line = line[m.start(3):m.end(3)]
            page.append(line)
            if m.lastindex == 4:  # open-close
                inText = False
        elif tag == '/text':
            if m.group(1):
                page.append(m.group(1))
            inText = False
        elif inText:
            page.append(line)
        elif tag == '/page':
            colon = title.find(':')
            if (colon < 0 or (title[:colon] in acceptedNamespaces) and id != last_id and
                    not redirect and not title.startswith(templateNamespace)):
                yield (id, revid, title, page)
                last_id = id
            id = ''
            revid = ''
            page = []
            inText = False
            redirect = False


def get_document(url, min_passage_length=200):
    target_title = url.split('/')[-1]
    url = f"https://en.wikipedia.org/wiki/Special:Export/{target_title}"
    response = requests.get(url)
    if response.status_code != 200:
        return Exception(f'There is no wiki page named {target_title}!')
    
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')
    text = str(soup)
    input = text.split('\n')
    urlbase = ''
    ordinal = 0  # page count
    jobs = []
    for id, revid, title, page in collect_pages(input):
        job = (id, revid, urlbase, title, page, ordinal)
        ordinal += 1
        jobs.append(job)
        
    if not jobs:
        return Exception('Bad Document error!')
        
    e = Extractor(*jobs[0][:-1])
    e.to_json = True
    
    with tempfile.TemporaryFile('w+') as tmp_f:
        e.extract(tmp_f)
        tmp_f.seek(os.SEEK_SET)
        page = json.load(tmp_f)
    
    tmp = re.sub(r'[=]{2}[\w|\s]+[=]{2}', '[SEP]', page['text'])
    subtitles = [s.strip('= ') for s in re.findall(r'[=]{2}[\w|\s]+[=]{2}', page['text'])]
    
    passages = {}
    used_subtitles = []
    for idx, passage in enumerate(tmp.split('[SEP]')):
        passage = re.sub(r'[=]{3,}', ' ', passage)
        passage = re.sub(r'\s+', ' ', passage).strip()
        
        if idx == 0:
            passages['background'] = passage
            continue
        
        if len(passage) < min_passage_length:
            continue
            
        passages[subtitles[idx-1]] = passage.strip('=')
        used_subtitles.append(subtitles[idx-1])
        
    return {
        'title': page['title'],
        'passages': passages,
        'subtitles': used_subtitles
    }
    
    
def get_passages(ret):
    """_convert wikipedia documents into passages in DSP manner_
    Args:
        ret (_type_): _description_
    Returns:
        _type_: _description_
    """
    psgs = []
    neg_titles = ["see also"]
    for sub_t, psg in ret['passages'].items():
        if sub_t.lower() in neg_titles: continue
        sub_t += "; "
        if sub_t == "background" + "; ":
            sub_t = ""

        txt_psg = sub_t + psg
        
        if len(txt_psg.split(" ")) > 100:
            for i in range(0, len(psg.split(" ")), 100):
                end_idx = min(i+100, len(psg.split(" ")))
                txt_psg = sub_t + " ".join(psg.split(" ")[i:end_idx])
                psgs += [txt_psg]
        else:
            psgs += [txt_psg]
    
    psgs = [ret['title'] + " | " + psg for psg in psgs]
    
    return psgs