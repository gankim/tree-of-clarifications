import re
import html

def remove_html_tags(text):
    clean = re.compile('<.*?>')
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = text.replace("\n", " ")
    
    return re.sub(clean, '', text)

def make_str_disambig(lst_disambig, ambigQA=True):
    
    str_disambigs = ""
    for idx, disambig in enumerate(lst_disambig):
        str_disambigs += f"\nDQ {idx+1}: {disambig['question']}"
        if ambigQA:
            str_disambigs += f"\nDA {idx+1}: {disambig['short_answers'][0]}"
        else:
            str_disambigs += f"\nDA {idx+1}: {disambig['answer']}"
    
    return str_disambigs
    

def parse_disambig(str_disambig):
    lst_disambig = str_disambig.split("\n")    
    if lst_disambig[0] == "":
        lst_disambig = lst_disambig[1:]
    disambigs = []
    for i in range(0, len(lst_disambig), 2):
        if i+1 >= len(lst_disambig):
            continue
        try:
            assert "DQ" in lst_disambig[i]
            assert "DA" in lst_disambig[i+1]
            
            question = re.split(r'DQ\s*\d*:', lst_disambig[i])[1]
            answer = re.split(r'DA\s*\d*:', lst_disambig[i+1])[1]
            disambigs.append({'question' : question.strip(), 
                            'answer'   : answer.strip()
                            }
            ) 
        except:
            continue
        
    return disambigs

def old_parse_disambig(str_disambig):
    lst_disambig = str_disambig.split("\nSub-")    
    
    disambigs = []
    for i in range(0, len(lst_disambig), 2):
        if i+1 >= len(lst_disambig):
            continue
        assert "Question" in lst_disambig[i]
        assert "Answer" in lst_disambig[i+1]
        
        question = re.split(r'Question\s*\d*:', lst_disambig[i])[1]
        answer = re.split(r'Answer\s*\d*:', lst_disambig[i+1])[1]
        disambigs.append({'question' : question.strip(), 
                        'answer'   : answer.strip()
                        }
        ) 
        
    return disambigs