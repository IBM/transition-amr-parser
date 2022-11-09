import re
import xml.etree.ElementTree as ET
from tqdm import tqdm
import penman
from penman.layout import Push
import copy
from penman import surface
from collections import Counter, defaultdict
from transition_amr_parser.clbar import yellow_font
from amr_io import AMR
from ipdb import set_trace
from copy import deepcopy
from transition_amr_parser.amr import protected_tokenizer
alignment_regex = re.compile('(-?[0-9]+)-(-?[0-9]+)')
def read_amr_penman(penman_strs):
    amrs = {}
    for penman_str in penman_strs:
        amr = AMR.from_penman(penman_str)
        amrs[amr.amr_id] = amr
    return amrs

def read_amr_add_sen_id(file_path,doc_id=None,add_id=False):
    with open(file_path) as fid:
        raw_amr = []
        raw_amrs = {}
        for line in tqdm(fid.readlines(), desc='Reading AMR'):
            if line.strip() == '':
                
                # From ::node, ::edge etc
                amr = AMR.from_metadata(raw_amr)
                
                raw_amrs[amr.sid] = amr
                raw_amr = []
            else:
                raw_amr.append(line)
                if add_id and '::id' in line:
                    continue

                if '::tok' in line and add_id:
                    raw_amr.append('# ::id '+doc_id+'.'+str(len(raw_amrs)+1))
                # elif '::snt' in line and add_id:
                #     raw_amr.append('# ::id '+doc_id+'.'+str(len(raw_amrs)+1))
            

    return raw_amrs

def read_amr_metadata(penman_strs,doc_id=None,add_id=False):
    amrs = {}
    for idx,penman_str in enumerate(penman_strs):
        p_strs = penman_str.split('\n')
        if add_id:
            p_strs = ['# ::id '+doc_id+'.'+str(idx)+'\n']+p_strs

        amr = AMR.from_metadata(p_strs)
        if amr is not None:
            if add_id:
                assert doc_id is not None, "provide doc id to add id to amr"
                amr.id = doc_id+'.'+str(idx)
                amr.amr_id = doc_id+'.'+str(idx)
                amrs[doc_id+'.'+str(idx)] = amr

            else:
                amrs[amr.amr_id] = amr
    return amrs

def read_amr3(file_path):
    with open(file_path) as fid:
        raw_amr = []
        raw_amrs = {}
        for line in tqdm(fid.readlines(), desc='Reading AMR'):
            if line.strip() == '':
                
                # From ::node, ::edge etc
                amr = AMR.from_metadata(raw_amr)
                raw_amrs[amr.sid] = amr
                raw_amr = []
            else:
                raw_amr.append(line)
    return raw_amrs



def get_all_vars(penman_str):
    in_quotes = False
    all_vars = []
    for (i,ch) in enumerate(penman_str):
        if ch == '"':
            if in_quotes:
                in_quotes = False
            else:
                in_quotes = True
        if in_quotes:
            continue
        if ch == '(':
            var = ''
            j = i+1
            while j < len(penman_str) and penman_str[j] not in [' ','\n']:
                var += penman_str[j]
                j += 1
            all_vars.append(var)
    return all_vars


def get_node_var(penman_str, node_id):
    """
    find node variable based on ids like 0.0.1
    """
    nid = '99990.0.0.0.0.0.0'
    cidx = []
    lvls = []
    all_vars = get_all_vars(penman_str)
    in_quotes = False
    for (i,ch) in enumerate(penman_str):
        if ch == '"':
            if in_quotes:
                in_quotes = False
            else:
                in_quotes = True
        if in_quotes:
            continue

        if ch == ":":
            idx = i
            while idx < len(penman_str) and penman_str[idx] != ' ':
                idx += 1
            if idx+1 < len(penman_str) and penman_str[idx+1] != '(':
                var = ''
                j = idx+1
                while j < len(penman_str) and penman_str[j] not in [' ','\n']:
                    var += penman_str[j]
                    j += 1
                if var not in all_vars:
                    lnum = len(lvls)
                    if lnum >= len(cidx):
                        cidx.append(1)
                    else:
                        cidx[lnum] += 1                            
        if ch == '(':
            lnum = len(lvls)
            if lnum >= len(cidx):
                cidx.append(0)
            lvls.append(str(cidx[lnum]))
        
        if ch == ')':
            lnum = len(lvls)
            if lnum < len(cidx):
                cidx.pop()
            cidx[lnum-1] += 1
            lvls.pop()

        if ".".join(lvls) == node_id:
            j = i+1
            while penman_str[j] == ' ':
                j += 1
            var = ""
            while penman_str[j] != ' ':
                var += penman_str[j]
                j += 1
            return var

    return None

def from_metadata(penman_text, tokenize=False):
    """Read AMR from metadata (IBM style)"""

    # Read metadata from penman
    field_key = re.compile(f'::[A-Za-z]+')
    metadata = defaultdict(list)
    separator = None
    penman_str = ""
    for line in penman_text:
        if line.startswith('#'):
            line = line[2:].strip()
            start = 0
            for point in field_key.finditer(line):
                end = point.start()
                value = line[start:end]
                if value:
                    metadata[separator].append(value)
                separator = line[end:point.end()][2:]
                start = point.end()
            value = line[start:]
            if value:
                metadata[separator].append(value)
        else:
            penman_str += line.strip() + ' ' 
                
    # assert 'tok' in metadata, "AMR must contain field ::tok"
    if tokenize:
        assert 'snt' in metadata, "AMR must contain field ::snt"
        tokens, _ = protected_tokenizer(metadata['snt'][0])
    else:
        assert 'tok' in metadata, "AMR must contain field ::tok"
        assert len(metadata['tok']) == 1
        tokens = metadata['tok'][0].split()

    #print(penman_str)
        
    sid="000"
    nodes = {}
    nvars = {}
    alignments = {}
    edges = []
    root = None
    sentence_ends = []

    if 'short' in metadata:
        short_str = metadata["short"][0].split('\t')[1]
        short = eval(short_str)
        short = {str(k):v for k,v in short.items()}
        all_vars = list(short.values())
    else:
        short = None
        all_vars = get_all_vars(penman_str)
    
    for key, value in metadata.items():
        if key == 'edge':
            for items in value:
                items = items.split('\t')
                if len(items) == 6:
                    _, _, label, _, src, tgt = items
                    edges.append((src, f':{label}', tgt))
        elif key == 'node':
            for items in value:
                items = items.split('\t')
                if len(items) > 3:
                    _, node_id, node_name, alignment = items
                    start, end = alignment_regex.match(alignment).groups()
                    indices = list(range(int(start), int(end)))
                    alignments[node_id] = indices
                else:
                    _, node_id, node_name = items
                    alignments[node_id] = None
                nodes[node_id] = node_name
                if short is not None:
                    var = short[node_id]
                else:
                    var = get_node_var(penman_str, node_id)
                if var is not None and var+" / " not in penman_str:
                    nvars[node_id] = None
                else:
                    nvars[node_id] = var
                if var != None:
                    all_vars.remove(var)
        elif key == 'root':
            root = value[0].split('\t')[1]
        elif key == 'id':
            sid = value[0].strip()
        elif key == 'sentence_ends':
            sentence_ends = value.split()
            sentence_ends = [int(x) for x in sentence_ends]

    if len(all_vars):
        print("varaible not linked to nodes:")
        print(all_vars)
        print(penman_str)
    
    if len(nodes)==0 and len(edges)==0:
        return None

    return AMR(tokens, nodes, edges, root, penman=None,
               alignments=alignments, nvars=nvars, id=sid,sentence_ends=sentence_ends)

def fix_alignments(amr,removed_idx):
    for node_id,align in amr.alignments.items():
        if align is not None:
            num_decr = len([rm for rm in removed_idx if align[0]>rm])
            if num_decr>0:
                lst = [x-num_decr for x in align]
                amr.alignments[node_id] = lst 


def get_sen_ends(amr):
    amr.sentence_ends = []
    removed_idx = []
    new_tokens = []
    for idx,tok in enumerate(amr.tokens):
        if tok=='<ROOT>':
            removed_idx.append(idx)
        elif tok=='<next_sent>':
            amr.sentence_ends.append(len(new_tokens)-1)
            removed_idx.append(idx)
        
        else:
            new_tokens.append(tok)
    amr.sentence_ends.append(len(new_tokens)-1)
    amr.tokens = new_tokens
    fix_alignments(amr,removed_idx)

def remove_unicode(amr):
    for idx,tok in enumerate(amr.tokens):
        new_tok = tok.encode("ascii", "ignore")
        amr.tokens[idx] = new_tok.decode()
        if amr.tokens[idx]=='':
            amr.tokens[idx]='.'



def recent_member_by_sent(chain,sid,doc_id):
    def get_sid_fromstring(string):
         
        sid = [int(s) for s in re.findall(r'\d+', string)]
        assert len(sid)==1
        return sid[0]

    sid = get_sid_fromstring(sid)    
    diff = lambda chain : abs(get_sid_fromstring(chain[0].split('.')[0]) - sid)
    ent = min(chain, key=diff)
    fix = False
    if get_sid_fromstring(ent[0].split('.')[0]) > sid:
    #     print(doc_id," closest sent is higher than connecting node ",ent[0],sid)
        fix = True
    return ent[0]

    

def recent_member_by_align(chain,src_align,doc_id,rel=None):
 
    diff = lambda chain : abs(chain[1]-src_align)
    ent = min(chain, key=diff)
    fix = False
    if ent[1]>= src_align:
    #     print(doc_id," coref edge missing ",ent[1],src_align,rel)
        fix  = True      
    return ent[0]

#convert v0 coref edge to connect to most recent sibling in the chain
def make_pairwise_edges(damr,verbose=False):
    
    ents_chain = defaultdict(list)
    edges_to_delete = []
    nodes_to_delete = []
    doc_id = damr.amr_id
    # damr.edges.sort(key = lambda x: x[0])
    for idx,e in enumerate(damr.edges):
        if e[1] == ':coref-of':
            # if len(ents[e[2]])==0:
                #damr.edges[idx] = (e[0],':same-as',ents[e[2]][-1])
            # else:
            edges_to_delete.append(e)

            if e[0] in damr.alignments and damr.alignments[e[0]] is not None:
                ents_chain[e[2]].append((e[0],damr.alignments[e[0]][0]))
            else:
                #FIXME adding the src node of a coref edge with no alignments member of chain with closest sid
                # print(doc_id + '  ',e[0],' alignments is None  src node in coref edge, not adding it ')
                sid = e[0].split('.')[0]
                if len(ents_chain[e[2]]) >0 :
                    ent = recent_member_by_sent(ents_chain[e[2]],sid,doc_id)
                    damr.edges[idx] = (e[0],':same-as',ent)
                #FIXME
                else:
                    
                    print("coref edge missing, empty chain, edge not added")
                
            assert e[2].startswith('rel')
       

    
    #adding coref edges between most recent sibling in chain    
    for cents in ents_chain.values():
        cents.sort(key=lambda x:x[1])
        for idx in range(0,len(cents)-1):
            damr.edges.append((cents[idx+1][0],':same-as',cents[idx][0]))

    for e in edges_to_delete:
        while e in damr.edges:
            damr.edges.remove(e)

    #connecting all other edges involving chain to most recent member in the chain
    for idx,e in enumerate(damr.edges):
        #Both src and target are coref nodes
        if e[0] in ents_chain and e[2] in ents_chain:
            damr.edges[idx] = (ents_chain[e[0]][-1][0],e[1],ents_chain[e[2]][-1][0])
        
        elif e[2] in ents_chain.keys():
            #src node is a normal amr node
            if e[0] in damr.alignments and damr.alignments[e[0]] is not None:
                ent = recent_member_by_align(ents_chain[e[2]],damr.alignments[e[0]][0],doc_id,e[1])
                
            else:
                #FIXME assigning src node with no alignments to the recent member by sent in the coref chain
                # print(doc_id + '  ',e[0],' alignments is None ')
                sid = e[0].split('.')[0]
                ent = recent_member_by_sent(ents_chain[e[2]],sid,doc_id)
            damr.edges[idx] = (e[0],e[1],ent)

        elif e[0] in ents_chain.keys():
            if e[2] in damr.alignments and damr.alignments[e[2]] is not None:
                ent = recent_member_by_align(ents_chain[e[0]],damr.alignments[e[2]][0],doc_id,e[1])
            else:
                #FIXME assigning tgt node with no alignments to the recent member by sent in the coref chain
                # print(doc_id + '  ',e[0],' alignments is None ')
                sid = e[2].split('.')[0]
                ent = recent_member_by_sent(ents_chain[e[0]],sid,doc_id)
        
            damr.edges[idx] = (ent,e[1],e[2])

       
    for n in ents_chain.keys():
        while n in damr.nodes:
            del damr.nodes[n]
    
        
    
    return damr


