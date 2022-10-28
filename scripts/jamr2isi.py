import argparse

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

def add_jamr_ids(penman_str):
    """
    find node variable based on ids like 0.0.1
    """
    cidx = []
    lvls = []
    all_vars = get_all_vars(penman_str)
    in_quotes = False
    var2id = {}

    ret_penman = ""
    current_node_id = None
    i = 0
    #for (i,ch) in enumerate(penman_str):
    while i < len(penman_str):
        ch = penman_str[i]
        ret_penman += ch
        if ch == '"':
            if in_quotes:
                in_quotes = False
            else:
                in_quotes = True
        if in_quotes:
            i += 1
            continue

        if ch == ":":
            idx = i
            while idx < len(penman_str) and penman_str[idx] != ' ':
                idx += 1
            if idx+1 < len(penman_str) and penman_str[idx+1] != '(':
                var = ''
                j = idx+1
                while j < len(penman_str) and penman_str[j] not in [' ','\n',')']:
                    var += penman_str[j]
                    j += 1
                if var not in all_vars:
                    lnum = len(lvls)
                    if lnum >= len(cidx):
                        cidx.append(1)
                    else:
                        cidx[lnum] += 1
                    node_id = "#"+".".join(lvls)+"."+str(cidx[lnum]-1)+"#"
                    if node_id != current_node_id:
                        current_node_id = node_id
                        if current_node_id not in ret_penman:
                            ret_penman += penman_str[i+1:idx+1] + current_node_id + " / "
                            i = idx
                elif var in var2id:
                    ret_penman += penman_str[i+1:idx+1]
                    ret_penman += var2id[var]+var
                    i = j - 1
                            
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

        node_id = "#"+".".join(lvls)+"#"
        if node_id != current_node_id and len(lvls):
            current_node_id = node_id
            if current_node_id not in ret_penman:
                j = i+1
                while penman_str[j] == ' ':
                    j += 1
                ret_penman + penman_str[i+1:j]
                var = ""
                while penman_str[j] not in [' ','\n']:
                    var += penman_str[j]
                    j += 1
                var2id[var] = node_id
                i = j - 1
                ret_penman += node_id+var
        i += 1
        
    return ret_penman

def add_alns(penman, alns):
    ret_penman = ""
    in_id = False
    no_var = False
    i = 0
    node_id = ""
    while i < len(penman):
        c = penman[i]
        if c == '#':
            if not in_id:
                in_id = True
                node_id = ""
            else:
                in_id = False
                no_var = penman[i+1] == ' '
            i += 1
            continue
        if in_id:
            node_id += c
        else:
            ret_penman += c
        if c == '/' and penman[i-1] == ' ' and penman[i+1] == ' ':
            j = i + 2
            if penman[j] == '"':
                j += 1
                while penman[j] != '"':
                    j += 1
                j += 1
            else:
                while penman[j] not in [' ','\n',')']:
                    j += 1
            if no_var:
                ret_penman = ret_penman[:-2]
                ret_penman += penman[i+2:j]
            else:
                ret_penman += penman[i+1:j]
            i = j-1
            if node_id in alns:
                ret_penman += "~"+alns[node_id]
        i += 1
        
    return ret_penman
        
def argument_parser():

    parser = argparse.ArgumentParser(description='AMR alignment plotter')
    # input parameters
    parser.add_argument(
        "--in-amr",
        help="AMR file with JAMR alignments and metadata",
        type=str,
        required=True
    )
    parser.add_argument(
        "--out-amr",
        help="AMR output in isi format",
        type=str,
        required=False
    )
    args = parser.parse_args()
    return args

def main(args):

    fin = open(args.in_amr)
    fout = open(args.out_amr,'w')
    
    penman = ""
    tokens = ""
    metadata = ""
    alns = {}
    
    n = 0
    for line in fin:
        if line.strip() == "":
            penman = add_jamr_ids(penman)
            penman = add_alns(penman,alns)
            fout.write(metadata)
            fout.write(tokens+"\n")
            fout.write(penman+"\n")
            alns = {}
            penman = ""
            tokens = ""
            metadata = ""
            n += 1
        elif line[0] != '#':
            penman += line.rstrip()+"\n"
        elif "::node" in line:
            if len(line.strip().split('\t')) == 4:
                _,node_id,_,aln = line.strip().split('\t')
                alns[node_id] = aln.replace('-',',')
        elif "::tok" in line:
            tokens = line.strip()
        else:
            if "::edge" not in line and "::root" not in line and "::alignments" not in line:
                metadata += line

        
if __name__ == '__main__':

    # Argument handling
    main(argument_parser())
