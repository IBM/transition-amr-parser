
import os
import sys
import argparse
import string
import json

LINETYPEPREFIXES = {
   'snt': '# ::snt ',
   'tok': '# ::tok ',
   'alignments': '# ::alignments ',
   'node': '# ::node\t',
   'root': '# ::root\t',
   'edge': '# ::edge\t',
   'amrline': ('(', ' ', '\t')
}

BTAG = 'B-'
ITAG = 'I-'
OTAG = 'O'
STARTMENTION = '[unused1]'
ENDMENTION = '[unused2]'
AMRNODESTART = '('
AMRNODEEND = ')'
AMRTYPESPLITTER = '/'
AMRARGSPLITTER = ':'
AMRPREFIX = 'amr:'
NAMEDENTITYOP = ':name'
NAMETYPE = 'name'
NONNERTYPESWITHNAME = ['name', 'word', 'letter']
WIKIOP = ':wiki'
AMRTYPEJOINER = '/'
AMRTYPEPAD = ' '
ANCESTORJOINER = '#'
NESTRINGOP = ":op"
EXPLORETHRESHOLDLIST = [0.0, 0.75, 0.85, 0.9, 0.95, 0.97, 0.99, 0.995, 0.997, 0.999]
OUTPUTPRECISION = 5
ROOTTOKEN = '<root>'
MAXNEGFLOAT = 0.0 - sys.float_info.max

def printerr(s, **kwargs):
   print(s, file=sys.stderr, **kwargs)
   sys.stderr.flush()
   return

# enddef printerr()


def parseclargs():
   parser=argparse.ArgumentParser('retyper')
   parser.add_argument('-i', '--inputfile',        action='store', required=True, help='path to the AMR parse file to retype')
   parser.add_argument('-m', '--modeldirectory',   action='store', help='path to the directory with the BERT model files')
   parser.add_argument('-l', '--labelsfile',       action='store', help='path to the labels.txt file for the BERT model')
   parser.add_argument('-o', '--outputfile',       action='store', help='name of the output file for retyped AMR parses')
   parser.add_argument('-b', '--biooutputfile',    action='store', help='name of the output file for IOB-formatted sentences from the input AMRs')
   parser.add_argument('-j', '--jsonoutputfile',   action='store', help='name of the output file for json-formatted mention structures')
   parser.add_argument('-t', '--scorethreshold',   action='store', type=float, help='retyper score threshold (decimal number between 0.0 and 1.0); retyper will ignore retypes if their model score is below threshold)')
   parser.add_argument('-d', '--debug',            action='store_true', help='turn on verbose debug output')
   parser.add_argument('-n', '--noclobber',        action='store_true', help='do not overwrite output files if they already exist')
   parser.add_argument('-x', '--explorethreshold', action='store_true', help='iterate the retyper over a (fixed) list of threshold scores')
   parser.add_argument('--skipretyper',            action='store_true', help="don't load the retyper model; find NE spans, generate IOB, and reconstruct the AMR parse, but don't retype")
   parser.add_argument('--padnodetypes',           action='store_true', help="add space after the variable type operator ('/'); if not specified, retyper will determine dynamically whether padding is needed on AMR generation to match input AMRs")
   parser.add_argument('--retypeamrtypesonly',     action='store_true', help="don't retype if the original type is not among the fixed list of AMR NE types")
   parser.add_argument('--withdictionary',         action='store_true', help='use the dictionary-extended BERT architecture (required if the model was trained with dictionary features)')
   parser.add_argument('--outputscores',           action='store_true', help='add model scores to the output AMRs (will show as comments on retyped nodes)')
   parser.add_argument('--wikify',                 action='store_true', help='use BLINK to add :wiki links to the AMRs')
   parser.add_argument('--blinkmodels',            action='store', help='path to the BLINK model directory; if not specified, will attempt to link from cache only')
   parser.add_argument('--blinkcachepath',         action='store', help='path to the BLINK cache directory')
   parser.add_argument('--blinkcrossencoder',      action='store_true', help='use BLINK cross-encoder reranking')
   parser.add_argument('--blinkthreshold',         action='store', type=float, help='BLINK score threshold for accepting/rejecting links')
   args=parser.parse_args()

   if not args.skipretyper:
      if not args.modeldirectory:
         printerr('must specify a retyper model directory')
         sys.exit(1)
      if not args.labelsfile:
         printerr('must specify a retyper model labels file')
         sys.exit(1)
      if not os.path.exists(args.modeldirectory):
         printerr(f'model directory {args.modeldirectory} not found')
         sys.exit(2)
      if not os.path.exists(args.labelsfile):
         printerr(f'labels file {args.labelsfile} not found')
         sys.exit(2)

   if args.blinkmodels:
      if not os.path.exists(args.blinkmodels):
         printerr(f'BLINK model directory {args.modeldirectory} not found')
         sys.exit(2)
   if args.blinkcachepath:
      if not os.path.exists(args.blinkcachepath):
         printerr(f'BLINK cache directory {args.blinkcachepath} not found')
         sys.exit(3)
   if args.inputfile:
      if not os.path.exists(args.inputfile):
         printerr(f'input file {args.inputfile} not found')
         sys.exit(4)
   if args.outputfile:
      if args.noclobber and os.path.exists(args.outputfile):
         printerr(f'output file {args.outputfile} already exists')
         sys.exit(5)
   if args.biooutputfile:
      if args.noclobber and os.path.exists(args.biooutputfile):
         printerr(f'bio output file {args.biooutputfile} already exists')
         sys.exit(6)

   return args

# enddef parseclargs()

def printoutput(string, outputfile, **kwargs):
   if outputfile:
      print(string, file=outputfile, **kwargs)
   else:
      print(string, **kwargs)

# enddef printoutput

class ParamStore:
   def __init__(self):
      self.pstore = dict()
      self.pstore['typejoiner'] = AMRTYPEJOINER
      self.pstore['typepad'] = ''
      self.pstore['retypeamronly'] = False
      self.pstore['retypetypes'] = list()
      self.pstore['debug'] = False
      self.pstore['skipretyper'] = False
      self.pstore['withdictionary'] = False
      self.pstore['outputscores'] = False
      self.pstore['explorethreshold'] = False
      self.pstore['noclobber'] = False
      self.pstore['blinkfastmode'] = True
      self.pstore['scorethreshold'] = 0.0
      self.pstore['blinkthreshold'] = MAXNEGFLOAT

   def getparam(self, paramname):
      return self.pstore[paramname]

   def setparam(self, paramname, paramvalue):
      self.pstore[paramname] = paramvalue

   def tostring(self):
      s = '\n'.join([f'pstore[{p}] == {v}' for p, v in self.pstore.items()])
      return s

# endclass ParamStore

class Node:
   def __init__(self, aid, ntype, nstart, nend, nodeline):
      self.aid = aid
      self.ntype = ntype
      if nstart.isdigit():
         self.start = int(nstart)
      if nend.isdigit():
         self.end = int(nend)
      self.nodeline = nodeline
      self.parsepiece = -1
      self.retype = self.ntype
      self.entitylink = None
      self.namenode = None
      self.score = 1.0

   def setretype(self, retype):
      self.retype = retype

   def setscore(self, score):
      self.score = score

   def setparsepiece(self, pp):
      self.parsepiece = pp

   def setentitylink(self, el):
      self.entitylink = el

   def setnamenode(self, nn):
      self.namenode = nn

   def tostring(self):
      s = f'self.aid = {self.aid}\n'
      s += f'self.ntype = {self.ntype}\n'
      s += f'self.start = {self.start}\n'
      s += f'self.end = {self.end}\n'
      s += f'self.nodeline = {self.nodeline}\n'
      s += f'self.parsepiece = {self.parsepiece}\n'
      s += f'self.entitylink = {self.entitylink}\n'
      return s

# endclass Node

class Edge:
   def __init__(self, etype, arg1type, arg1id, arg2type, arg2id, edgelinenum):
      self.etype = etype
      self.arg1type = arg1type
      self.arg1id = arg1id
      self.arg2type = arg2type
      self.arg2id = arg2id
      self.edgelinenum = edgelinenum
      self.parsepiece = -1
      self.retype = self.etype
      self.score = 1.0

   def setretype(self, retype):
      self.retype = retype

   def setscore(self, score):
      self.score = score

   def setparsepiece(self, pp):
      self.parsepiece = pp

   def tostring(self):
      s = f'self.etype = {self.etype}\n'
      s += f'self.arg1type = {self.arg1type}\n'
      s += f'self.arg1id = {self.arg1id}\n'
      s += f'self.arg2type = {self.arg2type}\n'
      s += f'self.arg2id = {self.arg2id}\n'
      s += f'self.edgelinenum = {self.edgelinenum}\n'
      s += f'self.parsepiece = {self.parsepiece}\n'
      return s

# endclass Edge

class ParsePiece:
   def __init__(self, head, ptype, tail, tailparent, tailparentpath):
      self.head = head
      self.ptype = ptype
      self.tail = tail
      self.tailparent = tailparent
      self.tailparentpath = tailparentpath
      self.nestart = -1
      self.neend = -1

   def setnespan(self, nestart, neend):
      self.nestart = nestart
      self.neend = neend

   def tostring(self):
      s = f"self.head = '{self.head}'\n"
      s += f"self.ptype = '{self.ptype}'\n"
      s += f"self.tail = '{self.tail}'\n"
      s += f"self.tailparent = '{self.tailparent}'\n"
      s += f"self.tailparentpath = '{self.tailparentpath}'\n"
      s += f"self.nespan = '{self.nestart}-{self.neend}'\n"
      return s

# endclass ParsePiece

class Stack:
   def __init__(self):
      self.stack = []

   def push(self, item):
      self.stack.append(item)

   def pop(self):
      if not self.stack:
         return None
      else:
         topval = self.stack[-1]
         self.stack = self.stack[:-1]
         return topval

   def top(self):
      if not self.stack:
         return None
      else:
         return self.stack[-1]

   def dump(self):
      return ANCESTORJOINER.join([str(i) for i in self.stack])

# endclass Stack

def addnode(nodelinenum, line):
   NUMFIELDS = 4
   TYPEFIELD = 0
   IDFIELD = 1
   TYPEFIELD = 2
   SPANFIELD = 3

   fields = line.strip().split('\t')
   if len(fields) != NUMFIELDS:
      return None

   aid = fields[IDFIELD].strip()

   nodetype = fields[TYPEFIELD]
   if nodetype.startswith('"'):		# ignore nodes that don't refer to types
      return None			# (such as the string ops of names)

   start = 0
   end = 0
   nspan = fields[SPANFIELD]
   spanfields = nspan.split('-')
   if len(spanfields) != 2:
      return None
   else:
      start = spanfields[0].strip()
      end = spanfields[1].strip()

   return Node(aid, nodetype, start, end, nodelinenum)

def addedge(edgelinenum, line):
   NUMFIELDS = 6
   EDGEFIELD = 0
   TYPEFIELD = 1
   ARG1TYPEFIELD = 2
   ARG2TYPEFIELD = 3
   ARG1IDFIELD = 4
   ARG2IDFIELD = 5

   fields = line.strip().split('\t')
   if len(fields) != NUMFIELDS:
      return None

   edgetype = fields[TYPEFIELD].strip()
   arg1type = fields[ARG1TYPEFIELD].strip()
   arg2type = fields[ARG2TYPEFIELD].strip()
   arg1id = fields[ARG1IDFIELD].strip()
   arg2id = fields[ARG2IDFIELD].strip()

   return Edge(edgetype, arg1type, arg1id, arg2type, arg2id, edgelinenum)

def protectedcount(s, c):
   DOUBLEQUOTE = '"'

   if not c in s:
      return 0

   count = 0
   isinsidequote = False

   for ch in s:
      if ch == DOUBLEQUOTE:
         isinsidequote = not isinsidequote
      if ch == c:
         if not isinsidequote:
            count += 1

   return count

# end def protectedcount

def protectedsplit(string, splitter):
# don't split string on splitter if splitter is inside a double-quoted string
   DOUBLEQUOTE = '"'

   if not splitter in string:
      return [string]

   if splitter == DOUBLEQUOTE:
      return string.split(splitter)

   strlist = list()
   currstr = ''
   insidequote = False
   for ch in string:
      if ch == DOUBLEQUOTE:
         if insidequote:
            insidequote = False
         else:
            insidequote = True

      if ch == splitter:
         if insidequote:
            currstr += ch
         else:
            strlist.append(currstr)
            currstr = ''
      else:
         currstr += ch

   strlist.append(currstr)

   return strlist

# end def protectedsplit

def getparsepieces(amrparse, params):
   parsepieces = dict()
   tmppp = protectedsplit(amrparse, AMRNODESTART)[1:]	# starts with '(', so [0] will be ''

   if params.getparam('debug'):
      printerr(amrparse)
      printerr(tmppp)

   parents = Stack()

   j = 0
   prevtail = ''
   for n in tmppp:
      tmppt = protectedsplit(n, AMRTYPESPLITTER)

      if params.getparam('debug'):
         printerr(tmppt)

      if len(tmppt) > 1:
         for _ in range(protectedcount(prevtail, AMRNODEEND)):
            ptailparent = parents.pop()

         phead = tmppt[0]
         parents.push(phead)
         tmptail = tmppt[1].split()

         if params.getparam('debug'):
            printerr(tmptail)

         pptype = tmptail[0]
         tmptail = tmppt[1].split(pptype)
         params.setparam('typepad', tmptail[0])
         pptail = pptype.join(tmptail[1:])

         for _ in range(protectedcount(pptype, AMRNODEEND)):
            ptailparent = parents.pop()

         ptailparent = parents.top()
         ptailparentpath = parents.dump()

         parsepieces[j] = ParsePiece(head=phead, ptype=pptype, tail=pptail, tailparent=ptailparent, tailparentpath=ptailparentpath)
         j += 1
         prevtail = pptail

   return parsepieces

# end def getparsepieces()

def getparentfrompath(pathstr):
   if not pathstr:
      return None

   ancestors = pathstr.split(ANCESTORJOINER)
   if len(ancestors) < 2:
      return None
   else:
      return ancestors[-2]

def getwikiid(pstr):
   parts = pstr.split(WIKIOP)
   if len(parts) <= 1:
      return None

   tstr = parts[1].strip()
   if not tstr.startswith('"'):
      return None

   parts = tstr.split('"')

   if len(parts) <= 1:
      return None
   
   wid = parts[1].strip()

   return wid

def getnespan(ppstr, edgelist, nodelist):
   if NESTRINGOP not in ppstr:
      return None

# really long mention spans cause BLINK to crash
# (there's a limit, but it's buried in the config
# file of the BLINK model, and BLINK just allows
# a fatal crash if the limit is exceeded)
   if NESTRINGOP+'10' in ppstr:
      return None

   tokstrs = list()
   pstrs = ppstr.split()
   strisop = False
   for pstr in pstrs:
      if isop(pstr):
         strisop = True
         continue
      if strisop:
         if pstr[0] == '"':
            tmpstrs = pstr.split('"')
            tokstr = '"'+tmpstrs[1]+'"'
            tokstrs.append(tokstr)
            strisop = False

   if not tokstrs:
      return None

   nodeids = dict()
   for tokstr in tokstrs:
      nodeids[tokstr] = list()

   # don't want to use sets because I want to maintain the order
   for edge in edgelist:
      if isop(edge.arg1type) and edge.arg2type in tokstrs:
         if edge.arg1id not in nodeids[edge.arg2type]:
            nodeids[edge.arg2type].append(edge.arg1id)

   # find a node id in the first tok's node ids 
   # that's among the node ids for all the other tokstrs
   candidatenodes = list()
   for t1node in nodeids[tokstrs[0]]:
      t1nodeok = True
      for tok in tokstrs[1:]:
         if t1node not in nodeids[tok]:
            t1nodeok = False
            break

      if t1nodeok:
         candidatenodes.append(t1node)

   if not candidatenodes:
      return None

   spannode = None
   for nid, node in nodelist.items():
      if node.aid in candidatenodes and node.parsepiece == -1:
         if len(candidatenodes) > 1:                        # if more than one node has the same string
            if (node.end - node.start) == len(tokstrs):     # pick the one with the span with the "right" length
               spannode = nid                               # (will not work when the number of tokens != number of :ops)
               break
         else:
            spannode = nid
            break

   return spannode

def isop(opstr):
   if not opstr:
      return False

   # the first char of NESTRINGOP (':') may be stripped (e.g., in an edge arg)
   if opstr[0] != NESTRINGOP[0]:
      opstr = NESTRINGOP[0] + opstr

   if not opstr.startswith(NESTRINGOP):
      return False

   intpart = opstr.split(NESTRINGOP, 1)[1]
   if not intpart:
      return False

   for ch in intpart:
      if not ch.isdigit():
         return False

   return True

def expandiob(ioblist):
   iobexp = list()
   insidemention = False
   for tok, tag in ioblist:
      if tag.startswith(BTAG):		# starting a new mention
         if insidemention:		      # but already in one, so close it
            iobexp.append((ENDMENTION, OTAG))
         else:
            insidemention = True
         iobexp.append((STARTMENTION, OTAG))
      elif tag.startswith(OTAG):
         if insidemention:
            iobexp.append((ENDMENTION, OTAG))
            insidemention = False

      iobexp.append((tok, tag))

   return iobexp

# end def expandiob()

def stripiob(ioblist):
   STRIPTAGPREFIX = '[unused'
   return [(tok, tag) for (tok, tag) in ioblist if not tok.startswith(STRIPTAGPREFIX)]

# end def stripiob()

def retypeiob(ioblist, retyper, params):

   retypediob = ioblist

   if retyper:
      tokens_only = [tok for (tok, _) in ioblist]
      retypediob = retyper.classify_input(tokens_only, return_probabilities=True)

   return retypediob

# end def retypeiob()

def dumpbiofile(ioblist, biofile):
   if biofile:
      for tok, iob in ioblist:
         if isinstance(iob, tuple):
            iobtag, score = iob
            print(f'{tok} {iobtag} ## {score}', file=biofile)
         else:
            print(f'{tok} {iob}', file=biofile)

   print('', file=biofile)

# end def dumpbiofile()

def dumpjsonfile(strippedioblist, blinker, jsonfile, jcount, threshold=MAXNEGFLOAT):
   nummentions = 0
   mention = list()
   linkedioblist = list()
   tag = ''
   toklist = [tok for tok, _ in strippedioblist]
   bidx = 0
   prevbidx = 0
   biobtag = ''
   bscore = None
   prevbiobtag = ''
   btok = ''
   for i, (tok, iob) in enumerate(strippedioblist):
      score = None
      if isinstance(iob, tuple):
         iobtag, score = iob
      else:
         iobtag = iob
      linkedioblist.append((tok, iob))
      elink = ''
      if iobtag.startswith(BTAG):
         if mention:
            elink = dumpjsonmention(toklist[:(i-len(mention))], mention, tag, toklist[i:], jcount, nummentions, blinker, jsonfile, threshold)
            nummentions += 1
            if elink:
               newtag = biobtag
               if bscore:
                  newtag = (biobtag, bscore)
               linkedioblist[bidx] = (btok, newtag+' :wiki "'+elink+'"')
         bidx = i
         biobtag = iobtag
         bscore = score
         btok = tok
         mention = [tok]
         tag = iobtag
      elif iobtag.startswith(ITAG):
         mention.append(tok)
         if not tag:
            tag = iobtag
      else:
         if mention:
            elink = dumpjsonmention(toklist[:(i-len(mention))], mention, tag, toklist[i:], jcount, nummentions, blinker, jsonfile, threshold)
            nummentions += 1
            if elink:
               newtag = biobtag
               if bscore:
                  newtag = (biobtag, bscore)
               linkedioblist[bidx] = (btok, newtag+' :wiki "'+elink+'"')
         mention = list()
         tag = ''

   if mention:
      elink = dumpjsonmention(toklist[:(i-len(mention))], mention, tag, toklist[i:], jcount, nummentions, blinker, jsonfile, threshold)
      if elink:
         newtag = biobtag
         if bscore:
            newtag = (biobtag, bscore)
         linkedioblist[bidx] = (btok, newtag+' :wiki "'+elink+'"')
      nummentions += 1

   return nummentions, linkedioblist

# end def dumpjsonfile()

def dumpjsonmention(leftcontext, mention, tag, rightcontext, sentcount, mentcount, blinker, jsonfile, threshold=MAXNEGFLOAT):
   wptitle = ''
   resurl = ''
   mstru = dict()
   mstru['id'] = f'{sentcount}.{mentcount}'
   mstru['label'] = 'unknown'
   mstru['label_id'] = -1
   mstru['context_left'] = ' '.join(leftcontext).lower()
   mstru['mention'] = ' '.join(mention).lower()
   rcstr = ' '.join(rightcontext).lower()
   if rcstr.endswith(ROOTTOKEN):
      peel = 0 - len(ROOTTOKEN) - 1    # extra 1 for the space added by join
      rcstr = rcstr[:peel]
   mstru['context_right'] = rcstr

   elink = ''
   tagelems = tag.split()
   if len(tagelems) > 1:
      elink = tagelems[1]

   if elink:
      mstru['Wikipedia_URL'] = 'en.wikipedia.org/wiki/'+elink

# Uncomment temporarily if you want to force context-free spans into the cache
#   mstru['context_left'] = ''
#   mstru['context_right'] = ''

   predictions = list()
   if blinker:
      predictions, scores = blinker.runblink([mstru])
      if not predictions: 	            # back off to no-context linking (should only happen in cache-only mode)
         mstru['context_left'] = ''
         mstru['context_right'] = ''
         predictions, scores = blinker.runblink([mstru])

   topscore = threshold - 1.0
   if len(predictions) > 0:
      # BLINK returns wptitles, but amr gold wiki links are stripped uris
      wptitle = predictions[0][0].replace(' ', '_')
      topscore = scores[0][0]
      
      wpid, resurl = blinker.getwpinfofortitle(wptitle)
      mstru['Wikipedia_URL'] = resurl
      mstru['label'] = wpid
      mstru['label_id'] = wpid
      
   if jsonfile:
      s = json.dumps(mstru)
      print(s, file=jsonfile)
      jsonfile.flush()

   if topscore < threshold:
      return '-'
   else:
      if resurl:
         return resurl.split('/')[-1]
      else:
         return wptitle

# end def dumpjsonmention()

def processparse(amrparse, toks, nodelist, edgelist, preamble, retyper, blinker, ofil, bfil, jfil, jcount, params):
   numretypes = 0
   numretypesdiff = 0

   parsepieces = getparsepieces(amrparse, params)
   if not parsepieces:
      return 0, 0, 0

   if params.getparam('debug'):
      for pid, pp in parsepieces.items():
         printerr(f'pp[{pid}] =\n{pp.tostring()}')

   parsepiecesneedingretyping = list()
   parentidx = -1
   for pid, pp in parsepieces.items():
      if pp.ptype == NAMETYPE and NESTRINGOP in pp.tail:
         namespannode = getnespan(pp.tail, edgelist, nodelist)
         if not namespannode:
            printerr(f'ERROR: No node with matching span found for\n{pp.head}{pp.tail}')
            continue
         ppnestart = nodelist[namespannode].start
         ppneend = nodelist[namespannode].end
         nodelist[namespannode].setparsepiece(pid)
         parent = getparentfrompath(pp.tailparentpath)
         for idx, ppiece in parsepieces.items():
            if ppiece.head == parent:
               if ppiece.ptype not in NONNERTYPESWITHNAME:
                  parentidx = idx
               break
         if parentidx >= 0:
            parsepieces[parentidx].setnespan(ppnestart, ppneend)
            parsepiecesneedingretyping.append(parentidx)

   # parsepieces are in the order of the edges (which I'm using to order the nodes needing retyping)
   parsepiecesneedingretyping = sorted(set(parsepiecesneedingretyping))

   if not parsepiecesneedingretyping:
      newpreamble = ''.join(preamble)
      printoutput(newpreamble, ofil, end='')
      printoutput(amrparse, ofil)
      return 0, 0, 0

   nodeidsneedingretyping = [(edge.arg1id, edge.arg2id) for edge in edgelist if edge.arg1type == NAMETYPE and edge.arg2type == NAMETYPE]

# the list comprehension would put the node indexes in the order of nodelist, 
# but I need them to be in the order they appear in the edgelist; so sad
#   nodesneedingretyping = [nodeindex for nodeindex, node in nodelist.items() if node.aid in nodeidsneedingretyping]
   nodesneedingretyping = list()
   for (nodeid, nameid) in nodeidsneedingretyping:
      for nodeindex, node in nodelist.items():
         if node.aid == nodeid:
            if node.ntype not in NONNERTYPESWITHNAME:
               nodesneedingretyping.append(nodeindex)
               for nnidx, nnode in nodelist.items():
                  if nnode.aid == nameid:
                     nodelist[nodeindex].setnamenode(nnidx)

   if len(nodesneedingretyping) != len(parsepiecesneedingretyping):
      nodesneedingretyping = list()
      parsepiecesneedingretyping = list()

   for nid in nodesneedingretyping:
      nnid = nodelist[nid].namenode
      namestart = nodelist[nnid].start
      nameend = nodelist[nnid].end
      for ppid in parsepiecesneedingretyping:
         if parsepieces[ppid].nestart == namestart and parsepieces[ppid].neend == nameend:
            nodelist[nid].setparsepiece(ppid)
            ptail = parsepieces[ppid].tail
            if WIKIOP in ptail:
               wikiid = getwikiid(ptail)
               nodelist[nid].setentitylink(wikiid)

   if params.getparam('debug'):
      for nid, node in nodelist.items():
         printerr(f'nodelist[{nid}] =\n{node.tostring()}')
      for i, edge in enumerate(edgelist):
         printerr(f'edgelist[{i}] = \n{edge.tostring()}')

   if params.getparam('debug'):
      printerr(f'parsepiecesneedingretyping = {parsepiecesneedingretyping}')
      printerr(f'nodesneedingretyping = {nodesneedingretyping}')

   iob = [(tok, OTAG) for tok in toks]
   for nid in nodesneedingretyping:
      n = nodelist[nid]
      if params.getparam('retypeamronly'):
         if not n.ntype in params.getparam('retypetypes'):
            continue
      ntyp = AMRPREFIX+n.ntype
      nent = ''
      if n.entitylink:
         nent += '\t'+n.entitylink
      if n.namenode:
         i = nodelist[n.namenode].start
         j = nodelist[n.namenode].end
      else:
         i = n.start
         j = n.end
      tok, _ = iob[i]
      iob[i] = (tok, BTAG+ntyp+nent)
      i += 1
      while i < j:
         tok, _ = iob[i]
         iob[i] = (tok, ITAG+ntyp+nent)
         i += 1

   if params.getparam('debug'):
      printerr(iob)

   iobexp = expandiob(iob)

   if params.getparam('debug'):
      printerr(toks)
      printerr(iobexp)

   retypediob = retypeiob(iobexp, retyper, params)

   if bfil:
      dumpbiofile(retypediob, bfil)

   jadd = 0
   linkedioblist = stripiob(retypediob)
   jadd, linkedioblist = dumpjsonfile(linkedioblist, blinker, jfil, jcount, threshold=params.getparam('blinkthreshold'))

   strippedretypediob = linkedioblist

   if len(strippedretypediob) != len(iob):
      printerr(f"retyped iob list doesn't align\n{iob}\n{strippedretypediob}")

   if params.getparam('debug'):
      printerr(strippedretypediob)

   newpreamble = list()

   newpreamble += [pl for pl in preamble if not (pl.startswith(LINETYPEPREFIXES['node']) or pl.startswith(LINETYPEPREFIXES['edge']) or pl.startswith(LINETYPEPREFIXES['root']))]
   nodelines = [pl for pl in preamble if pl.startswith(LINETYPEPREFIXES['node'])]
   rootline = [pl for pl in preamble if pl.startswith(LINETYPEPREFIXES['root'])][0]
   rootisnodenum = -1
   for i in range(len(nodelines)):
      if nodelines[i].replace(LINETYPEPREFIXES['node'], LINETYPEPREFIXES['root']) == rootline:
         rootisnodenum = i

   for nid in nodesneedingretyping:
      n = nodelist[nid]
      origtype = n.ntype
      start = 0
      if n.namenode:
         start = nodelist[n.namenode].start
      else:
         start = n.start
      retypedtoken, retypedtype = strippedretypediob[start]
      score = 1.0
      if isinstance(retypedtype, tuple):
         retypedbtag, score = retypedtype
      else:
         retypedbtag = retypedtype

      if retypedbtag.startswith(BTAG+AMRPREFIX):
         retypedbtag = retypedbtag[len(BTAG+AMRPREFIX):]
      if retypedbtag.startswith(ITAG+AMRPREFIX):
         retypedbtag = retypedbtag[len(ITAG+AMRPREFIX):]

      retype = origtype
      if score >= params.getparam('scorethreshold'):
         retype = retypedbtag.split('\t')[0]          # remove \t<entitylink> if it exists
         n.setscore(score)

      n.setretype(retype)
      retypenolink = retype.split(' :wiki')[0]
      nodelines[n.nodeline] = nodelines[n.nodeline].replace(origtype, retypenolink)
      if params.getparam('outputscores'):
         nodelines[n.nodeline] = nodelines[n.nodeline].rstrip() + f'\t## {n.score:.{OUTPUTPRECISION}f}\n'
      numretypes += 1
      if retype != origtype:
         numretypesdiff += 1

      ppid = n.parsepiece
      if ppid >= 0:
         if params.getparam('debug'):
            printerr(f'pp[{ppid}] =\n{parsepieces[ppid].tostring()}')
         parsepieces[ppid].ptype = retype
         if params.getparam('debug'):
            printerr(f'pp[{ppid}] =\n{parsepieces[ppid].tostring()}')

   newpreamble += nodelines

   if rootisnodenum >= 0:
      rootline = nodelines[rootisnodenum].replace(LINETYPEPREFIXES['node'], LINETYPEPREFIXES['root'])

   newpreamble.append(rootline)

   NUMFIELDS = 6
   EDGEFIELD = 0
   EDGENAME = 1
   ARG1TYPE = 2
   ARG2TYPE = 3
   ARG1NODE = 4
   ARG2NODE = 5
   edgelines = [pl for pl in preamble if pl.startswith(LINETYPEPREFIXES['edge'])]
   newedgelines = list()
   for edge in edgelines:
      fields = edge.strip().split('\t')
      if len(fields) != NUMFIELDS:
         newedgelines.append(edge)
      else:
         epref = fields[EDGEFIELD].strip()
         ename = fields[EDGENAME].strip()
         ea1type = fields[ARG1TYPE].strip()
         ea1node = fields[ARG1NODE].strip()
         ea2type = fields[ARG2TYPE].strip()
         ea2node = fields[ARG2NODE].strip()
         for nid in nodesneedingretyping:
            n = nodelist[nid]
            if ea1node == n.aid and ename == n.ntype:
               if n.retype:
                  ename = n.retype.split(' :wiki')[0]
            if ea2node == n.aid and ea2type == n.ntype:
               if n.retype:
                  ea2type = n.retype.split(' :wiki')[0]
         newedgelines.append('\t'.join([epref, ename, ea1type, ea2type, ea1node, ea2node+'\n']))

   newpreamble += newedgelines

   for pl in newpreamble:
      printoutput(pl, ofil, end='')

   newparsestrings = list()

   for _, pp in parsepieces.items():
      newparsestrings.append(f'{AMRNODESTART}{pp.head}{params.getparam("typejoiner")}{params.getparam("typepad")}{pp.ptype}{pp.tail}')
   for nps in newparsestrings:
      printoutput(nps, ofil, end='')
   printoutput('', ofil)

   if ofil:
      ofil.flush()

   return numretypes, numretypesdiff, jadd

# end def processparse



def main():
   params = ParamStore()
   clargs = parseclargs()

   if clargs.padnodetypes:
      params.setparam('typepad', AMRTYPEPAD)
   if clargs.debug:
      params.setparam('debug', True)
   if clargs.skipretyper:
      params.setparam('skipretyper', True)
   if clargs.wikify:
      params.setparam('wikify', True)
   if clargs.withdictionary:
      params.setparam('withdictionary', True)
   if clargs.outputscores:
      params.setparam('outputscores', True)
   if clargs.explorethreshold:
      params.setparam('explorethreshold', True)
   if clargs.noclobber:
      params.setparam('noclobber', True)
   if clargs.blinkcrossencoder:
      params.setparam('blinkfastmode', False)
   if clargs.scorethreshold:
      params.setparam('scorethreshold', clargs.scorethreshold)
   if isinstance(clargs.blinkthreshold, float):
      params.setparam('blinkthreshold', clargs.blinkthreshold)

   if clargs.retypeamrtypesonly:
      params.setparam('retypeamronly', True)
      lfile = open(clargs.labelsfile, 'r')
      retypetypes = list()
      for l in lfile:
         atyp = l.strip()
         if atyp.startswith(BTAG):
            atyp = atyp[len(BTAG):]
            if atyp.startswith(AMRPREFIX):
               atyp = atyp[len(AMRPREFIX):]
            retypetypes.append(atyp)
      params.setparam('retypetypes', retypetypes)

   if params.getparam('debug'):
      printerr(params.tostring())

   if params.getparam('skipretyper'):
      retyper = None
   else:
      printerr('Loading model...')
      if params.getparam('withdictionary'):
         from etyperwithdictionary import EntityTyper
      else:
         from etyper import EntityTyper

      retyper = EntityTyper(pathtoclasslabelsfile = clargs.labelsfile, pathtomodeldirectory = clargs.modeldirectory)

   blinker = None
   if clargs.wikify:
      from Blinker import Blinker
      blinker = Blinker(pathtomodeldirectory=clargs.blinkmodels, pathtocachedirectory=clargs.blinkcachepath, fastmode=params.getparam('blinkfastmode'), wikititleonly=True)

   threshlist = [params.getparam('scorethreshold')]

   if params.getparam('explorethreshold'):
      threshlist = EXPLORETHRESHOLDLIST

   for thresh in threshlist: 

      ifil = None
      ofil = None
      bfil = None
      jfil = None

      if clargs.inputfile:
         ifil = open(clargs.inputfile, 'r')
      if clargs.biooutputfile:
         bfil = open(clargs.biooutputfile, 'w')
      if clargs.jsonoutputfile:
         jfil = open(clargs.jsonoutputfile, 'w')

      if clargs.outputfile:
         if params.getparam('explorethreshold') or params.getparam('scorethreshold'):
            currofilname = f'{clargs.outputfile}_{thresh}.amr'
         else:
            currofilname = clargs.outputfile

         ofil = open(currofilname, 'w')

      params.setparam('scorethreshold', thresh)

      numparses = 0
      numretypes = 0
      numretypesdiff = 0
      jcount = 0

      if retyper:
         printerr(f'Retyping with threshold = {thresh}...')

      current_input = ''
      needsretyping = False
      current_snt = ''
      current_preamble = list()
      current_toks = list()
      nodelist = dict()
      edgelist = list()
      current_amrparse = ''
      i = 0
      nodelinenum = 0
      edgelinenum = 0
      processing = False

      for line in ifil:
         current_input += line

         if line == '\n':		   # must have blank line between amr parses
            if not processing:	# skip extra blank lines if already dumped previous parse
               continue

            numparses += 1

            if needsretyping:
               numretypesadd, numretypesdiffadd, numjadd = processparse(current_amrparse, current_toks, nodelist, edgelist, current_preamble, retyper, blinker, ofil, bfil, jfil, jcount, params)
               numretypes += numretypesadd
               numretypesdiff += numretypesdiffadd
               jcount += numjadd
            else:
               printoutput(current_input, ofil, end='')

            current_input = ''
            needsretyping = False
            current_snt = ''
            current_preamble = list()
            current_toks = list()
            nodelist = dict()
            edgelist = list()
            current_amrparse = ''
            i = 0
            nodelinenum = 0
            edgelinenum = 0
            processing = False
         elif line.startswith(LINETYPEPREFIXES['amrline']):
            processing = True
            if NAMEDENTITYOP in line:
               needsretyping = True
            current_amrparse += line
         else:
            current_preamble.append(line)
            processing = True
            if line.startswith(LINETYPEPREFIXES['snt']):
               current_snt = line[len(LINETYPEPREFIXES['snt']):]
            elif line.startswith(LINETYPEPREFIXES['tok']):
               current_toks = line[len(LINETYPEPREFIXES['tok']):].split()
            elif line.startswith(LINETYPEPREFIXES['node']):
               tmpnode = addnode(nodelinenum, line)
               nodelinenum += 1
               if tmpnode:
                  nodelist[i] = tmpnode
                  i += 1
            elif line.startswith(LINETYPEPREFIXES['edge']):
               tmpedge = addedge(edgelinenum, line)
               edgelinenum += 1
               if tmpedge:
                  edgelist.append(tmpedge)

      ifil.close()

      if processing:		# there's a parse left over (no blank line at end of file)
         if needsretyping:
            numretypesadd, numretypesdiffadd, numjadd = processparse(current_amrparse, current_toks, nodelist, edgelist, current_preamble, retyper, blinker, ofil, bfil, jfil, jcount, params)
            numretypes += numretypesadd
            numretypesdiff += numretypesdiffadd
            jcount += numjadd
         else:
            printoutput(current_input, ofil)

      printerr(f'{numparses} AMR parses processed')
      printerr(f'{numretypesdiff} / {numretypes} Named Entities processed')

      if bfil:
         bfil.close()
      if jfil:
         jfil.close()
      if ofil:
         ofil.close()
      if blinker:
         blinker.done()

# enddef main


if __name__ == '__main__':
   main()

