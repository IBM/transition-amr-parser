
import sys
import os
import glob
import argparse
import torch.cuda
import importlib.util
import logging
import string
import json
import numpy
#import LDConvert
import random
import hashlib

#BLINKMODULE = 'blink.main_dense'
#BLINKMODULEPATH = '/dccstor/kjbarker1/dev/BLINK/blink/main_dense.py'
#LDCONVERTMODULE = 'LDConverter'
#LDCONVERTPATH = '/u/kjbarker/dev/util/LDConvert.py'
#WPAPI = 'https://en.wikipedia.org/w/api.php'
MENTIONSTART = '[unused1]'
MENTIONEND = '[unused2]'
CONTEXTLEFTKEY = 'context_left'
CONTEXTRIGHTKEY = 'context_right'
MENTIONKEY = 'mention'
CACHEFILEPATTERN = 'linkcache_*.json'
CACHEIDKEY = 'link_cache_id'
MODEKEY = 'blink_mode'
CACHEDATAKEY = 'cache_data'
PREDICTIONSKEY = 'predictions'
SCORESKEY = 'scores'
FASTTAG = 'fast'
SLOWTAG = 'slow'
DELCHARS = string.punctuation + ' '

class LinkCache:

   def __init__(self, pathtocachedirectory, fastmode=True):
      if not pathtocachedirectory.endswith('/'):
         pathtocachedirectory += '/'
      self.pathtocachedirectory = pathtocachedirectory
      self.fastmode = fastmode
      self.linkcache = dict()
      jsonfileslist = glob.glob(f'{pathtocachedirectory}{CACHEFILEPATTERN}')
      for fname in jsonfileslist:
         jfil = open(fname, 'r')
         tmpcache = json.load(jfil)
         jfil.close()
         if not CACHEIDKEY in tmpcache:
            continue
         self.linkcache.update(tmpcache[CACHEDATAKEY])

      self.localcache = dict()

   def lookup(self, left, mention, right):
      gkl = list()
      gkslow = f'{SLOWTAG}#{self.generalize_key(left, mention, right)}'
      gkfast = f'{FASTTAG}#{self.generalize_key(left, mention, right)}'
      gkslowhashed = self.hash_key(gkslow)
      gkfasthashed = self.hash_key(gkfast)
      gkl.append(gkslowhashed)      # allow cached results from slow mode even if caller is in fast mode
      if self.fastmode:       # but don't allow fast mode results if caller is in slow mode
         gkl.append(gkfasthashed)
      for genkey in gkl:
         if self.localcache:
            if genkey in self.localcache[CACHEDATAKEY]:
               return self.localcache[CACHEDATAKEY][genkey][PREDICTIONSKEY], self.localcache[CACHEDATAKEY][genkey][SCORESKEY]
         if genkey in self.linkcache:
            return self.linkcache[genkey][PREDICTIONSKEY], self.linkcache[genkey][SCORESKEY]
      return None, None

   def addtocache(self, left, mention, right, predictions, scores, mode=None):
      gkmodetag = mode if mode else (FASTTAG if self.fastmode else SLOWTAG)
      gk = f'{gkmodetag}#{self.generalize_key(left, mention, right)}'
      gkhashed = self.hash_key(gk)
      linkdata = {PREDICTIONSKEY: predictions, SCORESKEY: scores}
      if CACHEDATAKEY not in self.localcache:
         self.localcache[CACHEDATAKEY] = dict()
      self.localcache[CACHEDATAKEY][gkhashed] = linkdata

   def hash_key(self, cachekey):
      return hashlib.sha3_256(cachekey.encode()).hexdigest()

   def generalize_key(self, left, mention, right):
      leftstr = ''.join([ch for ch in left if ch not in DELCHARS])
      mentionstr = ''.join([ch for ch in mention if ch not in DELCHARS])
      rightstr = ''.join([ch for ch in right if ch not in DELCHARS])
      retstr = f'{leftstr.lower()}#{mentionstr.lower()}#{rightstr.lower()}'
      return retstr

#   def flush(self):
#      if self.localcache and self.localcachename:
#         cfil = open(self.localcachename, 'w')
#         json.dump(self.localcache, cfil)
#         cfil.close()

   def done(self):
      if self.localcache:
         jsonfileslist = glob.glob(f'{self.pathtocachedirectory}/{CACHEFILEPATTERN}')
         i = len(jsonfileslist)
         rn = random.randint(0, 999)
         self.uid = f'{i:09d}_{rn:03d}'
         self.localcachename = f'{self.pathtocachedirectory}/linkcache_{self.uid}.json'
         while os.path.exists(self.localcachename):
            i += 1
            rn = random.randint(0, 999)
            self.uid = f'{i:09d}_{rn:03d}'
            self.localcachename = f'{self.pathtocachedirectory}/linkcache_{self.uid}.json'
         self.localcache[CACHEIDKEY] = self.uid
         self.localcache[MODEKEY] = FASTTAG if self.fastmode else SLOWTAG
#         self.flush()
         cfil = open(self.localcachename, 'w')
         json.dump(self.localcache, cfil)
         cfil.close()

# end class LinkCache

CACHEPATHENVVAR = 'BLINKERCACHEPATH'

class Blinker:
   def __init__(self, pathtocachedirectory=None, pathtomodeldirectory=None, fastmode=True, wikititleonly=False):
#      spec = importlib.util.spec_from_file_location(LDCONVERTMODULE, LDCONVERTPATH)
#      LDConvert = importlib.util.module_from_spec(spec)
#      spec.loader.exec_module(LDConvert)

      if pathtocachedirectory:
         self.cachedir = pathtocachedirectory
      else:
         if CACHEPATHENVVAR in os.environ:
            self.cachedir = os.environ[CACHEPATHENVVAR]
         else:
            self.cachedir = '.'

      if not self.cachedir.endswith('/'):
         self.cachedir += '/'

      self.cacheonlymode = False
      if pathtomodeldirectory:
         import run_blink
         self.run_blink = run_blink
         if not pathtomodeldirectory.endswith('/'):
            pathtomodeldirectory += '/'
      else:
         self.cacheonlymode = True
         print('Not using BLINK (Blinker in cache-only mode)', file=sys.stderr)

#      link_cache_directory = pathtomodeldirectory+"../linkcache"
      self.linkcache = LinkCache(self.cachedir, fastmode=fastmode)

      self.ldc = None
      if not wikititleonly:
         import LDConvert
         self.ldc = LDConvert.LDConverter()

      self.config = {
         "interactive": False,
         "fast": fastmode, # set this to True if no gpu available or if no faiss available
         "top_k": 16,
         "output_path": "logs/" # logging directory
      }
      if not self.cacheonlymode:
         self.config['biencoder_model'] = pathtomodeldirectory+'biencoder_wiki_large.bin'
         self.config["biencoder_config"] = pathtomodeldirectory+"biencoder_wiki_large.json"
         self.config["entity_catalogue"] = pathtomodeldirectory+"entity.jsonl"
         self.config["entity_encoding"] = pathtomodeldirectory+"all_entities_large.t7"
         self.config["crossencoder_model"] = pathtomodeldirectory+"crossencoder_wiki_large.bin"
         self.config["crossencoder_config"] = pathtomodeldirectory+"crossencoder_wiki_large.json"

      if not torch.cuda.is_available():   # override fastmode param if no gpu available
         self.config['fast'] = True
      if not self.cacheonlymode:
         if self.config['fast']:
            print('Using BLINK in fast mode (no cross-encoder reranking)', file=sys.stderr)
         else:
            print('Using BLINK in slow mode (bi-encoder candidate gen + cross-encoder reranking)', file=sys.stderr)

      args = argparse.Namespace(**self.config)
      self.logger = logging.getLogger()
      self.models = None      # load on demand (when mention+context not in cache)

   # enddef __init__

   def runblink(self, mentiondata):
#      self.config['test_entities'] = mentiondata
      args = argparse.Namespace(**self.config)

      results = dict()
      mentions2blink = list()
      for i, mstru in enumerate(mentiondata):
         (preds, scores) = self.linkcache.lookup(mstru[CONTEXTLEFTKEY], mstru[MENTIONKEY], mstru[CONTEXTRIGHTKEY])
         if all((preds, scores)):
            results[i] = (preds, scores)
         else:
            results[i] = None
            mentions2blink.append(mstru)

      predictions = list()
      scores = list()
      if mentions2blink and not self.cacheonlymode:
         if not self.models:
            print('Loading BLINK models...', file=sys.stderr)
            self.models = self.run_blink.load_models(args, logger=self.logger)
#            print('done.', file=sys.stderr)

         _, _, _, _, _, predictions, numpyscores = self.run_blink.run(args, self.logger, *self.models, test_data=mentions2blink)
         scores = list()
         for scoreslist in numpyscores:
            sl = list()
            for score in scoreslist:
               s = score
               if isinstance(score, numpy.float32):	# stupid numpy/json hack
                  s = score.item()
               sl.append(s)
            scores.append(sl)

         for i, mstru in enumerate(mentions2blink):
            s = scores[i]
#            if not isinstance(scores[i], list):
#               s = scores[i].tolist()
            self.linkcache.addtocache(mstru[CONTEXTLEFTKEY], mstru[MENTIONKEY], mstru[CONTEXTRIGHTKEY], predictions[i], s)

      j = 0
      retpreds = list()
      retscores = list()
      for i, mstru in enumerate(mentiondata):
         if results[i]:
            retpreds.append(results[i][0])
            retscores.append(results[i][1])
         else:
            currpreds = ['']
            currscores = [0.0]
            if predictions and scores:
               if predictions[j] and scores[j]:
                  currpreds = predictions[j]
                  flatscores = scores[j]
                  if not isinstance(scores[j], list):
                     flatscores = scores[j].tolist()
                  currscores = flatscores
            retpreds.append(currpreds)
            retscores.append(currscores)
            j += 1

# debug tmp:
#      self.linkcache.done()

      return retpreds, retscores

   # enddef runblink

   def runblinks(self, sentence):
      stoks = sentence.strip().split()
      lcontext = list()
      mention = list()
      rcontext = list()
      i = 0
      for tok in stoks:
         i += 1
         if tok == MENTIONSTART:
            break
         else:
            lcontext.append(tok)
      for tok in stoks[i:]:
         i += 1
         if tok == MENTIONEND:
            break
         else:
            mention.append(tok)
      rcontext = stoks[i:]
      mentionstr = ' '.join(mention).lower()

      mstru = dict()
#      mstru['id'] = f'{sentcount}.{mentcount}'
      mstru['label'] = 'unknown'
      mstru['label_id'] = -1
      mstru[CONTEXTLEFTKEY] = ' '.join(lcontext).lower()
      mstru[MENTIONKEY] = mentionstr
      mstru[CONTEXTRIGHTKEY] = ' '.join(rcontext).lower()

      predictions, scores = self.runblink([mstru])
      return mentionstr, predictions, scores

   # enddef runblinks

   def getwpinfofortitle(self, wptitle):
      wpid = ''
      resurl = ''
      if wptitle:
         if self.ldc:
            wpinfolist = self.ldc.get_WPinfo_for_WPtitles([wptitle])
            if wpinfolist:
               wpid, resurl = wpinfolist[0]

      return wpid, resurl

   def getdbpuriforwpid(self, wpid):
      dbpuri = ''
      if self.ldc:
         dbpurilist = self.ldc.get_DBPuris_for_WPids([wpid])
         if dbpurilist:
            dbpuri =dbpurilist[0]

      return dbpuri

   def done(self):
      self.linkcache.done()
      if self.ldc:
         self.ldc.flush()

# end class Blinker

def runtxt(blinker, inputfilename):
   if not os.path.exists(inputfilename):
      print(f'input file {inputfilename} not found', file=sys.stderr)
      return

   txtfil = open(inputfilename, 'r')
   for line in txtfil:
      mstrus = linksentenceall(blinker, line.strip())
      for mstru in mstrus:
         jstr = json.dumps(mstru)
         print(jstr)
         sys.stdout.flush()

   txtfil.close()
#   blinker.done()
#   blinker.ldc.flush()

# end runtxt()

MAXMENTIONWINDOW = 5
def runIOB(blinker, inputfilename):
   if not os.path.exists(inputfilename):
      print(f'input file {inputfilename} not found', file=sys.stderr)
      return

   stoks = list()
   iobfil = open(inputfilename, 'r')
   snum = 0
   for line in iobfil:
      if line.strip().startswith('#'):
         continue
      if not line.strip():
         taganddumpiob(blinker, snum, stoks)
         snum += 1
         stoks = list()
         continue

      fields = line.strip().split()
      tok = fields[0]
      tag = 'O'
      if len(fields) >= 2:
         tag = fields[1][0]
      stoks.append((tok, tag))

   if stoks:
      taganddumpiob(blinker, snum, stoks)

#   blinker.done()
#   blinker.ldc.flush()

# end runIOB()

def taganddumpiob(blinker, snum, stoks):
   toks = [t for (t, _) in stoks]
   mentions = list()
   insidespan = False
   spanstart = 0
   for i, (tok, tag) in enumerate(stoks):
      if tag == 'B':
         if insidespan:
            lcontext = toks[0:spanstart]
            mention = toks[spanstart:i]
            rcontext = toks[i:]
            mentions.append((lcontext, mention, rcontext))
         insidespan = True
         spanstart = i
      elif tag == 'O':
         if insidespan:
            lcontext = toks[0:spanstart]
            mention = toks[spanstart:i]
            rcontext = toks[i:]
            mentions.append((lcontext, mention, rcontext))
            insidespan = False

   s = ' '.join(toks)
   for (l, m, r) in mentions:
      mstru = dict()
      lstr = ' '.join(l)
      mstr = ' '.join(m)
      rstr = ' '.join(r)
      stagged = lstr + ' ' + MENTIONSTART + ' ' + mstr + ' ' + MENTIONEND + ' ' + rstr
      mention, predictions, scores = blinker.runblinks(stagged)
      if len(predictions) > 0:
         wptitle = predictions[0][0]
         wpid, resurl = blinker.getwpinfofortitle(wptitle)

         mstru['id'] = snum
         mstru['sentence'] = s
         mstru[CONTEXTLEFTKEY] = lstr
         mstru[MENTIONKEY] = mstr
         mstru[CONTEXTRIGHTKEY] = rstr
         mstru['Wikipedia_ID'] = wpid
         mstru['Wikipedia_URL'] = resurl
         mstru['Wikipedia_title'] = wptitle

         dbpuri = blinker.getdbpuriforwpid(wpid)
         if dbpuri:
            mstru['DBPedia_URI'] = dbpuri

         jstr = json.dumps(mstru)
         print(jstr)

   sys.stdout.flush()

# end taganddumpiob()

def linksentenceall(blinker, sentence):
   sid = 0
   if '\t' in sentence:
      fields = sentence.split('\t')
      sid = fields[0]
      sentence = fields[1]

   if '<' in sentence:
      sentence = sentence.replace('<', MENTIONSTART + ' ')
      sentence = sentence.replace('>', ' ' + MENTIONEND)

   j = sentence.find(MENTIONSTART)
   k = sentence.find(MENTIONEND) + len(MENTIONEND)
   if k < 0:
      k = len(sentence)
   mstrus = list()
   if j >= 0:
      while j >= 0:
         l = sentence[:j].replace(MENTIONSTART, '').replace(MENTIONEND, '').strip()
         m = sentence[j+len(MENTIONSTART):k]
         if MENTIONEND in m:
            m = m.replace(MENTIONEND, '').strip()
         r = sentence[k:].replace(MENTIONSTART, '').replace(MENTIONEND, '').strip()
         s = sentence.replace(MENTIONSTART, '').replace(MENTIONEND, '').strip()

         mstru = dict()
         mstru['id'] = sid
         mstru['sentence'] = s
         mstru['label'] = 'unknown'
         mstru['label_id'] = -1
         mstru[CONTEXTLEFTKEY] = l
         mstru[MENTIONKEY] = m
         mstru[CONTEXTRIGHTKEY] = r

         mstrus.append(mstru)

         j = sentence[k:].find(MENTIONSTART)
         if j > 0:
            j += k
         tmpk = sentence[k:].find(MENTIONEND)
         if tmpk < 0:
            k = len(sentence)
         else:
            k = tmpk + len(MENTIONEND) + k
   else:
      stoks = sentence.split()
      for i, tok in enumerate(stoks):
         l = stoks[:i]
         for j in range(len(stoks[i:])):
            k = i + j + 1
            if (k - i) > MAXMENTIONWINDOW:
               break
            m = stoks[i:k]
            r = stoks[k:]

            mstru = dict()
            mstru['id'] = sid
            mstru['sentence'] = sentence
            mstru['label'] = 'unknown'
            mstru['label_id'] = -1
            mstru[CONTEXTLEFTKEY] = ' '.join(l)
            mstru[MENTIONKEY] = ' '.join(m)
            mstru[CONTEXTRIGHTKEY] = ' '.join(r)

            mstrus.append(mstru)

   predictions, scores = blinker.runblink(mstrus)

   retstrus = list()
   for i, preds in enumerate(predictions):
      mstru = dict()
      wptitle = preds[0]
      topscore = scores[i][0]
      wpid, resurl = blinker.getwpinfofortitle(wptitle)

      mstru['id'] = mstrus[i]['id']
      mstru['sentence'] = mstrus[i]['sentence']
      mstru[CONTEXTLEFTKEY] = mstrus[i][CONTEXTLEFTKEY]
      mstru[MENTIONKEY] = mstrus[i][MENTIONKEY]
      mstru[CONTEXTRIGHTKEY] = mstrus[i][CONTEXTRIGHTKEY]
      mstru['Wikipedia_title'] = wptitle
      mstru['Link_score'] = topscore

      if wpid:
         mstru['Wikipedia_ID'] = wpid
      if resurl:
         mstru['Wikipedia_URL'] = resurl
      
      dbpuri = blinker.getdbpuriforwpid(wpid)
      if dbpuri:
         mstru['DBPedia_URI'] = dbpuri

      retstrus.append(mstru)

   return retstrus

# end def linksentenceall()

#BLINKMODELPATH = '/dccstor/ykt-parse/SHARED/MODELS/ELT/EL/BLINK/models'

def main():

   parser = argparse.ArgumentParser('Blinker')
   parser.add_argument('-i', '--inputfile',  action='store', help='path to inputfile (.txt or .iob format)')
   parser.add_argument('-o', '--outputfile', action='store', help='path to output file')
   parser.add_argument('-m', '--blinkmodels', action='store', help='path to BLINK models')
   parser.add_argument('-c', '--cachedirectory', action='store', help='path to cache directory')
   parser.add_argument('-r', '--rawoutput', action='store_true', help='output BLINK links (wikipedia titles) only (no mapping to other vocabularies)')
   parser.add_argument('-x', '--crossencoder', action='store_true', help='run cross-encoder reranking (slow mode)')
   args = parser.parse_args()

# TODO: allow non-cache version if blink models specified
#   if not args.blinkmodels and not args.cachedirectory:
#      print(f'no cache directory or path to blinkmodels specified: nothing to do', file=sys.stderr)
#      sys.exit(1)

   if args.blinkmodels:
      if not os.path.exists(args.blinkmodels):
         print(f'path to blink models {args.blinkmodels} not found', file=sys.stderr)
         sys.exit(2)

   if args.cachedirectory:
      if not os.path.exists(args.cachedirectory):
         print(f'path to cache directory {args.cachedirectory} not found', file=sys.stderr)
         sys.exit(3)

   infmt = ''
   if args.inputfile:
      if not os.path.exists(args.inputfile):
         print(f'input file {args.inputfile} not found', file=sys.stderr)
         sys.exit(4)
      else:
         if args.inputfile.endswith('.iob'):
            infmt = 'iob'
         elif args.inputfile.endswith('.txt'):
            infmt = 'txt'
         else:
            print(f'unknown format for input file {args.inputfile}; should be .iob or .txt', file=sys.stderr)
            sys.exit(5)

   if args.outputfile:
      if os.path.exists(args.outputfile):
         print(f'output file {args.outputfile} exists', file=sys.stderr)
         sys.exit(6)
      else:
         outfil = open(args.outputfile, 'w')

   wtitlesonly = False
   if args.rawoutput:
      wtitlesonly = True

   blinker = None
   fastmode = True
   if args.crossencoder:
      fastmode = False
   blinker = Blinker(pathtocachedirectory=args.cachedirectory, pathtomodeldirectory=args.blinkmodels, fastmode=fastmode, wikititleonly=wtitlesonly)

   if infmt == 'iob':
      runIOB(blinker, args.inputfile)
   elif infmt == 'txt':
      runtxt(blinker, args.inputfile)
   else:
      while True:
         try:
            s = input('BLINK> ')
            mstrus = linksentenceall(blinker, s.strip())

            for mstru in mstrus:
               mention = mstru[MENTIONKEY]
               wpid = ''
               if 'Wikipedia_ID' in mstru:
                  wpid = mstru['Wikipedia_ID']
               wpurl = ''
               if 'Wikipedia_URL' in mstru:
                  wpurl = mstru['Wikipedia_URL']
               wptitle = mstru['Wikipedia_title']
               linkscore = mstru['Link_score']
               dbpuri = ''
               if 'DBPedia_URI' in mstru:
                  dbpuri = mstru['DBPedia_URI']
               print(f'mention="{mention}"')
               print(f'\twptitle="{wptitle}"')
               print(f'\tlinkscore="{linkscore}"')
               print(f'\twpid="{wpid}"')
               print(f'\twpurl="{wpurl}"')
               print(f'\tdbpuri="{dbpuri}"')

         except EOFError:
            break

   blinker.done()

# end def main

if __name__ == '__main__':
   main()
