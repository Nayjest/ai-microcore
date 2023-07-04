"""
Make code review of changes using GPT-4 (any programming language):

Usage:

1. Make patch file:
git diff master..feature-branch > feature.patch

2. Run this app:
python.app.py <full-path-to-patch-file>

"""
import json
import sys
from microcore import *

microcore.llm_default_args['model'] = 'gpt-4'
file_name = sys.argv[1] if sys.argv[1:] else 'feature.patch'
full_diff = fs.read(file_name)
parts = ['diff -git' + i for i in full_diff.split('diff --git')]
i = 0
rng = [1, 100]
skip = []
for part in parts:
    i += 1
    if i < rng[0]:
        continue
    if i > rng[1]:
        break
    first_line = part.split('\n')[0].replace('diff -git', '').strip()
    if len(first_line) == 0:
        continue
    bskip = False
    for s in skip:
        if s in first_line:
            bskip = True
            break
    if bskip:
        continue
    a, b = first_line.split(' ')
    fn = b.replace('b/', '') + '.txt'
    print(fn)
    out = llm(tpl('code-review.j2', input=part))
    if len(out.strip()) < 10:
        continue
    try:
        lines = json.loads(out)
        out = '\nISS: ' + '\nISS:'.join(lines)
    except json.decoder.JSONDecodeError:
        pass
    fs.write('out/' + fn, out)
print('Done')
