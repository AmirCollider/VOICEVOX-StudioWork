"""
VoxTalk - write normal English, get an excited, human-sounding VOICEVOX voice.

Why this exists
  VOICEVOX is a Japanese voice engine. Given English it guesses: "me" becomes
  the letters "M E", and the fix people use - misspelling words until they
  sound right ("pro grami ngo") - breaks the words into pieces the engine
  then stresses one by one. Pushing speed and intonation up for energy makes
  the words run together. And Japanese text processing whispers the final
  vowel of many words ("...desu" -> "...des"), which is the tired breath at
  the end of every sentence.

What it does instead
  1. Every English word becomes katakana here, before the engine sees it:
     a dictionary for the words that matter (words.json, which you can edit),
     then e2k, an English-to-katakana model, for everything else.
  2. The words are grouped into accent phrases the way a Japanese speaker
     says loanwords - small words lean on the next real word - and written in
     VOICEVOX's own kana notation, so nothing is guessed.
  3. Nothing is whispered: every vowel is voiced. No tired breath.
  4. The melody is shaped per sentence: excitement comes from pitch movement,
     a rising end, emphasised words and stretched vowels - not from speed, so
     the words stay clear.
  5. Each mood can use a different Zundamon style (sweet, tsuntsun, teary...).

Needs: the VOICEVOX app running (it serves http://127.0.0.1:50021), and
  pip install numpy e2k

Run:
  python voxtalk.py script scripts/intro.txt      -> renders/intro/
      (each line's wav, full.wav, captions.srt per line, sentences.srt per
       sentence, timing.txt, and timing.json: every line, sentence and word
       with its start and end in full.wav - what a video should follow)
  python voxtalk.py say "Hello! I'm AmirCollider!" --mood excited
  python voxtalk.py kana "Hello! I'm AmirCollider!"   (shows the katakana, no engine needed)
  python voxtalk.py voices                         (lists the engine's voices)

Credit: put "VOICEVOX:ずんだもん" in every video that uses the voice.
"""

import argparse
import io
import json
import math
import os
import random
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import wave
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LN2_12 = math.log(2) / 12.0          # one semitone in VOICEVOX's log-pitch units
MIN_BG = 0.09                        # seconds: the shortest b or g that is still heard
FRAME = 256 / 24000                  # the engine times every sound in whole frames of this
UPSPEAK = 0.15                       # seconds the engine adds to the end of a question


# ==================================================================
# Katakana: the moras VOICEVOX accepts in its kana notation
# (taken from the engine's own mora table)
# ==================================================================
VALID = set("""
ァ ア ィ イ ゥ ウ ェ エ ォ オ カ ガ キ ギ ク グ ケ ゲ コ ゴ サ ザ シ ジ ス ズ セ ゼ ソ ゾ タ ダ チ ヂ ッ ツ ヅ テ デ ト ド
ナ ニ ヌ ネ ノ ハ バ パ ヒ ビ ピ フ ブ プ ヘ ベ ペ ホ ボ ポ マ ミ ム メ モ ャ ヤ ュ ユ ョ ヨ ラ リ ル レ ロ ヮ ワ ヰ ヱ ヲ ン
ヴ ヶ イェ ウィ ウゥ ウェ ウォ キィ キェ キャ キュ キョ ギィ ギェ ギャ ギュ ギョ クァ クィ クゥ クェ クォ クヮ グァ グィ
グゥ グェ グォ グヮ シェ シャ シュ ショ ジェ ジャ ジュ ジョ スィ ズィ チェ チャ チュ チョ ヂェ ヂャ ヂュ ヂョ ツァ ツィ
ツェ ツォ ティ テェ テャ テュ テョ ディ デェ デャ デュ デョ トゥ ドゥ ニィ ニェ ニャ ニュ ニョ ヒィ ヒェ ヒャ ヒュ ヒョ
ビィ ビェ ビャ ビュ ビョ ピィ ピェ ピャ ピュ ピョ ファ フィ フェ フォ ミィ ミェ ミャ ミュ ミョ リィ リェ リャ リュ リョ
ヴァ ヴィ ヴェ ヴォ ヴャ ヴュ ヴョ
""".split())
SMALL = set('ァィゥェォャュョヮ')
ROWS = {
    'a': 'アカガサザタダナハバパマヤラワァャヮ',
    'i': 'イキギシジチヂニヒビピミリヰィ',
    'u': 'ウクグスズツヅヌフブプムユルゥュヴ',
    'e': 'エケゲセゼテデネヘベペメレヱェ',
    'o': 'オコゴソゾトドノホボポモヨロヲォョ',
}
VOWEL_OF = {k: v for v, ks in ROWS.items() for k in ks}
VOWEL_KANA = {'a': 'ア', 'i': 'イ', 'u': 'ウ', 'e': 'エ', 'o': 'オ'}
BIG = {'ァ': 'ア', 'ィ': 'イ', 'ゥ': 'ウ', 'ェ': 'エ', 'ォ': 'オ', 'ャ': 'ヤ', 'ュ': 'ユ', 'ョ': 'ヨ', 'ヮ': 'ワ'}


def hira_to_kata(s):
    return ''.join(chr(ord(c) + 0x60) if 'ぁ' <= c <= 'ゖ' else c for c in s)


def moras_of(kana):
    """Split katakana into VOICEVOX moras. 'ー' becomes the vowel it lengthens
    (the kana notation has no long-vowel mark), and anything the engine would
    refuse is repaired instead of failing the whole line."""
    kana = hira_to_kata(kana)
    out = []
    i = 0
    while i < len(kana):
        c = kana[i]
        pair = kana[i:i + 2]
        if len(pair) == 2 and pair[1] in SMALL and pair in VALID:
            out.append(pair)
            i += 2
            continue
        if c == 'ー':
            if out:
                last = out[-1]
                v = VOWEL_OF.get(last[-1])
                if v:
                    out.append(VOWEL_KANA[v])
                elif last == 'ン':
                    out.append('ン')
            i += 1
            continue
        if c in SMALL:
            # a small kana that cannot combine: say it as the full-size one
            # ('フュ' is not a VOICEVOX mora -> 'フ' + 'ユ')
            out.append(BIG[c])
            i += 1
            continue
        if c in VALID:
            out.append(c)
        i += 1
    return out


def voiceable(moras):
    """A word with no vowel ('ンー' -> 'ン','ン') comes out of VOICEVOX as a
    hum about 40 dB below speech: silence, and the line jumps straight to
    the next word. Say it the way a Japanese hum is said: 'ンー' -> 'ウーン'."""
    if moras and all(m in ('ン', 'ッ') for m in moras):
        n = moras.count('ン')
        return ['ウ'] * n + ['ン'] if n else []
    return moras


def vowel(m):
    return VOWEL_OF.get(m[-1])


def special(moras, k):
    """Moras that cannot carry a Japanese accent: ン, ッ, the second half of a
    long vowel, the イ/ウ that closes a diphthong."""
    m = moras[k]
    if m in ('ン', 'ッ'):
        return True
    if k > 0 and m in 'アイウエオ':
        pv = vowel(moras[k - 1])
        if pv == VOWEL_OF.get(m) or (m == 'イ' and pv in ('a', 'e', 'o')) or (m == 'ウ' and pv == 'o'):
            return True
    return False


def loanword_accent(moras):
    """Where Japanese puts the accent in a borrowed word: on the third mora
    from the end, moved one earlier when that mora is a special one."""
    n = len(moras)
    if n <= 2:
        return 1
    k = n - 3                          # 0-based
    while k > 0 and special(moras, k):
        k -= 1
    return k + 1


# ==================================================================
# English -> katakana
# ==================================================================
NUM = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
       'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen']
TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']


def number_words(n):
    n = int(n)
    if n < 20:
        return NUM[n]
    if n < 100:
        return TENS[n // 10] + ('' if n % 10 == 0 else ' ' + NUM[n % 10])
    if n < 1000:
        return NUM[n // 100] + ' hundred' + ('' if n % 100 == 0 else ' ' + number_words(n % 100))
    if n < 1000000:
        return number_words(n // 1000) + ' thousand' + ('' if n % 1000 == 0 else ' ' + number_words(n % 1000))
    return ' '.join(NUM[int(d)] for d in str(n))


LETTERS = {'a': 'エー', 'b': 'ビー', 'c': 'シー', 'd': 'ディー', 'e': 'イー', 'f': 'エフ', 'g': 'ジー', 'h': 'エイチ',
           'i': 'アイ', 'j': 'ジェー', 'k': 'ケー', 'l': 'エル', 'm': 'エム', 'n': 'エヌ', 'o': 'オー', 'p': 'ピー',
           'q': 'キュー', 'r': 'アール', 's': 'エス', 't': 'ティー', 'u': 'ユー', 'v': 'ブイ', 'w': 'ダブリュー',
           'x': 'エックス', 'y': 'ワイ', 'z': 'ズィー'}

# Small words that lean on the next real word in speech ("i am" -> one phrase)
FUNCTION = set("""
a an the i me my mine you your yours we us our they them their he him his she her it its i'm i'll i've i'd
you're you'll you've you'd we're we'll it's that's there's here's let's is am are was were be been being do does
did don't doesn't didn't can can't could couldn't would wouldn't will won't should shouldn't have has had to of in
on at for from with by as or and but so if than then just not no too very this that these those what who how when
""".split())


class Kana:
    """English text -> katakana, word by word."""

    def __init__(self, words_json=None):
        self.words = {}
        path = words_json or os.path.join(HERE, 'words.json')
        if os.path.isfile(path):
            with open(path, encoding='utf-8') as f:
                self.words = {k.lower(): v for k, v in json.load(f).items() if not k.startswith('_')}
        try:
            from e2k import C2K
            self.c2k = C2K()
        except Exception:
            self.c2k = None

    def word(self, w):
        lw = w.lower().replace('’', "'")
        if lw in self.words:
            return self.words[lw]
        if re.fullmatch(r"[ァ-ヴーぁ-ゖ]+", w):            # already kana
            return hira_to_kata(w)
        if lw.isdigit():
            return ''.join(self.word(x) for x in number_words(lw).split())
        if len(lw) == 1 and lw in LETTERS:
            return LETTERS[lw]
        if w.isupper() and 1 < len(w) <= 4:                # PC, USA, AI -> letters
            return ''.join(LETTERS[c] for c in lw if c in LETTERS)
        bare = lw.replace("'", '')
        if bare in self.words:
            return self.words[bare]
        if self.c2k is not None:
            try:
                k = self.c2k(bare)
                if k:
                    return self.plural(bare, k)
            except Exception:
                pass
        return ''.join(LETTERS.get(c, '') for c in bare)

    def plural(self, bare, k):
        """e2k sometimes drops a plural s ('supporters' -> サポーター, the same
        as 'supporter', and the listener hears one supporter). Put it back."""
        if (len(bare) < 4 or not bare.endswith('s')
                or re.search(r'(ss|us|is|[sxz]es|[cs]hes|ges|ces)$', bare)
                or self.c2k(bare[:-1]) != k):
            return k
        if bare[-2] == 't' and k.endswith('ト'):
            return k[:-1] + 'ツ'
        return k + ('ス' if bare[-2] in 'pkf' else 'ズ')


# ==================================================================
# A line -> sentences -> words -> kana notation
# ==================================================================
SENT = re.compile(r'(\.{3,}|…|—|[.!?~♡❤]+)')
WORD = re.compile(r"\*?[A-Za-z0-9][A-Za-z0-9'’\-]*\*?|[ァ-ヴーぁ-ゖ]+|,|;|:")
PAUSE = re.compile(r'\(\s*pause\s+(\d+(?:\.\d*)?|\.\d+)\s*s?\s*\)', re.I)


def split_line(text, moods, sentence_gap=None):
    """Sentences with their moods and delivery markers. A [mood] inside the
    text switches the mood from that point on. The silence after a sentence
    follows its punctuation; sentence_gap sets it for every sentence, and a
    (pause 0.5) inside the text sets it at that point."""
    text = PAUSE.sub(lambda m: ' \0%s\0 ' % m.group(1), text)
    text = re.sub(r'\([^)]*\)', ' ', text)
    text = re.sub(r'\b([AaPp])\.\s?[Mm]\.', lambda m: (' エーエム.' if m.group(1) in 'Aa' else ' ピーエム.'), text)  # 3 a.m.
    segs, current, asked = [], list(moods), {}
    for k, chunk in enumerate(re.split(r'\[([^\]]*)\]', text)):
        if k % 2:
            current = [m.strip().lower() for m in re.split(r'[\s,]+', chunk) if m.strip()]
            continue
        for j, piece in enumerate(chunk.split('\0')):
            if j % 2:                                     # the seconds of a (pause N)
                if segs:
                    asked[len(segs) - 1] = float(piece)
                continue
            parts = SENT.split(piece)
            for i in range(0, len(parts), 2):
                body, sep = parts[i], (parts[i + 1] if i + 1 < len(parts) else '')
                if not re.search(r'[A-Za-z0-9ァ-ヴぁ-ゖ]', body):
                    if segs and sep:
                        segs[-1]['sep'] += sep
                    continue
                segs.append({'body': body, 'sep': sep, 'moods': list(current)})
    for i, s in enumerate(segs):
        b, sep = s['body'], s['sep']
        s['tilde'] = '~' in b + sep
        s['heart'] = '♡' in b + sep or '❤' in b + sep
        s['question'] = '?' in sep
        s['exclaim'] = '!' in sep
        s['trail'] = sep.startswith('...') or sep.startswith('…')
        s['cut'] = sep.startswith('—')
        s['pause'] = 0.5 if s['trail'] else 0.08 if s['cut'] else 0.32 if s['question'] else 0.25
        if sentence_gap is not None and not s['cut']:
            s['pause'] = max(s['pause'], sentence_gap) if s['trail'] else sentence_gap
        if i in asked:
            s['pause'] = asked[i]
    if segs and len(segs) - 1 not in asked:
        segs[-1]['pause'] = 0.0
    return segs


def sentence_words(body, kana):
    """[(display_word, moras, flags)] and ',' for pauses inside a sentence."""
    out = []
    for tok in WORD.findall(body):
        if tok in (',', ';', ':'):
            out.append(',')
            continue
        emph = tok.startswith('*') and tok.endswith('*') and len(tok) > 2
        w = tok.strip('*')
        stretch = bool(re.search(r'([A-Za-z])\1\1', w))
        w2 = re.sub(r'([A-Za-z])\1{2,}', r'\1', w)       # Pleeease -> Please
        stutter = None
        m = re.fullmatch(r"([A-Za-z]{1,3})-(\1[A-Za-z']*)", w2, flags=re.I)
        if m:
            w2 = m.group(2)
            stutter = True
        if '-' in w2:                                       # well-known -> well + known
            moras = moras_of(''.join(kana.word(x) for x in w2.split('-') if x))
        else:
            moras = moras_of(kana.word(w2))
        moras = voiceable(moras)
        if not moras:
            continue
        if stutter:
            out.append({'word': w, 'moras': [moras[0]], 'stutter': True,
                        'emph': False, 'stretch': False, 'func': True})
            out.append(',')
        out.append({'word': w, 'moras': moras, 'emph': emph, 'stretch': stretch,
                    'func': w2.lower().replace('’', "'") in FUNCTION})
    return out


def build_kana(words, question):
    """Group words into accent phrases and write VOICEVOX kana notation.
    Returns (kana_text, [mora index ranges per word], total moras)."""
    phrases, spans = [], []
    pending = []                         # function words waiting for a content word
    idx = 0

    def flush(group, accent_word):
        nonlocal idx
        moras = []
        acc = 1
        for w in group:
            if w is accent_word:
                acc = len(moras) + loanword_accent(w['moras'])
            spans.append((w, idx + len(moras), idx + len(moras) + len(w['moras'])))
            moras += w['moras']
        if accent_word is None:
            acc = 1
        idx += len(moras)
        phrases.append({'moras': moras, 'accent': acc, 'pause': False})

    for w in words:
        if w == ',':
            if pending:
                flush(pending, None)
                pending = []
            if phrases:
                phrases[-1]['pause'] = True
            continue
        if w.get('func') and not w.get('emph'):
            pending.append(w)
            if sum(len(p['moras']) for p in pending) > 8:
                flush(pending, None)
                pending = []
            continue
        flush(pending + [w], w)
        pending = []
    if pending:
        if phrases and not phrases[-1]['pause'] and len(phrases[-1]['moras']) + sum(len(p['moras']) for p in pending) <= 12:
            last = phrases[-1]
            for p in pending:
                spans.append((p, idx, idx + len(p['moras'])))
                idx += len(p['moras'])
                last['moras'] += p['moras']
        else:
            flush(pending, None)
    text = ''
    for i, ph in enumerate(phrases):
        ms = ph['moras']
        text += ''.join(ms[:ph['accent']]) + "'" + ''.join(ms[ph['accent']:])
        if i == len(phrases) - 1:
            if question:
                text += '？'
        else:
            text += '、' if ph['pause'] else '/'
    return text, spans, idx


# ==================================================================
# VOICEVOX engine (HTTP, standard library only)
# ==================================================================
class Engine:
    def __init__(self, url='http://127.0.0.1:50021'):
        self.url = url.rstrip('/')
        self._template = {}
        self._speakers = None

    def _req(self, method, path, params=None, body=None, raw=False):
        q = ('?' + urllib.parse.urlencode(params)) if params else ''
        data = None
        headers = {}
        if body is not None:
            data = json.dumps(body).encode('utf-8')
            headers['Content-Type'] = 'application/json'
        req = urllib.request.Request(self.url + path + q, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                content = r.read()
        except urllib.error.HTTPError as e:
            raise RuntimeError('VOICEVOX said %s for %s: %s' % (e.code, path, e.read()[:300].decode('utf-8', 'ignore')))
        except urllib.error.URLError:
            raise RuntimeError('VOICEVOX is not answering at %s. Open the VOICEVOX app first '
                               '(it starts the engine), then run this again.' % self.url)
        return content if raw else json.loads(content.decode('utf-8'))

    def speakers(self):
        if self._speakers is None:
            self._speakers = self._req('GET', '/speakers')
        return self._speakers

    def style_id(self, speaker, style):
        names = []
        for sp in self.speakers():
            if sp['name'] == speaker:
                for st in sp['styles']:
                    names.append(st['name'])
                    if st['name'] == style:
                        return st['id']
                if sp['styles']:
                    return sp['styles'][0]['id']
        raise RuntimeError('No voice called "%s" in this VOICEVOX. Run: python voxtalk.py voices' % speaker)

    def accent_phrases(self, kana, sid):
        return self._req('POST', '/accent_phrases', {'text': kana, 'speaker': sid, 'is_kana': 'true'})

    def template(self, sid):
        if sid not in self._template:
            self._template[sid] = self._req('POST', '/audio_query', {'text': 'あ', 'speaker': sid})
        return json.loads(json.dumps(self._template[sid]))

    def synthesis(self, query, sid):
        return self._req('POST', '/synthesis', {'speaker': sid, 'enable_interrogative_upspeak': 'true'},
                         body=query, raw=True)


# ==================================================================
# Shaping a sentence's melody
# ==================================================================
def stretch_target(ms, accent=None):
    """Which syllable to hold when a word or a line is drawn out: the long
    vowel if there is one ("pu-RII-zu", "no-OU"), else the accented
    syllable, else the last real vowel. Never a final 'zu'/'to' - in
    katakana English those are only the tail of a consonant."""
    real = [m for m in ms if m['vowel'] not in ('N', 'cl', 'pau', 'sil')]
    if not real:
        return None
    for k in range(len(ms) - 1, 0, -1):
        a, b = ms[k - 1]['vowel'].lower(), ms[k]['vowel'].lower()
        if ms[k].get('consonant') is None and ms[k] in real and (
                a == b or (a == 'o' and b == 'u') or (a == 'e' and b == 'i')):
            return ms[k]
    if accent and 1 <= accent <= len(ms) and ms[accent - 1] in real:
        return ms[accent - 1]
    # the last vowel that is really a vowel: skip trailing consonant+u/o
    k = len(real) - 1
    while k > 0 and real[k].get('consonant') and real[k]['vowel'].lower() in ('u', 'o'):
        k -= 1
    return real[k]


def all_moras(aps):
    for ap in aps:
        for m in ap['moras']:
            yield m


def shape(aps, spans, mood, seg, seed):
    """Edit VOICEVOX's predicted pitches and lengths in place."""
    flat = list(all_moras(aps))
    if not flat:
        return
    rng = random.Random(seed)

    # 1. Nothing whispered: give every devoiced vowel a voice and a pitch,
    #    and every b and g enough time to be heard.
    voiced = [m['pitch'] for m in flat if m['pitch'] > 0]
    mean = float(np.mean(voiced)) if voiced else 5.8
    for i, m in enumerate(flat):
        if m['vowel'] in ('A', 'I', 'U', 'E', 'O'):
            m['vowel'] = m['vowel'].lower()
        if m['pitch'] <= 0 and m['vowel'] not in ('cl', 'pau', 'sil'):
            nb = [x['pitch'] for x in flat[max(0, i - 1):i + 2] if x['pitch'] > 0]
            m['pitch'] = float(np.mean(nb)) if nb else mean
        # Inside a phrase the engine makes b and g so short they melt into the
        # vowels: "the golden" comes out "the olden", "rainbow" "rainmow".
        # (Not d: held this long, a d is heard as a t - "paddle" -> "battle".)
        if m.get('consonant') in ('b', 'g') and (m.get('consonant_length') or 0) < MIN_BG:
            m['consonant_length'] = MIN_BG

    # 2. A lively melody: each accent phrase a little higher or lower than
    #    the last, the way people bounce through a sentence.
    bounce = float(mood.get('bounce', 0.8))
    for k, ap in enumerate(aps):
        off = (rng.uniform(-1, 1) * bounce + (0.6 * bounce if k == 0 else 0)) * LN2_12
        for m in ap['moras']:
            if m['pitch'] > 0:
                m['pitch'] += off

    # 3. Emphasised (*word*) and stretched (Pleeease) words.
    for w, a, b in spans:
        ms = flat[a:b]
        if w.get('emph'):
            for m in ms:
                if m['pitch'] > 0:
                    m['pitch'] += 2.5 * LN2_12
                m['vowel_length'] *= 1.2
        if w.get('stretch'):
            m = stretch_target(ms, loanword_accent(w['moras']))
            if m is not None:
                m['vowel_length'] *= 2.6
                if m['pitch'] > 0:
                    m['pitch'] += 1.0 * LN2_12
        if w.get('stutter'):
            for m in ms:
                m['vowel_length'] *= 0.6

    # 4. The end of the sentence: rise, stretch or fall.
    end_rise = float(mood.get('end_rise', 0.0))
    stretch_end = float(mood.get('end_stretch', 1.0))
    if seg['tilde']:
        end_rise += 2.5
        stretch_end *= 1.7
    if seg['heart']:
        end_rise += 1.5
        stretch_end *= 1.3
    if seg['exclaim']:
        end_rise += float(mood.get('exclaim_rise', 1.5))
    if seg['trail']:
        end_rise -= 2.0
        stretch_end *= 1.4
    tail = [m for m in flat if m['vowel'] not in ('cl',)][-2:]
    if tail:
        steps = [0.5, 1.0] if len(tail) == 2 else [1.0]
        for m, s in zip(tail, steps):
            if m['pitch'] > 0:
                m['pitch'] += end_rise * s * LN2_12
        last_word = max(spans, key=lambda x: x[2]) if spans else None
        hold = stretch_target(flat[last_word[1]:last_word[2]]) if last_word else None
        if hold is not None:
            hold['vowel_length'] *= stretch_end

    # 5. Whole-sentence pitch offset of the mood, in semitones.
    st = float(mood.get('pitch', 0.0))
    if st:
        for m in flat:
            if m['pitch'] > 0:
                m['pitch'] += st * LN2_12


def wav_to_array(data):
    with wave.open(io.BytesIO(data)) as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def trim_bounds(x, sr, thresh=0.01, keep=0.03):
    a = np.abs(x)
    on = np.where(a > thresh * max(a.max(), 1e-6))[0]
    if len(on) == 0:
        return 0, len(x)
    k = int(keep * sr)
    return max(0, on[0] - k), min(len(x), on[-1] + k)


def trim(x, sr, thresh=0.01, keep=0.03):
    a, b = trim_bounds(x, sr, thresh, keep)
    return x[a:b]


def mora_times(aps, q):
    """Where each mora starts and ends in the engine's output, in seconds,
    and how long the output is. The engine divides every sound by the speed
    and rounds it to whole frames; a question gets one more sound at its end."""
    speed = float(q.get('speedScale', 1.0))

    def frames(sec):
        return int(np.round(sec / speed / FRAME))

    t = frames(q.get('prePhonemeLength', 0.0))
    out = []
    for ap in aps:
        for m in ap['moras']:
            start = t
            t += frames(m.get('consonant_length') or 0.0) + frames(m['vowel_length'])
            out.append([start, t])
        if ap.get('is_interrogative') and ap['moras']:
            t += frames(UPSPEAK)
            out[-1][1] = t
        if ap.get('pause_mora'):
            pause = q.get('pauseLength')
            pause = ap['pause_mora']['vowel_length'] if pause is None else pause
            t += frames(pause * float(q.get('pauseLengthScale', 1.0)))
    t += frames(q.get('postPhonemeLength', 0.0))
    return [(a * FRAME, b * FRAME) for a, b in out], t * FRAME


def caption_text(text):
    return re.sub(r'\s+', ' ', text.replace('*', '')).strip()


# ==================================================================
# The program
# ==================================================================
class VoxTalk:
    def __init__(self, config=None):
        with open(config or os.path.join(HERE, 'moods.json'), encoding='utf-8') as f:
            cfg = json.load(f)
        self.cfg = cfg
        self.moods = {k: v for k, v in cfg['moods'].items() if not k.startswith('_')}
        self.kana = Kana()
        self.engine = Engine(cfg.get('engine', 'http://127.0.0.1:50021'))

    def mood(self, names):
        m = dict(self.moods.get('normal', {}))
        for n in names:
            if n in self.moods:
                m.update(self.moods[n])
        return m

    def plan(self, text, moods=(), sentence_gap=None):
        """What will be said: sentences, kana, mood - no engine needed."""
        plan = []
        for seg in split_line(text, moods or ['normal'], sentence_gap):
            words = sentence_words(seg['body'], self.kana)
            if not any(isinstance(w, dict) for w in words):
                continue
            kana, spans, n = build_kana(words, seg['question'])
            plan.append({'seg': seg, 'words': words, 'kana': kana, 'spans': spans, 'moras': n,
                         'mood': self.mood(seg['moods'])})
        return plan

    def speak(self, text, moods=(), sentence_gap=None):
        return self.speak_timed(text, moods, sentence_gap)[:2]

    def speak_timed(self, text, moods=(), sentence_gap=None):
        """The audio, and when each sentence and word is said in it:
        [{'text', 'start', 'end', 'words': [{'word', 'start', 'end'}]}]."""
        out, sentences, n = [], [], 0
        sr = 24000
        for i, p in enumerate(self.plan(text, moods, sentence_gap)):
            mood = p['mood']
            sid = self.engine.style_id(self.cfg.get('speaker', 'ずんだもん'),
                                       mood.get('style', self.cfg.get('style', 'ノーマル')))
            aps = self.engine.accent_phrases(p['kana'], sid)
            shape(aps, p['spans'], mood, p['seg'], seed=zlib.crc32(p['kana'].encode('utf-8')))
            q = self.engine.template(sid)
            q['accent_phrases'] = aps
            q['speedScale'] = float(mood.get('speed', 1.0))
            q['intonationScale'] = float(mood.get('intonation', 1.3))
            q['pitchScale'] = 0.0                    # pitch is moved in semitones by shape()
            q['volumeScale'] = float(mood.get('volume', 1.0))
            q['prePhonemeLength'] = 0.05
            q['postPhonemeLength'] = 0.05
            q['pauseLengthScale'] = float(mood.get('pauses', 0.8))
            q.pop('kana', None)
            x, sr = wav_to_array(self.engine.synthesis(q, sid))
            a, b = trim_bounds(x, sr)
            times, total = mora_times(aps, q)
            scale = len(x) / sr / total if total else 1.0    # 1.0 unless the engine times differently
            start, length = n / sr, (b - a) / sr

            def at(sec):
                return start + min(max(sec * scale - a / sr, 0.0), length)

            words = []
            for w, j, k in sorted(p['spans'], key=lambda s: s[1]):
                if words and words[-1].pop('stutter', False):  # "W-" and "Wait" are one word
                    words[-1]['end'] = at(times[k - 1][1])
                    continue
                words.append({'word': w['word'], 'start': at(times[j][0]), 'end': at(times[k - 1][1])})
                if w.get('stutter'):
                    words[-1]['stutter'] = True
            for w in words:
                w.pop('stutter', None)
            sentences.append({'text': caption_text(p['seg']['body'] + p['seg']['sep']),
                              'start': start, 'end': start + length, 'words': words})
            out.append(x[a:b])
            out.append(np.zeros(int(p['seg']['pause'] * sr), np.float32))
            n += (b - a) + len(out[-1])
        if not out:
            return np.zeros(int(0.2 * sr), np.float32), sr, []
        return np.concatenate(out), sr, sentences


def write_wav(path, x, sr):
    x = np.clip(x, -1, 1)
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x * 32767).astype(np.int16).tobytes())


def normalize(x, peak=0.89):
    m = float(np.max(np.abs(x))) if len(x) else 0.0
    return x * (peak / m) if m > 0 else x


# Script format (same as KoriVoice):  ME [mood]: text   |  (pause 1.0)  |  gap = 0.6
#   sentence_gap = 0.5   sets the silence between the sentences of every line
#   ME: One! (pause 0.5) Two!   sets it at one place in a line
LINE = re.compile(r'^\s*(?:(?P<who>[A-Za-z_][\w\-]*)\s*)?(?:\[(?P<moods>[^\]]*)\])?\s*:\s*(?P<text>.+)$')


def parse_script(text):
    gap, sentence_gap, items = 0.6, None, []
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith('#'):
            continue
        m = re.match(r'^\(\s*pause\s+([\d.]+)\s*s?\s*\)$', s, re.I)
        if m:
            items.append({'pause': float(m.group(1))})
            continue
        m = re.match(r'^(sentence_gap|gap)\s*=\s*([\d.]+)$', s, re.I)
        if m:
            if m.group(1).lower() == 'gap':
                gap = float(m.group(2))
            else:
                sentence_gap = float(m.group(2))
            continue
        m = LINE.match(s)
        if m and (m.group('who') or m.group('moods')):
            moods = [x.lower() for x in re.split(r'[\s,]+', m.group('moods') or '') if x]
            items.append({'who': (m.group('who') or 'ME').upper(), 'moods': moods, 'text': m.group('text').strip()})
        else:
            items.append({'who': 'ME', 'moods': [], 'text': s})
    return gap, sentence_gap, items


def srt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return '%02d:%02d:%02d,%03d' % (h, m, s, ms)


def render_script(vt, path, out_dir):
    with open(path, encoding='utf-8-sig') as f:
        gap, sentence_gap, items = parse_script(f.read())
    os.makedirs(out_dir, exist_ok=True)
    pieces, timing, lines, t, sr, n = [], [], [], 0.0, 24000, 0
    total = sum(1 for it in items if 'text' in it)
    for it in items:
        if 'pause' in it:
            pieces.append(np.zeros(int(it['pause'] * sr), np.float32))
            t += len(pieces[-1]) / sr
            continue
        n += 1
        print('[%d/%d] %s: %s' % (n, total, ' '.join(it['moods']) or 'normal', it['text']))
        x, sr, sentences = vt.speak_timed(it['text'], it['moods'], sentence_gap)
        name = '%02d_%s.wav' % (n, it['who'])
        lines.append((name, x))
        if pieces:
            pieces.append(np.zeros(int(gap * sr), np.float32))
            t += len(pieces[-1]) / sr
        caption = caption_text(re.sub(r'\[[^\]]*\]|\([^)]*\)', '', it['text']))
        for s in sentences:                       # line time -> time in full.wav
            s['start'], s['end'] = t + s['start'], t + s['end']
            for w in s['words']:
                w['start'], w['end'] = t + w['start'], t + w['end']
        timing.append((t, t + len(x) / sr, it['who'], caption, name, sentences))
        pieces.append(x)
        t += len(x) / sr
    full = np.concatenate(pieces + [np.zeros(int(0.3 * sr), np.float32)])
    # One gain for everything: a whispered line stays quieter than a shout.
    peak = float(np.max(np.abs(full))) or 1.0
    gain = 0.89 / peak
    write_wav(os.path.join(out_dir, 'full.wav'), full * gain, sr)
    for name, x in lines:
        write_wav(os.path.join(out_dir, name), x * gain, sr)
    with open(os.path.join(out_dir, 'captions.srt'), 'w', encoding='utf-8') as f:
        for i, (a, b, _w, c, _n, _s) in enumerate(timing, 1):
            f.write('%d\n%s --> %s\n%s\n\n' % (i, srt_time(a), srt_time(b), c))
    # The same, one caption per sentence: what a video should follow.
    with open(os.path.join(out_dir, 'sentences.srt'), 'w', encoding='utf-8') as f:
        cues = [s for row in timing for s in row[5]]
        for i, s in enumerate(cues, 1):
            f.write('%d\n%s --> %s\n%s\n\n' % (i, srt_time(s['start']), srt_time(s['end']), s['text']))
    # Every line, sentence and word with its start and end in full.wav (seconds).
    r3 = lambda v: round(v, 3)
    with open(os.path.join(out_dir, 'timing.json'), 'w', encoding='utf-8') as f:
        json.dump({'length': r3(len(full) / sr), 'lines': [
            {'file': nm, 'who': w, 'text': c, 'start': r3(a), 'end': r3(b), 'sentences': [
                {'text': s['text'], 'start': r3(s['start']), 'end': r3(s['end']), 'words': [
                    {'word': wd['word'], 'start': r3(wd['start']), 'end': r3(wd['end'])} for wd in s['words']]}
                for s in ss]}
            for a, b, w, c, nm, ss in timing]}, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, 'timing.txt'), 'w', encoding='utf-8') as f:
        for a, b, w, c, _n, _s in timing:
            f.write('%6.2fs - %6.2fs  %-4s %s\n' % (a, b, w, c))
    print('\n%d lines, %.1f s -> %s' % (len(timing), len(full) / sr, out_dir))
    print('Remember the credit in the video: VOICEVOX:ずんだもん')


def main():
    ap = argparse.ArgumentParser(prog='voxtalk', description=__doc__.split('\n')[1])
    sub = ap.add_subparsers(dest='cmd')
    s = sub.add_parser('script'); s.add_argument('file'); s.add_argument('--out')
    s = sub.add_parser('say'); s.add_argument('text'); s.add_argument('--mood', action='append', default=[]); s.add_argument('--out')
    s = sub.add_parser('kana'); s.add_argument('text'); s.add_argument('--mood', action='append', default=[])
    sub.add_parser('voices')
    a = ap.parse_args()
    if a.cmd is None:
        ap.print_help()
        return
    vt = VoxTalk()
    try:
        if a.cmd == 'kana':
            for p in vt.plan(a.text, a.mood):
                print('%-12s %s' % ('/'.join(p['seg']['moods']), p['kana']))
            if vt.kana.c2k is None:
                print('(e2k is not installed: unknown words are spelled letter by letter. pip install e2k)')
        elif a.cmd == 'voices':
            for sp in vt.engine.speakers():
                print(sp['name'] + ':  ' + ', '.join('%s=%d' % (st['name'], st['id']) for st in sp['styles']))
        elif a.cmd == 'say':
            x, sr = vt.speak(a.text, a.mood)
            out = a.out or os.path.join(HERE, 'renders', 'say.wav')
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            write_wav(out, normalize(x), sr)
            print('wrote ' + out)
        elif a.cmd == 'script':
            name = os.path.splitext(os.path.basename(a.file))[0]
            render_script(vt, a.file, a.out or os.path.join(HERE, 'renders', name))
    except RuntimeError as e:
        print('\n' + str(e))
        sys.exit(1)


if __name__ == '__main__':
    main()
