"""Phrase-level Whisper transcription + per-word timing (syllable-weighted within each phrase)."""
import sys, re, json, numpy as np, soundfile as sf, sherpa_onnx
wav, out, M = sys.argv[1], sys.argv[2], sys.argv[3]
a, sr = sf.read(wav, dtype="float32")
if sr != 16000:
    import scipy.signal as ss; a16 = ss.resample_poly(a, 16000, sr).astype("float32")
else: a16 = a
rec = sherpa_onnx.OfflineRecognizer.from_whisper(encoder=M+"small-encoder.int8.onnx", decoder=M+"small-decoder.int8.onnx",
        tokens=M+"small-tokens.txt", language="en", task="transcribe", num_threads=4)
hop = 160; n = len(a16)//hop
db = 20*np.log10(np.sqrt(np.mean(a16[:n*hop].reshape(n,hop)**2,1))+1e-9)
speech = db > -40
# phrases: speech separated by >=0.09s quiet; merge so phrases are 1.5-6 s
segs=[]; i=0
while i<n:
    if speech[i]:
        j=i
        while j<n and (speech[j:j+9].any()): j+=1
        segs.append([i,j]); i=j
    else: i+=1
merged=[]
for s in segs:
    if merged and (s[0]-merged[-1][1])<12 and (s[1]-merged[-1][0])<450: merged[-1][1]=s[1]
    else: merged.append(s)
def syl(w):
    w=w.lower(); w=re.sub(r"[^a-z0-9]","",w)
    if not w: return 1
    if w.isdigit(): return 2*len(w)
    return max(1,len(re.findall(r"[aeiouy]+",w)))
words=[]
for s,e in merged:
    st=rec.create_stream(); st.accept_waveform(16000,a16[s*hop:e*hop]); rec.decode_stream(st)
    txt=st.result.text.strip(); toks=txt.split()
    if not toks: continue
    # active frames inside the phrase -> distribute words over speaking time
    act=np.flatnonzero(speech[s:e])+s
    t0,t1=act[0]*0.01,(act[-1]+1)*0.01
    w=np.array([syl(t)+0.6 for t in toks],float); c=np.concatenate([[0],np.cumsum(w)])/w.sum()
    for k,t in enumerate(toks):
        words.append({"w":t,"s":round(t0+c[k]*(t1-t0),3),"e":round(t0+c[k+1]*(t1-t0),3)})
    print(f"[{t0:6.2f}-{t1:6.2f}] {txt}")
json.dump(words,open(out,"w"),indent=0)
