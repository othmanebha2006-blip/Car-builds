import numpy as np, soundfile as sf, sys, scipy.signal as ss
S=sys.argv[1]; SR=44100
ev=[(1,2.130),(1,10.630),(1,61.430),(2,1.200),(3,3.270),(3,50.300),(2,63.500)]
PRE,POST=0.35,1.25
sos=ss.butter(6,1500,'hp',fs=SR,output='sos')
segs=[]
for i,t in ev:
    a,sr=sf.read(f"{S}/v{i}/a44.wav"); assert sr==SR
    s=a[int((t-PRE)*SR):int((t+POST)*SR)]
    segs.append(s)
hp=[ss.sosfiltfilt(sos,s) for s in segs]
# reference = occurrence with the strongest HF burst relative to its own pre-roll
score=[np.sqrt((h[int(PRE*SR):int((PRE+0.4)*SR)]**2).mean())/np.sqrt((h[:int(0.25*SR)]**2).mean()+1e-9) for h in hp]
ref=int(np.argmax(score)); print("burst/pre-roll ratio per occurrence:",[round(x,1) for x in score],"-> reference",ev[ref])
L=int(0.15*SR); lags=[]; corrs=[]
r=hp[ref][int((PRE-0.05)*SR):int((PRE+0.6)*SR)]
for h in hp:
    c=ss.correlate(h,r,mode='valid'); c0=int((PRE-0.05)*SR)
    lo,hi=c0-L,c0+L
    k=lo+int(np.argmax(c[lo:hi])); lags.append(k-c0)
    seg=h[k:k+len(r)]; corrs.append(float(np.dot(seg,r)/np.sqrt(np.dot(seg,seg)*np.dot(r,r))))
print("alignment lag (ms):",[round(x/SR*1000,1) for x in lags])
print("normalised correlation with reference (1.0 = identical):",[round(c,2) for c in corrs])
N=min(len(s) for s in segs)-int(0.12*SR)
al=np.array([s[max(0,lag):max(0,lag)+N] if lag>=0 else np.concatenate([np.zeros(-lag),s[:N+lag]]) for s,lag in zip(segs,lags)])
nper=2048; hopn=512
Z=[ss.stft(x,fs=SR,nperseg=nper,noverlap=nper-hopn)[2] for x in al]
mags=np.abs(np.array(Z))
low=np.sort(mags,axis=0)[1]                 # 2nd-smallest of 7: what every copy shares (the effect), voice drops out
Xr=Z[ref]
mask=np.minimum(1.0, low/(np.abs(Xr)+1e-12))
out=ss.istft(Xr*mask,fs=SR,nperseg=nper,noverlap=nper-hopn)[1][:N]
# trim: start at the flash start (time PRE in the segment), keep until the effect has decayed
t0=int(PRE*SR)
env=np.convolve(out**2,np.ones(441)/441,'same'); pk=env.max()
on=int(np.argmax(env>pk*1e-3)); print("effect onset relative to flash start: %+.3f s"%((on-t0)/SR))
end=len(env)-int(np.argmax(env[::-1]>pk*1e-4))
clip=out[min(on,t0)-int(0.005*SR):end]
fade=int(0.02*SR); clip[-fade:]*=np.linspace(1,0,fade)
sf.write(f"{S}/shine_ref_extracted.wav",clip/np.abs(clip).max()*0.9,SR)
print("extracted length %.2f s, starts %.3f s before/after flash start"%(len(clip)/SR,(min(on,t0)-int(0.005*SR)-t0)/SR))
# how much voice is left: energy below 1 kHz relative to the effect band
f,tt,Zc=ss.stft(clip,fs=SR,nperseg=2048); P=np.abs(Zc)**2
print("energy <1kHz vs 2-5kHz (dB): %.1f"%(10*np.log10(P[f<1000].sum()/P[(f>2000)&(f<5000)].sum())))
# reference loudness: effect RMS vs the narrator's speech RMS in video 1
a,_=sf.read(f"{S}/v1/a44.wav"); speech=a[int(12*SR):int(40*SR)]
eff=out[t0:t0+int(0.6*SR)]
print("REF effect RMS / speech RMS = %.3f (%.1f dB)"%(np.sqrt((eff**2).mean())/np.sqrt((speech**2).mean()),20*np.log10(np.sqrt((eff**2).mean())/np.sqrt((speech**2).mean()))))
np.save(f"{S}/shine_ratio.npy",np.array([np.sqrt((eff**2).mean())/np.sqrt((speech**2).mean())]))
