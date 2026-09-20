import cv2, numpy as np, glob, operator
panels=[]
for name in ['image114','image35','image73']:
    f=glob.glob(f'data/**/{name}.jpg',recursive=True)[0]
    img=cv2.imread(f,cv2.IMREAD_GRAYSCALE|cv2.IMREAD_IGNORE_ORIENTATION)
    th=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    n,lab,st,_=cv2.connectedComponentsWithStats(th,4)
    b=1+int(np.argmax(st[1:,cv2.CC_STAT_AREA]))
    m=((lab==b)*255).astype(np.uint8)
    cs,_=cv2.findContours(m,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    c=sorted(cs,key=cv2.contourArea,reverse=True)[0]; ca=cv2.contourArea(c)
    br,_=max(enumerate([p[0][0]+p[0][1] for p in c]),key=operator.itemgetter(1))
    tl,_=min(enumerate([p[0][0]+p[0][1] for p in c]),key=operator.itemgetter(1))
    bl,_=min(enumerate([p[0][0]-p[0][1] for p in c]),key=operator.itemgetter(1))
    tr,_=max(enumerate([p[0][0]-p[0][1] for p in c]),key=operator.itemgetter(1))
    q=np.array([c[tl][0],c[tr][0],c[br][0],c[bl][0]])
    x,y=q[:,0].astype(float),q[:,1].astype(float)
    qa=0.5*abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    vis=cv2.cvtColor(m,cv2.COLOR_GRAY2BGR)
    vis[m>0]=(90,90,90)                       # the blob, grey
    cv2.polylines(vis,[q.reshape(-1,1,2)],True,(0,0,255),4)   # quad from corners, red
    for p in q: cv2.circle(vis,tuple(p),10,(0,255,255),-1)
    cv2.putText(vis,f'{name}  ratio={qa/ca:.2f}',(12,38),cv2.FONT_HERSHEY_SIMPLEX,1.1,(255,255,255),3)
    h=430; vis=cv2.resize(vis,(int(vis.shape[1]*h/vis.shape[0]),h))
    panels.append(vis)
w=max(p.shape[1] for p in panels)
panels=[cv2.copyMakeBorder(p,0,0,0,w-p.shape[1],cv2.BORDER_CONSTANT,value=(0,0,0)) for p in panels]
cv2.imwrite("ratio.png", np.hstack(panels))

print('grey = biggest blob, red = quad through the 4 detected corners')