"""Make a synchronized three-way video with native placement and lift telemetry."""
import argparse
from pathlib import Path
import subprocess
import sys
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"sequential-repair"))
from compose_stages import load, decode, placement, FPS, BG, FG, MUTED, BAD, OBJECTS, RULE

PANEL, HEADER, INFO, FOOTER = 336, 78, 96, 50
SIZE=(3*PANEL,HEADER+PANEL+INFO+FOOTER)
ARMS=("stock","bp","wsp")
LABELS=("原版 SmolVLA","BP · 梯度更新","WSP · Astra 直接写数")
ACCENTS=("#b1cbe0","#9edbc9","#d1baff")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        p.add_argument(arm,type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("--title",required=True)
    p.add_argument("--subtitle",default="同一起姿与指令；BP 和 WSP 使用同一动作示范、同一可改权重")
    a=p.parse_args()
    records=[load(getattr(a,k)) for k in ARMS]
    ref=records[0][1]
    fields=("id","seed","init_index","layout","goal","instruction","first_prompt","prompt_mode")
    assert all(all(r.get(k)==ref.get(k) for k in fields) and s==records[0][4]
               for _,r,_,_,s in records)
    end=max(r["steps"] for _,r,*_ in records)
    poster=max((r.get("stable_released_at") or r.get("first_native_goal") or 0) for _,r,*_ in records) or end
    font="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    fonts={n:ImageFont.truetype(font,n,index=2) for n in (14,15,16,18,22,27)}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    enc=subprocess.Popen(["ffmpeg","-nostdin","-v","error","-y","-f","rawvideo","-pix_fmt","rgb24",
            "-s",f"{SIZE[0]}x{SIZE[1]}","-r",str(FPS),"-i","pipe:0","-an","-c:v","libx264","-crf","19",
            "-pix_fmt","yuv420p","-movflags","+faststart",str(a.out)],stdin=subprocess.PIPE)
    readers=[]
    try:
        readers=[decode(path,r["steps"]) for path,r,*_ in records]
        frames=[None]*3
        for step in range(end+1):
            canvas=Image.new("RGB",SIZE,BG); d=ImageDraw.Draw(canvas)
            def label(x,y,txt,n=16,color=FG,**kw):
                d.text((x,y),txt,font=fonts[n],fill=color,**kw)
            label(20,8,a.title,27)
            label(20,47,a.subtitle,15,MUTED)
            label(SIZE[0]-20,22,f"{step/FPS:05.2f} 秒  /  第 {step} 步",18,anchor="rt")
            for i,(_,r,tr,native,_) in enumerate(records):
                t=min(step,r["steps"]); x=i*PANEL
                if t==step:
                    pixels=readers[i].stdout.read(native*native*3)
                    assert len(pixels)==native*native*3
                    frames[i]=Image.frombytes("RGB",(native,native),pixels).resize((PANEL,PANEL),Image.Resampling.LANCZOS)
                canvas.paste(frames[i],(x,HEADER))
                if step>t:
                    d.rounded_rectangle((x+11,HEADER+10,x+161,HEADER+35),radius=4,fill=BG)
                    label(x+17,HEADER+11,f"第 {t} 步结束 · 定格",14)
                y=HEADER+PANEL
                d.rectangle((x+17,y+12,x+20,y+34),fill=ACCENTS[i])
                label(x+29,y+7,LABELS[i],22,ACCENTS[i])
                lift=r.get("first_lift_step")
                if lift is not None and t>=lift:
                    first=r["first_lift"]
                    label(x+18,y+44,f"首抬：{OBJECTS.get(first,first)} · 第 {lift} 步",15,FG if first==r["goal"] else BAD)
                else:
                    label(x+18,y+44,"无持续双指抬起记录",15,MUTED)
                status,color=placement(r,tr[t-1] if t else None,t)
                label(x+18,y+67,status,16,color)
                if i:
                    d.line((x,HEADER,x,HEADER+PANEL+INFO),fill=RULE,width=2)
            d.line((0,SIZE[1]-FOOTER,SIZE[0],SIZE[1]-FOOTER),fill=RULE,width=2)
            label(20,SIZE[1]-38,"稳定放置：目标连续 40 步（2 秒）在柜顶，且未被双指夹持。",15,MUTED)
            if step==poster:
                canvas.save(a.out.with_suffix(".png"))
            enc.stdin.write(canvas.tobytes())
        enc.stdin.close()
        codes=[q.wait() for q in [*readers,enc]]
        if any(codes):
            raise RuntimeError(f"Video encoder/decoder exit codes: {codes}")
    finally:
        for proc in [*readers,enc]:
            if proc.poll() is None:
                proc.terminate()
    print(a.out)

if __name__=="__main__":
    main()
