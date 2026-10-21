"""Animate recorded Minecraft poses; optionally add actual RGB. No inferred map."""
import argparse
import bisect
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def font(size):
    for name in ('/System/Library/Fonts/Supplemental/Arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('sequences')
    p.add_argument('--rgb-root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--geometry-family', default='f0000')
    p.add_argument('--transfer-list', type=Path)
    a = p.parse_args()
    entries = [e for e in json.loads(Path(a.sequences).read_text()) if e['family'] == a.geometry_family]
    entries.sort(key=lambda e: e['route'])
    if len(entries) != 4 or not all(e['is_minecraft'] for e in entries):
        raise ValueError('Need four recorded Minecraft routes in the chosen family')
    paths = sorted({f['frame'] for e in entries for f in e['frames']})
    if a.transfer_list:
        a.transfer_list.parent.mkdir(parents=True, exist_ok=True)
        a.transfer_list.write_text('\n'.join(paths)+'\n')
    if a.rgb_root:
        missing = [path for path in paths if not (a.rgb_root/path).is_file()]
        if missing:
            raise FileNotFoundError(f'{len(missing)} RGB frames missing; first: {missing[0]}')
    frames = []
    final_tick = max(e['frames'][-1]['elapsed_ticks'] for e in entries)
    sample_ticks = list(range(0, final_tick+1, 16))
    if sample_ticks[-1] != final_tick:
        sample_ticks.append(final_tick)
    colors = ['#2463c7','#ae5531','#28836d','#8655ae']
    for tick in sample_ticks:
        canvas = Image.new('RGB', (1440, 1020), '#f2f5f9')
        d = ImageDraw.Draw(canvas)
        d.text((28,18), 'Minecraft exploration | four routes through the same world', font=font(30), fill='#13283b')
        d.text((28,61), 'Recorded simulator positions • scripted movement • 4× playback • no learned cell activity shown', font=font(20), fill='#46586a')
        for index,e in enumerate(entries):
            left = 20+(index%2)*710; top = 105+(index//2)*440
            d.rounded_rectangle((left,top,left+690,top+420),radius=12,fill='white',outline='#c5d2de')
            times = [f['elapsed_ticks'] for f in e['frames']]
            current = max(0,bisect.bisect_right(times,tick)-1)
            f = e['frames'][current]; pose = f['pose']
            role = 'probe training' if e['route'] < 2 else 'held-out route'
            d.text((left+14,top+12), f'Route {e["route"]} · {role}', font=font(24), fill=colors[index])
            if a.rgb_root:
                with Image.open(a.rgb_root/f['frame']) as im:
                    canvas.paste(im.convert('RGB').resize((448,280)),(left+12,top+60))
                x0,y0,w,h = left+475,top+60,195,280
            else:
                x0,y0,w,h = left+30,top+60,620,280
            d.rectangle((x0,y0,x0+w,y0+h),fill='#f7f9fc',outline='#c5d2de')
            def xy(pose):
                return (x0+(pose['x']+8)/16*w, y0+h-(pose['z']+1)/13*h)
            for x in range(-8,9,2):
                px=x0+(x+8)/16*w;d.line((px,y0,px,y0+h),fill='#e0e7ee')
            for z in range(0,13,2):
                py=y0+h-(z+1)/13*h;d.line((x0,py,x0+w,py),fill='#e0e7ee')
            points = [xy(v['pose']) for v in e['frames'][:current+1]]
            if len(points)>1:d.line(points,fill=colors[index],width=3)
            x,y=xy(pose);d.ellipse((x-5,y-5,x+5,y+5),fill=colors[index])
            import math
            yaw=math.radians(pose['yaw'])
            d.line((x,y,x-15*math.sin(yaw),y-15*math.cos(yaw)),fill='#13283b',width=3)
            ended = tick >= times[-1]
            d.text((left+14,top+352),f't = {min(tick,times[-1])/20:.1f} s   X = {pose["x"]:.2f}   Z = {pose["z"]:.2f}'+('   FINISHED' if ended else ''),font=font(18),fill='#13283b')
            d.text((left+14,top+382),e['episode']+'  |  world axes: +X right, +Z up',font=font(16),fill='#46586a')
        d.text((28,995), 'RGB + recorded path' if a.rgb_root else 'POSE-ONLY PREVIEW: first-person images have not been downloaded.',font=font(16),fill='#46586a')
        frames.append(canvas)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    frames[0].save(a.output.with_suffix('.png'))
    frames[0].save(a.output,save_all=True,append_images=frames[1:],duration=200,loop=0,optimize=False)
    a.output.with_suffix('.json').write_text(json.dumps(dict(source=str(a.sequences),episodes=[e['episode'] for e in entries],rgb=bool(a.rgb_root),playback_speed=4,frames=len(frames),description='Actual simulator poses; scripted paths. No model activation or navigation-performance claim.'),indent=2))
    print(a.output)


if __name__ == '__main__':main()
