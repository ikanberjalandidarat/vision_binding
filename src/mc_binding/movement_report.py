"""Visualize measured movement; retain the entire raw frame below a telemetry header."""
import html
import json
from pathlib import Path
from PIL import Image, ImageDraw


def movement_report(root):
    root=Path(root);data=json.loads((root/'approach.json').read_text())
    trajectory=data['trajectory'];frames=[r for r in trajectory if 'frame' in r]
    if not frames:
        raise ValueError('No recorded frames')
    oracle=data['trial'].get('choice_source')=='ground_truth_not_model'
    label='CONTROLLER ONLY - known correct target' if oracle else 'Saved model choice - no ongoing visual inference'
    animation=[]
    for r in frames:
        with Image.open(root/r['frame']) as im:raw=im.convert('RGB')
        canvas=Image.new('RGB',(raw.width,raw.height+76),'#152333')
        canvas.paste(raw,(0,76));draw=ImageDraw.Draw(canvas)
        pose=r['pose']
        lines=[label, 'Goal: '+data['trial']['goal_color']+' | step '+str(r['tick']),
               f"X {pose['x']:.2f}   Y {pose['y']:.2f}   Z {pose['z']:.2f}   yaw {pose['yaw']:.1f}",
               f"Distance to requested waypoint: {r['distance_to_requested']:.2f} blocks"]
        for i,line in enumerate(lines):draw.text((7,4+i*17),line,fill='white')
        animation.append(canvas)
    durations=[max(20,(b['tick']-a['tick'])*50) for a,b in zip(frames,frames[1:])]+[1000]
    animation[0].save(root/'movement.gif',save_all=True,append_images=animation[1:],duration=durations,loop=0,optimize=False)
    # Top-down measured X/Z trajectory; Minecraft yaw=0 faces +Z.
    points=[(float(r['pose']['x']),float(r['pose']['z'])) for r in trajectory]
    goals=data['waypoints'];objects=data['objects']
    xs=[p[0] for p in points+goals]+[o['bounds'][j][0] for o in objects for j in (0,1)]
    zs=[p[1] for p in points+goals]+[o['bounds'][j][2] for o in objects for j in (0,1)]
    xmin,xmax=min(xs)-2,max(xs)+2;zmin,zmax=min(zs)-2,max(zs)+2
    canvas=Image.new('RGB',(800,650),'white');draw=ImageDraw.Draw(canvas)
    scale=min(660/(xmax-xmin),440/(zmax-zmin))
    def xy(p):return (70+(p[0]-xmin)*scale,530-(p[1]-zmin)*scale)
    draw.text((30,15),'MEASURED TRAJECTORY - top-down world coordinates',fill='black')
    draw.text((30,35),label,fill='black')
    draw.text((30,55),'X increases right; Z increases up. This is not screen left/right.',fill='black')
    for o in objects:
        lo,hi=o['bounds'];a=xy((lo[0],hi[2]));b=xy((hi[0],lo[2]))
        draw.rectangle((*a,*b),fill=o['color'],outline='black')
        draw.text((a[0],a[1]-14),o['color']+' '+o['type'],fill='black')
    for i,g in enumerate(goals):
        x,y=xy(g);draw.ellipse((x-5,y-5,x+5,y+5),outline='black',width=2)
        draw.text((x+8,y),'requested' if i==data['goal_side_in_rendered_scene'] else 'other',fill='black')
    line=[xy(p) for p in points]
    if len(line)>1:draw.line(line,fill='#793fb5',width=3)
    for p,name in ((line[0],'START'),(line[-1],'END')):
        x,y=p;draw.ellipse((x-4,y-4,x+4,y+4),fill='black');draw.text((x+7,y+5),name,fill='black')
    draw.text((30,565),f"Run state: {data['state']} | reached selected: {data.get('reached_selected_waypoint','not established')}",fill='black')
    draw.text((30,585),f"Correct-goal arrival: {data.get('task_success','not established')} | end distance: {trajectory[-1]['distance_to_requested']:.2f} blocks",fill='black')
    draw.text((30,610),'Arrival = within 0.8 blocks of the waypoint 2 blocks in front of the object.',fill='black')
    canvas.save(root/'trajectory.png')
    rows=''.join(f'<tr><td>{r["tick"]}</td><td>{r["pose"]["x"]:.2f}</td><td>{r["pose"]["y"]:.2f}</td><td>{r["pose"]["z"]:.2f}</td><td>{r["distance_to_requested"]:.2f}</td><td><a href="{html.escape(r["frame"])}">Raw frame</a></td></tr>' for r in frames)
    (root/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Minecraft movement</title><style>body{font:17px system-ui;max-width:1050px;margin:30px auto;line-height:1.6}img{max-width:100%}td,th{padding:8px}</style>'
        +'<h1>Minecraft approach recording</h1><p>'+label+'</p><p>Full raw first-person view, including inventory bar. Telemetry is added ABOVE the view, not over the pixels. This is not a third-person view of the avatar.</p>'
        +'<img src="movement.gif" alt="Recorded Minecraft movement with XYZ telemetry"><img src="trajectory.png" alt="Measured X Z trajectory">'
        +'<p>Playback uses 20 recorded simulator steps per second, with a final pause; it does not measure inference latency. Known world coordinates guide this controller. No training, patches during movement, or autonomous visual navigation is demonstrated.</p>'
        +'<p>State: '+html.escape(data['state'])+'; correct-goal arrival: '+str(data.get('task_success','not established'))+'</p><p><a href="approach.json">Full per-step actions and telemetry</a></p>'
        +'<table><tr><th>Step</th><th>X</th><th>Y</th><th>Z</th><th>Goal distance</th><th>Frame</th></tr>'+rows+'</table>')
