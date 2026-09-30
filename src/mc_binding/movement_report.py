"""Visualize measured movement; retain the entire raw frame below a telemetry header."""
import math
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
    # Rotate the world map into the INITIAL camera's ground-plane basis.
    points=[(float(r['pose']['x']),float(r['pose']['z'])) for r in trajectory]
    goals=data['waypoints'];objects=data['objects']
    yaw=math.radians(float(trajectory[0]['pose']['yaw']))
    def project(p):
        x,z=p[0]-points[0][0],p[1]-points[0][1]
        return (-math.cos(yaw)*x-math.sin(yaw)*z, -math.sin(yaw)*x+math.cos(yaw)*z)
    corners=[[(o['bounds'][i][0],o['bounds'][j][2]) for i,j in ((0,0),(0,1),(1,1),(1,0))] for o in objects]
    projected=[project(p) for p in points+goals+[p for box in corners for p in box]]
    xmin,xmax=min(p[0] for p in projected)-2,max(p[0] for p in projected)+2
    zmin,zmax=min(p[1] for p in projected)-2,max(p[1] for p in projected)+2
    canvas=Image.new('RGB',(800,650),'white');draw=ImageDraw.Draw(canvas)
    scale=min(660/(xmax-xmin),440/(zmax-zmin))
    def xy(p):
        right,forward=project(p)
        return (70+(right-xmin)*scale,530-(forward-zmin)*scale)
    draw.text((30,15),'MEASURED TRAJECTORY - aligned to the STARTING camera',fill='black')
    draw.text((30,35),label,fill='black')
    draw.text((30,55),'Up = initial forward; right = initial camera right. Arrows show heading.',fill='black')
    for o,box in zip(objects,corners):
        polygon=[xy(p) for p in box]
        draw.polygon(polygon,fill=o['color'],outline='black')
        draw.text((min(p[0] for p in polygon),min(p[1] for p in polygon)-14),o['color']+' '+o['type'],fill='black')
    for i,g in enumerate(goals):
        x,y=xy(g);draw.ellipse((x-5,y-5,x+5,y+5),outline='black',width=2)
        draw.text((x+8,y),'requested' if i==data['goal_side_in_rendered_scene'] else 'other',fill='black')
    line=[xy(p) for p in points]
    if len(line)>1:draw.line(line,fill='#793fb5',width=3)
    for p,name in ((line[0],'START'),(line[-1],'END')):
        x,y=p;draw.ellipse((x-4,y-4,x+4,y+4),fill='black');draw.text((x+7,y+5),name,fill='black')
    for r in (trajectory[0],trajectory[-1]):
        pose=r['pose'];angle=math.radians(pose['yaw']);x,y=xy((pose['x'],pose['z']))
        ex,ey=xy((pose['x']-math.sin(angle),pose['z']+math.cos(angle)))
        draw.line((x,y,ex,ey),fill='#157f86',width=3)
        direction=math.atan2(ey-y,ex-x)
        draw.polygon([(ex,ey),(ex-9*math.cos(direction-.5),ey-9*math.sin(direction-.5)),
                      (ex-9*math.cos(direction+.5),ey-9*math.sin(direction+.5))],fill='#157f86')
    draw.text((30,565),f"Run state: {data['state']} | reached selected: {data.get('reached_selected_waypoint','not established')}",fill='black')
    draw.text((30,585),f"Correct-goal arrival: {data.get('task_success','not established')} | end distance: {trajectory[-1]['distance_to_requested']:.2f} blocks",fill='black')
    draw.text((30,610),'Arrival = within 0.8 blocks of the waypoint 2 blocks in front of the object.',fill='black')
    canvas.save(root/'trajectory.png')
    region_html=''
    selected=data['trial'].get('choice_side',data['goal_side_in_rendered_scene'])
    obj=objects[selected]
    if 'bbox_raw' in obj:
        with Image.open(root/frames[0]['frame']) as im:initial=im.convert('RGB')
        preview=Image.new('RGB',(initial.width,initial.height+48),'#152333');preview.paste(initial,(0,48))
        pen=ImageDraw.Draw(preview)
        pen.text((7,5),'KNOWN TARGET REGION - scene annotation, not model attention',fill='white')
        pen.text((7,24),obj['color']+' '+obj['type']+' | initial pose only',fill='white')
        x0,y0,x1,y1=obj['bbox_raw'];pen.rectangle((x0,y0+48,x1-1,y1+47),outline='#00ffff',width=2)
        preview.save(root/'destination-region.png')
        region_html='<h2>Where is the selected destination?</h2><img src="destination-region.png" alt="Known selected object box in the initial raw frame"><p>Cyan is the saved scene-annotation box, not a model detector, attention map or patch mask. It is shown only at the initial pose; moving-camera alignment has not been computed.</p>'
    rows=''.join(f'<tr><td>{r["tick"]}</td><td>{r["pose"]["x"]:.2f}</td><td>{r["pose"]["y"]:.2f}</td><td>{r["pose"]["z"]:.2f}</td><td>{r["distance_to_requested"]:.2f}</td><td><a href="{html.escape(r["frame"])}">Raw frame</a></td></tr>' for r in frames)
    (root/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Minecraft movement</title><style>body{font:17px system-ui;max-width:1050px;margin:30px auto;line-height:1.6}img{max-width:100%}td,th{padding:8px}</style>'
        +'<h1>Minecraft approach recording</h1><p>'+label+'</p><p>Full raw first-person view, including inventory bar. Telemetry is added ABOVE the view, not over the pixels. This is not a third-person view of the avatar.</p>'
        +region_html+'<img src="movement.gif" alt="Recorded Minecraft movement with XYZ telemetry"><img src="trajectory.png" alt="Measured X Z trajectory">'
        +'<p>Playback uses 20 recorded simulator steps per second, with a final pause; it does not measure inference latency. Known world coordinates guide this controller. No training, patches during movement, or autonomous visual navigation is demonstrated.</p>'
        +'<p>State: '+html.escape(data['state'])+'; correct-goal arrival: '+str(data.get('task_success','not established'))+'</p><p><a href="approach.json">Full per-step actions and telemetry</a></p>'
        +'<table><tr><th>Step</th><th>X</th><th>Y</th><th>Z</th><th>Goal distance</th><th>Frame</th></tr>'+rows+'</table>')
