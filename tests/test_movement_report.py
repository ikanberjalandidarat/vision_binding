"""Synthetic reporting test; does not validate Minecraft motion."""
import json
from PIL import Image
from mc_binding.movement_report import movement_report


def test_full_view_gif_and_measured_trajectory(tmp_path):
    rows=[]
    for t,z in ((0,0),(5,1),(10,2)):
        im=Image.new('RGB',(448,280),(t*10,40,90));name=f'frame-{t:04d}.png';im.save(tmp_path/name)
        rows.append(dict(tick=t,pose=dict(x=0,y=200,z=z,yaw=0),distance_to_requested=3-z,frame=name))
    report=dict(state='complete',trial=dict(goal_color='yellow',choice_source='ground_truth_not_model'),trajectory=rows,
                objects=[dict(color='yellow',type='pillar',bounds=[[0,200,5],[1,204,6]])],waypoints=[[0,3]],
                goal_side_in_rendered_scene=0,reached_selected_waypoint=False,task_success=False)
    (tmp_path/'approach.json').write_text(json.dumps(report))
    originals=[(tmp_path/r['frame']).read_bytes() for r in rows]
    movement_report(tmp_path)
    assert originals==[(tmp_path/r['frame']).read_bytes() for r in rows]
    with Image.open(tmp_path/'movement.gif') as gif:
        assert gif.size==(448,356) and gif.n_frames==3
        assert gif.info['duration']==250
    assert 'correct-goal arrival: False' in (tmp_path/'report.html').read_text()
    assert (tmp_path/'trajectory.png').exists()
