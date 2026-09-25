"""Candidate backend preflight only. No unvalidated frames enter benchmark datasets."""
from pathlib import Path
import numpy as np
from PIL import Image
from ..io import atomic_json, environment, file_hash
from ..scenes import family


def smoke_scene(seed):
    spec = family(0, seed, 'debug')['contexts']['recipient']
    # Isolate the arena from natural terrain and light it independently of the sky.
    for obj in spec['objects']:
        for point in obj['blocks'] + obj['bounds']:
            point[1] += 196
    spec['camera'] = {'position': [0.5, 200, 0.5], 'yaw': 0, 'pitch': 0,
                      'position_convention': 'player_feet; eye height not calibrated'}
    commands = [
        '/gamerule sendCommandFeedback false', '/gamemode creative',
        '/gamerule doDaylightCycle false', '/gamerule doWeatherCycle false',
        '/gamerule doMobSpawning false', '/time set 6000', '/weather clear',
        '/fill -20 200 -5 20 216 25 minecraft:air',
        '/fill -20 199 -5 20 199 25 minecraft:stone',
        '/fill -20 200 -5 -20 210 25 minecraft:white_concrete',
        '/fill 20 200 -5 20 210 25 minecraft:white_concrete',
        '/fill -20 200 -5 20 210 -5 minecraft:white_concrete',
        '/fill -20 200 25 20 210 25 minecraft:white_concrete',
        '/fill -20 205 -5 20 205 25 minecraft:sea_lantern',
    ]
    for obj in spec['objects']:
        for x, y, z in obj['blocks']:
            commands.append(f"/setblock {x} {y} {z} minecraft:{obj['color']}_wool")
    commands.append('/tp @p 0.5 200 0.5 0 0')
    return spec, commands


def json_value(value):
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    return value


def smoke(output, seed=731):
    try:
        from minestudio.simulator import MinecraftSim
        from minestudio.simulator.callbacks import CommandsCallback
    except ImportError as exc:
        raise RuntimeError('MineStudio is not installed. See docs/OSCAR.md for the separate rendering environment.') from exc
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    spec, commands = smoke_scene(seed)
    report = {'environment': environment(), 'commands': commands, 'scene': spec, 'is_minecraft': True, 'labels_validated': False, 'camera_calibrated': False, 'captures': []}
    sim = None
    try:
        sim = MinecraftSim(action_type='env', obs_size=(448, 280), render_size=(448, 280), seed=seed, callbacks=[CommandsCallback(commands=commands)])
        for repeat in range(2):
            obs, info = sim.reset()
            for _ in range(100):
                obs, _, done, truncated, info = sim.step(sim.noop_action())
                if done or truncated:
                    raise RuntimeError('Simulator terminated during settle')
            frame = np.asarray(obs['image'])
            if frame.shape != (280, 448, 3) or frame.dtype != np.uint8 or frame.std() < 1:
                raise RuntimeError('Unexpected image dimensions/type or blank render')
            path = root/f'reset-{repeat}.png'
            Image.fromarray(frame).save(path)
            report['captures'].append({'file': path.name, 'sha256': file_hash(path), 'info_keys': sorted(info), 'location_stats': json_value(info.get('location_stats')), 'player_pos': json_value(info.get('player_pos'))})
        action = sim.noop_action()
        action['camera'] = np.array([0., 15.], dtype=np.float32)
        obs, _, _, _, info = sim.step(action)
        for _ in range(20):
            obs, _, done, truncated, info = sim.step(sim.noop_action())
            if done or truncated:
                raise RuntimeError('Simulator terminated after camera turn')
        Image.fromarray(np.asarray(obs['image'])).save(root/'camera-turn.png')
        report['camera_turn'] = {'requested_action': [0., 15.], 'player_pos': json_value(info.get('player_pos')), 'location_stats': json_value(info.get('location_stats'))}
        report['state'] = 'captured_unvalidated'
        report['repeat_hash_equal'] = report['captures'][0]['sha256'] == report['captures'][1]['sha256']
    except BaseException as exc:
        report.update(state='error', error=str(exc))
        raise
    finally:
        atomic_json(root/'smoke.json', report)
        if sim is not None:
            sim.close()
