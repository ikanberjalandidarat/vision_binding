import argparse
import json
import shutil
import subprocess
from pathlib import Path
from .io import atomic_json, environment


def main():
    parser = argparse.ArgumentParser(description='Minecraft binding backbone; fixtures are non-Minecraft data')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('visual-extract','visual-train','visual-evaluate'):
        p=sub.add_parser(name)
        p.add_argument('--output',required=True)
        p.add_argument('--config',default='configs/visual_readout.json')
        if name!='visual-extract': p.add_argument('--cache',required=True)
        if name!='visual-train':
            p.add_argument('--dataset',required=True)
            p.add_argument('--reviewed-captures',action='store_true')
        if name=='visual-evaluate':
            p.add_argument('--readout',required=True)
            p.add_argument('--patch',action='store_true')
    doctor = sub.add_parser('doctor')
    doctor.add_argument('--output', default='runs/environment.json')
    for name in ('smoke', 'generate'):
        p = sub.add_parser(name)
        p.add_argument('--output', required=True)
        p.add_argument('--backend', choices=['fixture', 'minestudio'], default='fixture')
        p.add_argument('--n', type=int, default=2 if name == 'smoke' else 20)
        p.add_argument('--seed', type=int, default=731)
        p.add_argument('--split', choices=['debug', 'calibration', 'test'], default='debug')
    p = sub.add_parser('validate')
    p.add_argument('dataset')
    p = sub.add_parser('capture-pairs')
    p.add_argument('--output', required=True)
    p.add_argument('--n', type=int, default=4)
    p.add_argument('--seed', type=int, default=731)
    p = sub.add_parser('capture-vision')
    p.add_argument('--output', required=True)
    p.add_argument('--n', type=int, default=4)
    p.add_argument('--seed', type=int, default=731)
    p.add_argument('--scene-set', choices=['original','depth_spacing_v1','replication_v1'], default='original')
    p = sub.add_parser('capture-binding')
    p.add_argument('--output', required=True)
    p.add_argument('--n', type=int, default=1)
    p.add_argument('--seed', type=int, default=731)
    for name in ('binding-baseline','binding-patch','binding-diagnostic'):
        p = sub.add_parser(name)
        p.add_argument('--dataset', required=True)
        p.add_argument('--output', required=True)
        p.add_argument('--config', default='configs/binding_baseline.json')
        p.add_argument('--reviewed-captures', action='store_true')
    p = sub.add_parser('vision-pilot')
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--config', default='configs/vision_pilot.json')
    p.add_argument('--reviewed-captures', action='store_true')
    p = sub.add_parser('destination')
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--config', default='configs/vision_replication_v.json')
    p.add_argument('--reviewed-captures', action='store_true')
    p = sub.add_parser('approach')
    p.add_argument('--dataset', required=True)
    p.add_argument('--decisions', required=True)
    p.add_argument('--trial', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--max-steps', type=int, default=400)
    p = sub.add_parser('controller-smoke')
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--family', default='f0000')
    p.add_argument('--goal-color', default='yellow')
    p.add_argument('--max-steps', type=int, default=400)
    p = sub.add_parser('recognize')
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--config', default='configs/qwen_pilot.json')
    p.add_argument('--reviewed-captures', action='store_true')
    p = sub.add_parser('q-pilot')
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--reviewed-captures', action='store_true')
    for name in ('baseline', 'patch'):
        p = sub.add_parser(name)
        p.add_argument('--dataset', required=True)
        p.add_argument('--output', required=True)
        p.add_argument('--config', default='configs/qwen_pilot.json')
        p.add_argument('--allow-fixture', action='store_true')
    p = sub.add_parser('analyze')
    p.add_argument('run')
    args = parser.parse_args()
    if args.command.startswith('visual-'):
        from .visual_readout import extract, train, evaluate
        config=json.loads(Path(args.config).read_text())
        if args.command=='visual-extract': extract(args.dataset,args.output,config,args.reviewed_captures)
        elif args.command=='visual-train': train(args.cache,args.output,config)
        else: evaluate(args.dataset,args.cache,args.readout,args.output,config,args.reviewed_captures,args.patch)
    elif args.command == 'doctor':
        report = environment()
        report['executables'] = {k: shutil.which(k) for k in ('java', 'nvidia-smi', 'sbatch', 'xvfb-run')}
        for name, command in [('java', ['java', '-version']), ('gpu', ['nvidia-smi'])]:
            if shutil.which(command[0]):
                result = subprocess.run(command, capture_output=True, text=True, timeout=20)
                report[name] = {'returncode': result.returncode, 'output': result.stdout+result.stderr}
        atomic_json(args.output, report)
        print(json.dumps(report, indent=2))
    elif args.command in ('smoke', 'generate'):
        if args.backend == 'minestudio':
            if args.command != 'smoke':
                parser.error('Minecraft dataset generation is gated on render/reset/camera and annotation validation. Only the real simulator smoke is implemented.')
            from .backends.minestudio_smoke import smoke
            smoke(args.output, args.seed)
        else:
            from .dataset import generate
            print(generate(args.output, args.n, args.seed, args.split))
    elif args.command == 'capture-pairs':
        from .capture_pairs import capture_pairs
        capture_pairs(args.output, args.n, args.seed)
    elif args.command == 'capture-vision':
        from .capture_pairs import capture_pairs
        capture_pairs(args.output, args.n, args.seed, vision=True, scene_set=args.scene_set)
    elif args.command == 'capture-binding':
        from .capture_pairs import capture_pairs
        capture_pairs(args.output, args.n, args.seed, binding=True, scene_set='binding_v1')
    elif args.command == 'binding-diagnostic':
        from .binding_diagnostic import run
        run(args.dataset, args.output, json.loads(Path(args.config).read_text()), args.reviewed_captures)
    elif args.command in ('binding-baseline','binding-patch'):
        from .binding_experiment import binding_experiment
        binding_experiment(args.dataset, args.output, json.loads(Path(args.config).read_text()), args.reviewed_captures, args.command=='binding-patch')
    elif args.command == 'vision-pilot':
        from .vision_pilot import vision_pilot
        vision_pilot(args.dataset, args.output, json.loads(Path(args.config).read_text()), args.reviewed_captures)
    elif args.command == 'destination':
        from .destination import destination
        destination(args.dataset, args.output, json.loads(Path(args.config).read_text()), args.reviewed_captures)
    elif args.command == 'approach':
        from .approach import approach
        approach(args.dataset, args.decisions, args.trial, args.output, args.max_steps)
    elif args.command == 'controller-smoke':
        from .approach import controller_smoke
        controller_smoke(args.dataset, args.output, args.family, args.goal_color, args.max_steps)
    elif args.command == 'recognize':
        from .recognition import recognize
        recognize(args.dataset, args.output, json.loads(Path(args.config).read_text()), args.reviewed_captures)
    elif args.command == 'q-pilot':
        from .q_pilot import q_pilot
        q_pilot(args.dataset, args.output, json.loads(Path(args.config).read_text()), args.reviewed_captures)
    elif args.command == 'validate':
        from .dataset import validate
        data = validate(args.dataset)
        print(f"Validated {len(data['families'])} families; is_minecraft={data['is_minecraft']}")
    elif args.command in ('baseline', 'patch'):
        from .runner import run
        config = json.loads(Path(args.config).read_text())
        run(args.dataset, args.output, config, args.command == 'patch', args.allow_fixture)
    else:
        from .analysis import analyze
        print(json.dumps(analyze(args.run), indent=2))


if __name__ == '__main__':
    main()
