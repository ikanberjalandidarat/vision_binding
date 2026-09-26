import argparse
import json
import shutil
import subprocess
from pathlib import Path
from .io import atomic_json, environment


def main():
    parser = argparse.ArgumentParser(description='Minecraft binding backbone; fixtures are non-Minecraft data')
    sub = parser.add_subparsers(dest='command', required=True)
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
    if args.command == 'doctor':
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
