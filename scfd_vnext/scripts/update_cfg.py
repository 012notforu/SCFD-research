from pathlib import Path
import yaml

path = Path('cfg/defaults.yaml')
data = yaml.safe_load(path.read_text())

physics = data['physics']
physics['alpha'] = 0.2
physics['gamma'] = 0.5
physics['potential'] = {
    'kind': 'quadratic',
    'params': {
        'center': 0.0,
        'stiffness': 0.05,
    },
}
physics['cross_gradient']['enabled'] = False
physics['curvature_penalty']['enabled'] = False

path.write_text(yaml.safe_dump(data, sort_keys=False))
