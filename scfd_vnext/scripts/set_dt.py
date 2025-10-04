from pathlib import Path
import yaml

path = Path('cfg/defaults.yaml')
data = yaml.safe_load(path.read_text())
data['integration']['dt'] = 0.0005
path.write_text(yaml.safe_dump(data, sort_keys=False))
