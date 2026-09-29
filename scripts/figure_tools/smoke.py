"""Exercise scriptable figure tools sequentially with bounded, CPU-only exports."""
from pathlib import Path
import argparse
import datetime
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfd_io import sha
from replot_sod_promo import memory_gate
import cairosvg
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parent
DOC = ROOT / 'docs/figure_tools'


def drawio(path, training=False):
    root = ET.Element('mxfile', host='app.diagrams.net')
    diagram = ET.SubElement(root, 'diagram', id='nnlci', name='Training' if training else 'Inference')
    model = ET.SubElement(diagram, 'mxGraphModel', page='1', pageWidth='1000', pageHeight='500')
    cells = ET.SubElement(model, 'root')
    ET.SubElement(cells, 'mxCell', id='0')
    ET.SubElement(cells, 'mxCell', id='1', parent='0')
    nodes = [('lf1', 'LF1 CFD', 30, 60, '#edf3ff'), ('lf2', 'LF2 CFD', 30, 180, '#eff9f1'),
             ('align', 'Align + stencils', 250, 120, '#eff7f8'), ('net', 'NNLCI', 470, 120, '#eff7f8'),
             ('out', 'Prediction', 690, 120, '#eff7f8')]
    edges = [('lf1', 'align'), ('lf2', 'align'), ('align', 'net'), ('net', 'out')]
    if training:
        nodes += [('hf', 'Training HF', 470, 290, '#fff1f2'), ('loss', 'Loss', 690, 290, '#fff1f2')]
        edges += [('hf', 'loss'), ('out', 'loss')]
    for name, label, x, y, color in nodes:
        node = ET.SubElement(cells, 'mxCell', id=name, value=label, vertex='1', parent='1',
            style=f'rounded=1;whiteSpace=wrap;html=0;fillColor={color};strokeColor=#9caeba;fontSize=18;')
        ET.SubElement(node, 'mxGeometry', x=str(x), y=str(y), width='170', height='65', attrib={'as':'geometry'})
    for i, (source, target) in enumerate(edges):
        edge = ET.SubElement(cells, 'mxCell', id=f'e{i}', edge='1', parent='1', source=source, target=target,
                             style='edgeStyle=orthogonalEdgeStyle;endArrow=block;strokeColor=#8799a8;')
        ET.SubElement(edge, 'mxGeometry', relative='1', attrib={'as':'geometry'})
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / '.smoke/figure-tools')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    DOC.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, NNLCI_FIGURE_OUTPUT=str(out), MPLCONFIGDIR='/tmp/nnlci-matplotlib',
               XDG_CACHE_HOME='/tmp/nnlci-figure-cache', XDG_CONFIG_HOME='/tmp/nnlci-figure-config',
               DRAWIO_DISABLE_UPDATE='true', LIBGL_ALWAYS_SOFTWARE='1', OPENBLAS_NUM_THREADS='1')
    report = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'execution': 'Sequential CPU/headless exports; no NVIDIA GPU', 'tests': []}

    def run(name, command):
        available = memory_gate()
        if available < 1500:
            raise RuntimeError(f'{name}: deferred below 1500 MiB effective available memory')
        log = out / (name+'.log')
        print(name, flush=True)
        with log.open('w') as stream:
            result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=180)
        report['tests'].append({'name': name, 'command': command, 'returncode': result.returncode,
                                'memory_before_mib': available, 'log_sha256': sha(log)})
        if result.returncode:
            print(log.read_text()[-4000:])
            raise RuntimeError(f'{name}: exit {result.returncode}')

    for name in ('layers', 'stencil'):
        source = SRC / f'{name}.scad'
        for ext in ('stl', 'svg', 'png'):
            command = ['openscad', '-o', str(out / f'openscad_{name}.{ext}')]
            if ext == 'svg':
                command += ['-D', 'flat=true']
            if ext == 'png':
                command = ['xvfb-run', '-a']+command+['--imgsize=1600,1000', '--viewall', '--autocenter', '--projection=o']
            run(f'openscad_{name}_{ext}', command+[str(source)])
        cairosvg.svg2pdf(url=str(out / f'openscad_{name}.svg'), write_to=str(out / f'openscad_{name}.pdf'))
    for name in ('pipeline', 'training'):
        for ext in ('svg', 'pdf', 'png'):
            run(f'graphviz_{name}_{ext}', ['dot', f'-T{ext}', '-Gdpi=240', str(SRC / f'{name}.dot'), '-o', str(out / f'graphviz_{name}.{ext}')])
        source = SRC / f'{name}.drawio'
        drawio(source, training=name=='training')
        for ext in ('svg', 'pdf', 'png'):
            run(f'drawio_{name}_{ext}', ['xvfb-run', '-a', 'drawio', '--no-sandbox', '--disable-gpu',
                '--disable-dev-shm-usage', '--disable-update', '--user-data-dir=/tmp/nnlci-drawio-profile',
                '--export', '--format', ext, '--scale', '2', '--output', str(out / f'drawio_{name}.{ext}'), str(source)])
    run('freecad', [sys.executable, str(SRC / 'freecad_examples.py')])
    for name in ('domain', 'sampling'):
        cairosvg.svg2pdf(url=str(out / f'freecad_{name}.svg'), write_to=str(out / f'freecad_{name}.pdf'))
        cairosvg.svg2png(url=str(out / f'freecad_{name}.svg'), write_to=str(out / f'freecad_{name}.png'), output_width=2000, output_height=1240)
    run('svg_python', [sys.executable, str(SRC.parent / 'render_promo_mechanism.py')])
    run('blender', ['blender', '-b', '-t', '2', '--python', str(SRC / 'blender_examples.py')])
    # Validate actual exported files, not just process exit status.
    exports = []
    for file in sorted(out.iterdir()):
        if file.suffix not in ('.svg', '.pdf', '.png', '.step', '.FCStd', '.stl', '.blend'):
            continue
        assert file.stat().st_size > 100, file
        item = {'file': file.name, 'bytes': file.stat().st_size, 'sha256': sha(file)}
        if file.suffix == '.png':
            with Image.open(file) as im:
                im.load()
                item['dimensions'] = list(im.size)
                assert im.convert('L').getextrema()[0] < im.convert('L').getextrema()[1], file
        elif file.suffix == '.svg':
            assert ET.parse(file).getroot().tag.endswith('svg'), file
        elif file.suffix == '.pdf':
            assert file.read_bytes().startswith(b'%PDF'), file
        exports.append(item)
    report['exports'] = exports
    report['versions'] = subprocess.check_output(['dpkg-query', '-W', '-f=${Package} ${Version}\n',
        'openscad', 'graphviz', 'freecad', 'draw.io', 'blender', 'python3-cairosvg', 'python3-matplotlib'], text=True).splitlines()
    report['source_sha256'] = {str(p.relative_to(ROOT)):sha(p) for p in sorted(SRC.iterdir()) if p.is_file()}
    report['godot'] = 'Not installed: optional, no interactive visualization requested'
    (DOC / 'smoke_report.json').write_text(json.dumps(report, indent=2)+'\n')
    files = ['openscad_layers.png', 'openscad_stencil.png', 'graphviz_pipeline.png', 'graphviz_training.png',
             'drawio_pipeline.png', 'drawio_training.png', 'freecad_domain.png', 'freecad_sampling.png',
             'blender_layers.png', 'blender_samples.png']
    sheet = Image.new('RGB', (1200, 1300), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, name in enumerate(files):
        x, y = (i % 2)*600, (i//2)*260
        with Image.open(out / name) as im:
            tile = ImageOps.contain(im.convert('RGB'), (560, 216))
            sheet.paste(tile, (x+20+(560-tile.width)//2, y+30))
        draw.text((x+20, y+8), name, fill='#162536')
    sheet.save(DOC / 'tool_previews.png')
    print(f'Validated {len(exports)} exports; report: {DOC / "smoke_report.json"}')


if __name__ == '__main__':
    main()
