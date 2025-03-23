
import sys, os
join = os.path.join

def set_write_dir(_writedir):
    global writedir 
    writedir = _writedir
    global imsdir
    imsdir = join(writedir, "ims")

writedir = "/home/dan/usb_twitching/sparseml/writing"
imsdir = join(writedir, "ims")

import matplotlib.pyplot as plt

module_config = {
    "dpi": 300,
    "svg": False
}

def save_figure(name, origin='', fig=None, config={}):
    if fig is None:
        fig = plt.gcf()
    svg = config.get('svg', module_config['svg'])
    dpi = config.get('dpi', module_config['dpi'])
    bbox_inches='tight'
    kwargs = {"dpi": dpi, "bbox_inches": bbox_inches, "transparent": True}

    _name = name + '.png'
    svg_name = name + '.svg'
    _target = join(imsdir, _name)
    svg_target = join(imsdir, svg_name)
    with open(_target, 'w') as f:
        print("writing figure to:")
        print(_target)
        if svg:
            print(svg_target)
        if fig:
            fig.savefig(_target, **kwargs)
            if svg:
                fig.savefig(svg_target, **kwargs)
