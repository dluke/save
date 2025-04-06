# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# In response to a suggestion: 
# Make production quality drawings of typical trajectories for each subset

# %% 
import os 
join = os.path.join
import numpy as np
import matplotlib.pyplot as plt
import pili
import shapeplot
import _fj

# %% 
# linearised?
subsetdata = _fj.load_subsets()

# %% 

def translate(tr, x, y):
    new = tr.copy()
    new['x'] = tr['x'] + x
    new['y'] = tr['y'] + y
    new['trail_x'] = tr['trail_x'] + x
    new['trail_y'] = tr['trail_y'] + y
    return new


norm = np.linalg.norm
def angle(v1,v2):
    v1_u, v2_u = v1/norm(v1,axis=1)[:,np.newaxis], v2/norm(v2,axis=1)[:,np.newaxis]
    return np.sign(np.cross(v1_u, v2_u)) * np.arccos(np.clip(np.sum(v1_u * v2_u, axis=1), -1.0, 1.0))


ex = np.array([1.,0.])

# %% 
candidate = subsetdata['candidate']
top = subsetdata['top']
walking = subsetdata['walking']
# ax = plt.gca()
# shapeplot.longtracks(ax, candidate)

# %% 
ax = plt.gca()
i = 1
tr = top[i]
_tr = tr.cut(0,1000)
_tr = _tr.shrink_projected_axis_by_radius()
_tr = translate(_tr, -tr['x'][0], -tr['y'][0])


_angle = angle(_tr.get_body_projection()[0], ex)

def rotate(tr, angle):
    transform = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
    print(transform)
    head = tr.get_head()[:,:2]
    trail = tr.get_trail()[:,:2]
    rhead = np.array([np.dot(transform, h) for h in head])
    rtrail = np.array([np.dot(transform, h) for h in trail])
    new = tr.copy()
    new['x'] = rhead[:,0]
    new['y'] = rhead[:,1]
    new['trail_x'] = rtrail[:,0]
    new['trail_y'] = rtrail[:,1]
    return new

_tr = rotate(_tr, -1 * _angle)

shapeplot.longtracks(ax, [_tr])

# %% 

ax= plt.gca()
# wi = 1
wi = 7
wtrack = walking[wi]
_wtrack = wtrack.cut(200,6000)
_wtrack
print(_wtrack.size)
# print(_wtrack["length"])
# print(_wtrack["width"])
# head = _wtrack.get_head()
# trail = _wtrack.get_trail()
# length = np.linalg.norm(head-trail, axis=1)
# np.mean(length), np.mean(_wtrack["length"])
# _wtrack = _wtrack.shrink_projected_axis_by_radius()
# _wtrack
_wtrack = translate(_wtrack, -wtrack['x'][0], -wtrack['y'][0])
shapeplot.longtracks(ax, [_wtrack])

# %% 
setting = {"draw_trail_line" : True, "draw_final_caps": True, "lwidth" : 8}

def publication_draw_track(ax, tr, setting=setting):
    lwidth = setting.get("lwidth")
    style = {"alpha" : 0.4, "linewidth" : 6}
    hlstyle = {"linewidth" : lwidth}
    tlstyle = {"linewidth" : lwidth, "alpha" : 0.5}
    
    R = np.mean(tr["width"])/2
    R = 0.50

    def _get_axis(xy, txy):
        x, y = xy
        tx, ty = txy
        ax_x, ax_y = x-tx, y-ty
        norm = np.sqrt(ax_x**2 + ax_y**2)
        ax_x, ax_y = ax_x/norm, ax_y/norm
        return np.array(ax_x), np.array(ax_y)
    #
    x = tr['x']
    y = tr['y']
    trail_x = tr["trail_x"]
    trail_y = tr["trail_y"]
    #
    def _draw_caps_at(ax, tr, idx):
        x = tr['x']
        y = tr['y']
        trail_x = tr["trail_x"]
        trail_y = tr["trail_y"]
        # must be arrays, even if length 1 
        xy = (np.array([x[idx]]), np.array([y[idx]]))
        txy = (np.array([trail_x[idx]]), np.array([trail_y[idx]]))
        capsule = (xy, txy, _get_axis(xy, txy))
        shapeplot.capsdraw(ax, *capsule, R=R, hashead=True, color='k', style=style)
    _draw_caps_at(ax, tr, 0)
    if setting.get("draw_final_caps"):
        _draw_caps_at(ax, tr, -1)
    # draw tracks on top
    ax.plot(x,y, **hlstyle)
    if setting.get("draw_trail_line", True):
        ax.plot(trail_x,trail_y, **tlstyle)
    ax.set_aspect("equal")

# fig, axes = plt.subplots(2,1, figsize=(10,2*4))
fig, axes = plt.subplots(2,1, sharex=True, sharey=True, figsize=(20,2*4))


ax = axes[0]
publication_draw_track(ax, _tr)

ax = axes[1]
wsetting = setting.copy()
wsetting["draw_trail_line"] = False
wsetting["draw_final_caps"] = False
wsetting["lwidth"] = 4
publication_draw_track(ax, _wtrack, wsetting)
out =  join(pili.root, "../writing/draw/", "example_tracks.svg")
print("writing to ", out)
plt.savefig(out)

# %%
# print out length and width
dimensions = _wtrack["length"][0] + np.mean(_wtrack["width"]), _wtrack["width"][0]
dimensions = _wtrack["length"][0] + 1.0, _wtrack["width"][0]
print(dimensions)
dimensions = tr["length"][0], tr["width"][0]
print(dimensions)
# %%

# %%
