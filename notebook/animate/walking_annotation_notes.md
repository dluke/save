FJ tracking annotation key and notes

These notes are pared with data `./walking_meta.json` which is an attempt
to annotate Fanjin data by hand. 
Researcher: Daniel Barton (daluke.barton@gmail.com)
Date: 2021/11/09

First trajectories are coarse grained and filtered by their minimum aspect ratio
after coarse graining to separate the dominant crawling behaviour from out-of-plane 
movements. Some very short trajectories are also removed during this step (?).

The ~300 (~10%) low aspect ratio trajectories are then classified by hand 
. The aspect ratio of of the bacterium projected onto the surface
is often used in place of the angle with the surface. This approach relies
on (i) identifying the real length of the body and (ii) the imaging technique
and body tracking algorithm are accurate for out-of-plane bodies.
The purpose of classifying the trajectories by eye into walking/crawling is 
to provide a reference to test automatic classification algorithms against.

Trajectories here are classified according to their tracking data, without
access to the original image data. Note that is difficult to distinguish the
persistent walking of a long bacterium with crawling of a short bacterium.

# annotation tags

The annotation tags are described

* walking: small aspect ratio, aspect ratio and orientation may vary rapidly, 
    one pole appears to be travelling a significantly larger distance.
* crawling: large and constant aspect ratio, high persistence
* static: bacteria moves not at all or so slowly that it travels by less 0.5 microns
    + horizontal/vertical: the orientation in which the bacterium appears to be static
* static_pole: bacterium appears to have one static pole and one moving pole
* transition: bacterium appears to transition between between walking and crawling
    states. Note some walking bacteria frequently tilt and become persistent as if 
    crawling but may or may not touch down their rear pole.
* persistent: a tag given to trajectories that are highly persistent despite having
    some walking characteristics and might be persistent walkers.
* short: trajectory is too short to make a reliable guess of behaviour type
 

A question mark '?' at the end of any tag indicates some uncertainty in the 
classification. E.g. `["crawling?"]` indicates the trajectory might be crawling 
or walking but crawling is preferred. `["transition?"]` indicates the possibility
of a transition between walking and crawling states.

# particular interest

Keep a note of any trajectories here that we might want to go back and look at later.

2925 - fast persistent track, is it walking or crawling? 
Similar (2944)