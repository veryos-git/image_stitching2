
this is an image stitcher app. my⠈team⠁wants to evaluate different tech stacks/ pipelines⢀than using superpoint, not only because superpoints license is heavily restricted. please implement a comparisan mode and use all different options possible. here is some research

Best candidates
Model	Type	License	Robustness	Compute	My use case for it
XFeat	sparse + semi-dense	Apache-2.0 weights available	★★★★	very low	Start here
XFeat + LightGlue	sparse contextual	Apache-2.0 ecosystem	★★★★½	low	Strong sparse matching
DISK + LightGlue	sparse contextual	Apache-2.0	★★★★½	medium	Alternative to SuperPoint
ALIKED + LightGlue	sparse contextual	BSD-3 + Apache-2	★★★★½	low/medium	Very interesting
LoFTR-family	detector-free / semi-dense	varies	★★★★★	medium/high	Low texture / difficult images
RoMa-family	dense	check specific model/repo	★★★★★	high	When you need maximum correspondence density
MASt3R	dense / geometry-aware	CC BY-NC-SA	★★★★★	high	Research only if commercial licensing matters

A particularly important detail: LightGlue itself is Apache-2.0. It's SuperPoint that introduces the restrictive licensing issue in the standard SuperPoint+LightGlue combination. The official LightGlue repository explicitly says its code and pretrained weights are Apache-2.0; it also supports DISK (Apache-2.0) and ALIKED (BSD-3-Clause).

So you can simply replace:

SuperPoint → SuperGlue

with something like:

ALIKED → LightGlue

or

DISK → LightGlue

without inheriting SuperPoint's license.


 provide a webgui / webbrowser app, where i can upload images and run them to compare the different stacks.
 by default use the demo content in the webapp, but allow the user to upload custom content aswell