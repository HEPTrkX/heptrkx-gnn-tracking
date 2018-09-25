# HEP.TrkX Graph Neural Networks for Particle Tracking

This repository contains the Graph Neural Network code which accompanies
the HEP.TrkX CTD 2018 presentation and paper (link to come).

The CTD results are implemented in Jupyter notebooks.
There is also some more recent work on TrackML challenge data.

## CTD 2018 results

The following notebooks were made using this ACTS QCD dataset with
pileup $\mu = 10$ and particle PT $>$ 1 GeV.

*MPNN_HitClassifier.ipynb* - Binary hit classifier model with partially-labeled
graph.

*MPNN_Seg_ACTS.ipynb* - Segment classifier model with simple 10-track events.

*MPNN_Seg_ACTS_fullEvents.ipynb* - Segment classifier model on variable-sized
events.
