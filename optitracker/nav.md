
OptiTracker
---

# Fix smoothing

- Currently, smoothing applied when computing avg'd position of the marker set.

[[OptiTracker.py:235]]
- Instead, this should be done before return, by:

[[OptiTracker.py:249]]

- But, how to properly index sub-elements (markers)? I.e.,

[[tests/live_test_optitracker/ExpAssets/Resources/code/test_OptiTracker.py:14]]

- Logic likely will be similar to how individual frames are handled here:

[[OptiTracker.py:236]]
