"""Independent observed planning geometry; it does not authorize transfer motion."""
from copy import deepcopy

import pytest

from qumus_simlab.records.geometry import pickup_coverage

OUTER = [( .04748744493060019, .008599069285984444),(.017292317365699496,.026217581485745736),
    (-.03121388866837218,.013080603306407028),(-.035277010417425765,-.020051132544668285),
    (-.0009479841218435651,-.031803178755714104)]


INNER = [(-.0026662936191434827,-.01600247956045338),(.0025595384013992087,-.004476021666275501),
    (.0034361149030225514,.006728532836970328),(-.00057819597352651,.015890802417706847),
    (-.005663220983236841,.006306109165595552),(-.00704045372268417,-.004896286348579779)]


R = [.0025197212868569027,-.0013733086843707608]


SIGMA = .007597901760464169


REQUIRED = .007790426432018735


BINDING = {'source':'observed-source','carried':['observed-hbn'], 'source_image':'source.png',
           'stamp_image':'film.png','source_at':100.,'film_at':100.2,'evaluated_at':100.4}


def bound(binding=BINDING, required=REQUIRED):
    return {'binding':deepcopy(binding),'frame':'camera','required_clearance_mm':required,'provenance':'full_pair_plus_age'}


def test_unknown_budget_is_nominal_only_not_zero():
    c = pickup_coverage(OUTER,INNER,frame='assembly',binding=BINDING)
    assert c['state'] == 'nominal_only' and c['required_clearance_mm'] is None and c['headroom_mm'] is None


@pytest.mark.parametrize('change',['mismatch','stale','nan','negative','unknown','frame','polygon','binding_nan'])
def test_unknown_invalid_stale_or_mismatched_bound_cannot_admit(change):
    b, binding, frame, outer = bound(), deepcopy(BINDING), 'camera', OUTER
    if change == 'mismatch':
        binding['carried'] = ['another']
    elif change == 'stale':
        binding['evaluated_at'] += 1.
    elif change == 'nan':
        b['required_clearance_mm'] = float('nan')
    elif change == 'negative':
        b['required_clearance_mm'] = -.1
    elif change == 'unknown':
        b['required_clearance_mm'] = None
    elif change == 'frame':
        frame = ''
    elif change == 'binding_nan':
        binding['evaluated_at'] = float('nan')
    else:
        outer = [(0.,0.),(float('inf'),0.),(1.,1.)]
    result = pickup_coverage(outer,INNER,frame=frame,binding=binding,required_bound=b)
    assert result['state'] == 'unresolved'
    import json
    json.dumps(result,allow_nan=False)


def test_planning_coverage_failure_remains_independent_from_transfer_success():
    current = pickup_coverage(OUTER, INNER, frame="camera", binding=BINDING, required_bound=bound())
    assert current["state"] == "bound_failed"
    assert current["nominal_clearance_mm"] == pytest.approx(.005296096453326317)
    centre = pickup_coverage(OUTER, INNER, frame="camera", binding=BINDING, required_bound=bound(),
                             placement_mm=[-v for v in R])
    assert centre["state"] == "bound_failed"
    assert centre["nominal_clearance_mm"] == pytest.approx(.0072803396013333835)
