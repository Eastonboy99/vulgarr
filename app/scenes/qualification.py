"""Duration-based benchmark metrics. Any-overlap scene recall is insufficient."""
import math
from app.common.intervals import merge_intervals


def interval_metrics(expected, predicted):
    def normalize(windows):
        if any(len(w)!=2 or any(type(t) not in (int,float) or not math.isfinite(t) for t in w)
               or not 0 <= w[0] < w[1] for w in windows):
            raise ValueError("Qualification intervals must be finite, nonnegative and increasing")
        return merge_intervals([(float(a),float(b)) for a,b in windows],merge_gap_seconds=0)
    truth,actual=normalize(expected),normalize(predicted)
    def subtract(windows,other):
        result=list(windows)
        for a,b in other:
            result=[(x,y) for s,t in result for x,y in
                ([(s,t)] if b<=s or a>=t else [(s,min(t,a)),(max(s,b),t)]) if y>x]
        return result
    missed,extra=subtract(truth,actual),subtract(actual,truth)
    seconds=lambda windows:sum(b-a for a,b in windows)
    required,filtered=seconds(truth),seconds(actual)
    covered=required-seconds(missed)
    return {"expected_seconds":required,"filtered_seconds":filtered,"covered_seconds":covered,
        "missed_seconds":seconds(missed),"extra_filtered_seconds":seconds(extra),
        "duration_recall":covered/required if required else None,
        "duration_precision":covered/filtered if filtered else None,
        "missed_intervals":missed,"extra_filtered_intervals":extra}
