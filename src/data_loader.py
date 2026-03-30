from lightkurve import search_lightcurve
from preprocessing import preprocess_lightcurve

def get_segments(target, mission="Kepler"):
    search_result = search_lightcurve(target, mission=mission)

    lc_collection = search_result.download_all()
    lc = lc_collection.stitch()

    segments = preprocess_lightcurve(lc)

    return segments