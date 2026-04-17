"""Pre-flight sun-safety check shared by all planner entry points."""

import warnings

from astropy.time import Time

from ..coordinates import Coordinates
from ..exceptions import PointingWarning
from ..site import Site


def _check_field_sun_safety(
    ra: float,
    dec: float,
    start_time: Time,
    site: Site,
) -> None:
    """Quick pre-flight check that a field center is not near the sun.

    This is a lightweight check that warns before expensive trajectory
    generation. It never blocks trajectory generation. Violations are
    reported as warnings.

    Parameters
    ----------
    ra : float
        Right Ascension of the field center in degrees.
    dec : float
        Declination of the field center in degrees.
    start_time : Time
        Observation start time.
    site : Site
        Site configuration with sun avoidance settings.

    Warns
    -----
    PointingWarning
        If the field center is within the sun exclusion radius.
    """
    if not site.sun_avoidance.enabled:
        return
    coords = Coordinates(site)
    az, el = coords.radec_to_altaz(ra, dec, start_time)
    sun_az, sun_alt = coords.get_sun_altaz(start_time)
    sep = coords.angular_separation(az, el, sun_az, sun_alt)
    if sep <= site.sun_avoidance.exclusion_radius:
        warnings.warn(
            f"EXCLUSION ZONE: Field center passes {sep:.1f}\u00b0 from the Sun "
            f"(exclusion radius: {site.sun_avoidance.exclusion_radius}\u00b0) "
            f"at {start_time.iso}. The telescope hardware may refuse this trajectory.",
            PointingWarning,
            stacklevel=2,
        )
