Site Configuration
==================

.. automodule:: fyst_pointing.site
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: fyst_pointing.get_fyst_site

Convenience Constants
---------------------

.. py:data:: fyst_pointing.FYST_LOCATION

   Pre-computed :class:`~astropy.coordinates.EarthLocation` for the FYST
   telescope.  Equivalent to ``get_fyst_site().location``.  Useful for
   quick calculations where a full :class:`Site` object is not needed.

   ::

       from fyst_pointing import FYST_LOCATION

       print(FYST_LOCATION.lat)   # -22d59m08.3004s
       print(FYST_LOCATION.lon)   # -67d44m25.0008s
