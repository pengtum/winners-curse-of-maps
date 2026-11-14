"""ExCP -- exchange-distribution conformal prediction: a general GeoCP.

Validity comes from the exchange distribution Pi implied by the source of randomness
(sampling design, random split, covariate shift, random localization); everything else
(localizer, score, engine, decision layer) only changes efficiency. See
theory/GeoCP_framework.md for the theorems and excp.family.TABLE for how every GeoCP variant
is a parameter setting of this API.
"""
from . import weights, localizers, scores, engines, select, designs, family, robust, cluster  # noqa: F401

__version__ = "0.1.0"
