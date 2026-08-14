"""Parser-alias scaffold for the six IMR flagship families.
These are deliberately conservative snippets to be merged into the I4 parser layer.
"""
FLAGSHIP_ALIASES = {
    "IMR-QD-005": ["holonomy", "wilson", "parallel transport", "trace"],
    "IMR-LG-006": ["period", "de rham", "winding", "cohomology", "globally exact"],
    "IMR-AN-005": ["limit of integrals", "lim int", "integral of limit", "interchange"],
    "IMR-AM-005": ["theta", "theta_eff", "vacuum angle", "topological angle"],
    "IMR-BV-005": ["euler-lagrange", "boundary", "canonical momentum", "surface term"],
    "IMR-TN-005": ["charge vector", "dyon", "electric", "magnetic", "dsz"],
}

FLAGSHIP_EXPECTED_PATTERNS = {
    "IMR-QD-005": ["trace=1", "g^-1 U g", "conjugation"],
    "IMR-LG-006": ["period=2*pi", "not globally exact"],
    "IMR-AN-005": ["limit_of_integrals=1", "integral_of_limit=0", "interchange_valid=no"],
    "IMR-AM-005": ["theta+pi", "theta+pi/2", "measure_jacobian"],
    "IMR-BV-005": ["bulk unchanged", "p=xdot+2*x", "p=xdot+3*x^2"],
    "IMR-TN-005": ["(1,-2)", "(3,1)", "DSZ=5"],
}
