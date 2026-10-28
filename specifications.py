import numpy as np


def _num(na=(), zero=(), valid=None, log=False):
    """Description of a numeric or ordinal variable.
    Args:
        na: codes meaning don't know / refused / missing -> NaN
        zero: codes meaning "none" -> 0
        valid: (low, high) documented range of real answers; outside -> NaN
        log: apply log1p if the distribution is skewed
    Returns:
        dict with the above information
    """
    return {
        "kind": "num",
        "na": tuple(na),
        "zero": tuple(zero),
        "valid": valid,
        "log": log,
    }


def _cat(na=()):
    """Description of a categorical variable.
    Args:
        na: codes meaning don't know / refused / missing -> NaN
    Returns:
        dict with the above information
    """
    return {"kind": "cat", "na": tuple(na), "zero": (), "valid": None, "log": False}


SPECS = {
    # Geography and demographics
    "_STATE": _cat(),
    "MSCODE": _cat(),
    "SEX": _cat(na=(7, 9)),
    "_AGEG5YR": _cat(na=(14,)),
    "_AGE80": _num(),
    "_RACE": _cat(na=(9,)),
    "MARITAL": _cat(na=(9,)),
    "EDUCA": _num(na=(9,)),
    "INCOME2": _num(na=(77, 99)),
    "EMPLOY1": _cat(na=(9,)),
    "RENTHOM1": _cat(na=(7, 9)),
    "VETERAN3": _cat(na=(7, 9)),
    "INTERNET": _cat(na=(7, 9)),
    "_CHLDCNT": _num(na=(9,)),
    "PREGNANT": _cat(na=(7, 9)),
    "HH_ADULTS": _num(),
    "IS_PARTIAL": _cat(),
    # Body measurements
    "HTM4": _num(valid=(0.91, 2.44)),
    "WTKG3": _num(valid=(23, 295)),
    "_BMI5": _num(valid=(12, 100)),
    "_BMI5CAT": _cat(),
    # Health status and healthy days
    "GENHLTH": _num(na=(7, 9)),
    "PHYSHLTH": _num(na=(77, 99), zero=(88,), valid=(0, 30)),
    "MENTHLTH": _num(na=(77, 99), zero=(88,), valid=(0, 30)),
    "POORHLTH": _num(na=(77, 99), zero=(88,), valid=(0, 30)),
    # Health care access
    "HLTHPLN1": _cat(na=(7, 9)),
    "PERSDOC2": _cat(na=(7, 9)),
    "MEDCOST": _cat(na=(7, 9)),
    "CHECKUP1": _cat(na=(7, 9)),
    # Blood pressure and cholesterol
    "BPHIGH4": _cat(na=(7, 9)),
    "BPMEDS": _cat(na=(7, 9)),
    "_CHOLCHK": _cat(na=(9,)),
    "TOLDHI2": _cat(na=(7, 9)),
    # Chronic conditions
    "CVDSTRK3": _cat(na=(7, 9)),
    "_ASTHMS1": _cat(na=(9,)),
    "CHCSCNCR": _cat(na=(7, 9)),
    "CHCOCNCR": _cat(na=(7, 9)),
    "CHCCOPD1": _cat(na=(7, 9)),
    "HAVARTH3": _cat(na=(7, 9)),
    "ADDEPEV2": _cat(na=(7, 9)),
    "CHCKIDNY": _cat(na=(7, 9)),
    "DIABETE3": _cat(na=(7, 9)),
    # Disability
    "QLACTLM2": _cat(na=(7, 9)),
    "USEEQUIP": _cat(na=(7, 9)),
    "BLIND": _cat(na=(7, 9)),
    "DECIDE": _cat(na=(7, 9)),
    "DIFFWALK": _cat(na=(7, 9)),
    "DIFFDRES": _cat(na=(7, 9)),
    "DIFFALON": _cat(na=(7, 9)),
    # Tobacco
    "_SMOKER3": _cat(na=(9,)),
    "STOPSMK2": _cat(na=(7, 9)),
    "LASTSMK2": _cat(na=(77, 99)),
    "USENOW3": _cat(na=(7, 9)),
    # Alcohol
    "DROCDY3_": _num(valid=(0, 899), log=True),
    "_DRNKWEK": _num(valid=(0, 98999), log=True),
    "_RFBING5": _cat(na=(9,)),
    "_RFDRHV5": _cat(na=(9,)),
    # Fruits and vegetables
    "_FRUTSUM": _num(valid=(0, 999.98), log=True),
    "_VEGESUM": _num(valid=(0, 999.98), log=True),
    "_FRTLT1": _cat(na=(9,)),
    "_VEGLT1": _cat(na=(9,)),
    # Physical activity
    "_TOTINDA": _cat(na=(9,)),
    "_PACAT1": _cat(na=(9,)),
    "_PASTRNG": _cat(na=(9,)),
    "PA1MIN_": _num(valid=(0, np.inf), log=True),
    "PA1VIGM_": _num(valid=(0, np.inf), log=True),
    "STRFREQ_": _num(valid=(0, 98), log=True),
    # Arthritis burden
    "_LMTACT1": _cat(na=(9,)),
    "_LMTWRK1": _cat(na=(9,)),
    "_LMTSCL1": _cat(na=(9,)),
    "JOINPAIN": _num(na=(77, 99), valid=(0, 10)),
    # Seatbelt, immunization, HIV testing
    "SEATBELT": _cat(na=(7, 9)),
    "FLUSHOT6": _cat(na=(7, 9)),
    "PNEUVAC3": _cat(na=(7, 9)),
    "HIVTST6": _cat(na=(7, 9)),
}
