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
    # --- Geography and demographics ---------------------------------------
    "_STATE": _cat(),  # state FIPS code
    "MSCODE": _cat(),  # metropolitan status (landline only)
    "SEX": _cat(na=(7, 9)),
    "_AGEG5YR": _cat(na=(14,)),  # 13 age groups: non-linear age effect
    "_AGE80": _num(),  # imputed age, 80 = 80 or older
    "_RACE": _cat(na=(9,)),
    "MARITAL": _cat(na=(9,)),
    "EDUCA": _num(na=(9,)),  # ordinal 1-6
    "INCOME2": _num(na=(77, 99)),  # ordinal 1-8
    "EMPLOY1": _cat(na=(9,)),  # 7 = retired is a real answer here
    "RENTHOM1": _cat(na=(7, 9)),
    "VETERAN3": _cat(na=(7, 9)),
    "INTERNET": _cat(na=(7, 9)),
    "_CHLDCNT": _num(na=(9,)),  # 1 = none ... 6 = five or more
    "PREGNANT": _cat(na=(7, 9)),
    "HH_ADULTS": _num(),  # derived, see _special_columns
    "IS_PARTIAL": _cat(),  # derived, see _special_columns
    # --- Body measurements ---------------------------------------------------
    # The course file already applies the implied decimals for these (HTM4 in
    # meters, WTKG3 in kg, _BMI5 as BMI), unlike the raw CDC file.
    "HTM4": _num(valid=(0.91, 2.44)),
    "WTKG3": _num(valid=(23, 295)),
    "_BMI5": _num(valid=(12, 100)),
    "_BMI5CAT": _cat(),
    # --- Health status and healthy days -------------------------------------
    "GENHLTH": _num(na=(7, 9)),  # ordinal 1-5; use _cat() for one-hot
    "PHYSHLTH": _num(na=(77, 99), zero=(88,), valid=(0, 30)),
    "MENTHLTH": _num(na=(77, 99), zero=(88,), valid=(0, 30)),
    "POORHLTH": _num(na=(77, 99), zero=(88,), valid=(0, 30)),
    # --- Health care access --------------------------------------------------
    "HLTHPLN1": _cat(na=(7, 9)),
    "PERSDOC2": _cat(na=(7, 9)),
    "MEDCOST": _cat(na=(7, 9)),
    "CHECKUP1": _cat(na=(7, 9)),  # 8 = never is a real answer
    # --- Blood pressure and cholesterol --------------------------------------
    "BPHIGH4": _cat(na=(7, 9)),
    "BPMEDS": _cat(na=(7, 9)),
    "_CHOLCHK": _cat(na=(9,)),
    "TOLDHI2": _cat(na=(7, 9)),
    # --- Chronic conditions (heart attack / CHD excluded: target) -----------
    "CVDSTRK3": _cat(na=(7, 9)),
    "_ASTHMS1": _cat(na=(9,)),  # current / former / never
    "CHCSCNCR": _cat(na=(7, 9)),
    "CHCOCNCR": _cat(na=(7, 9)),
    "CHCCOPD1": _cat(na=(7, 9)),
    "HAVARTH3": _cat(na=(7, 9)),
    "ADDEPEV2": _cat(na=(7, 9)),
    "CHCKIDNY": _cat(na=(7, 9)),
    "DIABETE3": _cat(na=(7, 9)),
    # --- Disability -----------------------------------------------------------
    "QLACTLM2": _cat(na=(7, 9)),
    "USEEQUIP": _cat(na=(7, 9)),
    "BLIND": _cat(na=(7, 9)),
    "DECIDE": _cat(na=(7, 9)),
    "DIFFWALK": _cat(na=(7, 9)),
    "DIFFDRES": _cat(na=(7, 9)),
    "DIFFALON": _cat(na=(7, 9)),
    # --- Tobacco --------------------------------------------------------------
    "_SMOKER3": _cat(na=(9,)),  # daily / some days / former / never
    "STOPSMK2": _cat(na=(7, 9)),  # current smokers only (skip pattern)
    "LASTSMK2": _cat(na=(77, 99)),  # former smokers only (skip pattern)
    "USENOW3": _cat(na=(7, 9)),
    # --- Alcohol --------------------------------------------------------------
    # These two keep the raw CDC scale in the course file (100 = 1 per day).
    "DROCDY3_": _num(valid=(0, 899), log=True),  # 900 = don't know / missing
    "_DRNKWEK": _num(valid=(0, 98999), log=True),  # 99900 = don't know / missing
    "_RFBING5": _cat(na=(9,)),
    "_RFDRHV5": _cat(na=(9,)),
    # --- Fruits and vegetables ------------------------------------------------
    "_FRUTSUM": _num(valid=(0, 999.98), log=True),  # decimals applied: times per day
    "_VEGESUM": _num(valid=(0, 999.98), log=True),
    "_FRTLT1": _cat(na=(9,)),
    "_VEGLT1": _cat(na=(9,)),
    # --- Physical activity ----------------------------------------------------
    "_TOTINDA": _cat(na=(9,)),
    "_PACAT1": _cat(na=(9,)),
    "_PASTRNG": _cat(na=(9,)),
    "PA1MIN_": _num(valid=(0, np.inf), log=True),
    "PA1VIGM_": _num(valid=(0, np.inf), log=True),
    "STRFREQ_": _num(valid=(0, 98), log=True),  # decimals applied; 99 = DK code
    # --- Arthritis burden -----------------------------------------------------
    "_LMTACT1": _cat(na=(9,)),
    "_LMTWRK1": _cat(na=(9,)),
    "_LMTSCL1": _cat(na=(9,)),
    "JOINPAIN": _num(na=(77, 99), valid=(0, 10)),
    # --- Seatbelt, immunization, HIV testing ---------------------------------
    "SEATBELT": _cat(na=(7, 9)),  # 8 = never drives is a real answer
    "FLUSHOT6": _cat(na=(7, 9)),
    "PNEUVAC3": _cat(na=(7, 9)),
    "HIVTST6": _cat(na=(7, 9)),
}
