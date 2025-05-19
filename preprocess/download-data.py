"""Download raw data from Zenodo."""

### Extract the capacity factor data for the wind and solar resources in the ISOs
### !! no interpolation. just extract the points within each ISOs
scenarios = ["rcp45hotter"]
startyear = {"historic": 2001, "future": 2020}
endyear = {"historic": 2020, "future": 2059}


def make_input_fpath(var, period, iy):
    """Generate the input file path for the given variable, period, and year."""
    return f"data/raw/{var}/{period}/{var}_gen_cf_{iy}.nc"


if __name__ == "__main__":
    for var in ["wind", "solar"]:
        for period in scenarios:
            if period == "historic":
                sy = startyear[period]
                ey = endyear[period]
            else:
                sy = startyear["future"]
                ey = endyear["future"]
            for iy in range(sy, ey + 1):
                fpath = make_input_fpath(var, period, iy)
                print("Downloading raw data file:", fpath)
                # TODO: Actually download it
