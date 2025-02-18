import numpy as np

PARAM_LABELS = {
    "ACC": "Accessory",
    "ABP": "Absolute Peak Pos in Laser*2",
    "ADC": "External Analog Signals",
    "ADT": "Additional Data Treatment",
    "AG2": "Actual Signal Gain Channel 2",
    "AN1": "Analog Signal 1",
    "AN2": "Analog Signal 2",
    "APF": "Apodization Function",
    "APR": "ATR Pressure",
    "APT": "Aperture Setting",
    "AQM": "Acquisition Mode",
    "ARG": "Actual Reference Gain",
    "ARS": "Number of Reference Scans",
    "ASG": "Actual Signal Gain",
    "ASS": "Number of Sample Scans",
    "BBW": "Number of Bad Backward Scans",
    "BFW": "Number of Bad Forward Scans",
    "BLD": "Building",
    "BMS": "Beamsplitter",
    "CAM": "Coaddition Mode",
    "CFE": "Low Intensity Power Mode with DTGS",
    "CHN": "Measurement Channel",
    "CNM": "Operator Name",
    "COR": "Correlation Test Mode",
    "CPG": "Character Encoding Code Page",
    "CPY": "Company",
    "CRR": "Correlation Rejection Reason",
    "CSF": "Y Scaling Factor",
    "DAQ": "Data Acquisition Status",
    "DAT": "Date of Measurement",
    "DEL": "Delay Before Measurement",
    "DLY": "Stabilization Delay",
    "DPF": "Data Point Format",
    "DPM": "Department",
    "DTC": "Detector",
    "DUR": "Duration (sec)",
    "DXU": "X Units",
    "DYU": "Y Units",
    "EXP": "Experiment",
    "FOC": "Focal Length",
    "FXV": "First X Value",
    "GBW": "Number of Good Backward Scans",
    "GFW": "Number of Good Forward Scans",
    "HFF": "Digital Filter High Folding Limit",
    "HFL": "High Folding Limit",
    "HFQ": "End Frequency Limit for File",
    "HFW": "Wanted High Freq Limit",
    "HPF": "High Pass Filter",
    "HUM": "Relative Humidity Interferometer",
    "INS": "Instrument Type",
    "IST": "Instrument Status",
    "LCT": "Location",
    "LFF": "Digital Filter Low Folding Limit",
    "LFL": "Low Folding Limit",
    "LFQ": "Start Frequency Limit for File",
    "LFW": "Wanted Low Freq Limit",
    "LPF": "Low Pass Filter",
    "LPV": "Variable Low Pass Filter (cm-1)",
    "LWN": "Laser Wavenumber",
    "LXV": "Last X Value",
    "MNY": "Y Minimum",
    "MVD": "Max Velocity Deviation",
    "MXY": "Y Maximum",
    "NFL": "Nominal FW Peak Pos in Points",
    "NLA": "NL Alpha",
    "NLB": "NL Beta",
    "NLI": "Nonlinearity Correction",
    "NPT": "Number of Data Points",
    "NSN": "Scan Number",
    "NSR": "Number of Background Scans",
    "NSS": "Number of Sample Scans",
    "OPF": "Optical Filter Setting",
    "P2A": "Peak Amplitude Channel 2",
    "P2K": "Backward Peak Location Channel 2",
    "P2L": "Peak Location Channel 2",
    "P2R": "Backward Peak Amplitude Channel 2",
    "PGN": "Preamplifier Gain",
    "PGR": "Reference Preamplifier Gain",
    "PHR": "Phase Resolution",
    "PHZ": "Phase Correction Mode",
    "PKA": "Peak Amplitude",
    "PKL": "Peak Location",
    "PLF": "Result Spectrum Type",
    "PRA": "Backward Peak Amplitude",
    "PRL": "Backward Peak Location",
    "PRS": "Pressure Interferometer (hPa)",
    "RCH": "Reference Measurement Channel",
    "RDX": "Extended Ready Check",
    "RDY": "Ready Check",
    "RES": "Resolution (cm-1)",
    "RG2": "Signal Gain, Background 2nd Channel",
    "RGN": "Reference Signal Gain",
    "RSN": "Running Sample Number",
    "SFM": "Sample Form",
    "SG2": "Signal Gain, Sample 2nd Channel",
    "SGN": "Sample Signal Gain",
    "SNM": "Sample Name",
    "SON": "External Sync",
    "SOT": "Sample Scans or Time",
    "SPO": "Sample Number",
    "SPZ": "Stored Phase Mode",
    "SRC": "Source",
    "SRN": "Instrument Serial Number",
    "SRT": "Start Time (sec)",
    "SSM": "Sample Spacing Multiplier",
    "SSP": "Sample Spacing Divisor",
    "STR": "Scans or Time (Reference)",
    "TCL": "Command Line for Additional Data Treatment",
    "TDL": "To Do List",
    "TIM": "Time of Measurement",
    "TPX": "Total Points X",
    "TSC": "Scanner Temperature",
    "UID": "Universally Unique Identifier",
    "VEL": "Scanner Velocity",
    "VSN": "Firmware Version",
    "WAS": "Tr.Rec. Slices",
    "WDV": "Transient Recorder",
    "WIB": "Tr.Rec.Input Range 2nd channel",
    "WIR": "Tr.Rec.Input Range",
    "WPD": "Tr.Rec. Stab. Delay after Stepping",
    "WRC": "Tr.Rec. Repeat Count",
    "WSS": "Tr.Rec. Sampling Source",
    "WTD": "Tr.Rec. trigger Delay in points",
    "WTR": "Tr.Rec. Resolution",
    "WXD": "Tr.Rec. Experiment Delay",
    "WXP": "Tr.Rec. Trigger Mode",
    "XPP": "Experiment Path",
    "XSM": "Xs Sampling Mode",
    "ZFF": "Zero Filling Factor",
}


CODE_0 = {
    0: "",
    1: "Real Part of Complex Data",
    2: "Imaginary Part of Complex Data",
    3: "",  # Amplitude - leave blank because it is redundant when forming a label
}


CODE_1 = {
    0: "",
    1: "Sample",
    2: "Reference",
    3: "",  # Ratioed - leave blank because it is redundant when forming a label
}


CODE_2 = {
    0: "",
    1: "Data Status Parameters",
    2: "Instrument Status Parameters",
    3: "Acquisition Parameters",
    4: "Fourier Transform Parameters",
    5: "Plot and Display Parameters",
    6: "Optical Parameters",
    7: "GC Parameters",
    8: "Library Search Parameters",
    9: "Communication Parameters",
    10: "Sample Origin Parameters",
    11: "Lab and Process Parameters",
}


CODE_3 = {
    0: "",
    1: "Spectrum",
    2: "Interferogram",
    3: "Phase",
    4: "Absorbance",
    5: "Transmittance",
    6: "Kubelka-Munk",
    7: "Trace (Intensity over time)",
    8: "gc File, Series of Interferograms",
    9: "gc File, Series of Spectra",
    10: "Raman",
    11: "Emisson",
    12: "Reflectance",
    13: "Directory",
    14: "Power",
    15: "log Reflectance",
    16: "ATR",
    17: "Photoacoustic",
    18: "Result of Arithmatics, looks like Transmittance",
    19: "Result of Arithmatics, looks like Absorbance",
    33: "2-Channel",
    34: "Interferogram 2-Channel",
    36: "Absorbance 2-Channel",
    65: "3-Channel",
    97: "4-Channel",
}


CODE_4 = {
    0: "",
    1: "First Derivative",
    2: "Second Derivative",
    3: "n-th Derivative",
}


CODE_5 = {
    0: "",
    1: "Compound Information",
    2: "(Series)",
    3: "Molecular Structure",
    4: "Macro",
    5: "File Log",
}

# care must be taken when using 3-letter keys to avoid cross contaminating with 3-char parameter keys
CODE_3_ABR = {
    0: "",
    1: "",  # sc - single channel  leave blank because it is redundant when paired with sm, rf
    2: "ig",
    3: "ph",
    4: "a",
    5: "t",
    6: "km",
    7: "tr",
    8: "gcig",
    9: "gcsc",
    10: "ra",
    11: "e",
    12: "r",
    13: "dir",
    14: "pw",
    15: "logr",
    16: "atr",
    17: "pas",
    18: "arit",
    19: "aria",
    33: "_2ch",  # sm/rf
    34: "ig_2ch",  # sm/rf
    36: "a_2ch",
    65: "_3ch",  # sm/rf
    97: "_4ch",  # sm/rf
}


TYPE_CODE_LABELS = [CODE_0, CODE_1, CODE_2, CODE_3, CODE_4, CODE_5]


STRUCT_3D_INFO_BLOCK = [
    {"key": "nss", "fmt": "i", "dtype": np.int32},
    {"key": "nsr", "fmt": "i", "dtype": np.int32},
    {"key": "nsn", "fmt": "i", "dtype": np.int32},
    {"key": "npt", "fmt": "i", "dtype": np.int32},
    {"key": "gfw", "fmt": "i", "dtype": np.int32},
    {"key": "gbw", "fmt": "i", "dtype": np.int32},
    {"key": "bfw", "fmt": "i", "dtype": np.int32},
    {"key": "bbw", "fmt": "i", "dtype": np.int32},
    {"key": "hfl", "fmt": "d", "dtype": np.float64},
    {"key": "lfl", "fmt": "d", "dtype": np.float64},
    {"key": "hff", "fmt": "d", "dtype": np.float64},
    {"key": "lff", "fmt": "d", "dtype": np.float64},
    {"key": "filter_size", "fmt": "i", "dtype": np.int32},
    {"key": "filter_type", "fmt": "i", "dtype": np.int32},
    {"key": "fxv", "fmt": "d", "dtype": np.float64},
    {"key": "lxv", "fmt": "d", "dtype": np.float64},
    {"key": "mny", "fmt": "d", "dtype": np.float64},
    {"key": "mxy", "fmt": "d", "dtype": np.float64},
    {"key": "csf", "fmt": "d", "dtype": np.float64},
    {"key": "pka", "fmt": "d", "dtype": np.float64},
    {"key": "pra", "fmt": "d", "dtype": np.float64},
    {"key": "pkl", "fmt": "i", "dtype": np.int32},
    {"key": "prl", "fmt": "i", "dtype": np.int32},
    {"key": "srt", "fmt": "d", "dtype": np.float64},
    {"key": "ert", "fmt": "d", "dtype": np.float64},
]


Y_LABELS = {
    "sm": "Sample Spectrum",
    "rf": "Reference Spectrum",
    "igsm": "Sample Interferogram",
    "igrf": "Reference Interferogram",
    "phsm": "Sample Phase",
    "phrf": "Reference Phase",
    "a": "Absorbance",
    "t": "Transmittance",
    "r": "Reflectance",
    "km": "Kubelka-Munk",
    "tr": "Trace (Intensity over Time)",
    "gcig": "gc File (Series of Interferograms)",
    "gcsc": "gc File (Series of Spectra)",
    "ra": "Raman",
    "e": "Emission",
    "dir": "Directory",
    "p": "Power",
    "logr": "log(Reflectance)",
    "atr": "ATR",
    "pas": "Photoacoustic",
}


XUN_LABELS = {
    "wl": "Wavelength",
    "wn": "Wavenumber",
    "f": "Frequency",
    "pnt": "Points",
    "min": "Minutes",
    "logwn": "Log Wavenumber",
}
