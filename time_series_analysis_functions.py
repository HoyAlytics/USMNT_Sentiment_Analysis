# %%
# to install kats
# conda install -c conda-forge kats=0.2.0
# Or see env.yaml

from statsmodels.tsa.seasonal import seasonal_decompose
from kats.detectors.cusum_detection import CUSUMDetector
import matplotlib.pyplot as plt

# %%
def change_detect(df, historical, scan, stepper, p_val = 0.01):

    # Get historical window (past)
    historical_window = historical

    # Get scan window (compare)
    scan_window = scan

    # Get sliding step
    step = stepper

    # Initialize changepoints
    changepoints = []

    # Create sliding window
    n = len(df)
    for end_idx in range(historical_window + scan_window, n, step):
        tsd = df[end_idx - (historical_window + scan_window) : end_idx]
        changepoints += CUSUMDetector(tsd).detector(interest_window=[historical_window, historical_window + scan_window], threshold = p_val)

    # Plot the results
    detector = CUSUMDetector(df) # we are not really using this detector
    detector.detector() # this call to detector is not being used for anything
    detector.plot(changepoints)
    plt.xticks(rotation=45)
    plt.show()

    return changepoints



# %%
