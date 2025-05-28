# Improving the Predictability of the US Seasonal Surface Temperature With Convolutional Neural Networks Trained on CESM2 LENS
https://doi.org/10.1029/2024JD040961

## Abstract
To better understand and improve the prediction of the seasonal surface temperature (TS) across the United States, we employed convolutional neural network (CNN) models trained on the Community Earth System Model Version 2 Large Ensemble (CESM2 LENS). We used lagged sea surface temperatures (SST) over the tropical Pacific region, containing the information of the El Niño Southern Oscillation (ENSO), as input for the CNN models. ENSO is the principal driver of variability in seasonal US surface temperatures (TSUS) and employing CNN models allows for spatiotemporal aspects of ENSO to be analyzed to make seasonal TSUS predictions. For predicting TSUS, the CNN models exhibited significantly improved skill over standard statistical multilinear regression (MLR) models and dynamical forecasts across most regions in the US, for lead times ranging from 1 to 6 months. Furthermore, we employed the CNN models to predict seasonal TSUS during extreme ENSO events. For these events, the CNN models outperformed the MLR models in predicting the effects on seasonal TSUS, suggesting that the CNN models are able to capture the ENSO-TSUS teleconnection more effectively. Results from a heatmap analysis demonstrate that the CNN models utilize spatial features of ENSO rather than solely the magnitude of the ENSO events, indicating that the improved skill of seasonal TSUS is due to analyzing spatial variation in ENSO events. The proposed CNN model demonstrates a promising improvement in prediction skill compared to existing methods, suggesting a potential path forward for enhancing TSUS forecast skill from subseasonal to seasonal timescales.

## Requirements
Python (python3 version 3.9.13+)<br/>
Tensorflow (version 2.12+)<br/>
Cartopy (version 0.22+)
