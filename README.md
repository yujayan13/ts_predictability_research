# Understanding the predictability of the U.S. Seasonal Surface Temperature with Convolutional Neural Networks trained on CESM2 LENS

## Abstract
To better understand the potential predictability of the seasonal surface temperature (TS) over the United States, we utilize Convolutional Neural Network (CNN) models trained on the Community Earth System Model Version 2 Large Ensemble (CESM2 LENS). We use lagged sea surface temperature (SST) over the region in the tropical Pacific containing the information of the El Niño Southern Oscillation (ENSO) as input for the CNN models. ENSO is the primary driver of variability in seasonal United States surface temperatures (TSUS) and utilizing CNN models allows for spatiotemporal aspects of ENSO to be analyzed to make seasonal TSUS predictions. For predicting TSUS, the CNN models show significantly improved skill over standard statistical multilinear regression (MLR) models throughout most regions in the US over a one to six-month lead time. We also use the CNN models to predict seasonal TSUS during extreme ENSO events. For these events, the CNN models better predict the effects on seasonal TSUS than the MLR models, suggesting that the CNN models are able to capture the ENSO-TSUS teleconnection more effectively. To better understand the source of predictability, a heatmap analysis is conducted. It demonstrates that the CNN models utilize spatial features of ENSO, rather than solely the magnitude of the ENSO events, indicating that increased predictability of seasonal TSUS can be derived from analyzing spatial variation in ENSO events. The proposed CNN model demonstrates a promising improvement in prediction skill compared to existing methods, suggesting a potential path forward for improving seasonal TSUS forecasts.

## Requirements
Python (python3 version 3.9.13+)<br/>
Tensorflow (version 2.12+)<br/>
Cartopy (version 0.22+)
