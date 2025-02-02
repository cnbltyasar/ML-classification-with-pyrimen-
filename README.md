# ML-classification-with-pyrimen-
This notebook is a complete machine‐learning pipeline that loads EEG (or similar multichannel) data from MATLAB files, preprocesses it (filtering and covariance estimation), “recenters” the covariance matrices via parallel transport, and then compares several classifiers applied in the “tangent space” of the covariance matrices. In other words, the code:

Loads and Preprocesses the Data:

It loads data from a MATLAB file (with variables such as adg_180, emre_180, adg_360, emre_360, and emre_plus_adg_540).
It plots some raw (or filtered) EEG signals to check the data.
It applies band‐pass filtering (using MNE’s filter functions) to isolate frequencies between 2 and 30 Hz.
Compute Covariance Matrices:

For a given “epoch” (e.g. a 1‑second segment), the notebook uses the pyriemann.estimation.Covariances transformer to compute a covariance matrix from the filtered data.
There are helper functions called get_conv_matrix, get_conv_matrix360, and get_conv_matrix540 that do the following:
Filter the raw data.
Extract a time window (epoch) based on given start time and duration.
Compute the covariance matrices from that epoch.
Compute the “mean” covariance (using the Riemannian mean via mean_riemann).
Create labels by concatenating a binary vector (with ones in specific segments) a different number of times (2, 4, or 6 times) to match the data’s expected label dimensions.
“Recenter” the covariance matrices by performing parallel transport from each covariance matrix to the mean (using an invsqrtm transformation). This is implemented in the function recenter_data, which internally calls the helper function parallel_transport_covariance_matrix.
Classification in Tangent Space:

The code uses the pyriemann.tangentspace.TangentSpace transformer to project the covariance matrices (which lie on the manifold of symmetric positive‐definite matrices) to a Euclidean space.
A number of classifiers are then wrapped in pipelines with the tangent space projection. The classifiers compared include:
Linear Discriminant Analysis (LDA)
Logistic Regression (LR)
Support Vector Classifier (SVC, with linear kernel)
Lasso (a linear model with L1 regularization)
Minimum Distance to Mean (MDM, which is intrinsic to the Riemannian approach)
Decision Tree Classifier (DTC)
K-Nearest Neighbors (KNN)
Random Forest Classifier (RFC)
For each classifier, a 10-fold cross‑validation is repeated 20 times (with different random shuffles), and the mean accuracy is computed. These average accuracies are stored in an accuracy matrix.
Plotting and Comparing Models:

After computing the average accuracies for each classifier, the notebook plots a scatter/line plot (with markers) of “Machine Learning Model vs Accuracy”.
Labels (e.g. “ldac”, “lrc”, “SVC”, etc.) are added as annotations so that one can see which classifier achieved which accuracy.
Similar classification experiments are then performed for different data sets (e.g. adg_180, emre_180, adg_360, emre_360, and emre_plus_adg_540) so that one can compare performance when the data (or the label configuration) is changed.
