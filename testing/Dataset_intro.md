## Introduction to Cover Type Dataset

The Forest Cover Type Dataset consists of forest observations for 30 x 30 meter cells. Independent variables were derived from data
originally obtained from US Geological Survey (USGS) and USFS data. This study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado.  These areas represent forests with minimal human-caused disturbances, so that existing forest cover types are more a result of ecological processes rather than forest management practices.

### Attribute Information

12 measures expanded to 54 columns: Wilderness Area and Soil Type are already one-hot-encoded. Total number of observations equals 581012.

| Name | Data Type | Measurement | Description |
|----------|-------------|------|------|
Elevation                             |  quantitative  | meters                      | Elevation in meters
Aspect                                |  quantitative  | azimuth                     | Aspect in degrees azimuth
Slope                                 |  quantitative  | degrees                     | Slope in degrees
Horizontal_Distance_To_Hydrology      |  quantitative  | meters                      | Horz Dist to nearest surface water features
Vertical_Distance_To_Hydrology        |  quantitative  | meters                      | Vert Dist to nearest surface water features
Horizontal_Distance_To_Roadways       |  quantitative  | meters                      | Horz Dist to nearest roadway
Hillshade_9am                         |  quantitative  | 0 to 255 index              | Hillshade index at 9am, summer solstice
Hillshade_Noon                        |  quantitative  | 0 to 255 index              | Hillshade index at noon, summer soltice
Hillshade_3pm                         |  quantitative  | 0 to 255 index              | Hillshade index at 3pm, summer solstice
Horizontal_Distance_To_Fire_Points    |  quantitative  | meters                      | Horz Dist to nearest wildfire ignition points
Wilderness_Area (4 binary columns)    |  qualitative   | 0 (absence) or 1 (presence) | Wilderness area designation
Soil_Type (40 binary columns)         |  qualitative   | 0 (absence) or 1 (presence) | Soil Type designation


Based on attributes we predict the cover type for each observation.

### Forest Cover Type Classes:

0. Spruce/Fir
1. Lodgepole Pine
2. Ponderosa Pine
3. Cottonwood/Willow
4. Aspen
5. Douglas-fir
6. Krummholz

You can find more information here: https://archive.ics.uci.edu/ml/datasets/covertype

The Data is great for learning and experimenting purposes for many reasons:
 - it has many instances (over 500k)
 - it is numerical and clean (no missing values)
 - it is multiclass (7 classes)
 - it is highly imbalanced (2 most common classes ~85% in total, least represented class ~0,5%)
 - it can be downloaded with a single line of code: `sklearn.dataset.fetch_covtype()`

 



## Class Imbalance in Multiclass Classification

Imbalanced classification are the predictions where the distribution of instances across classes is not equal. This creates a challenge in Machine Learning. Many traditional algorithms are less effective when dealing with disproportion in class representation. The biggest problem is predicting the least represented classes. Chosing the proper metric to evaluate models can also be confusing.

Possible ways of dealing with the challenges:
 - using proper evaluation metrics
 - artificial oversampling of the least represented instances. This should be performed only on the train set.

In this report, we show various types of metrics' averages. We'll use f1 for evaluating trained models.




## Evaluation Metrics

Placeholder...

