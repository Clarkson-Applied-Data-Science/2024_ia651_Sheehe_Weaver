# Machine Learning Final Project
2024_ia651_Sheehe_Weaver

IA 651: Machine Learning

Instructor: Professor Michael Gilbert

Final Project

Kelsey Sheehe & Sarah Weaver

## Project Overview

[Project Code (.ipynb file)](IA651_Final_Project.ipynb)

### Importance

The goal for this project is to create a formula to predict a team's goal_differential (goals - goals_conceded) using the selected data set (illustrated below). Predicting a team's goal differential can be key for being able to rank teams and make predictions on team's success in the post season. Being able to out-score opponents is largely impactful towards winning games and being a well ranked team. 

### Process

To tackle this project our team used a variety of machine learning techniques learned though out this course. To start we evaluated and examined out selected data. Multiple figures were created to better understand each variable and their relationship with others. The Exploratory Data Analysis was key in determining what processes we should try on our data. Given high correlation for some of the features we opted to perform Principal Component Analysis (PCA) to attempt to reduce the number number of features and determine what features are most important to our model. 

(Add more about Model Fitting, Validation/Metrics, Model Fit process)

## Table of Contents

1. [Dataset Link & Data Overview](#dataset-link--data-overview)
2. [Exploratory Data Analysis](#exploratory-data-analysis)
3. [Principal Component Analysis](#principal-component-analysis)

## Dataset Link & Data Overview

[National Women’s Soccer League Team Statistics](https://data.scorenetwork.org/soccer/nwsl-team-stats.html#data)

The data for this project has been sourced from the [Score Sports Data Repository](https://data.scorenetwork.org/) and focuses on data collected from 2016 to 2022 (excluding 2020 due to cancellation of season because of COVID) on each NWSL team during their regular season. [NWSL](https://www.nwslsoccer.com/) is the National Women's Soccer League in the United States, founded in 2012, and currently hosts 14 different teams. 

The dataset utilized has the following parameters:

|Variable |Description|
|---------|-----------|
|team_name | Name of NWSL team|
|season | Regular season year of team's statistic|
|games_played | Number of games team played in season|
|goal_differential | Goals scored - goals conceded|
|goals | Number of goals scores|
|goals_conceded | Number of goals conceded|
|cross_accuracy | Percent of crosses that were successful|
|goal_conversion_pct | Percent of shots scored|
|pass_pct | Pass accuracy|
|pass_pct_opposition_half | Pass accuracy in opposition half|
|possession_pct | Percentage of overall ball possession the team had during the season|
|shot_accuracy | Percentage of shots on target|
|tackle_success_pct | Percent of successful tackles|

Note: Because goal_differential is equal to goals - goals_conceded, we will remove the goals and goals_conceded columns from our dataset.

Below is a sample of this dataset without any modifications:

|Team Name |season |games_played |goal_differential |goals |goals_conceded |cross_accuracy |goal_conversion_pct |pass_pct |pass_pct_opposition_half |possession_pct |shot_accuracy |tackle_success_pct| 
|---------|-----------|---------|-----------|---------|-----------|---------|-----------|---------|-----------|---------|-----------|---------|
|Boston Breakers |2016 |20 |-33 |14 |47 |25.57 |8.97 |67.38 |57.86 |47 |42.95 |77.42 |
|Boston Breakers |2017 |24 |-11 |24 |35 |23.70 |12.37 |72.53 |61.42 |48 |42.78 |73.49 |
|Chicago Red Stars |2016 |21 |3 |25 |22 |21.19 |11.79 |67.35 |57.74 |36 |48.58 |84.32 |
|Chicago Red Stars |2017 |25 |2 |33 |31 |21.08 |13.10 |69.23 |61.52 |47 |49.60 |71.29 |
|Chicago Red Stars |2018 |25 |8 |38 |30 |25.96 |13.67 |71.63 |64.55 |51 |45.68 |67.97 |


## Exploratory Data Analysis

![Image of correlation matrix](correlation_matrix_2.png)

## Principal Component Analysis

After performing PCA on the dataset and setting *n_components* equal to 0.8. Meaning we want the number of Principal Components that would describe at least 80% of the variance in the data. Using this value our model calculated that the top 4 Principal Components explain 90.98% of the variance in our data, as illustrated below in the image.

![Image of cumulative explained variance PCA](Cumulative_Explained_Variance_Image.png)

Below is a sample data frame using these 4 Principal Components (PCs):

|PC 1 |PC 2 |PC 3 |PC 4 |
|-----|-----|-----|-----|
|-3.299368 |0.018155 |-1.766622 |0.130626 |
|-1.065873 |-0.554368 |-0.079226 |0.044649 |
|-3.125800 |1.138610 |-0.690204 |-1.550228 |
|-1.187709 |0.103420 |1.240969 |-1.022356 |
|0.023972 |-0.000238 |0.591931 |1.025205 |
|... |... |... |... |

To better interpret this information, we created a single loading chart, as shown below for PC1 and PC2 (the two PCs that explain the largest amount of the variance):

![PCA Single Loading Chart](PCA_Single_Loading_Chart.png)

From this chart, we can see that pass_pct and pass_pct_opposition_half contribute to the first principal component, and shot_accuracy and goal_conversion_pct contribute to the second.

Using this new dataset and applying linear regression we calculated the weight and intercept for these 4 PCs. The following table displays the calculated results:

|   |PC 1 |PC 2 |PC 3 |PC 4 |
|---|-----|-----|-----|-----|
|Weight |5.59397384684628 |1.1208677444194222 |4.265529209544452 |0.12437191275278267 |
|Intercept |-2.443881915550918 |1.596366457623861 |3.982261158683513 |2.014217812686465 |

To validate and understand these results we calculated the Mean Squared Error and the R² Value using a train test split. With a train test split of 20:80 and a random state of 42, the following calculation were made:

**Training MSE:** 189.6294515647826

**Test MSE:** 231.39369800935484

**Training R Squared:** 3.6*10⁻⁵

**Test R Squared:** -0.007156

Unfortunately, we calculated a very high MSE and R Squared values that are very close to zero. This shows weakness in our model and encouraged us to look at other methods for this project. 