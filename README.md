# Machine Learning Final Project
2024_ia651_Sheehe_Weaver

IA 651: Machine Learning

Instructor: Professor Michael Gilbert

Final Project

Kelsey Sheehe & Sarah Weaver

## Project Overview

### Importance

The goal for this project is to create a formula to predict a team's goal_differential (goals - goals_conceded) using the selected data set (illustrated below). Predicting a team's goal differential can be key for being able to rank teams and make predictions on team's success in the post season. Being able to out-score opponents is largely impactful towards winning games and being a well ranked team. 

### Process

To tackle this project our team used a variety of machine learning techniques learned though out this course. To start we evaluated and examined out selected data. Multiple figures were created to better understand each variable and their relationship with others. The Exploratory Data Analysis was key in determining what processes we should try on our data. Given high correlation for some of the features we opted to perform Principal Component Analysis (PCA) to attempt to reduce the number number of features and determine what features are most important to our model. 

(Add more about Model Fitting, Validation/Metrics, Model Fit process)

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

## Exploratory Data Analysis

![Image of correlation matrix](correlation_matrix_2.png)