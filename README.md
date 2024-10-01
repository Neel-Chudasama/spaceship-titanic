# Spaceship Titanic 

Hello ! 

![image](https://github.com/user-attachments/assets/99708a45-2365-4916-82be-ffc6794feab0)

**Title**  
Predicting Spaceship Titanic Passenger's Transport with Machine Learning

Welcome to the year 2912, where your data science skills are needed to solve a cosmic mystery. We've received a transmission from four lightyears away and things aren't looking good.

The Spaceship Titanic was an interstellar passenger liner launched a month ago. With almost 13,000 passengers on board, the vessel set out on its maiden voyage transporting emigrants from our solar system to three newly habitable exoplanets orbiting nearby stars.

While rounding Alpha Centauri en route to its first destination—the torrid 55 Cancri E—the unwary Spaceship Titanic collided with a spacetime anomaly hidden within a dust cloud. Sadly, it met a similar fate as its namesake from 1000 years before. Though the ship stayed intact, almost half of the passengers were transported to an alternate dimension!

To help rescue crews and retrieve the lost passengers, I am challenged to predict which passengers were transported by the anomaly using records recovered from the spaceship’s damaged computer system.

**Objective**   
The objective of this project is to build a machine learning model that can accurately predict passenger Transport based on various features such as age, gender, class, fare, etc.

In this notebook, I will conduct a Binary Classification on the Titanic Spaceship Dataset (which can be found here: https://www.kaggle.com/competitions/spaceship-titanic/data) to predict whether a person will be transported to an alternative dimension or not.

**Conclusion**   
There were a few very few useful features in the dataset which helped the model with its predictions.

With the data given, I was able to glean out new information and features that I feel were very useful for the model in its predictions.

I used different classifiers machine learning techniques for predictions and chose the most optimal one for further exploration.

Then I have compared all the preddictions given by different classifier models and selected the most optimal model.

To reduce overfitting from the model I conducted Hyper-Parameter Tunning on my highest performing model - xgboost models.

Finally, I used the most optimal model found through my gridsearch to make predictions on the test dataset.
