# Introduction

![img](leo_rsos.png) [Low earth orbit](https://en.wikipedia.org/wiki/Low_Earth_orbit) (LEO) is a crowded place and the number of resident space objects (RSOs) entering LEO is rapidly escalating. With this population boom also comes the inevitable increase in close encounters (conjunctions) between objects. The heart of the space situational awareness (SSA) problem is to predict where RSOs are and where they are going; everything from a spec of paint to the International Space Station.

The current state of the art methods for orbit prediction are physics based models which require extremely accurate knowledge of the object's trajectory, the environment in which it operates, and the intent of the object to maneuver. In practice we do not have access to this data. Trajectories are measured infrequently from noisy ground based radar systems, our understanding of space weather and atmospheric density is limited, and satellite operators are not keen on sharing their plans to maneuver.

This project aims to be an experimental lab and playground for using ML to improve SSA and provides an end-to-end pipeline to:

-   ETL orbit data about RSOs in LEO from the USSTRATCOM Space Track API.
-   Make orbit predictions based on a physical model.
-   Train and use machine learning models to learn the error in physics based orbit prediction models.
-   Quickly perform a temporal-spatial search of orbit predictions to identify conjunctions based on parameterized queries.
-   Visualize when and where conjunctions may occur.


# Components

The SSA solution is currently composed of two components.


## Orbit Prediction

The [orbital prediction component](orbit_prediction/README.md) combines physics and machine learning models to predict the future path of RSOs.


## Conjunction search

The [conjunction search component](conjunction_search/README.md) combs through future orbit predictions to determine when and where two RSOs may come close to each other based on user provided search parameters.

# Live Demo

The SSA solution demo can be accessed at https://spaceorbits.net/


# License

The IBM Space Situational Awareness solution is licensed under the Apache 2.0 license. Full license text is available at [LICENSE](LICENSE).
