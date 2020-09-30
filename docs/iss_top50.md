# The Problem with Space Debris

On [September 22, 2020 at 5:19 PM EDT](https://blogs.nasa.gov/spacestation/2020/09/22/station-boosts-orbit-to-avoid-space-debris/) the astronauts aboard the International Space Station (ISS) fired the station's propulsion system for 150 seconds then they made their way to their [Soyuz MS-16 spacecraft](https://en.wikipedia.org/wiki/Soyuz_MS-16) in case they needed to make a last minute escape. The reason for the unplanned maneuver and the crew heading for the exits? A [piece of debris](https://spacenews.com/space-station-maneuvers-to-avoid-debris/) from a Japanese [H-2A](https://en.wikipedia.org/wiki/H-IIA) rocket body that launched in 2008 was predicted at the last minute to come within 1.39 kilometers of the ISS. ![img](images/leo_objects.png) *Objects in LEO ([Image Source](https://astriagraph.spacetech-ibm.com))*

The [U.S. military currently tracks](https://www.space-track.org/auth/login) around 15,000 anthropogenic space objects (ASOs) in low Earth orbit (LEO), which are objects that have an altitude between 300 - 2,000 km. These objects range in size from about a softball to the ISS and travel at speeds between 7-8 kilometers per second. Unfortunately we cannot reliably track objects smaller than 5-10 cm and there are an estimated 400,000 - 500,000 of those. While objects smaller than 5 cm might not sound too dangerous we need to remember that they are traveling at a rate 10 times faster than a bullet as we can see from the results of firing a 1 cm diameter aluminum sphere into a solid block of aluminum 8 cm thick. ![img](images/collision.png) *[Image Source](https://swfound.org/media/99971/wright-space-debris_situation.pdf)*


# The 50 Most Dangerous ASOs

A group of 11 teams of international researchers has recently been working to determine what are the most dangerous ASOs currently in Earth's orbit. Their determination is based on the likelihood the objects will collide with other objects thus spawning more space debris and possibly more collisions. ![img](images/top50.jpg) *50 Most Dangerous ASOs [Image Source](https://www.forbes.com/sites/jonathanocallaghan/2020/09/10/experts-reveal-the-50-most-dangerous-pieces-of-space-junk-orbiting-earth-right-now/#61f7c0397c21)*

Will any of these objects come close to the ISS in the coming days? We can use the [orbit prediction](https://github.com/IBM/spacetech-ssa/tree/master/orbit_prediction) and [conjunction search](https://github.com/IBM/spacetech-ssa/tree/master/conjunction_search) components of the [space situational awareness](https://github.com/IBM/spacetech-ssa) project to find out.


# Predicting where Space Objects will be in the Future

For objects we can track we still have the problem of determining where they will be in the future so we can take corrective action if need be. State-of-the-art methods for orbit prediction rely on physics-based models, which to be successful, require extremely accurate data of the ASO and the environment in which it operates. The trouble is, the location data we get about ASOs from terrestrial based sensors comes infrequently and is noisy and our understanding of phenomena like space weather and atmospheric density are in their nascent stages. Our approach is to improve orbit prediction using machine learning methods, not by using ML models to predict orbits, but to create models that learn when physical models get orbit prediction wrong. This methodology allows us to take advantage of the scientific underpinning of the physical models and reduces the search space of machine learning models we have to comb through. ![img](images/data_flow.png) Detailed instructions on installing all the requisite dependencies and the API/CLI for each component can be found [here](https://github.com/IBM/spacetech-ssa/blob/master/orbit_prediction/README.md) along with a [demo script](https://github.com/IBM/spacetech-ssa/blob/master/orbit_prediction/pipeline_demo.sh) that can run the pipeline end-to-end, but a brief outline of the steps are that we:

1.  ETL historic orbit data from the U.S. military for all the objects they track in LEO.
2.  Use a physics based orbital dynamics model and the historical orbit data to build a training dataset of the physics model's prediction errors.
3.  Train a basic machine learning model to estimate the physics model's prediction errors.
4.  Combine the physics and ML models into one orbit predictor that we use to predict the future location of the ISS and the top 50 most dangerous ASOs for the next 3 days. We do this by utilizing the `--norad_id_file` flag in the `pred_orbits` module and pass a text file that has the NORAD IDs for only the ISS and the dangerous APOs.


# Determining Close Approaches

Now that we have predictions on where the ISS and the top 50 most dangerous APOs are going to be over the next 3 days, we want to be able answer queries like:

-   What are the 10 ASOs that will come the closest to the ISS in the next 3 days?
-   For the same time period, how which ASOs will come within a radius of X to the ISS?

The conjunction (a fancy word for "close approach") search [service](https://github.com/IBM/spacetech-ssa/tree/master/conjunction_search) can answer just these kinds of questions for us. A demo that offers a UI and interactive 3-dimensional visualization of the results can be found [here](https://spaceorbits.net).

|         |            |                       |                              |
|-------- |----------- |---------------------- |----------------------------- |
| NORAD ID | Object Name | Conjunction Time Start | Conjunction Distance (meters) |

~~----------~~--------------~~-------------------------~~--------------------------&#x2013;&#x2014;|

| 27386 | ENVISAT      | 10-01-2020 03:15 PM UTC | 369,737.35 |
| 12092 | SL-8 R/B     | 10-01-2020 10:55 PM UTC | 375,925.37 |
| 31793 | SL-16 R/B    | 09-30-2020 02:25 AM UTC | 423,383.99 |
| 28353 | SL-16 R/B    | 10-01-2020 01:35 PM UTC | 455,599.06 |
| 17590 | SL-16 R/B    | 10-02-2020 06:05 PM UTC | 558,856.18 |
| 27006 | SL-16 R/B    | 09-30-2020 05:05 AM UTC | 583,697.49 |
| 23405 | SL-16 R/B    | 09-30-2020 05:45 AM UTC | 585,471.95 |
| 22566 | SL-16 R/B    | 10-01-2020 03:15 PM UTC | 592,660.31 |
| 23704 | COSMOS 2322  | 09-30-2020 10:55 AM UTC | 600,219.08 |
| 10693 | SL-8 R/B     | 09-30-2020 08:45 PM UTC | 623,340.42 |
| 12092 | SL-8 R/B     | 10-01-2020 10:35 AM UTC | 624,199.21 |
| 17590 | SL-16 R/B    | 10-01-2020 12:55 PM UTC | 630,876.18 |
| 23180 | SL-8 R/B     | 10-01-2020 06:35 AM UTC | 651,200.03 |
| 26070 | SL-16 R/B    | 09-30-2020 08:35 PM UTC | 678,455.24 |
| 22803 | SL-16 R/B    | 09-30-2020 08:35 PM UTC | 684,910.18 |
| 25400 | SL-16 R/B    | 10-01-2020 01:05 AM UTC | 693,473.56 |
| 16182 | SL-16 R/B    | 10-01-2020 10:45 PM UTC | 700,430.99 |
| 23705 | SL-16 R/B    | 10-01-2020 12:55 AM UTC | 712,241.47 |
| 10693 | SL-8 R/B     | 09-30-2020 07:05 PM UTC | 736,288.17 |
| 23405 | SL-16 R/B    | 09-30-2020 11:35 PM UTC | 739,164.32 |
| 17590 | SL-16 R/B    | 09-30-2020 12:15 AM UTC | 741,546.11 |
| 27601 | H-2A R/B     | 10-01-2020 03:05 PM UTC | 753,468.40 |
| 24277 | ADEOS        | 10-02-2020 09:15 AM UTC | 754,023.85 |
| 26070 | SL-16 R/B    | 09-30-2020 10:15 PM UTC | 757,447.65 |
| 22285 | SL-16 R/B    | 10-02-2020 12:25 AM UTC | 767,046.32 |
| 23180 | SL-8 R/B     | 10-02-2020 10:25 AM UTC | 772,569.95 |
| 25861 | SL-16 R/B    | 10-02-2020 12:25 AM UTC | 775,389.73 |
| 23405 | SL-16 R/B    | 10-02-2020 01:45 PM UTC | 777,714.67 |
| 16292 | SL-8 R/B     | 10-01-2020 04:45 PM UTC | 778,000.92 |
| 23405 | SL-16 R/B    | 10-02-2020 12:05 PM UTC | 783,861.55 |
| 23405 | SL-16 R/B    | 10-02-2020 10:25 AM UTC | 785,631.62 |
| 31793 | SL-16 R/B    | 10-02-2020 05:45 AM UTC | 791,674.27 |
| 17974 | SL-16 R/B    | 09-30-2020 01:55 AM UTC | 792,994.72 |
| 17973 | COSMOS 1844  | 09-30-2020 03:05 PM UTC | 797,064.22 |
| 23405 | SL-16 R/B    | 10-01-2020 05:15 PM UTC | 801,276.04 |
| 15334 | SL-12 R/B(2) | 10-02-2020 05:15 PM UTC | 803,540.23 |
| 22285 | SL-16 R/B    | 10-02-2020 04:35 PM UTC | 805,199.61 |
| 23405 | SL-16 R/B    | 10-02-2020 08:15 AM UTC | 806,186.78 |
| 24298 | SL-16 R/B    | 09-30-2020 10:55 AM UTC | 810,292.65 |
| 23180 | SL-8 R/B     | 10-02-2020 10:45 PM UTC | 815,350.53 |
| 10693 | SL-8 R/B     | 10-02-2020 10:15 AM UTC | 818,049.64 |
| 23704 | COSMOS 2322  | 10-01-2020 11:25 PM UTC | 844,900.70 |
| 16292 | SL-8 R/B     | 09-30-2020 03:15 PM UTC | 853,125.25 |
| 27387 | ARIANE 5 R/B | 10-02-2020 08:05 AM UTC | 853,645.97 |
| 17590 | SL-16 R/B    | 09-30-2020 06:55 PM UTC | 860,686.26 |
| 20624 | COSMOS 2082  | 09-30-2020 02:35 AM UTC | 868,331.93 |
| 19650 | SL-16 R/B    | 10-02-2020 01:25 PM UTC | 868,386.54 |
| 16292 | SL-8 R/B     | 10-01-2020 04:55 PM UTC | 877,493.73 |
| 10693 | SL-8 R/B     | 09-30-2020 03:25 AM UTC | 881,547.31 |
| 27387 | ARIANE 5 R/B | 10-02-2020 12:35 PM UTC | 883,244.20 |
