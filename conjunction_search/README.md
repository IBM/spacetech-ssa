# Introduction

Once we have made predictions on where RSOs in LEO are going to be in the future using the tools in the [orbit prediction package](../orbit_prediction/README.md), we need a way to quickly perform spatial-temporal search over the prediction space to determine if objects may come too close to each other. This package provides:

-   A [module](conjunction_search/conjunction_search.py) for using [k-d trees](https://en.wikipedia.org/wiki/K-d_tree) to quickly perform spatial-temporal search to answer questions like:
    -   What are the `k` RSOs that will come closest to the given RSO over the prediction window?
    -   For a given radius `r`, how many RSOs will come within distance `r` of the given RSO over the prediction window?
-   A [library](conjunction_search/czml.py) for building [CZML](https://github.com/AnalyticalGraphicsInc/czml-writer/wiki/CZML-Guide) JSON documents to display predicted orbits and conjunctions in a [CesiumJS](https://cesium.com/cesiumjs/) UI.
-   A [Flask](https://flask.palletsprojects.com/en/1.1.x/) application providing an HTTP API for serving CZML and JSON conjunction search results.
-   A CesiumJS UI for querying and visualizing RSO conjunctions. `r` of the given RSO over the prediction window?


# Development

A [Makefile](Makefile) is provided to automate the tasks of building and running the web service in a Docker container.


## Building and Running the Service

The docker container can be built with

```sh
make build
```

and the application can be run locally via

```sh
make run_dev
```

Then you should be able to access the conjunction search UI at <http://localhost:8080>.


## Environment Variables

The service expects that you have a file named `cos-vars.env` that sets environment variables that are used inside the docker container to either use test data or retrieve the needed pickle file from [this COS bucket](https://cloud.ibm.com/objectstorage/crn%3Av1%3Abluemix%3Apublic%3Acloud-object-storage%3Aglobal%3Aa%2F30fa06c653644d7ba6b9948af64b7ab3%3A204afe80-9e57-413b-904a-a98e978c92c1%3A%3A?bucket=ag-ssa-preds&bucketRegion=us-east&endpoint=s3.us-east.cloud-object-storage.appdomain.cloud&paneId=bucket_overview). You can create this file by

```sh
cp cos-vars.env.example cos-vars.env
```

and then supplying the values for the following environment variables:

-   `DEV`: If set to `true`, uses the test data otherwise pulls the data from COS
-   `COS_ENDPOINT`: The [endpoint](https://cloud.ibm.com/docs/cloud-object-storage?topic=cloud-object-storage-endpoints#endpoints-region) used by the client to access the bucket.
-   `COS_API_KEY_ID`: The API secret key to authenticate with.
-   `COS_INSTANCE_CRN`: The bucket instance CRN.
-   `COS_BUCKET`: The COS bucket that the CSV file is in.
-   `COS_FILENAME`: The filename of the pickle file in the `COS_BUCKET`.
