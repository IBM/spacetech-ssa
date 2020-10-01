// Copyright 2020 IBM Corporation

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


var viewer = new Cesium.Viewer("cesiumContainer", {
  shouldAnimate: true,
});

function updateTextInput(id, val) {
    document.getElementById(id).value=val;
}

function hideShow(hideDivId, showDivId) {
    var hideDiv = document.getElementById(hideDivId);
    var showDiv = document.getElementById(showDivId);
    hideDiv.style.display = "none";
    showDiv.style.display = "block";
}

function showOrbits() {
    viewer.dataSources.removeAll();
    var CZMLURL = "/conjunction_search/";
    var asoId = document.getElementById("asoSelect").value;
    CZMLURL += asoId;
    CZMLURL += "?";
    if (document.getElementById("knnSearch").checked) {
        var kNNs = document.getElementById("numKNNs").value;
        CZMLURL += "k=";
        CZMLURL += kNNs;
    } else {
        var radius = document.getElementById("radius").value;
        CZMLURL += "radius=";
        CZMLURL += radius;
    }
    viewer.dataSources.add(
        Cesium.CzmlDataSource.load(CZMLURL)
    );
}
