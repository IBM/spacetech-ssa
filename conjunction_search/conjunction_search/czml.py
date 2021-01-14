# Copyright 2020 IBM Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import pandas as pd
import datetime as dt


class CZMLBuilder:
    """Class for building CZML JSON documents to be used to
    drive a Cesium UI for displaying the results of a conjunction
    search.

    :param czml_df: A pandas DataFrame containing the orbit predictions
                    and related data for the ASOs that matched the
                    conjunction search query.
    :type czml_df: pandas.DataFrame
    """

    FORMAT_STR = '%Y-%m-%dT%H:%M:%S.%fZ'

    def __init__(self, czml_df):
        self.czml_df = czml_df
        # Find the row in the DataFrame that the conjunction search query
        # was run for
        self.query_row = self.czml_df[self.czml_df.conj_dists.isna()].iloc[0]
        self.pred_start_dt = self.query_row.pred_start_dt
        self.pred_end_dt = self.query_row.pred_end_dt
        # Calculate the interval in seconds that predictions are made
        time_interval = (self.pred_end_dt - self.pred_start_dt)
        pred_span_seconds = time_interval.total_seconds()
        num_pred_timesteps = self.query_row.orbit_preds.shape[0] - 1
        self.pred_interval = pred_span_seconds / num_pred_timesteps
        # Get strings of timestamps that are used multiple times
        self.pred_start_str = self._format_dt(self.pred_start_dt)
        self.pred_end_str = self._format_dt(self.pred_end_dt)
        self.pred_span = f'{self.pred_start_str}/{self.pred_end_str}'
        # Build a human readable result of the conjunction search
        self.conj_descs = self._build_conj_descs()

    def _format_dt(self, dt, format_str=FORMAT_STR):
        """Represent a datetime object as a string in the format
        specified in the CZML spec.

        :param dt: The datetime object to format as a strign
        :type dt: datetime.datetime

        :param format_str: The format string to use.  Defaults to the format
                           used in the CZML spec
        :type format_str: str

        :return: String representation of the date time object
        :rtype: str
        """
        return dt.strftime(format_str)

    def _build_conj_descs(self):
        """Creates a human readable DataFrame for describing
        the conjunctions between the query ASO and the search matching
        ASOs.
        """
        conj_descs = []
        cols = ['aso_id',
                'ASO',
                'Conjunction Window Start',
                'Predicted Conjunction Distance (m)']
        dist_col = cols[-1]
        pretty_ts_fmt = "%H:%M %m/%d/%y"
        for _, row in self.czml_df.iterrows():
            if row.aso_id == self.query_row.aso_id:
                continue
            conj_spans = [self._get_conj_span(ts, pretty_ts_fmt)
                          for ts in row.conj_timesteps]
            for d, (st, et) in zip(row.conj_dists, conj_spans):
                conj_desc = (row.aso_id, row.aso_name, st, int(d))
                conj_descs.append(conj_desc)
        conj_descs_df = pd.DataFrame(conj_descs, columns=cols)
        conj_descs_df.sort_values(by=dist_col,
                                  inplace=True)
        dists = conj_descs_df[dist_col].apply(lambda d: f'{d:,}')
        conj_descs_df[dist_col] = dists
        return conj_descs_df

    def _build_document_node(self, multiplier=10, doc_name=None):
        """Builds a node that tells Cesium what kind of CZML document
        it is getting.

        :param multiplier: The factor to speed up the animation by
        :type multiplier: float
        :param doc_name: The name of the document
        :type doc_name: str

        :return: A dictionary representing the CZML document node
        :rtype: dict
        """
        if not doc_name:
            doc_name = f'orbital_predictions_{self.pred_span}'

        document_node = {
            'id': 'document',
            'name': doc_name,
            'version': '1.0',
            'clock': {
                'interval': self.pred_span,
                'currentTime': self.pred_start_str,
                'multiplier': multiplier,
                'range': 'LOOP_STOP',
                'step': 'SYSTEM_CLOCK_MULTIPLIER'
            }
        }

        return document_node

    def _get_conj_span(self, timestep, format_str=FORMAT_STR):
        """Calculates the start and end timestamps of the
        prediction timestep.

        :param timestep: The prediction timestep to calculate the
                         start and end timestamps for
        :type timestep: float

        :return: The start and end timestamps for the prediction
                 timestep
        :rtype: (str, str)
        """
        elapsed_seconds = (self.pred_interval * timestep) - 5*60
        conj_start = self.pred_start_dt + dt.timedelta(seconds=elapsed_seconds)
        conj_end = conj_start + dt.timedelta(seconds=self.pred_interval)
        conj_start_str = self._format_dt(conj_start, format_str)
        conj_end_str = self._format_dt(conj_end, format_str)
        return conj_start_str, conj_end_str

    def _get_line_intervals(self, conj_spans):
        """Calculates the intervals for showing conjunction lines
        between ASOs in the Cesium UI

        :param conj_spans: A list of start and end timestamps for
                           when the conjunctions occur
        :type conj_spans: [(str, str)]

        :return: A list of dicts used to draw the conjunction lines
                 in the UI at the correct time intervals
        :rtype: [dict]
        """
        first_int = {
            'interval': f'0000-01-01T00:00:00Z/{conj_spans[0][0]}',
            'boolean': False
        }
        intervals = [first_int]
        prev_end = None
        for conj_start, conj_end in conj_spans:
            if prev_end:
                between_int = {
                    'interval': f'{prev_end}/{conj_start}',
                    'boolean': False
                }
                intervals.append(between_int)

            conj_int = {
                'interval': f'{conj_start}/{conj_end}',
                'boolean': True
            }
            intervals.append(conj_int)
            prev_end = conj_end
        last_int = {
            'interval': f'{conj_spans[-1][1]}/9999-12-31T24:00:00Z',
            'boolean': False
        }
        intervals.append(last_int)
        return intervals

    def _build_conj_node(self, row):
        """Builds a node to display a line connecting the two ASOs
        when the predicted conjunction will occur.

        :param row: The row from the orbital prediction DataFrame representing
                    the ASO.
        :type rows: pandas.Series

        :return: A dictionary representing the CZML conjunction  node
        :rtype: dict
        """
        conj_spans = [self._get_conj_span(ts) for ts in row.conj_timesteps]
        conj_node = {
            'id': f'conj_{row.aso_id}_to_{self.query_row.aso_id}',
            'name': f'{row.aso_name} to {self.query_row.aso_name}',
            'parent': 'conjunction_connections',
            'availability': [f'{st}/{et}' for (st, et) in conj_spans],
            'polyline': {
                'show': self._get_line_intervals(conj_spans),
                'width': 1,
                'material': {
                    'solidColor': {
                        'color': {
                            'rgba': [255, 0, 0, 255]
                        }
                    }
                },
                'arcType': 'NONE',
                'positions': {
                    'references': [f'{row.aso_id}#position',
                                   f'{self.query_row.aso_id}#position']
                }
            }

        }
        return conj_node

    def _get_conj_desc(self, row):
        """Builds an HTML table of every conjunction search
        match the given row has with the query ASO.

        :param row: The row from the orbital prediction DataFrame representing
                    the ASO.
        :type rows: pandas.Series

        :return: HTML table of conjunction search results
        :rtype: str
        """
        if row.aso_id == self.query_row.aso_id:
            desc_df = self.conj_descs
        else:
            desc_df = self.conj_descs[self.conj_descs.aso_id == row.aso_id]
        return desc_df.drop(columns=['aso_id']).to_html(index=False,
                                                        justify='center')

    def _build_aso_pred_node(self, row):
        """Builds a node representing the predicted orbit of a single ASO

        :param row: The row from the orbital prediction DataFrame representing
                    the ASO.
        :type rows: pandas.Series

        :return: A dictionary representing the CZML ASO node
        :rtype: dict
        """
        orbit_pred_intervals = []
        for idx, op in enumerate(row.orbit_preds.tolist()):
            op_interval = [idx*self.pred_interval] + op[:3]
            orbit_pred_intervals += op_interval
        if row.aso_id == self.query_row.aso_id:
            path_color = [63, 191, 63, 255]
        else:
            path_color = [127, 63, 191, 255]

        aso_node = {
            'id': row.aso_id,
            'name': row.aso_name,
            'availability': self.pred_span,
            'description': self._get_conj_desc(row),
            'billboard': {
                'eyeOffset': {
                    'cartesian': [0, 0, 0]
                },
                'horizontalOrigin': 'CENTER',
                'image': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAADJSURBVDhPnZHRDcMgEEMZjVEYpaNklIzSEfLfD4qNnXAJSFWfhO7w2Zc0Tf9QG2rXrEzSUeZLOGm47WoH95x3Hl3jEgilvDgsOQUTqsNl68ezEwn1vae6lceSEEYvvWNT/Rxc4CXQNGadho1NXoJ+9iaqc2xi2xbt23PJCDIB6TQjOC6Bho/sDy3fBQT8PrVhibU7yBFcEPaRxOoeTwbwByCOYf9VGp1BYI1BA+EeHhmfzKbBoJEQwn1yzUZtyspIQUha85MpkNIXB7GizqDEECsAAAAASUVORK5CYII=', # noqa E501
                'pixelOffset': {
                    'cartesian2': [0, 0]
                },
                'scale': 1.5,
                'show': True,
                'verticalOrigin': 'CENTER'
            },
            'label': {
                'fillColor': {
                    'rgba': path_color
                },
                'font': '11pt Lucida Console',
                'horizontalOrigin': 'LEFT',
                'outlineColor': {
                    'rgba': [0, 0, 0, 255]
                },
                'outlineWidth': 2,
                'pixelOffset': {
                    'cartesian2': [12, 0]
                },
                'show': True,
                'style': 'FILL_AND_OUTLINE',
                'text': row.aso_name,
                'verticalOrigin': 'CENTER'
            },
            'path': {
                'show': [
                    {
                        'interval': self.pred_span,
                        'boolean': True
                    }
                ],
                'width': 1,
                'material': {
                    'solidColor': {
                        'color': {
                            'rgba': path_color
                        }
                    }
                },
                'resolution': 120,
                'leadTime':  3600,
                'trailTime': 3600
            },
            'position': {
                'interpolationAlgorithm': 'LAGRANGE',
                'interpolationDegree': 5,
                'referenceFrame': 'INERTIAL',
                'epoch': self.pred_start_str,
                'cartesian': orbit_pred_intervals
            }
        }

        return aso_node

    def build_czml_doc(self, outfile=None):
        """Builds a CZML document describing the orbits of all of the ASOs
        that matched the conjunction search query.

        :param outfile: Optional file path to save the CZML document to
        :type outfile: str

        :return: If an `outfile` is passed returns None, otherwise returns a
                 a list of CZML nodes built from the `czml_df`.
        :rtype: None or [dict]
        """
        czml_doc = []
        doc_node = self._build_document_node()
        czml_doc.append(doc_node)
        conj_root_node = {
            'id': 'conjunction_connections',
            'name': 'Conjunctions'
        }
        conj_nodes = [conj_root_node]
        aso_nodes = []

        for _, row in self.czml_df.iterrows():
            # Only build conjunction nodes if the row
            # is not the conjunction search query ASO
            if row.aso_id != self.query_row.aso_id:
                conj_node = self._build_conj_node(row)
                conj_nodes.append(conj_node)
            aso_node = self._build_aso_pred_node(row)
            aso_nodes.append(aso_node)

        # Only add the conjunction root node if
        # there are conjunction search results
        if len(conj_nodes) > 1:
            czml_doc += conj_nodes
        czml_doc += aso_nodes

        if outfile:
            with open(outfile, 'w') as czml_file:
                json.dump(czml_doc, czml_file)
        else:
            return czml_doc
