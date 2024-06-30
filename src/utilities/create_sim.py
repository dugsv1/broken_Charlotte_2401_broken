import re
import numpy as np
import pandas as pd
from pathlib import Path


class Sim:

    regex_call_row = r'CALL\|([^|]+)\|(.*)(?=\|)'
    regex_custom_data_row = r'CUSTOM-DATA\|([^|]+)\|(.*)(?=\|)'
    regex_audit_data = r'CUSTOM-DATA\|([^|]+).*\n*((?:\n.*)+?)(?=\nCALL|\Z)'
    regex_on_scene = r'BENCHMARK\|OnScene\|([^|]+)\|(\d+(?:\.\d*)?)\|([^|]+)\|'
    regex_in_service = r'^BENCHMARK\|SceneTime\|([^|]+)\|([\d.]+)\|([^|]+)\|'
    regex_committed_hours = r'^BENCHMARK\|CommittedHours\|([^|]+)\|([\d.]+)\|([^|]+)\|'
    regex_first_arrival = r'^BENCHMARK\|FirstArrival\|([^|]+)\|([\d.]+)\|([^|]+)'
    regex_full_compliment = r'^BENCHMARK\|FullComplement\|([^|]+)\|([\d.]+)\|([^|]+)'
    regex_tt = r'^BENCHMARK\|TurnoutTime\|([^|]+)\|([\d.]+)\|([^|]+)'
    regex_route = r'^BENCHMARK\|Route\|([^|]+)\|([\d.]+)\|([^|]+)\|Route\|([0-9,.]+)(?=,)'
    regex_coords = r"Call\|CALL: .*\[(\-?\d+\.\d+,\-?\d+\.\d+)\]"

    def __init__(self, ResultsTextFile, c3s_xml=None):
        self.results_txt = ResultsTextFile
        self.analyze = self._analyze_text_file()
        self.xml_path = c3s_xml if c3s_xml is not None else None

    def _analyze_text_file(self):
        """
        Analyze text file and output it's size and shape.

        Args:
            None. This will automatically run on initiation.

        Returns:
           Number of columns, including the CALL and CUSTOM DATA pipes, within the incoming text file. This will be printed using _print_config_summary.
        """

        with open(self.results_txt, 'r') as file:
            required_lines = [None] * 4  # initialize list with None values
            for i, line in enumerate(file):
                if i < len(required_lines):
                    required_lines[i] = line
                else:
                    break  # stop reading the file once all required lines are found

        header_line, metadata_call_line, call_line, custom_data_line = required_lines
        # Check if the lines match the expected format

        if not re.match(r'^HEADER\|', header_line) or not re.match(r'^METADATA-Call\|', metadata_call_line) \
                or not re.match(r'^CALL\|', call_line) or not re.match(r'^CUSTOM-DATA\|', custom_data_line):
            raise ValueError("Invalid text file format.")

        # Extract header columns
        call_columns = call_line.split('|')
        custom_data_columns = custom_data_line.split('|')

        project_dict = {
            'HEADER: ': header_line,
            'call_columns_count': len(call_columns),
            'custom_data_columns_count': len(custom_data_columns),
        }

        self._print_config_summary(project_dict)
        return project_dict

    def _extract_model_number(self):
        with open(self.results_txt) as f:
            first_line = f.readline()
            pattern = re.compile(r'^HEADER\|(.*?)\|')
            match = pattern.search(first_line)

        assert match.group(
            1) != "", "Check imported text file, doesn't contain a model number so likely a RW Dataset"
        return match.group(1)

    def _get_matches(self):
        """
        This function defines the REGEX expression that grabs data from the text file from "CALL" row to "CALL" row.

        Args:
            None.

        Returns:
           Match iterator objects that are used later and then iterated upon.
        """
        ifile = open(self.results_txt, 'r')
        text = ifile.read()
        ifile.close()

        pattern = re.compile(
            r'CALL\|\d\d\d\d\d\d.*\n*((?:\n.*)+?)(?=\nCALL|\Z)', re.MULTILINE)
        matches = pattern.finditer(text)
        return matches

    def _process_match_for_calls(self, match):
        """
        This function processes each of the benchmark and call components and creates a list that is then appended into a dataframe.

        Args:
            match: Regex match. This is an iterator object returned from the _get_matches function.

        Returns:
           Returns the list containing call details as well as a list that contains calls that did not have certain benchmarks. This is the audit list.
           The return is a tuple, but the final output will be a single dataframe or a tuple of dataframes depending on the input boolean on the
           sim_results_to_df function.
        """

        big_string = match.group(0)
        # this pattern separates the call row from the big string that contained the entire data for one call
        pattern2 = re.compile(self.regex_call_row)
        call_row_iterator = pattern2.finditer(big_string)
        # this pattern separates the custom data from the big string that contained the entire data
        pattern3 = re.compile(self.regex_custom_data_row)
        custom_data_row_iterator = pattern3.finditer(big_string)
        # this pattern searches the big string for only the audit and benchmark data by searching from CUSTOM-DATA until but not including CALL
        pattern4 = re.compile(self.regex_audit_data)
        extra_data_iterator = pattern4.finditer(big_string, re.MULTILINE)
        # We need to caputre the call number from just the audit data selection
        pattern5 = re.compile(self.regex_on_scene)
        on_scene_iterator = pattern5.finditer(big_string, re.MULTILINE)
        # this captures the coordinates found in the audit|0| row
        pattern6 = re.compile(self.regex_coords)
        coordinates_iterator = pattern6.finditer(big_string, re.MULTILINE)
        # capture the first arriving unit and their response time
        pattern7 = re.compile(self.regex_first_arrival, re.MULTILINE)
        first_unit_iterator = pattern7.search(big_string)
        # capture the FULL COMPLIMENT after the last unit arrives
        pattern9 = re.compile(self.regex_full_compliment, re.MULTILINE)
        full_compliment_iterator = pattern9.finditer(big_string)

        # anything we want broken out into the main dataframe has to be appended as a single item into the temp list in a for loop below
        call_row = custom_data = call_num = cords = extra_data = first_unit = None
        first_arrival_response_time = full_compliment_unit = full_compliment_time = onscene_time = onscene_unit = None
        # have to iterate each object separately, tried to iterate as a tuple but they must be different lengths. Their components are being kept in temp_list.
        for row in call_row_iterator:
            call_row = row.group(0)
            call_num = row.group(1)
        for row_b in custom_data_row_iterator:
            custom_data = row_b.group(2)
        for row_c in on_scene_iterator:
            onscene_unit = row_c.group(1)
            onscene_time = row_c.group(2)
        for row_d in coordinates_iterator:
            cords = row_d.group(1)
        for row_e in extra_data_iterator:
            extra_data = row_e.group(2)
        # we have to use search instead of find iter, because it was grabbing the last arrival time. Search stops on first match so it cannot be iterated thru in a for loop
        if pattern7.search(big_string):
            first_unit = first_unit_iterator.group(1)
            first_arrival_response_time = first_unit_iterator.group(2)
            # first_unit = row_f.group(1)
        for row_h in full_compliment_iterator:
            full_compliment_unit = row_h.group(1)
            full_compliment_time = row_h.group(2)

        # appending to be added to main dataframe These are different sizes than the initiation dataframe analysis because our REGEX skipped over the first phrase
        # in each row which was "CALL" and "CUSTOM-DATA"
        call_column_count = self.analyze['call_columns_count']
        custom_data_column_count = self.analyze['custom_data_columns_count']
        total_columns = call_column_count + custom_data_column_count + 6

        t_list = [None] * total_columns
        call_row_items = call_row.split('|')
        for i, item in enumerate(call_row_items):
            t_list[i] = item

        custom_data_items = custom_data.split('|')
        for i, item in enumerate(custom_data_items):
            t_list[call_column_count + i] = item

        t_list[call_column_count + custom_data_column_count + 1] = first_unit
        t_list[call_column_count + custom_data_column_count +
               2] = first_arrival_response_time
        t_list[call_column_count + custom_data_column_count +
               3] = full_compliment_unit
        t_list[call_column_count + custom_data_column_count +
               4] = full_compliment_time
        t_list[call_column_count + custom_data_column_count + 5] = extra_data

        self.audit_columns = {
            -5: "FirstArrivalUnit",
            -4: "FirstArrivalTime",
            -3: "FullComplementUnit",
            -2: "FullComplementTime",
            -1: "Audit and Benchmark Info",
        }

        return t_list

    def _create_dataframes(self, data_list, c3s_col_names=None):
        df_main = pd.DataFrame(data_list)
        # these are extracted from the imported xml file
        if c3s_col_names:
            mapping = c3s_col_names['Incident-Related Fields']
            if mapping:
                df_main.rename(
                    columns=mapping, inplace=True)

        # these fixed columns are custom extracted by me from the data
        df_main.rename(columns=self.audit_columns, inplace=True)
        df_main = self._apply_audit_benchmark_mapping(df_main)
        df_main = self._remove_empty_columns(df_main)

        return df_main

    def _apply_audit_benchmark_mapping(self, df_main):
        mapping = self.audit_columns
        length = len(mapping)
        # lenght will be the starting point back from the end of the df
        col_names = list(df_main)
        col_names[-5:] = list(mapping.values())
        df_main.columns = col_names
        return df_main

    def _remove_empty_columns(self, df):
        """
        Removes all empty columns within the dataframe before returning it.

        Args:
            df: A pandas dataframe containing CALL, CUSTOM-DATA and extra info.

        Returns:
            A dataframe stripped of none, null, or "" strings
        """
        empty_columns = [col for col in df.columns if all(
            df[col].apply(lambda x: x == '' or x is None))]
        df.drop(columns=empty_columns, inplace=True)
        return df

    def _create_calls_df(self):
        """
        Process txt file and return either a tuple of dataframes or a single dataframe.

        Args:
            audit (bool, optional): If True, return a tuple of dataframes. If False, return a single dataframe.
                Default is False.

        Returns:
            pd.DataFrame or tuple of pd.DataFrame: The processed data. If to_tuple is True, returns an additional dataframe where
            first arriving or full compliment failed to benchmark.
            Otherwise, returns a single dataframe.
        """
        model_number = self._extract_model_number()
        matches = self._get_matches()

        data_list = []
        for match in matches:
            t_list = self._process_match_for_calls(match)
            data_list.append(t_list)

        c3s_col_names = c3s_mapping(
            self.xml_path) if self.xml_path is not None else None

        return self._create_dataframes(data_list, c3s_col_names)

    def _create_responses_df(self):
        """
        Finishes converting a dictionary into a dataframe.

        Args:
            none

        Returns:
            Dataframe: A dataframe formated to show the call number and unit as dual primary keys.

        """
        model_number = self._extract_model_number()
        print(model_number)
        matches = self._get_matches()

        call_dictionary = {}
        for match in matches:
            call_dictionary.update(self._process_match_for_responses(match))

        df = pd.DataFrame.from_dict(call_dictionary, orient='index')
        df.index
        df.astype({'TurnoutTime': 'float',
                   'OnScene': 'float',
                   'InService': 'float',
                   'FirstArrivalTime': 'float',
                   'FullComplimentTime': 'float',
                   'CommittedHours': 'float',
                   })

        df.index = pd.MultiIndex.from_tuples(
            [(ix[0], ix[1]) for ix in df.index.tolist()])
        idx_0 = df.index.levels[0].unique()

        # =============================================================================
        # Assigning columns to calculate travel times
        df['TravelTime'] = df['OnScene'].astype(
            float) - df['TurnoutTime'].astype(float)
        return df

    def _process_match_for_responses(self, match):
        """
        Performs regex searching through each match iter to create a response based dataframe which dual keys.

        Args:
            match (callable_iterator): regex match iterator.

        Returns:
            dict: A dictionary of all key:value items extracted from that call.
        """
        from collections import defaultdict
        small_string = match.group(0)
        t_call_dictionary = defaultdict(dict)
        call_num = re.search("(?<=CALL\|)[^|]+", small_string).group(0)

        match_assigned = re.finditer(
            r'^BENCHMARK\|Assigned\|([^|]+)\|.*?\|([^|]+)', small_string, flags=re.M | re.S)
        for sub in match_assigned:
            unit = sub.group(1)
            unit_type = ''
            assert call_num == sub.group(
                2), f"""In response dataframe building, call numbers at the benchmark for Assigned are not matching call number for the CALL|<>| text \n
            call_num = {call_num}\n
            sub.group(2) = {sub.group(2)}"""
            t_call_dictionary[(call_num, unit)] = {}

            for char in unit:
                if char.isalpha():
                    unit_type += char

        match_on_scene = re.finditer(
            r'^BENCHMARK\|OnScene\|([^|]+)\|(\d+(?:\.\d*)?)\|([^|]+)\|', small_string, flags=re.M | re.S)
        for sub in match_on_scene:
            unit = sub.group(1)
            on_scene = sub.group(2)
            assert call_num == sub.group(
                3), "In response dataframe building, call numbers at the benchmark row for OnScene are not matching call number for the CALL|<>| text"
            t_call_dictionary[(call_num, unit)]['OnScene'] = on_scene

        match_in_service = re.finditer(
            r'^BENCHMARK\|SceneTime\|([^|]+)\|([\d.]+)\|([^|]+)\|', small_string, flags=re.M | re.S)
        for sub in match_in_service:
            unit = sub.group(1)
            in_service = sub.group(2)
            assert call_num == sub.group(
                3), "In response dataframe building, call numbers at the benchmark row for SceneTime are not matching call number for the CALL|<>| text"
            t_call_dictionary[(call_num, unit)]['InService'] = in_service

        match_first_arrival = re.finditer(
            r'^BENCHMARK\|FirstArrival\|([^|]+)\|([\d.]+)\|([^|]+)', small_string, flags=re.M | re.S)
        for sub in match_first_arrival:
            unit = sub.group(1)
            first_arrival_time = sub.group(2)
            assert call_num == sub.group(
                3), "In response dataframe building, call numbers at the benchmark row for FirstArrival are not matching call number for the CALL|<>| text"
            t_call_dictionary[(call_num, unit)
                              ]['FirstArrivalTime'] = first_arrival_time

        match_full_compliment = re.finditer(
            r'^BENCHMARK\|FullComplement\|([^|]+)\|([\d.]+)\|([^|]+)', small_string, flags=re.M | re.S)
        for sub in match_full_compliment:
            unit = sub.group(1)
            full_compliment_time = sub.group(2)
            assert call_num == sub.group(
                3), "In response dataframe building, call numbers at the benchmark row for FullComplement are not matching call number for the CALL|<>| text"
            t_call_dictionary[(call_num, unit)
                              ]['FullComplimentTime'] = full_compliment_time

        match_tt = re.finditer(
            r'^BENCHMARK\|TurnoutTime\|([^|]+)\|([\d.]+)\|([^|]+)', small_string, flags=re.M | re.S)
        for sub in match_tt:
            unit = sub.group(1)
            turnout_time = sub.group(2)
            assert call_num == sub.group(
                3), "In response dataframe building, call numbers at the benchmark row for TurnoutTime are not matching call number for the CALL|<>| text"
            t_call_dictionary[(call_num, unit)
                              ]['TurnoutTime'] = turnout_time

        match_committed_hours = re.finditer(
            r'^BENCHMARK\|CommittedHours\|([^|]+)\|([\d.]+)\|([^|]+)', small_string, flags=re.M | re.S)
        for sub in match_committed_hours:
            unit = sub.group(1)
            committed_hours = sub.group(2)
            assert call_num == sub.group(
                3), "In response dataframe building, call numbers at the benchmark row for CommittedHours are not matching call number for the CALL|<>| text"
            t_call_dictionary[(call_num, unit)
                              ]['CommittedHours'] = committed_hours

        match_route = re.finditer(
            r'^BENCHMARK\|Route\|([^|]+)\|([\d.]+)\|([^|]+)\|Route\|([0-9,.]+)(?=,)', small_string, flags=re.M | re.S)
        for sub in match_route:
            unit = sub.group(1)
            unit_miles = sub.group(2)
            assert call_num == sub.group(
                3), "In response dataframe building, call numbers at the benchmark row for Route are not matching call number for the CALL|<>| text"
            route = sub.group(4)
            t_call_dictionary[(call_num, unit)
                              ]['UnitMiles'] = unit_miles
            t_call_dictionary[(call_num, unit)
                              ]['Route'] = route

        return t_call_dictionary

    def _print_config_summary(self, project_dict):
        print("\nCall File Configuration Summary:")
        print("---------------------------------")
        print(f"{project_dict['HEADER: ']}")
        print(f"Call columns count: {project_dict['call_columns_count']}")
        print(
            f"Custom data columns count: {project_dict['custom_data_columns_count']}")
        print("\nCall file configuration setup is successful.\n")


def c3s_mapping(xml_path):
    """
    Pass through the C3S "C3CallFieldMappings.xml" file that is generated after data import.

    Args:
        xml_path (string): A string to the the xml settings file.

    Returns:
        dict: A dictionary of column numbers and associated column names, as named by C3S, present in this models XML mapping.
    """
    import re
    from lxml import etree

    # Read the file
    with open(xml_path, "r") as file:
        xml_content = file.read()

    # Replace problematic characters in the attribute values
    xml_content = re.sub(r'="([^"]*?)"', r"='\1'", xml_content)

    # Parse the XML
    root = etree.fromstring(xml_content)

    config = {
        'Incident-Related Fields': {
            1: 'UniqueId',
            2: 'NatureCode',
            3: 'DispatchDate',
            4: 'DispatchTime',
            5: 'StreetAddr',
            6: 'XLoc',
            7: 'YLoc',
            8: 'TransportFlg',
            9: 'CustomContent1',
            10: 'CancelFlg',
            11: 'CallReceived',
            12: 'FirstEffAction ',
            13: 'FirstResponding',
            14: 'FirstArrival',
            15: 'FullComplement',
            16: 'Shift',
            17: 'Battalion',
            18: 'Division',
            19: 'DispatchNature',
            20: 'SituationFound'
        },
        'Response-Related Fields': {
            1: 'UniqueId',
            2: 'UnitId',
            3: 'UnitType',
            4: 'NumStaff',
            5: 'FromQtrs',
            6: 'Dispatched',
            7: 'Responding',
            8: 'OnScene',
            9: 'ClearScene',
            10: 'PtDest',
            11: 'PtHandoff',
            12: 'InService ',
            13: 'InQuarters',
            14: 'Urgency',
            15: 'Assigned',
        }
    }
    attribute_names = ['UniqueId', 'NatureCode']
    xml_aggregate = ".//Aggregate[@Type='C3m.Model.DataConversion.C3mFieldMapping, C3SimModel, Version=2.10.0.0, Culture=neutral, PublicKeyToken=null']"
    xml_destin_attribute = ".//Field[@Name='DestinAttribute']/Aggregate/Field[@Name='Name']/Atom"

    found_fields = {}
    for group_name, field_group in config.items():
        found_fields[group_name] = {}
        for field_number, field_name in field_group.items():
            field_found = False
            for field_mapping in root.findall(xml_aggregate):
                destin_attribute = field_mapping.find(
                    xml_destin_attribute).text
                if field_name in destin_attribute:
                    found_fields[group_name][field_number] = field_name
                    field_found = True
                    break
            if not field_found:
                print(f"    * {field_name} not found in the XML file.")

    return found_fields


if __name__ == "__main__":
    
    
    df_calls = Sim(Path("/app/data/call text files/Results.txt"))._create_calls_df()
    print(df_calls)
