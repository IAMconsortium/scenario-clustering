from pathlib import Path
import logging

import yaml

import pandas as pd
from pyam import IamDataFrame
from nomenclature.processor import Processor, RequiredDataValidator
from nomenclature.codelist import CodeList

MODULE_PATH = Path(__file__).parent

# logger = logging.getLogger(__name__)

class IMPClusterProcessor(Processor):
    """Identifies whether an IamDataFrame represents run(s) belongs to any of 
    the MVP set of clusters, based on IMP scenario clusters.
    
    Contact (method): Mark Dekker (mark.dekker@pbl.nl)
    Contact (implementation): Gabriel Sher (gabriel.sher@pbl.nl)"""
    cluster_name: str
    code_list : CodeList
    sdg_scenarios: list
    req_data_validator : RequiredDataValidator

    def __init__(self, cluster_name):
        super().__init__(
            cluster_name = cluster_name,
            code_list = CodeList.from_directory(
                name='code_list',
                path=MODULE_PATH / 'definitions' / 'variable',
                file_glob_pattern=f'*{cluster_name.lower()}*'
            ),
            sdg_scenarios = list(yaml.safe_load(open(
                MODULE_PATH / 'definitions' / 'metadata' / 'sdg_scenarios' / 'imp_sdg.yaml'
            ))),
            req_data_validator = RequiredDataValidator.from_file(
                MODULE_PATH / 'definitions' / 'requirements' / f'req_data_{cluster_name.lower()}.yaml',
            )
        )

    def apply(self, df: IamDataFrame) -> IamDataFrame:
        for ms in df.index:
            (model, scenario) = ms
            idx = pd.MultiIndex.from_tuples([ms], names=['model', 'scenario'])
            META_0 = pd.Series(0, index=idx, name=self.cluster_name)
            META_1 = pd.Series(1, index=idx, name=self.cluster_name)

            # (0) Check if all variables are available
            sub_df = df.filter(model=model, scenario=scenario)
            missing_data = self.req_data_validator.check_required_data_per_model(sub_df, model)

            if missing_data:
                # Most clusters: report missing var, move to next model/scen pair
                if self.cluster_name != 'HighRen':
                    # logging.info(f'{model}, {scenario} failed "{self.cluster_name}" required variable check: missing {[d.variable.values.item() for d in missing_data]}')
                    df.set_meta(META_0, self.cluster_name, index=idx)
                    continue
                # HighRen cluster: check for alternative variables
                alt_req_data_validator = RequiredDataValidator.from_file(
                    MODULE_PATH / 'definitions' / 'requirements' / f'req_data_highren_alt.yaml',
                )
                missing_alt_data = alt_req_data_validator.check_required_data_per_model(sub_df, model)
                if missing_alt_data:
                    # logging.info(f'{model}, {scenario} failed "{self.cluster_name}" required variable check: missing {[d.variable.values.item() for d in missing_data + missing_alt_data]}')
                    df.set_meta(META_0, self.cluster_name, index=idx)
                    continue
                # Store that we need to use alt variables
                highren_alt_var = True
            else:
                highren_alt_var = False

            # (1) Hand-pick scenarios for SDG cluster
            if self.cluster_name == 'SDG':
                ms_str = f'{model}|{scenario}'
                if ms_str in self.sdg_scenarios:
                    df.set_meta(META_1, self.cluster_name, index=idx)
                else:
                    df.set_meta(META_0, self.cluster_name, index=idx)
                continue
            # (2) Apply variable-based criteria for other clusters
            #       First assume that it does belong to the cluster, then disprove
            df.set_meta(META_1, self.cluster_name, index=idx)
            for varname, code in self.code_list.mapping.items():
                # Apply conditions to variables
                if '@' in varname:
                    varname = varname.split('@')[0]

                attrs = code.extra_attributes
                
                bounds = {}
                bounds[f'{attrs['bound']}_bound'] = attrs['threshold']

                slice_kwargs = {}
                slice_kwargs['index'] = idx
                slice_kwargs['region'] = 'World'

                # TODO: possibly handle mismatched units (e.g. Gt CO2 as well as Mt)
                slice_kwargs['measurand'] = (varname, attrs['unit'])

                not_valid = None
                if attrs['type'] == 'abs':
                    # Take year directly
                    slice_kwargs['year'] = attrs['year']

                    not_valid = sub_df.validate(
                        **bounds,
                        **slice_kwargs,
                    )
                elif attrs['type'] == 'change':
                    # Take year as end of range, read it relative to start of range
                    slice_kwargs['year'] = int(attrs['year'].split('-')[1])
                    offset_year = int(attrs['year'].split('-')[0])

                    not_valid = sub_df.offset(year=offset_year).validate(
                        **bounds,
                        **slice_kwargs,
                    )
                elif attrs['type'] == 'share':
                    # Take year directly
                    slice_kwargs['year'] = attrs['year']

                    # Handle special case: Primary Energy|Renewables (incl. Biomass) not present
                    if self.cluster_name == 'HighRen' and highren_alt_var:
                            vars = ['Primary Energy|Biomass', 'Primary Energy|Non-Biomass Renewables']
                    else:
                            vars = [varname]

                    share_df = self.calc_share(
                        df=sub_df,
                        vars=vars,
                        denom_var=varname.split('|')[0],
                        slice_kwargs=slice_kwargs
                    )
                    
                    not_valid = share_df.validate(
                        **bounds,
                        **slice_kwargs,
                    )
                elif attrs['type'] == 'cum':
                    # Extract relevant slice
                    start, end = [int(y) for y in attrs['year'].split('-')]
                    slice_kwargs['year'] = range(start, end)

                    data = sub_df.filter(**slice_kwargs).interpolate(df.time).data

                    total_neg = 0
                    for i, row in data.iterrows():
                        if row['value'] < 0:
                            total_neg -= row['value']
                    
                    if total_neg <= attrs['threshold']:
                        not_valid = True
                if not_valid is not None:
                    # Have scenarios where a criterion is not met: Report, exit loop
                    df.set_meta(META_0, self.cluster_name, index=idx)
                    # logging.info(f'{model,scenario} failed {self.cluster_name} variable validation check (var: {varname})')
                    break
        return df
    
    def calc_share(self, df, vars, denom_var, slice_kwargs):
        unit = slice_kwargs['measurand'][1]

        var_data = []
        for var in vars:
            # Reset the measurand in case variable name is non-default
            slice_kwargs['measurand'] = (var, unit)
            
            # Fetch data
            var_data.append(df.filter(**slice_kwargs).data)

        # Fetch superset data (denominator)
        slice_kwargs['measurand'] = (denom_var, unit)
        denom_data = df.filter(**slice_kwargs).data

        # Calculate share and cast to df
        share_data = denom_data.copy()
        share_data['value'] = sum([v['value'] for v in var_data]) / denom_data['value']
        share_df = IamDataFrame(data=share_data, meta=df.meta)

        return share_df