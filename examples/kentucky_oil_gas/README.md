# KENTUCKY OIL & GAS PRODUCTION ANALYSIS
Author: Jonathan M. Frame
Date: February 3rd 2025
Dataset Source: USGS ScienceBase (https://www.sciencebase.gov/catalog/item/664b5d86d34e1955f5a47206)

## PROJECT OVERVIEW
This project analyzes oil and gas production in Kentucky using geostatistics 
and spatial analysis. The dataset provides quarter-mile resolution cells 
representing oil and gas exploration and production across the state.

Key Objectives:
- Visualizing production zones (oil, gas, both, or unknown).
- Applying kriging interpolation to estimate probabilities at unsampled locations.
- Assessing spatial uncertainty through variance maps.

## DATASET INFORMATION
Dataset Name: Oil and Gas Exploration and Production in the State of Kentucky
Source: U.S. Geological Survey (USGS)
Time Span: 1866 - 2005
Coordinate System: NAD83 (EPSG:4269)

File Contents (kycells05g):
- kycells05g.shp   - Shapefile containing quarter-mile cell polygons.
- kycells05g.dbf   - Attribute table with well production data.
- kycells05g.prj   - Projection metadata.
- kycells05g.shx   - Index file for spatial data.

Attributes of Interest:
- `CELLSYMB` - Production status:
    1: Oil-producing wells (Green)
    2: Gas-producing wells (Red)
    3: Oil & gas-producing wells (Gold)
    4: Unknown or no production (Charcoal)
- `CC83XCOORD`, `CC83YCOORD` - Coordinates in NAD83.