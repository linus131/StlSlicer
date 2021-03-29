# StlSlicer
Slice stl file and generate movepath
Explicit code for orientation is not added
The orientiation is clockwise wrt the normal of the triangle (clockwise for inner surfaces, counterclockwise for outer surfaces)

how to run this program
program.exe [input file] [slice height] [parallel or serial] [write or nowrite] [if write : output filename] 
example:
untitled22.exe c:\rustfiles\all_shapesb.stl 0.1 parallel write c:\rustfiles\movepath.csv

the disconnected loops are separated by NaN,NaN,NaN in output file
can be plotted in matlab using:
movepath = csvread('filename')
plot3(movepath(:,1),movepath(:,2),movepath(:,3),'b-');



What it does now ->
This takes STL file and defines a continuous path of deposition for 3D printers or removal for CNC machines.

What I plan to add ->
1. Define infill
2. Generate Gcode for 3D printing
3. Add simple GUI
