# StlSlicer
<b>Slice stl file and generate contour movepath</b>

<b>how to run this program</b> <br>
program.exe [input file] [slice height] [parallel or serial] [write or nowrite] [if write : output filename] [orient or noorient] [if orient: clockwise or anticlockwise]
example:
untitled22.exe c:\rustfiles\all_shapesb.stl 0.1 parallel write c:\rustfiles\movepath.csv orient clockwise

Process planning is an important step in additive manufacturing processes that involves the creation of a deposition toolpath from input geometry. The most commonly used input geometry file formats are in the form of unstructured triangular meshes. Previous research works have presented different algorithms for slicing an unstructured triangular mesh by a series of uniformly spaced parallel planes. Recent research works show that the use of hashtable data structures results in an optimal algorithm that improves the efficiency of the slicing process. The presented algorithm minimizes the use of hashtable in the slicing process by using graph-based data structures. It was found that the slicing can be sped up by about 10 times by using the graph-based algorithm compared to the previous algorithm. The use of parallel processing on multi-core processors was explored to further speed up the slicing process. The proposed efficient algorithm can be useful in process planning optimization.

Sunil Bhandari,
A graph-based algorithm for slicing unstructured mesh files,
Additive Manufacturing Letters,
2022,
100056,
ISSN 2772-3690,
https://doi.org/10.1016/j.addlet.2022.100056.
(https://www.sciencedirect.com/science/article/pii/S2772369022000305)
