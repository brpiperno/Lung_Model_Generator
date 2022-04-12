# Ben Piperno: February 5th, 2019

# _______________________________________________________
#                      IMPORTS
# import glob
import vtk
import numpy as np
import sys
import math
import vtktools as v
import gtools as g

# ________________________________________________________
#                     Program Description

"""
Necessary TODO
- fix facing to adjust based on face selection
- confirm rotation function is fixed
- Smooth bifurcations to prevent overlapping
- branch face projection from most appropriate face
- use literature backed segment angling change
- look into literature backed radius smoothing
- look into literature bifurcation geometry
- implement intersection correction function objects
- implement new vertex creation scheme (no phase shift in circle creation)

Optional
- Speed up calculations by storing more in memory
- create smoothing expression
- create 3-child base segment
- organize code with local functions and anything else necessary
- Parameterize number of loops

Lung Generator
Takes in a csv file of a geometric representation of a lung airway structure
creats a surface mesh of the representation in a .vtp format
takes in the following parameters:
    - path to desired directory (containing .csv files)
    - number of generations to go to (trachea is generation 0)
    - number of edge loops in each segment (minimum 3)
    - number of vertices in each loop (currently only 4)
    - smoothing curve parameters (tbd)
"""

# Console test input:
# python C:\Users\BRPip\Desktop\RESIST\lung_generator\lung_generator.py
# C:\Users\BRPip\Desktop\RESIST\m11\
# _______________________________________________________
#                      CONSTANTS
SMOOTHPAR = 0  # this is current unused
LENGTH_FACTOR = 1
# flat percentage of each each segment's length to make
#  to account for overlaps in measuring


# variable explorer variables
csvpath = r'C:\Users\BRPip\Desktop\Archive\RESIST\m11\m11_AirwayTreeTable.csv'
loop_size = 4
gens = 23  # number of generations deep to go, gen 1 is trachea
subdivs = 0
RADIAL_FACTOR = 0*subdivs  # not yet implemented
# _______________________________________________________
#                      functions


def generate(csvpath, gens, subdivision_level, rendermesh=True):
    # CLEAN UP DATA
    # Columns:
    # \Parent \ Length \ Radius \ CentroidX,Y,Z \ DirectionX,Y,Z
    # Direction is unit vector that describes the direction of the distal end

    data = np.genfromtxt(csvpath, delimiter=",", skip_header=1,
                         usecols=(1, 2, 3, 5, 6, 7, 8, 9, 10))
    data[:, 0] -= 1

    # CALCULATE VERTEX LOCATIONS
    vertices = np.zeros((max_row(data, gens), loop_size*3, 3))
    for i in range(vertices.shape[0], 0, -1):
        sys.stdout.write("\rSegment {} of {}".format(i-1, vertices.shape[0]))
        vertices[i-1] = mk_segment(data[i-1], loop_size,
                                   proximal_verts(children(i-1, data, gens),
                                                  vertices, loop_size))
    sys.stdout.write("\r{} Segments Calculated  ".format(vertices.shape[0]))

    # ARRANGE FACES
    faces = list()
    base1 = [int(math.floor(n/2)
             + loop_size*(n % 2)) for n in range(loop_size*2)]
    base1 += [0, loop_size]
    # base = [[0, 4, 1, 5, 2, 6, 3, 7, 0, 4],    <----- No Child
    #         [4, 8, 5, 9, 6, 10, 7, 11, 4, 8],
    #         [9, 8, 10, 11]]
    # base = [[0, 4, 1, 5, 2, 6, 3, 7, 0, 4],    <----- Two Children
    #         [5, 6, 9, 10, 8, 11, 4, 7]
    for i in range(len(vertices)):
        # Assumption- segments have 0 or 2 children
        # TODO: make trachea cap
        if len(children(i, data, gens)) == 0:
            first = [base1[j] + loop_size*3*i for j in range(len(base1))]
            second = [j+loop_size for j in first]
            base3 = [9, 8, 10, 11]
            third = [j+loop_size*i*3 for j in base3]
            faces.append([first, second, third])
        elif len(children(i, data, gens)) == 2:  # side 0 and 2
            first = [base1[j] + loop_size*3*i for j in range(len(base1))]
            base2 = [5, 4, 8, 9, 11, 10, 6, 7]
            second = [base2[j] + loop_size*3*i for j in range(len(base2))]
            faces.append([first, second])

    # INITIALIZE MESH OBJECT
    polydata = vtk.vtkPolyData()

    # INITIALIZE CELL ARRAY OBJECT
    polys = vtk.vtkCellArray()

    # INSERT POINTS INTO POINTS OBJECT
    vtkpoints = vtk.vtkPoints()
    for i in range(len(vertices)):
        for j in range(loop_size*3):
            vtkpoints.InsertNextPoint(vertices[i][j])

    # INSERT EACH SEGMENT INTO MESH (using trianglestrip mesh topology)
    for i in faces:
        for j in i:
            for k in range(len(j)-2):
                polys.InsertNextCell(mkVtkIdList(j[k:k+3]))

    # Assign the cells to the polydata
    polydata.SetPoints(vtkpoints)
    polydata.SetPolys(polys)

    # Clean the mesh - remove duplicate vertices and edges
    cleanPolyData = vtk.vtkCleanPolyData()
    cleanPolyData.SetInputData(polydata)

    # SUBDIVIDE THE MESH
    smooth_loop = vtk.vtkLoopSubdivisionFilter()
    smooth_loop.SetNumberOfSubdivisions(subdivision_level)
    smooth_loop.SetInputConnection(cleanPolyData.GetOutputPort())
    # smooth_loop = cleanPolyData

    # RENDER OR EXPORT
    if rendermesh:
        v.render(smooth_loop)
    else:
        v.export(smooth_loop, csvpath)
    return


def mk_segment(seginfo, loop_size, child_verts):
    # given list of information about current segment,
    # number of vertices in a single loop
    # any connecting child vertices (3D array)
    # Returns a 2D array of numbers representing the vertices in the segment
    # represented by vertices from proximal to distal
    length = seginfo[1]  # length of segment
    radius = seginfo[2]  # radius of segment
    centroid = seginfo[3:6]  # spacial center of segment
    dirv = seginfo[6:9]  # direction vector

    # Create list of center points for each loop using length
    centers = [[0, 0, .5*length*LENGTH_FACTOR],
               [0, 0, 0],
               [0, 0, -.5*length*LENGTH_FACTOR]]

    # Create verts to store vertices
    verts = np.zeros((loop_size*len(centers), 3))

    # Create vertices at origin with given radius
    for i in range(len(centers)):
        start = i*loop_size
        end = start+loop_size
        verts[start:end] = g.create_circle(loop_size, radius, centers[i][2])

    # ROTATE SEGMENT
    if dirv[2] == 1:  # direction is entirely up
        verts = g.rotate_yz(verts, np.pi, 0)
    elif dirv[2] == -1:  # direction is entirely down
        pass
    else:
        verts = g.rotate_pts(verts, dirv)

    # TRANSLATE SEGMENT
    for point in range(len(verts)):
        for axis in range(3):
            verts[point][axis] += centroid[axis]

    # REPLACE VERTS IF NECESSARY
    if len(child_verts) != 0:
        verts = bifurcate(verts, child_verts, loop_size)

    # ROUND VERTS
    for i in range(verts.shape[0]):
        for j in range(verts.shape[1]):
            verts[i][j] = round(verts[i][j], 2)
    return verts


def bifurcate(orgnl, child_verts, loop_size):
    # replace distal verts of a segment with it's children's proximal verts
    # calculate the center points of each side of the original
    # TODO: currently doesn't account for 3rd child, skips over it
    # TODO: absract code for other loop_sizes

    # TODO: Explain what these are
    ref = [[4, 5, 8, 9], [5, 6, 9, 10], [6, 7, 10, 11], [7, 4, 11, 8]]
    sides = np.zeros((loop_size, loop_size, 3))
    for i in range(loop_size):
        for j in range(loop_size):
            for k in range(3):
                sides[i][j][k] = orgnl[ref[i][j]][k]

    # TODO: Explain what this is
    side_center = [np.mean(sides[i], axis=0) for i in range(loop_size)]

    # calculate the distances of the child_vert[0] to each side center
    # child_vert[0] is the spacial center of one of the children, to be used
    # to compare to the centroid of each side of the distal end of the segment
    distances = list()
    for side in side_center:
        distances.append(g.calc_dist(np.mean(child_verts[0], axis=0), side)
                         + g.calc_dist(np.mean(child_verts[0], axis=0), side))
    val, idx = min((val, idx) for (idx, val) in enumerate(distances, 0))

    # find the closest sets of vertices on that face
    v_d = np.zeros((child_verts.shape[0], loop_size))
    for c in range(child_verts.shape[0]):  # iterate over number of children
        for r in range(loop_size):
            rolled = np.roll(child_verts[c], r)
            # total distance between each side and its new match
            for i in range(loop_size):
                v_d[c][r] += g.calc_dist(rolled[i], sides[idx - 2*c][i])

    # rearrange the child vertices according to their roll
    new_child = [np.roll(child_verts[c], np.argmin(v_d[c]),
                         axis=0) for c in range(child_verts.shape[0])]

    # assign the child verts to the main segment
    for i in range(loop_size):
        orgnl[ref[idx][i]] = new_child[0][i]
        orgnl[ref[idx-2][i]] = new_child[1][i]
    return orgnl


def generation(n, data):
    # Returns the the generation number of an index in a the tree
    row = n
    gen_count = 0
    while row >= 0:
        # set row equal to the parent of the row
        row = int(data[row][0])
        gen_count += 1
    return gen_count


def children(row, data, gens):
    # return a list of list of indices of children of a given row
    child_list = list()
    for i in range(data.shape[0]):
        if (data[i][0] == row) and (generation(i, data) <= gens):
            child_list.append(i)
    return child_list


def proximal_verts(indices, vertices, loop_size):
    # returns vertices of proximal side of each given segment (by index)
    # use the given list of vertices and the loop_size to pull correct items
    # TODO: reconfigure to be modular with number of edge loops (currently 3)
    # if the vertices are 0, then return an empty list
    proximalverts = np.zeros((len(indices), loop_size, 3))
    for i in range(len(indices)):
        proximalverts[i] = vertices[indices[i]][:loop_size - vertices.shape[1]]
    return proximalverts


def max_row(data, gen):
    # Finds the last row of of a particular generation in the list of segments
    # Assumption: All segments of a row greater than the current segment are of
    #     the same generation or greater
    index = 0
    while generation(index, data) <= gen:
        index += 1
    return index


def mkVtkIdList(it):
    # Makes a vtkIdList from a Python iterable.
    vil = vtk.vtkIdList()
    for i in it:
        vil.InsertNextId(int(i))
    return vil


if __name__ == "__main__":
    """command line use: python lung_generator.py
    [directory with .csv file(s),
    num_generations = max_number_of_generations,
    number of subdivisions (catmull-clark)
    smoothing parameters = tbd]"""
    # for csvpath in glob.glob(sys.argv[1]+os.path.sep+'*.csv'):
    print('\nProcessing: {0}'.format(csvpath))
    generate(csvpath, gens, subdivs, True)
    print('\n{0} completed\n'.format(csvpath))
