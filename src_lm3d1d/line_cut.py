from collections import deque
import networkx as nx
import dolfin as df
import numpy as np
import itertools
import operator

# ---
# Once we have a marked volume in the mesh we want to clasify interior
# and exterior facets for the purpose of penalizing jumps and what not
def build_graph(mesh, volume_cells):
    '''
    Graph representation of cell-cell connectivity over facets. In graph 
    cell is a node; common facet between cells mounts to an edge. Edges 
    can be clasified during graph building: those connected to just one 
    cell are guaranteed to exterior.
    '''
    assert isinstance(volume_cells, set)
    
    assert mesh.mpi_comm().tompi4py().size == 1  # Only serial
    
    cdim = mesh.topology().dim()
    fdim = cdim - 1

    mesh.init(cdim, fdim)
    mesh.init(fdim, cdim)

    f2c, c2f = mesh.topology()(fdim, cdim), mesh.topology()(cdim, fdim)

    G = nx.Graph()
    volume_facets = np.unique(np.hstack(map(c2f, volume_cells)))

    vc = set(volume_cells)

    interiors, exteriors = [], []
    for f in volume_facets:
        cells = f2c(f)

        # Definitely an exterior
        if len(cells) == 1:
            exteriors.append(f)
        else:
            # We are always after a subset
            if set(cells) <= volume_cells:
                G.add_edges_from(itertools.combinations(cells, 2))
                interiors.append(f)
            else:
                exteriors.append(f)

    return (interiors, exteriors), G


def ie_facets_volume(mesh, volume_cells):
    '''Interior, exterior facets of the marked volume (for each connected component)'''
    _, G = build_graph(mesh, volume_cells)

    ccs = nx.connected_components(G)

    interiors, exteriors = [], []
    for cc in ccs:
        (i, e), _ = build_graph(mesh, cc)
        interiors.append(i)
        exteriors.append(e)

    return interiors, exteriors

# ---
# Some CGAL like functionality where, unlike with dolfin, I can control
# tolerance for what is 0
def norm(vec):
    '''Handle cross product'''
    if not vec.shape:  # Scalar
        return np.abs(vec)
    return np.linalg.norm(vec, 2)


def tri_area(A, B, C):
    '''Area of triangle'''
    cross = np.cross(B-A, C-A)
    return 0.5*norm(cross)


def point_is_inside_tri(tri, P, tol):
    '''Is point inside triangle'''
    A, B, C = tri
    return (tri_area(A, B, P) + tri_area(B, C, P) +  tri_area(C, A, P)) < tri_area(A, B, C) + tol


def seg_seg_intersect(seg0, seg1, tol):
    '''Two segments intersect?'''
    A, B = seg0
    X, Y = seg1

    # Same?
    if (norm(A-X) < tol and norm(B-Y) < tol) or (norm(A-Y) < tol and norm(B-X) < tol):
        return True

    # System is A + (B-A)*s = X + (Y-X)*t
    dir0, dir1 = B-A, X-Y
    # Singularity
    if norm(np.cross(dir0, dir1)) < tol:
        return False

    # We have it
    mat = np.c_[dir0, dir1]
    vec = X-A
    if len(vec) == 2:
        x = np.linalg.solve(mat, vec)
    else:
        # Planarity
        if norm(np.cross(dir0, dir1).dot(vec)) > tol:
            return False
        else:
            x = np.linalg.lstsq(mat, vec)[0]
    return all((-tol < xi < 1+tol) for xi in x)


def seg_is_inside_tri(tri, seg, tol):
    '''Both points of seg are inside triangle'''
    return all(point_is_inside_tri(tri, P, tol) for P in seg)


def tri_seg_intersect(tri, seg, tol):
    '''Is triangle intersected by line'''
    # Both point in
    if seg_is_inside_tri(tri, seg, tol):
        return True

    # Then some edge must be hit
    A, B, C = tri
    edges = ((A, B), (B, C), (C, A))
    return any(seg_seg_intersect(seg, edge, tol) for edge in edges)


def tet_volume(A, B, C, D):
    '''Area of tetrahedron'''
    return abs((D-A).dot(np.cross(B-A, C-A)))/6.


def point_is_inside_tet(tet, P, tol):
    '''Is point inside tetrahedron'''
    A, B, C, D = tet
    return (tet_volume(A, B, C, P) +
            tet_volume(B, C, D, P) +
            tet_volume(C, D, A, P) +
            tet_volume(D, A, B, P)) < tet_volume(A, B, C, D) + tol


def tet_seg_intersect(tet, seg, tol):
    '''Is tetrahedron intersected by line'''
    # Both point in
    if all(point_is_inside_tet(tet, P, tol) for P in seg):
        return True

    # Then some edge must be hit
    A, B, C, D = tet
    tris = ((A, B, C), (B, C, D), (C, D, A), (D, A, B))
    return any(tri_seg_intersect(tri, seg, tol) for tri in tris)


def cell_collisions(point, point_is_inside_cell, tree, c2f, f2c, tol):
    '''My predicate collision checking'''
    basic = set(tree.compute_entity_collisions(df.Point(point)))
    # Which others are connected to them
    patch = reduce(operator.or_,
                   (set(f2c(facet)) for cell in basic for facet in c2f(cell)))
    # Which are new?
    patch.difference_update(basic)
    # Of these, which really contain the point
    basic.update((c for c in patch if point_is_inside_cell(c, point, tol)))

    return basic


def is_simply_connected(mesh, volumes=None):
    '''Volumes define proper subdomain'''
    if volumes is None:
        volumes = range(mesh.num_cells())
    _,  graph = build_graph(mesh, set(list(volumes)))
    return len(list(nx.connected_components(graph))) == 1


# --
# Finally, collision detection between curve and cells
# The main idea is to keep looking for points between 1d mesh nodes until
# the cells containing points are connected
def curve_cover(curve, mesh, tol=1E-13):
    '''Mesh cells intersected by the curve'''
    assert curve.topology().dim() == 1
    assert curve.geometry().dim() == mesh.geometry().dim()
    # So the main idea only works if curve is connected
    assert is_simply_connected(curve)

    # If this is not property of the graph then break into into connected
    # components, curve_cover components and then do whatever you want.
    # I don't want to do this here because maybe you want different colors
    # for different covers or something like that. This is a tool to do it.
    
    tol = tol*mesh.hmin()  # FIXME: Note so sure about this
    
    tdim = mesh.topology().dim()
    mesh.init(tdim-1)
    mesh.init(tdim, tdim-1)
    mesh.init(tdim-1, tdim)
    # Two cells are neighors/connected if they share a facet
    c2f = mesh.topology()(tdim, tdim-1)
    f2c = mesh.topology()(tdim-1, tdim)
    
    tree = mesh.bounding_box_tree()
    
    points = list(curve.coordinates())
    x_cells = mesh.coordinates()[mesh.cells()]

    if tdim == 2:
        point_is_inside_cell = lambda c, p, tol: point_is_inside_tri(x_cells[c], p, tol)
    else:
        point_is_inside_cell = lambda c, p, tol: point_is_inside_tet(x_cells[c], p, tol)
        
    # Initial cells are those that are hit by points
    cells_with_points = {i: cell_collisions(point, point_is_inside_cell, tree, c2f, f2c, tol)
                         for i, point in enumerate(points)}
    # Everybody must be found
    assert all(len(v) for v in cells_with_points.values())

    mid_idx = len(points)
    segments = deque(map(tuple, curve.cells()))
    while segments:
        A, B = segments.pop()
        # Is this a covered segment
        if any(set(c2f(cell_A)) & set(c2f(cell_B))
               for cell_A in cells_with_points[A] for cell_B in cells_with_points[B]):
            continue

        # Try to cover it by midpoint
        mid = 0.5*(points[A] + points[B])
        cells_with_points[mid_idx] = cell_collisions(mid, point_is_inside_cell, tree, c2f, f2c, tol)

        # Add the children
        segments.appendleft((A, mid_idx))
        segments.appendleft((B, mid_idx))
        
        points.append(mid)        
        mid_idx += 1
        
    # Cells of cover
    return np.fromiter(reduce(operator.or_, cells_with_points.values()),
                       dtype='uintp')
