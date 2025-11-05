using Orbit
using Orbit.Comodo
using Orbit.Comodo.GLMakie
using Orbit.Comodo.GLMakie.Colors
using Orbit.GeometryBasics
using Orbit.Rotations
using Orbit.Statistics
using Orbit.LinearAlgebra
using Orbit.DelaunayTriangulation
using Orbit.FileIO
using Orbit.ZipFile

GLMakie.closeall()

# ============================================================================
# CONSTANTS
# ============================================================================

const MM_TO_CM = 0.1
const OVERHANG_THRESHOLD_DEGREES = 35.0
const SUPPORT_DIAMETER_MM = 2.0
const BUILD_PLATE_CLEARANCE = 0.2
const SUPPORT_DENSITY_REDUCTION = 2.0
const MIN_FEATURE_SIZE_MM = 0.4

# ============================================================================
# MESH LOADING
# ============================================================================

"""
Load a 3D mesh from file (STL/3MF/AMF formats supported).
Returns faces and vertices with duplicates merged and unused vertices removed.
"""
function load_mesh_from_file(test_case::Int)
    # Select file based on test case
    mesh_file_path = if test_case == 1
        joinpath(comododir(), "assets", "stl", "stanford_bunny_low.stl")
    elseif test_case == 2
        "C:\\Users\\Karl\\Downloads\\Cervical Cage v1.0 v2(1).stl"
    elseif test_case == 3
        "C:\\Users\\Karl\\Downloads\\Cylix3D_BabyDeadpool3MF.3mf"
    elseif test_case == 4
        "C:\\Users\\Karl\\Downloads\\Cervical Cage v1.0 v2.amf"
    else
        error("Invalid test case: $test_case.")
    end

    # Load the mesh file
    file_extension = lowercase(splitext(mesh_file_path)[2])
    raw_mesh = if file_extension in [".3mf", ".amf"]
        load_3d_file(mesh_file_path)
    else
        load(mesh_file_path)  # Standard FileIO for STL
    end
   
    # Merge vertices and remove unused ones
    faces = tofaces(GeometryBasics.faces(raw_mesh))
    vertices = topoints(coordinates(raw_mesh))
    merged_faces, merged_vertices = mergevertices(faces, vertices)
    
    return remove_unused_vertices(merged_faces, merged_vertices)
end

"""
Load 3MF/AMF files by extracting mesh data from XML content.
3MF files are zipped XML, AMF files are plain XML.
"""
function load_3d_file(filepath::String)
    file_extension = lowercase(splitext(filepath)[2])
    
    if file_extension == ".3mf"
        # 3MF: Extract XML from zip archive
        reader = ZipFile.Reader(filepath)
        model_entry = findfirst(f -> endswith(f.name, ".model"), reader.files)
        model_entry === nothing && error("No 3D model found in 3MF file")
        xml_content = String(read(reader.files[model_entry])) # raw XML text
        close(reader)

        # Parse vertices: <vertex x="1.23" y="4.56" z="7.89">
        vertex_regex = r"<vertex\s+x=\"([^\"]+)\"\s+y=\"([^\"]+)\"\s+z=\"([^\"]+)\""
        vertex_matches = collect(eachmatch(vertex_regex, xml_content))
        
        # Parse triangles: <triangle v1="0" v2="1" v3="2"> (indices are 0-based)
        triangle_regex = r"<triangle\s+v1=\"(\d+)\"\s+v2=\"(\d+)\"\s+v3=\"(\d+)\""
        triangle_matches = collect(eachmatch(triangle_regex, xml_content))
        
        vertices = Vector{Point{3,Float64}}(undef, length(vertex_matches))
        faces = Vector{TriangleFace{Int}}(undef, length(triangle_matches))

        for (i, match) in enumerate(vertex_matches)
            x, y, z = parse.(Float64, match.captures)
            vertices[i] = Point{3,Float64}(x, y, z)
        end
        
        for (i, match) in enumerate(triangle_matches)
            v1, v2, v3 = parse.(Int, match.captures) .+ 1 # adjust to 1-based indexing
            faces[i] = TriangleFace{Int}(v1, v2, v3)
        end
        
    elseif file_extension == ".amf"
        xml_content = String(read(filepath))
        
        # Parse AMF vertices and triangles
        vertex_pattern = r"<vertex[^>]*>.*?<x>([^<]+)</x>.*?<y>([^<]+)</y>.*?<z>([^<]+)</z>.*?</vertex>"s
        vertex_matches = collect(eachmatch(vertex_pattern, xml_content))
        
        triangle_pattern = r"<triangle>.*?<v1>(\d+)</v1>.*?<v2>(\d+)</v2>.*?<v3>(\d+)</v3>.*?</triangle>"s
        triangle_matches = collect(eachmatch(triangle_pattern, xml_content))
        
        vertices = Vector{Point{3,Float64}}(undef, length(vertex_matches))
        faces = Vector{TriangleFace{Int}}(undef, length(triangle_matches))
        
        for (i, match) in enumerate(vertex_matches)
            x, y, z = parse.(Float64, match.captures)
            vertices[i] = Point{3,Float64}(x, y, z)
        end
        
        for (i, match) in enumerate(triangle_matches)
            v1, v2, v3 = parse.(Int, match.captures) .+ 1
            faces[i] = TriangleFace{Int}(v1, v2, v3)
        end
    else
        error("Unsupported file format: $file_extension")
    end
    
    println("Loaded $(length(vertices)) vertices and $(length(faces)) faces from $(uppercase(file_extension[2:end]))")
    return GeometryBasics.Mesh(vertices, faces)
end

# ============================================================================
# ORIENTATION AND GEOMETRY
# ============================================================================

"""
Create test orientations using a geodesic sphere.
Returns faces and vertices sorted by Z-coordinate (top-first).
"""
function create_test_orientations(subdivisions=2, radius=1.0)
    # Generate geodesic sphere for uniform orientation sampling
    sphere_faces, sphere_vertices = geosphere(subdivisions, radius)

    # Sort vertices by Z-coordinate (highest first for top-down processing)
    z_coordinates = [vertex[3] for vertex in sphere_vertices]
    sorted_indices = reverse(sortperm(z_coordinates))
    
    # Create reindexing map for faces
    reindex_map = sortperm(sorted_indices)

    # Apply sorting
    sorted_vertices = sphere_vertices[sorted_indices]
    sorted_faces = [TriangleFace{Int}(reindex_map[f[1]], reindex_map[f[2]], reindex_map[f[3]]) for f in sphere_faces]
    
    # Create opposite hemisphere vertices 
    opposite_vertices = [-vertex for vertex in sorted_vertices]
    
    return sorted_faces, sorted_vertices, opposite_vertices
end

"""
Orient part for 3D printing by aligning build direction with +Z axis.
Places the part on the build plate (Z=0).
"""
function orient_part_for_printing(vertices, build_direction)
    up_vector = Point{3, Float64}(0.0, 0.0, 1.0)
    part_center = mean(vertices)

    # Calculate rotation to align build direction with +Z
    rotation = rotation_between(up_vector, normalizevector(build_direction))
    
    # Apply rotation around part centre
    rotated_vertices = deepcopy(vertices)
    min_z = Float64(Inf)
    max_z = -Float64(Inf)
    
    # Track Z bounds
    for (i, vertex) in enumerate(rotated_vertices)
        rotated_vertices[i] = rotation * (vertex - part_center)
        z_coord = rotated_vertices[i][3]
        min_z = min(min_z, z_coord)
        max_z = max(max_z, z_coord)
    end
    
    # Move part to build plate (z = 0)
    build_plate_vertices = [Point{3, Float64}(v[1], v[2], v[3] - min_z) for v in rotated_vertices]
    part_height = max_z - min_z
    
    return build_plate_vertices, part_height
end

"""
Calculate axis-aligned bounding box volume and edges for visualisation.
Returns volume in mm³ and 12 edge lines for wireframe display.
"""
function calculate_bounding_box_metrics(vertices)
    # Bounds in each dimension
    x_min, x_max = extrema(v[1] for v in vertices)
    y_min, y_max = extrema(v[2] for v in vertices)
    z_min, z_max = extrema(v[3] for v in vertices)
    
    volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
    
    # Bounding box edges for visualisation
    bottom_corners = [
        Point{3,Float64}(x_min, y_min, z_min), Point{3,Float64}(x_max, y_min, z_min),
        Point{3,Float64}(x_max, y_max, z_min), Point{3,Float64}(x_min, y_max, z_min)
    ]
    
    top_corners = [
        Point{3,Float64}(x_min, y_min, z_max), Point{3,Float64}(x_max, y_min, z_max),
        Point{3,Float64}(x_max, y_max, z_max), Point{3,Float64}(x_min, y_max, z_max)
    ]
    
    box_edges = []
    for i in 1:4
        next_corner = mod1(i + 1, 4)
        push!(box_edges, [bottom_corners[i], bottom_corners[next_corner]])  # Bottom edges
        push!(box_edges, [top_corners[i], top_corners[next_corner]])        # Top edges
        push!(box_edges, [bottom_corners[i], top_corners[i]])               # Vertical edges
    end
    
    return volume, box_edges
end

# ============================================================================
# SHADOW PROJECTION
# ============================================================================

"""
Calculate the XY projection area (shadow) of the mesh using ray casting.
Uses a grid of rays cast upward to detect mesh coverage.
"""
function calculate_shadow_projection_area(faces, vertices, grid_resolution=0.5)
    length(vertices) < 3 && return 0.0

    # XY bounds
    xs = [v[1] for v in vertices]
    ys = [v[2] for v in vertices]
    x_min, x_max = extrema(xs)
    y_min, y_max = extrema(ys)

    # Circular mask to reduce unnecessary ray tests
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    radius_squared = maximum(((vx - center_x)^2 + (vy - center_y)^2) for (vx, vy) in zip(xs, ys))

    # Ray casting parameters
    z_min = minimum(v[3] for v in vertices)
    ray_origin_z = z_min - 1.0  # Start 1mm below part
    ray_direction = Vec{3,Float64}(0.0, 0.0, 1.0)

    # Count grid cells that hit the mesh
    hits_count = 0
    for y in y_min:grid_resolution:y_max
        dy_squared = (y - center_y)^2
        for x in x_min:grid_resolution:x_max
            # Skip points outside circular mask
            ((x - center_x)^2 + dy_squared) > radius_squared && continue
            
            # Cast ray and check for intersection
            origin = Point{3,Float64}(x, y, ray_origin_z)
            ray_hits, _ = ray_triangle_intersect(faces, vertices, origin, ray_direction; rayType=:ray, triSide=0)
            !isempty(ray_hits) && (hits_count += 1)
        end
    end

    return hits_count * grid_resolution^2
end

# ============================================================================
# SURFACE AREA CALCULATIONS
# ============================================================================

"""
Calculate area of faces pointing upward (within angle_threshold of +Z).
Used to determine surface quality requirements.
"""
function calculate_up_facing_area(faces, vertices, angle_threshold=45.0)
    up_direction = Vec{3,Float64}(0.0, 0.0, 1.0)
    threshold_radians = deg2rad(angle_threshold)
    face_normals = facenormal(faces, vertices)
    
    total_up_area = 0.0
    for (face, normal) in zip(faces, face_normals)
        angle = acos(clamp(dot(normal, up_direction), -1.0, 1.0))
        
        if angle < threshold_radians
            # Calculate triangle area
            v1, v2, v3 = vertices[face[1]], vertices[face[2]], vertices[face[3]]
            edge1, edge2 = v2 - v1, v3 - v1
            total_up_area += 0.5 * norm(cross(edge1, edge2))
        end
    end
    
    return total_up_area
end

"""
Calculate area of faces pointing downward (within angle_threshold of -Z).
These faces typically require support structures.
"""
function calculate_down_facing_area(faces, vertices, angle_threshold=45.0)
    down_direction = Vec{3,Float64}(0.0, 0.0, -1.0)
    threshold_radians = deg2rad(angle_threshold)
    face_normals = facenormal(faces, vertices)
    
    total_down_area = 0.0
    for (face, normal) in zip(faces, face_normals)
        angle = acos(clamp(dot(normal, down_direction), -1.0, 1.0))
        
        if angle < threshold_radians
            # Calculate triangle area
            v1, v2, v3 = vertices[face[1]], vertices[face[2]], vertices[face[3]]
            edge1, edge2 = v2 - v1, v3 - v1
            total_down_area += 0.5 * norm(cross(edge1, edge2))
        end
    end
    
    return total_down_area
end

"""
Calculate the height of the part's center of gravity.
Lower values indicate better stability during printing.
"""
function calculate_gravity_center_height(vertices)
    return mean(vertices)[3]
end

# ============================================================================
# SUPPORT STRUCTURE GENERATION
# ============================================================================

"""
Identify where support structures are needed and generate support geometry.
Returns external supports (overhangs), internal supports (voids), and affected face indices.
"""
function identify_support_requirements(faces, vertices, overhang_threshold=OVERHANG_THRESHOLD_DEGREES, 
                                      support_diameter=SUPPORT_DIAMETER_MM, density_reduction=SUPPORT_DENSITY_REDUCTION,
                                      support_type="Lattice")

    # Constants
    up_dir = Vec{3,Float64}(0, 0, 1)
    threshold_rad = deg2rad(overhang_threshold)
    normals = facenormal(faces, vertices)
    
    # Mesh bounds
    x_min, x_max = extrema(v[1] for v in vertices)
    y_min, y_max = extrema(v[2] for v in vertices)
    z_min, z_max = extrema(v[3] for v in vertices)
    
    # Support grid
    grid_spacing = support_diameter * max(1.0, density_reduction / 2.0)
    x_points = collect(x_min:grid_spacing:x_max)
    y_points = collect(y_min:grid_spacing:y_max)
    
    external_support_lines = Vector{Vector{Point{3,Float64}}}()
    internal_support_lines = Vector{Vector{Point{3,Float64}}}()
    external_face_indices = Int[]
    internal_face_indices = Int[]
    
    # Identify support columns
    support_columns = Set{Tuple{Int,Int}}() 
    column_heights = Dict{Tuple{Int,Int}, Float64}()

    # Check each grid point for overhangs
    for (xi, x) in enumerate(x_points)
        for (yi, y) in enumerate(y_points)
            # Cast ray upward from below the part
            ray_origin = Point{3,Float64}(x, y, z_min - 1.0)
            ray_dir = Vec{3,Float64}(0, 0, 1)
            
            ray_hits, face_indices = ray_triangle_intersect(faces, vertices, ray_origin, ray_dir; rayType=:ray, triSide=0)
            
            isempty(ray_hits) && continue
            
            # Sort hits by Z (distance along the ray)
            sorted_indices = sortperm([h[3] for h in ray_hits])
            sorted_hits = ray_hits[sorted_indices]
            sorted_faces = face_indices[sorted_indices]
            
            # Check first hit for overhang
            for (hit_point, face_idx) in zip(sorted_hits, sorted_faces)
                face_normal = normals[face_idx]
                angle = acos(clamp(dot(face_normal, up_dir), -1, 1))
                
                if angle > threshold_rad && hit_point[3] > BUILD_PLATE_CLEARANCE
                    push!(support_columns, (xi, yi))
                    column_heights[(xi, yi)] = max(get(column_heights, (xi, yi), 0.0), hit_point[3])
                    push!(external_face_indices, face_idx)
                    break 
                end
            end
        end
    end

   # Generate support structure based on type
    if !isempty(support_columns)
        if support_type == "Linear"
            # Simple vertical pillars
            for (xi, yi) in support_columns
                x, y = x_points[xi], y_points[yi]
                max_height = column_heights[(xi, yi)]
                
                support_bottom = Point{3,Float64}(x, y, 0.0)
                support_top = Point{3,Float64}(x, y, max_height)
                push!(external_support_lines, [support_bottom, support_top])
            end
        else  # Lattice
            generate_lattice_supports!(external_support_lines, support_columns, column_heights, x_points, y_points, grid_spacing)
        end
    end
    
    # Find internal supports (voids requiring support)
    find_internal_supports!(internal_support_lines, internal_face_indices, faces, vertices, normals, x_points, y_points, z_min, threshold_rad)
    
    # Ensure face indices match support lines
    while length(external_face_indices) < length(external_support_lines)
        push!(external_face_indices, external_face_indices[1])
    end
    
    return external_support_lines, internal_support_lines, external_face_indices, internal_face_indices
end

"""
Generate lattice-style support structure with horizontal connections.
"""
function generate_lattice_supports!(support_lines, support_columns, column_heights, x_points, y_points, grid_spacing)
    # Vertical pillars
    for (xi, yi) in support_columns
        x, y = x_points[xi], y_points[yi]
        max_height = column_heights[(xi, yi)]
        
        support_bottom = Point{3,Float64}(x, y, 0.0)
        support_top = Point{3,Float64}(x, y, max_height)
        push!(support_lines, [support_bottom, support_top])
    end
    
    # Horizontal connections at regular intervals
    max_support_height = maximum(values(column_heights))
    lattice_layer_spacing = grid_spacing * 1.5
    z_layers = collect(0.0:lattice_layer_spacing:max_support_height)
    
    for z_layer in z_layers[2:end-1]
        for (xi, yi) in support_columns
            column_heights[(xi, yi)] < z_layer && continue
            
            x, y = x_points[xi], y_points[yi]
            current_point = Point{3,Float64}(x, y, z_layer)
            
            # Connect to neighboring columns
            for (dx, dy) in [(1, 0), (0, 1)]
                neighbor = (xi + dx, yi + dy)
                if neighbor in support_columns && 
                   neighbor[1] <= length(x_points) && 
                   neighbor[2] <= length(y_points) &&
                   column_heights[neighbor] >= z_layer
                    
                   neighbor_x = x_points[neighbor[1]]
                   neighbor_y = y_points[neighbor[2]]
                   neighbor_point = Point{3,Float64}(neighbor_x, neighbor_y, z_layer)
                   push!(support_lines, [current_point, neighbor_point])
                end
            end
        end
    end
end

"""
Find internal voids that require support structures.
"""
function find_internal_supports!(support_lines, face_indices, faces, vertices, normals, x_points, y_points, z_min, threshold_rad)
    up_dir = Vec{3,Float64}(0, 0, 1)
    
    # Sparser grid for internal supports
    for x in x_points[1:2:end]
        for y in y_points[1:2:end]
            ray_origin = Point{3,Float64}(x, y, z_min - 1.0)
            ray_dir = Vec{3,Float64}(0, 0, 1)
            
            ray_hits, hit_faces = ray_triangle_intersect(faces, vertices, ray_origin, ray_dir; rayType=:ray, triSide=0)
            
            length(ray_hits) <= 2 && continue
            
            # Sort hits by Z
            sorted_indices = sortperm([h[3] for h in ray_hits])
            sorted_hits = ray_hits[sorted_indices]
            sorted_faces = hit_faces[sorted_indices]
            
            # Check ceiling surfaces (even-numbered hits)
            for i in 2:2:length(sorted_hits)-1
                face_normal = normals[sorted_faces[i]]
                angle = acos(clamp(dot(face_normal, up_dir), -1, 1))
                
                if angle > threshold_rad
                    support_bottom = sorted_hits[i-1]
                    support_top = sorted_hits[i]
                    gap = support_top[3] - support_bottom[3]
                    
                    # Only add support if gap is significant
                    if gap > MIN_FEATURE_SIZE_MM * 2
                        push!(support_lines, [support_bottom, support_top])
                        push!(face_indices, sorted_faces[i])
                    end
                end
            end
        end
    end
end

"""
Calculate material volume for support structures.
Returns both external and internal volumes
"""
function calculate_support_material_volumes(external_supports, internal_supports; support_diameter=SUPPORT_DIAMETER_MM)
    cross_section_area = π * (support_diameter / 2)^2
    
    # External volume (accounts for overlap in horizontal segments)
    external_volume = 0.0
    for line in external_supports
        length = norm(line[2] - line[1])
        is_horizontal = abs(line[2][3] - line[1][3]) < 0.1
        external_volume += cross_section_area * length * (is_horizontal ? 0.8 : 1.0)
    end
    
    # Internal volume
    internal_volume = sum(cross_section_area * norm(line[2] - line[1]) for line in internal_supports; init=0.0)
    
    return external_volume, internal_volume
end

# ============================================================================
# SLICING
# ============================================================================

"""
Slice the mesh at a given Z height and return contours, area, and perimeter.
"""
function slice_mesh_at_z_height(faces, vertices, z_height)
    # Define slicing plane
    plane_point = Point{3,Float64}(0.0, 0.0, z_height)
    plane_normal = Vec{3,Float64}(0.0, 0.0, 1.0)
    
    # Find all triangle-plane intersections
    intersection_segments = find_triangle_plane_intersections(faces, vertices, plane_point, plane_normal)
    isempty(intersection_segments) && return [], 0.0, 0.0
    
    # Build contours from segments
    contours = build_contours_from_segments(intersection_segments)
    total_area = calculate_contour_area(contours)
    total_perimeter = calculate_contour_perimeter(contours)
    
    return contours, total_area, total_perimeter
end

"""
Find intersection segments between triangles and a plane.
"""
function find_triangle_plane_intersections(faces, vertices, plane_point, plane_normal)
    intersection_segments = Vector{Vector{Point{3,Float64}}}()
    tolerance = 1e-10
    
    for face in faces
        v1, v2, v3 = vertices[face[1]], vertices[face[2]], vertices[face[3]]
        triangle_vertices = [v1, v2, v3]
        
        # Calculate signed distances from plane
        distances = [dot(v - plane_point, plane_normal) for v in triangle_vertices]
        
        # Check if triangle intersects plane
        above = distances .> tolerance
        below = distances .< -tolerance
        (all(above) || all(below)) && continue
        
        # Find intersection points
        intersection_points = Point{3,Float64}[]
        
        for i in 1:3
            j = mod1(i + 1, 3)
            d1, d2 = distances[i], distances[j]
            
            if (d1 > tolerance && d2 < -tolerance) || (d1 < -tolerance && d2 > tolerance)
                # Edge crosses plane
                t = d1 / (d1 - d2)
                point = triangle_vertices[i] + t * (triangle_vertices[j] - triangle_vertices[i])
                push!(intersection_points, point)
            elseif abs(d1) <= tolerance
                # Vertex on plane
                push!(intersection_points, triangle_vertices[i])
            end
        end
        
        # Remove duplicate points
        unique_points = Point{3,Float64}[]
        for point in intersection_points
            if !any(norm(point - p) < tolerance for p in unique_points)
                push!(unique_points, point)
            end
        end
        
        # Add segment if we have exactly 2 points
        length(unique_points) == 2 && push!(intersection_segments, unique_points)
    end
    
    return intersection_segments
end

"""
Build closed contours from intersection segments.
"""
function build_contours_from_segments(segments)
    isempty(segments) && return []
    
    # Build adjacency graph
    adjacency = Dict{Point{3,Float64}, Vector{Point{3,Float64}}}()
    tolerance = 1e-8
    
    # Find matching points within tolerance
    function find_matching_point(target, point_dict)
        for existing in keys(point_dict)
            norm(target - existing) < tolerance && return existing
        end
        return nothing
    end
    
    # Build graph from segments
    for segment in segments
        p1, p2 = segment[1], segment[2]
        
        # Find or create vertices
        existing_p1 = find_matching_point(p1, adjacency)
        existing_p2 = find_matching_point(p2, adjacency)
        
        point1 = existing_p1 === nothing ? p1 : existing_p1
        point2 = existing_p2 === nothing ? p2 : existing_p2
        
        # Add edges
        !haskey(adjacency, point1) && (adjacency[point1] = Point{3,Float64}[])
        !haskey(adjacency, point2) && (adjacency[point2] = Point{3,Float64}[])
        
        push!(adjacency[point1], point2)
        push!(adjacency[point2], point1)
    end
    
    # Extract contours from graph
    visited_edges = Set{Tuple{Point{3,Float64}, Point{3,Float64}}}()
    contours = []
    
    for start_point in keys(adjacency)
        for neighbor in adjacency[start_point]
            # Create normalised edge key
            edge_key = norm(start_point) <= norm(neighbor) ? 
                      (start_point, neighbor) : (neighbor, start_point)
            edge_key in visited_edges && continue
            
            # Build contour
            contour = [start_point]
            current = neighbor
            push!(visited_edges, edge_key)
            
            # Follow edges until closed or dead end
            while current != start_point && haskey(adjacency, current)
                push!(contour, current)
                
                # Find unvisited edge
                next_point = nothing
                for candidate in adjacency[current]
                    candidate_edge = norm(current) <= norm(candidate) ? 
                                   (current, candidate) : (candidate, current)
                    
                    if !(candidate_edge in visited_edges)
                        next_point = candidate
                        push!(visited_edges, candidate_edge)
                        break
                    end
                end
                
                next_point === nothing && break
                current = next_point
            end
            
            # Add valid contours (at least 3 points)
            if length(contour) >= 3
                # Close contour if needed
                if norm(contour[end] - contour[1]) < tolerance
                    push!(contour, contour[1])
                end
                push!(contours, contour)
            end
        end
    end
    
    return contours
end

"""
Calculate the area enclosed by contours using the shoelace formula.
"""
function calculate_contour_area(contours)
    total_area = 0.0
    
    for contour in contours
        length(contour) < 3 && continue
        
        # Project to XY plane and calculate area
        xy_points = [Point{2,Float64}(p[1], p[2]) for p in contour]
        area = 0.0
        n = length(xy_points)
        
        # Shoelace formula
        for i in 1:n-1
            area += xy_points[i][1] * xy_points[i+1][2] - xy_points[i+1][1] * xy_points[i][2]
        end
        
        # Close the polygon if needed
        if n > 2 && xy_points[1] != xy_points[end]
            area += xy_points[end][1] * xy_points[1][2] - xy_points[1][1] * xy_points[end][2]
        end
        
        total_area += abs(area) / 2.0
    end
    
    return total_area
end

"""
Calculate the perimeter of all contours.
"""
function calculate_contour_perimeter(contours)
    total_perimeter = 0.0
    
    for contour in contours
        length(contour) < 2 && continue
        
        # Sum distances between consecutive points
        for i in 1:length(contour)-1
            total_perimeter += norm(contour[i+1] - contour[i])
        end
        
        # Close the contour if needed
        if length(contour) > 2 && contour[1] != contour[end]
            total_perimeter += norm(contour[1] - contour[end])
        end
    end
    
    return total_perimeter
end


"""
Find the maximum cross-sectional area in the XY plane.
Samples the mesh at multiple Z heights.
"""
function find_max_xy_cross_section(faces, vertices, num_slices=50)
    # Get Z range
    z_coords = [v[3] for v in vertices]
    z_min, z_max = extrema(z_coords)
    z_range = z_max - z_min
    
    z_range < 1e-6 && return 0.0, z_min
    
    # Sample heights
    margin = z_range * 0.01
    z_heights = range(z_min + margin, z_max - margin, num_slices)
    
    # Find maximum area
    max_area = 0.0
    max_z_height = z_min
    
    for z_height in z_heights
        _, area, _ = slice_mesh_at_z_height(faces, vertices, z_height)
        if area > max_area
            max_area = area
            max_z_height = z_height
        end
    end
    
    return max_area, max_z_height
end

"""
Create a visual plane mesh for slicing visualisation.
"""
function create_cutting_plane_mesh(vertices, z_height, margin_factor=0.2)
    isempty(vertices) && return Point{3,Float64}[], TriangleFace{Int}[]
    
    # Get XY bounds
    x_min, x_max = extrema(v[1] for v in vertices)
    y_min, y_max = extrema(v[2] for v in vertices)
    
    # Add margin for better visualisation
    margin = margin_factor * max(x_max - x_min, y_max - y_min)
    
    # Create rectangle at z_height
    plane_vertices = [
        Point{3,Float64}(x_min - margin, y_min - margin, z_height),
        Point{3,Float64}(x_max + margin, y_min - margin, z_height),
        Point{3,Float64}(x_max + margin, y_max + margin, z_height),
        Point{3,Float64}(x_min - margin, y_max + margin, z_height)
    ]
    
    plane_faces = [TriangleFace{Int}(1, 2, 3), TriangleFace{Int}(1, 3, 4)]
    
    return plane_vertices, plane_faces
end

# ============================================================================
# TOOL CUTTING
# ============================================================================

"""
Apply vertical offset to all vertices.
"""
function apply_z_offset(vertices, z_offset)
    return [Point{3,Float64}(v[1], v[2], v[3] + z_offset) for v in vertices]
end

"""
Calculate optimal bandsaw cutting plane to remove maximum supports.
"""
function calculate_optimal_bandsaw_plane(vertices, support_lines, z_offset=0.0)
    isempty(support_lines) && return nothing, 0.0, 0.0
    
    # Get part bounds with offset
    offset_vertices = apply_z_offset(vertices, z_offset)
    z_min = minimum(v[3] for v in offset_vertices)
    
    # Analyse support distribution
    support_bounds = analyse_support_distribution(support_lines)
    isempty(support_bounds.z_values) && return nothing, 0.0, 0.0
    
    # Find best cutting plane
    best_plane = nothing
    best_supports_cut = 0
    best_angle = 0.0
    
    clearance = 0.8  # Minimum distance from part
    
    # Try horizontal cuts first
    for height_offset in [0.0, -1.0, -2.0]
        cut_height = z_min - clearance + height_offset
        cut_height < 0.2 && continue  # Don't cut below build plate
        
        plane_point = [support_bounds.center_x, support_bounds.center_y, cut_height]
        plane_normal = [0.0, 0.0, 1.0]
        
        supports_cut = count_supports_cut(support_lines, plane_point, plane_normal)
        
        if supports_cut > best_supports_cut
            best_supports_cut = supports_cut
            best_angle = 0.0
            best_plane = (point=plane_point, normal=plane_normal, direction="horizontal", supports_cut=supports_cut)
        end
    end
    
    # Try angled cuts for better access
    available_height = z_min + z_offset - 0.2
    if available_height > 1.0
        best_plane, best_supports_cut, best_angle = 
            optimise_angled_cuts(support_lines, support_bounds, offset_vertices, clearance, best_plane, best_supports_cut, best_angle)
    end
    
    # Estimate removable volume
    best_volume = best_supports_cut * π * (SUPPORT_DIAMETER_MM / 2)^2 * 2.5
    
    return best_plane, best_volume, best_angle
end

"""
Analyse support distribution for bandsaw optimisation.
"""
function analyse_support_distribution(support_lines)
    support_points = Float64[]
    x_values = Float64[]
    y_values = Float64[]
    z_values = Float64[]
    
    for line in support_lines
        for point in line
            push!(z_values, point[3])
            push!(x_values, point[1])
            push!(y_values, point[2])
        end
    end
    
    if isempty(z_values)
        return (z_values=Float64[], center_x=0.0, center_y=0.0,
                x_range=(0.0, 0.0), y_range=(0.0, 0.0), z_range=(0.0, 0.0))
    end
    
    return (
        z_values = z_values,
        center_x = mean(x_values),
        center_y = mean(y_values),
        x_range = extrema(x_values),
        y_range = extrema(y_values),
        z_range = extrema(z_values)
    )
end

"""
Optimise angled bandsaw cuts for maximum support removal.
"""
function optimise_angled_cuts(support_lines, support_bounds, offset_vertices, clearance, current_best_plane, current_best_count, current_best_angle)
    best_plane = current_best_plane
    best_supports_cut = current_best_count
    best_angle = current_best_angle
    
    z_min = minimum(v[3] for v in offset_vertices)
    base_height = z_min - clearance
    
    # Test different cutting directions and angles
    for direction in ["x", "y", "diagonal"]
        for angle_deg in 5:2:35  # Test angles from 5° to 35°
            angle_rad = deg2rad(angle_deg)
            
            # Calculate plane parameters based on direction
            plane_data = calculate_angled_plane(direction, angle_rad, support_bounds, base_height)
            plane_data === nothing && continue
            
            # Check for better cut
            supports_cut = count_supports_cut(support_lines, plane_data.point, plane_data.normal)
            min_distance = minimum_part_distance(offset_vertices, plane_data.point, plane_data.normal)
            
            if supports_cut > best_supports_cut && min_distance >= clearance
                best_supports_cut = supports_cut
                best_angle = angle_deg
                best_plane = (point=plane_data.point, normal=plane_data.normal, direction=direction, supports_cut=supports_cut)
            end
        end
    end
    
    return best_plane, best_supports_cut, best_angle
end

"""
Calculate angled cutting plane parameters.
"""
function calculate_angled_plane(direction, angle_rad, support_bounds, base_height)
    if direction == "x"
        # Cut along X axis
        plane_normal = [sin(angle_rad), 0.0, cos(angle_rad)]
        max_sweep = (support_bounds.x_range[2] - support_bounds.x_range[1]) * 0.8
        height_drop = tan(angle_rad) * max_sweep
        
        base_height - height_drop < 0.2 && return nothing
        
        plane_point = [support_bounds.center_x, support_bounds.center_y, base_height]
        
    elseif direction == "y"
        # Cut along Y axis
        plane_normal = [0.0, sin(angle_rad), cos(angle_rad)]
        max_sweep = (support_bounds.y_range[2] - support_bounds.y_range[1]) * 0.8
        height_drop = tan(angle_rad) * max_sweep
        
        base_height - height_drop < 0.2 && return nothing
        
        plane_point = [support_bounds.center_x, support_bounds.center_y, base_height]
        
    else  # diagonal
        plane_normal = [sin(angle_rad)*0.707, sin(angle_rad)*0.707, cos(angle_rad)]
        max_sweep = max(support_bounds.x_range[2] - support_bounds.x_range[1],
                       support_bounds.y_range[2] - support_bounds.y_range[1]) * 0.6
        height_drop = tan(angle_rad) * max_sweep
        
        base_height - height_drop < 0.2 && return nothing
        
        plane_point = [support_bounds.center_x, support_bounds.center_y, base_height]
    end
    
    return (point=plane_point, normal=plane_normal)
end

"""
Count how many support lines would be cut by a plane.
"""
function count_supports_cut(support_lines, plane_point, plane_normal)
    supports_cut = 0
    
    for line in support_lines
        start_point, end_point = line[1], line[2]
        
        # Calculate signed distances from plane
        d1 = dot([start_point[1] - plane_point[1], 
                 start_point[2] - plane_point[2], 
                 start_point[3] - plane_point[3]], plane_normal)
        d2 = dot([end_point[1] - plane_point[1], 
                 end_point[2] - plane_point[2], 
                 end_point[3] - plane_point[3]], plane_normal)
        
        # Line crosses plane if endpoints are on opposite sides
        (d1 * d2 <= 0) && (supports_cut += 1)
    end
    
    return supports_cut
end

"""
Calculate minimum distance from plane to part surface.
"""
function minimum_part_distance(vertices, plane_point, plane_normal)
    min_distance = Inf
    
    for vertex in vertices
        distance = abs(dot([vertex[1] - plane_point[1], 
                          vertex[2] - plane_point[2], 
                          vertex[3] - plane_point[3]], plane_normal))
        min_distance = min(min_distance, distance)
    end
    
    return min_distance
end

"""
Create visualisation mesh for bandsaw cutting plane.
"""
function create_bandsaw_plane_mesh(plane_data, vertices, z_offset=0.0)
    plane_data === nothing && return Point{3,Float64}[], TriangleFace{Int}[]
    
    # Get part bounds for sizing
    offset_vertices = apply_z_offset(vertices, z_offset)
    x_min, x_max = extrema(v[1] for v in offset_vertices)
    y_min, y_max = extrema(v[2] for v in offset_vertices)
    
    # Size plane appropriately
    x_size = (x_max - x_min) * 1.3
    y_size = (y_max - y_min) * 1.3
    max_size = max(x_size, y_size)
    
    if plane_data.direction == "horizontal"
        # Simple horizontal rectangle
        plane_vertices = [
            Point{3,Float64}(x_min - x_size*0.15, y_min - y_size*0.15, plane_data.point[3]),
            Point{3,Float64}(x_max + x_size*0.15, y_min - y_size*0.15, plane_data.point[3]),
            Point{3,Float64}(x_max + x_size*0.15, y_max + y_size*0.15, plane_data.point[3]),
            Point{3,Float64}(x_min - x_size*0.15, y_max + y_size*0.15, plane_data.point[3])
        ]

    else
        # Angled plane - orthogonal vectors
        plane_normal = plane_data.normal
        plane_point = plane_data.point
        
        # Find perpendicular vectors in the plane
        if abs(plane_normal[3]) > 0.8
            u = [1.0, 0.0, 0.0]
        else
            u = cross([0.0, 0.0, 1.0], plane_normal)
        end
        
        v1 = cross(plane_normal, u)
        norm(v1) > 0 ? (v1 = v1 ./ norm(v1)) : (v1 = [1.0, 0.0, 0.0])
        v2 = cross(plane_normal, v1)
        v2 = v2 ./ norm(v2)
        
        # Create plane corners
        half_size = max_size * 0.6
        plane_vertices = [
            Point{3,Float64}(plane_point + half_size * v1 + half_size * v2),
            Point{3,Float64}(plane_point - half_size * v1 + half_size * v2),
            Point{3,Float64}(plane_point - half_size * v1 - half_size * v2),
            Point{3,Float64}(plane_point + half_size * v1 - half_size * v2)
        ]
    end
    
    plane_faces = [TriangleFace{Int}(1, 2, 3), TriangleFace{Int}(1, 3, 4)]
    
    return plane_vertices, plane_faces
end

# ============================================================================
# SCORING
# ============================================================================

"""
Normalise metric values to [0, 1] range.
"""
function normalise_metric_values(raw_values)
    min_val, max_val = extrema(raw_values)
    range_val = max_val - min_val
    return range_val ≈ 0 ? zeros(length(raw_values)) : (raw_values .- min_val) ./ range_val
end

"""
Calculate weighted composite scores for orientation comparison.
"""
function calculate_weighted_scores(heights, shadows, volumes, max_crosssections, up_facing_areas, down_facing_areas, 
                                 external_supports, internal_supports, gravity_centers,
                                 weight_height, weight_shadow, weight_volume, weight_crosssection, 
                                 weight_up_facing, weight_down_facing, weight_external, weight_internal, weight_gravity)
    # Normalise all metrics
    norm_heights = normalise_metric_values(heights)
    norm_shadows = normalise_metric_values(shadows)
    norm_volumes = normalise_metric_values(volumes)
    norm_crosssections = normalise_metric_values(max_crosssections)
    norm_up_facing = 1.0 .- normalise_metric_values(up_facing_areas)  # Invert for optimisation
    norm_down_facing = normalise_metric_values(down_facing_areas)
    norm_external = normalise_metric_values(external_supports)
    norm_internal = normalise_metric_values(internal_supports)
    norm_gravity = normalise_metric_values(gravity_centers)
    
    return (weight_height * norm_heights + weight_shadow * norm_shadows + 
            weight_volume * norm_volumes + weight_crosssection * norm_crosssections +
            weight_up_facing * norm_up_facing + weight_down_facing * norm_down_facing +
            weight_external * norm_external + weight_internal * norm_internal +
            weight_gravity * norm_gravity)
end

# ============================================================================
# VISUALISATION 
# ============================================================================

"""
Update slice visualisation including contours and cutting plane.
"""
function update_slice_visualisation(orientation_index, z_height, oriented_meshes, part_faces, 
                                  slice_area_obs, slice_plane_vis, slice_contour_data_obs, show_slice_plane_obs, slice_axis)
    current_mesh = oriented_meshes[orientation_index]
    contours, area, perimeter = slice_mesh_at_z_height(part_faces, current_mesh, z_height)
    slice_area_obs[] = area
    
    # Update slice plane visualisation
    if show_slice_plane_obs[]
        plane_vertices, plane_faces = create_cutting_plane_mesh(current_mesh, z_height)
        if !isempty(plane_vertices)
            plane_mesh = GeometryBasics.Mesh(plane_vertices, plane_faces)
            slice_plane_vis[1] = plane_mesh
            slice_plane_vis.visible = true
        else
            slice_plane_vis.visible = false
        end
    else
        slice_plane_vis.visible = false
    end
    
    # Update contour visualisation
    if !isempty(contours)
        flattened_points = Point{3,Float64}[]
        for contour in contours
            append!(flattened_points, contour)
            push!(flattened_points, Point{3,Float64}(NaN, NaN, NaN))
        end
        slice_contour_data_obs[] = flattened_points
        
        # Auto-scale slice view
        valid_points = filter(p -> !any(isnan.(p)), flattened_points)
        if !isempty(valid_points)
            x_coords = [p[1] for p in valid_points]
            y_coords = [p[2] for p in valid_points]
            x_min, x_max = extrema(x_coords)
            y_min, y_max = extrema(y_coords)
            
            x_range, y_range = x_max - x_min, y_max - y_min
            margin_x, margin_y = x_range * 0.1, y_range * 0.1
            
            xlims!(slice_axis, x_min - margin_x, x_max + margin_x)
            ylims!(slice_axis, y_min - margin_y, y_max + margin_y)
        end
    else
        slice_contour_data_obs[] = Point{3,Float64}[]
        
        # Scale based on mesh bounds at this height
        z_coords = [v[3] for v in current_mesh]
        if z_height >= minimum(z_coords) && z_height <= maximum(z_coords)
            x_coords = [v[1] for v in current_mesh]
            y_coords = [v[2] for v in current_mesh]
            x_min, x_max = extrema(x_coords)
            y_min, y_max = extrema(y_coords)
            
            margin = 0.1 * max(x_max - x_min, y_max - y_min)
            xlims!(slice_axis, x_min - margin, x_max + margin)
            ylims!(slice_axis, y_min - margin, y_max + margin)
        end
    end
    
    return area, perimeter
end

"""
Create interactive spider/radar chart for performance metrics.
"""
function create_spider_chart_interactive(axis, metric_values_obs, metric_names, metric_colors)
    n_metrics = length(metric_names)
    angles = [2π * (i-1) / n_metrics for i in 1:n_metrics]
    
    # Draw grid circles
    for level in [0.2, 0.4, 0.6, 0.8, 1.0]
        circle_pts = [Point2f(level * cos(θ), level * sin(θ)) for θ in range(0, 2π, 101)]
        lines!(axis, circle_pts, color=(:gray, 0.3), linewidth=1)
    end
    
    # Draw radial lines
    for angle in angles
        lines!(axis, [Point2f(0, 0), Point2f(cos(angle), sin(angle))], color=(:gray, 0.3), linewidth=1)
    end
    
    # Draw metric lines and points
    for i in 1:n_metrics
        line_points = @lift begin
            inverted_val = 1.0 - $(metric_values_obs)[i]
            end_point = Point2f(inverted_val * cos(angles[i]), inverted_val * sin(angles[i]))
            [Point2f(0, 0), end_point]
        end
        
        point_pos = @lift begin
            inverted_val = 1.0 - $(metric_values_obs)[i]
            [Point2f(inverted_val * cos(angles[i]), inverted_val * sin(angles[i]))]
        end

        lines!(axis, line_points, color=metric_colors[i], linewidth=4)
        scatter!(axis, point_pos, color=metric_colors[i], markersize=8, 
                strokewidth=2, strokecolor=:white)
    end

    # Add labels
    weight_texts = []
    for (i, (name, color)) in enumerate(zip(metric_names, metric_colors))
        label_pos = 1.12 * Point2f(cos(angles[i]), sin(angles[i]))
        weight_pos = 1.28 * Point2f(cos(angles[i]), sin(angles[i]))
        
        text!(axis, label_pos, text=name, fontsize=8, color=color, align=(:center, :center), font="Arial Bold")
        push!(weight_texts, text!(axis, weight_pos, text="w=0.11", fontsize=7, color=color, align=(:center, :center)))
    end

    # Add scale labels
    for (val, pos) in zip(["0.2", "0.4", "0.6", "0.8", "1.0"], [0.25, 0.45, 0.65, 0.85, 1.05])
        text!(axis, Point2f(pos, 0), text=val, fontsize=8, color=:gray, align=(:left, :center))
    end
    
    axis.aspect = DataAspect()
    xlims!(axis, -1.4, 1.4)
    ylims!(axis, -1.4, 1.4)
    hidedecorations!(axis)
    hidespines!(axis)
    
    return weight_texts
end

# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

"""
Evaluates all orientations before visualisation.
"""
function analyse_all_orientations(test_case::Int, analysis_resolution=1.0, slice_resolution=50)
    part_faces, part_vertices = load_mesh_from_file(test_case)
    orientation_faces, orientation_vertices, opposite_vertices = create_test_orientations()
    
    num_orientations = length(orientation_vertices)
    println("Starting analysis of $num_orientations orientations...")
    
    # Pre-allocate result arrays
    results = (
        height = zeros(Float64, num_orientations),
        shadow = zeros(Float64, num_orientations),
        volume = zeros(Float64, num_orientations),
        max_crosssection = zeros(Float64, num_orientations),
        up_facing_area = zeros(Float64, num_orientations),
        down_facing_area = zeros(Float64, num_orientations),
        gravity_center = zeros(Float64, num_orientations),
        external_support = zeros(Float64, num_orientations),
        internal_support = zeros(Float64, num_orientations)
    )
    
    oriented_meshes = Vector{Vector{Point{3,Float64}}}(undef, num_orientations)
    external_support_data = Vector{Vector{Vector{Point{3,Float64}}}}(undef, num_orientations)
    internal_support_data = Vector{Vector{Vector{Point{3,Float64}}}}(undef, num_orientations)

    # Process orientations in parallel
    Threads.@threads for orientation_index in 1:num_orientations
        build_direction = orientation_vertices[orientation_index]
        
        # Orient mesh and calculate height
        oriented_mesh, height = orient_part_for_printing(part_vertices, build_direction)
        oriented_meshes[orientation_index] = oriented_mesh
        results.height[orientation_index] = height
        
        # Calculate all metrics using oriented mesh
        results.shadow[orientation_index] = 
            calculate_shadow_projection_area(part_faces, oriented_mesh, analysis_resolution)
        
        volume, _ = calculate_bounding_box_metrics(oriented_mesh)
        results.volume[orientation_index] = volume
        
        max_cross, _ = find_max_xy_cross_section(part_faces, oriented_mesh, slice_resolution)
        results.max_crosssection[orientation_index] = max_cross
        
        results.up_facing_area[orientation_index] = 
            calculate_up_facing_area(part_faces, oriented_mesh)
        results.down_facing_area[orientation_index] = 
            calculate_down_facing_area(part_faces, oriented_mesh)
        
        results.gravity_center[orientation_index] = 
            calculate_gravity_center_height(oriented_mesh)
        
        # Support calculations
        ext_supports, int_supports, _, _ = 
            identify_support_requirements(part_faces, oriented_mesh, OVERHANG_THRESHOLD_DEGREES, 
                                         SUPPORT_DIAMETER_MM, SUPPORT_DENSITY_REDUCTION, "Lattice")
        external_support_data[orientation_index] = ext_supports
        internal_support_data[orientation_index] = int_supports
        
        ext_vol, int_vol = calculate_support_material_volumes(ext_supports, int_supports)
        results.external_support[orientation_index] = ext_vol
        results.internal_support[orientation_index] = int_vol
        
        # Progress reporting
        if orientation_index % 10 == 0 || orientation_index == num_orientations
            println("Progress: $orientation_index/$num_orientations orientations analysed")
        end
    end

    # Calculate combined scores
    combined_scores = calculate_weighted_scores(
        results.height, results.shadow, results.volume, results.max_crosssection,
        results.up_facing_area, results.down_facing_area, results.external_support,
        results.internal_support, results.gravity_center,
        0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111
    )
    
    # Find best orientations
    best_orientations = (
        height = argmin(results.height),
        shadow = argmin(results.shadow),
        volume = argmin(results.volume),
        max_crosssection = argmin(results.max_crosssection),
        up_facing_area = argmax(results.up_facing_area),
        down_facing_area = argmin(results.down_facing_area),
        gravity_center = argmin(results.gravity_center),
        external_support = argmin(results.external_support),
        internal_support = argmin(results.internal_support),
        combined = argmin(combined_scores)
    )
    
    return (part_faces, part_vertices, orientation_faces, orientation_vertices, opposite_vertices, oriented_meshes,
            results.height, results.shadow, results.volume, results.max_crosssection,
            results.up_facing_area, results.down_facing_area, results.gravity_center,
            results.external_support, results.internal_support,
            external_support_data, internal_support_data, combined_scores, best_orientations)
end

# ============================================================================
# INTERACTIVE INTERFACE
# ============================================================================

"""
Provides interactive interface with GUI layout and controls.
"""
function create_interactive_interface(part_faces, part_vertices, orientation_faces, orientation_vertices, 
                                    opposite_vertices, oriented_meshes, heights, shadows, volumes, max_crosssections,
                                    up_facing_areas, down_facing_areas, gravity_centres, 
                                    external_volumes, internal_volumes, 
                                    external_supports, internal_supports, scores, optimal_orientations)
    
    main_figure = Figure(size=(1700, 1000), backgroundcolor=:white)
    
    # Initialise observables
    weight_observables = [Observable(0.111) for _ in 1:9]
    weight_locks = [Observable(false) for _ in 1:9]
    current_metric_values = [Observable("0.0") for _ in 1:9]
    weight_height, weight_shadow, weight_volume, weight_crosssection, weight_up_facing, 
    weight_down_facing, weight_gravity, weight_external_support, weight_internal_support = weight_observables
    
    current_scores = Observable(scores)
    best_combined_index = Observable(optimal_orientations.combined)
    selected_orientation_index = Observable(1)
    show_external_supports = Observable(true)
    show_internal_supports = Observable(true)
    slice_z_position = Observable(0.0)
    slice_area = Observable(0.0)
    show_slice_plane = Observable(true)
    z_offset = Observable(0.0) 
    support_type = Observable("Lattice")
    show_bandsaw_plane = Observable(false)

    # Layout structure
    main_controls_layout = GridLayout(main_figure[1:3, 1], tellwidth=true, width=260)
    
    part_view = AxisGeom(main_figure[1, 2:5], title="Live Orientation", aspect=:data)
    cross_section_view = AxisGeom(main_figure[1, 6], title="Cross-Section View", aspect=:data) 

    metrics_layout = GridLayout(main_figure[1, 7], tellwidth=true, width=180)

    performance_sphere_view = AxisGeom(main_figure[2, 6], title="Performance Sphere", aspect=:data)
    opposite_hemisphere_view = AxisGeom(main_figure[2, 7], title="Opposite Sphere", aspect=:data)
    performance_radar_view = Axis(main_figure[3, 6:7], title="Performance Radar")

    metrics_history_layout = GridLayout(main_figure[2:3, 2:5])
    metrics_history_view = Axis(metrics_history_layout[1:2, 1], title="Orientation Scanner")
    metrics_toggles_layout = GridLayout(metrics_history_layout[1:2, 2], tellwidth=true, width=120)

    # Metrics display labels
    metrics_labels = []
    metrics_data = [
        ("Height", "0.0mm"),
        ("Shadow", "0.0cm²"), 
        ("Volume", "0.0cm³"),
        ("Max XY", "0.0cm²"),
        ("Up-Face", "0.0cm²"),
        ("Down-Face", "0.0cm²"),
        ("Gravity", "0.0mm"),
        ("Ext.Sup", "0.0cm³"),
        ("Int.Sup", "0.0cm³"),
        ("Combined", "0.0"),
        ("Z Offset", "0.0mm"),
        ("Supports", "Ext: 0 | Int: 0")
    ]

    for (i, (label, value)) in enumerate(metrics_data)
        row = i
        Label(metrics_layout[row, 1], text=label, fontsize=9, halign=:right, width=60)
        metric_label = Label(metrics_layout[row, 2], text=value, fontsize=9, color=:blue, halign=:left, width=80)
        push!(metrics_labels, metric_label)
    end

    cross_section_view.elevation = π/2
    cross_section_view.azimuth = 0.0

    initial_mesh = oriented_meshes[1]
    overhang_colors = [pi - acos(dot(n, Point{3,Float64}(0,0,1))) for n in vertexnormal(part_faces, initial_mesh)]
    _, initial_box_edges = calculate_bounding_box_metrics(initial_mesh)

    part_mesh_plot = meshplot!(part_view, part_faces, initial_mesh; color=overhang_colors, colormap=:Spectral)
    shadow_mesh_plot = meshplot!(part_view, part_faces, [Point{3,Float64}(v[1], v[2], 0.0) for v in initial_mesh]; color=:black)
    bounding_box_lines = [lines!(part_view, edge, color=:red, linewidth=2) for edge in initial_box_edges]

    external_support_plot_data = Observable(Point{3,Float64}[])
    internal_support_plot_data = Observable(Point{3,Float64}[])
    external_support_visualisation = lines!(part_view, external_support_plot_data, color=:cyan, linewidth=3, transparency=true, alpha=0.8)
    internal_support_visualisation = lines!(part_view, internal_support_plot_data, color=:magenta, linewidth=2, transparency=true, alpha=0.6)

    original_oriented_meshes = deepcopy(oriented_meshes)
    original_external_supports = deepcopy(external_supports)
    original_internal_supports = deepcopy(internal_supports)

    initial_empty_mesh = GeometryBasics.Mesh(Point{3,Float64}[], TriangleFace{Int}[])
    slice_plane_visualisation = meshplot!(part_view, initial_empty_mesh, color=(:yellow, 0.3), strokewidth=2, strokecolor=:yellow)
    bandsaw_plane_visualisation = meshplot!(part_view, initial_empty_mesh, color=(:orange, 0.4), strokewidth=3, strokecolor=:red)

    sphere_surface = meshplot!(performance_sphere_view, orientation_faces, orientation_vertices; color=scores, strokewidth=0.3, transparency=true, colormap=Reverse(:Spectral))
    sphere_points = scatter!(performance_sphere_view, orientation_vertices, color=scores, markersize=5, colormap=Reverse(:Spectral))

    hidedecorations!(performance_sphere_view)
    hidespines!(performance_sphere_view)
    
    opposite_surface = meshplot!(opposite_hemisphere_view, orientation_faces, opposite_vertices; color=scores, strokewidth=0.3, transparency=true, colormap=Reverse(:Spectral), colorrange=sphere_surface.colorrange[])
    opposite_points = scatter!(opposite_hemisphere_view, opposite_vertices, color=scores, markersize=4, colormap=Reverse(:Spectral), colorrange=sphere_points.colorrange[])
    
    hidedecorations!(opposite_hemisphere_view)
    hidespines!(opposite_hemisphere_view)

    direction_indicator_line = lines!(performance_sphere_view, [Point{3,Float64}(0,0,0), orientation_vertices[1]], color=:purple, linewidth=6)
    direction_indicator_point = scatter!(performance_sphere_view, [Point{3,Float64}(0,0,0), orientation_vertices[1]], color=[:yellow, :red], markersize=[15, 18], strokewidth=3, strokecolor=:white)

    current_orientation = orientation_vertices[1]
    corresponding_opposite = opposite_vertices[1]
    opposite_direction_line = lines!(opposite_hemisphere_view, [Point{3,Float64}(0,0,0), corresponding_opposite], color=:purple, linewidth=6)
    opposite_direction_point = scatter!(opposite_hemisphere_view, [Point{3,Float64}(0,0,0), corresponding_opposite], color=[:yellow, :red], markersize=[12, 15], strokewidth=2, strokecolor=:white)

    slice_contour_data = Observable(Point{3,Float64}[])
    slice_contour_visualisation = lines!(cross_section_view, slice_contour_data, color=:red, linewidth=3, transparency=true, alpha=0.9)

    metric_names = ["Height", "Shadow", "Volume", "MaxXY", "UpArea", "DownArea", "Gravity", "Ext.Sup", "Int.Sup"]
    metric_colors = [:red, :green, :orange, :blue, :lime, :darkred, :purple, :cyan, :magenta]
    
    norm_metrics = [
        normalise_metric_values(heights),
        normalise_metric_values(shadows),
        normalise_metric_values(volumes),
        normalise_metric_values(max_crosssections),
        1.0 .- normalise_metric_values(up_facing_areas),
        normalise_metric_values(down_facing_areas),
        normalise_metric_values(gravity_centres),
        normalise_metric_values(external_volumes),
        normalise_metric_values(internal_volumes)
    ]
    
    current_spider_metrics = Observable([nm[1] for nm in norm_metrics])
    weight_text_elements = create_spider_chart_interactive(performance_radar_view, current_spider_metrics, metric_names, metric_colors)

    # Metrics history plot
    orientation_indices = 1:length(orientation_vertices)
    history_plots = []
    metric_visibility = [Observable(true) for _ in 1:9]  # All metrics visible by default
    
    for (i, (name, color)) in enumerate(zip(metric_names, metric_colors))
        metric_data = norm_metrics[i]
        line_plot = lines!(metrics_history_view, orientation_indices, metric_data, 
                          color=color, linewidth=2, label=name, visible=metric_visibility[i])
        push!(history_plots, line_plot)
    end
    
    # Current orientation indicator line
    current_orientation_line = vlines!(metrics_history_view, [1], color=:black, linewidth=3, linestyle=:dash)
    
    xlims!(metrics_history_view, 1, length(orientation_vertices))
    ylims!(metrics_history_view, 0, 1)
    metrics_history_view.xlabel = "Orientation Index"
    metrics_history_view.ylabel = "Normalised Metric Value"

    Label(metrics_toggles_layout[1, 1], "Show Metrics:", fontsize=10, halign=:center)
    
    metric_toggles = []
    for (i, (name, color)) in enumerate(zip(metric_names, metric_colors))
        toggle_row = GridLayout(metrics_toggles_layout[i+1, 1])
        toggle = Toggle(toggle_row[1, 1], active=true, width=16)
        Label(toggle_row[1, 2], name, fontsize=8, color=color, halign=:left, width=55)
        push!(metric_toggles, toggle)
        
        on(toggle.active) do is_active
            metric_visibility[i][] = is_active
            history_plots[i].visible = is_active
        end
    end

    # Main controls
    z_coords_initial = [v[3] for v in oriented_meshes[1]]
    z_min_initial, z_max_initial = extrema(z_coords_initial)

    viewing_modes = ["Live Orientation", "Best Height", "Best Shadow", "Best Bounding Box", 
                    "Best Max XY Cross-Section", "Best Up-Facing Area", "Best Down-Facing Area",
                    "Best Gravity Centre", "Best External Support", "Best Internal Support", "Best Combined"]

    row = 1
    
    Label(main_controls_layout[row, 1], "Mode:", fontsize=10, halign=:left)
    row += 1
    mode_dropdown = Menu(main_controls_layout[row, 1], options=viewing_modes, default="Live Orientation", width=240)
    row += 1
    
    Label(main_controls_layout[row, 1], "", fontsize=3)
    row += 1

    Label(main_controls_layout[row, 1], "Orientation:", fontsize=10, halign=:left)
    row += 1
    orientation_control = Slider(main_controls_layout[row, 1], range=1:length(orientation_vertices), startvalue=1, width=240)
    row += 1
    
    Label(main_controls_layout[row, 1], "", fontsize=3)
    row += 1

    Label(main_controls_layout[row, 1], "Slice Height:", fontsize=10, halign=:left)
    row += 1
    slice_height_control = Slider(main_controls_layout[row, 1], range=z_min_initial:0.1:z_max_initial, startvalue=(z_min_initial + z_max_initial)/2, width=240)
    row += 1
    slice_area_display = Label(main_controls_layout[row, 1], text=@lift("Area: $(round($(slice_area) * MM_TO_CM^2, digits=1)) cm²"), fontsize=9, halign=:left)
    row += 1
    
    Label(main_controls_layout[row, 1], "", fontsize=3)
    row += 1

    Label(main_controls_layout[row, 1], "Display Options:", fontsize=10, halign=:left)
    row += 1

    toggles_grid = GridLayout(main_controls_layout[row, 1])
    external_support_toggle = Toggle(toggles_grid[1, 1], active=true, width=16)
    Label(toggles_grid[1, 2], "Ext.Supports", fontsize=8, color=:cyan, width=60)
    internal_support_toggle = Toggle(toggles_grid[1, 3], active=true, width=16)
    Label(toggles_grid[1, 4], "Int.Supports", fontsize=8, color=:magenta, width=60)
    
    slice_plane_toggle = Toggle(toggles_grid[2, 1], active=true, width=16)
    Label(toggles_grid[2, 2], "Slice Plane", fontsize=8, color=:yellow, width=60)
    bandsaw_plane_toggle = Toggle(toggles_grid[2, 3], active=false, width=16)
    Label(toggles_grid[2, 4], "Bandsaw", fontsize=8, color=:orange, width=60)
    row += 1
    
    Label(main_controls_layout[row, 1], "", fontsize=3)
    row += 1

    Label(main_controls_layout[row, 1], "Support Type:", fontsize=10, halign=:left)
    row += 1
    support_type_dropdown = Menu(main_controls_layout[row, 1], options=["Lattice", "Linear"], default="Lattice", width=240)
    row += 1
    
    Label(main_controls_layout[row, 1], "", fontsize=3)
    row += 1

    Label(main_controls_layout[row, 1], "Z Offset:", fontsize=10, halign=:left)
    row += 1
    z_offset_control = Slider(main_controls_layout[row, 1], range=-10.0:0.1:50.0, startvalue=0.0, width=240)
    row += 1
    z_offset_display = Label(main_controls_layout[row, 1], text=@lift("Offset: $(round($(z_offset), digits=1)) mm"), fontsize=9, halign=:left)
    row += 1
    
    Label(main_controls_layout[row, 1], "", fontsize=3)
    row += 1

    # Weight controls
    Label(main_controls_layout[row, 1], "Optimisation Weights:", fontsize=11, halign=:center)
    row += 1

    weight_controls = [
        ("Height", :red, weight_height, weight_locks[1]),
        ("Shadow", :green, weight_shadow, weight_locks[2]), 
        ("Volume", :orange, weight_volume, weight_locks[3]),
        ("MaxXY", :blue, weight_crosssection, weight_locks[4]),
        ("UpArea", :lime, weight_up_facing, weight_locks[5]),
        ("DownArea", :darkred, weight_down_facing, weight_locks[6]),
        ("Gravity", :purple, weight_gravity, weight_locks[7]),
        ("ExtSup", :cyan, weight_external_support, weight_locks[8]),
        ("IntSup", :magenta, weight_internal_support, weight_locks[9])
    ]

    weight_sliders = []
    weight_lock_buttons = []
    weight_value_displays = []

    for (i, (label, color, weight_obs, lock_obs)) in enumerate(weight_controls)
        weight_row_grid = GridLayout(main_controls_layout[row, 1])
        
        Label(weight_row_grid[1, 1], label, fontsize=8, color=color, halign=:right, width=50)
        slider = Slider(weight_row_grid[1, 2], range=0.0:0.01:1.0, startvalue=0.111, width=120)
        lock_button = Button(weight_row_grid[1, 3], label="L", width=18, fontsize=6)
        weight_display = Button(weight_row_grid[1, 4], label="0.11", buttoncolor=:lightgreen, fontsize=6, width=25)
        
        push!(weight_sliders, slider)
        push!(weight_lock_buttons, lock_button)
        push!(weight_value_displays, weight_display)
        
        row += 1
    end

    # Support regeneration and visualisation
    function regenerate_supports_with_offset(orientation_index, offset_value, current_support_type)
        if offset_value == 0.0 && current_support_type == "Lattice"
            return original_external_supports[orientation_index], original_internal_supports[orientation_index]
        end
        
        offset_mesh = apply_z_offset(original_oriented_meshes[orientation_index], offset_value)
        
        ext_supports, int_supports, _, _ = identify_support_requirements(part_faces, offset_mesh, 
                                                                       OVERHANG_THRESHOLD_DEGREES, 
                                                                       SUPPORT_DIAMETER_MM, 
                                                                       SUPPORT_DENSITY_REDUCTION,
                                                                       current_support_type)
        
        return ext_supports, int_supports
    end

    function update_supports_visualisation(orientation_index, offset_value, display_mode, current_support_type)
        ext_supports, int_supports = regenerate_supports_with_offset(orientation_index, offset_value, current_support_type)
        
        if show_bandsaw_plane[]
            try
                current_mesh = original_oriented_meshes[orientation_index]
                plane_data, volume_removed, cutting_angle = calculate_optimal_bandsaw_plane(current_mesh, ext_supports, offset_value)
                
                if plane_data !== nothing
                    plane_vertices, plane_faces = create_bandsaw_plane_mesh(plane_data, current_mesh, offset_value)
                    if !isempty(plane_vertices) && !isempty(plane_faces)
                        bandsaw_mesh = GeometryBasics.Mesh(plane_vertices, plane_faces)
                        bandsaw_plane_visualisation[1] = bandsaw_mesh
                        bandsaw_plane_visualisation.visible = true
                    else
                        bandsaw_plane_visualisation.visible = false
                    end
                else
                    bandsaw_plane_visualisation.visible = false
                end
            catch e
                println("Bandsaw plane error: ", e)
                bandsaw_plane_visualisation.visible = false
            end
        else
            bandsaw_plane_visualisation.visible = false
        end
        
        if show_external_supports[] && (display_mode in ["Live Orientation", "Best External Support", "Best Combined"])
            flattened_external_points = Point{3,Float64}[]
            for support_line in ext_supports
                append!(flattened_external_points, support_line)
                push!(flattened_external_points, Point{3,Float64}(NaN, NaN, NaN))
            end
            external_support_plot_data[] = flattened_external_points
            external_support_visualisation.visible = true
        else
            external_support_plot_data[] = Point{3,Float64}[]
            external_support_visualisation.visible = false
        end
        
        if show_internal_supports[] && (display_mode in ["Live Orientation", "Best Internal Support", "Best Combined"])
            flattened_internal_points = Point{3,Float64}[]
            for support_line in int_supports
                append!(flattened_internal_points, support_line)
                push!(flattened_internal_points, Point{3,Float64}(NaN, NaN, NaN))
            end
            internal_support_plot_data[] = flattened_internal_points
            internal_support_visualisation.visible = true
        else
            internal_support_plot_data[] = Point{3,Float64}[]
            internal_support_visualisation.visible = false
        end
        
        return ext_supports, int_supports
    end

    function update_spider_chart(orientation_index, weights)
        current_spider_metrics[] = [nm[orientation_index] for nm in norm_metrics]
    end

    function update_live_metrics(orientation_index)
        metrics_values = [
            "$(round(heights[orientation_index], digits=1))mm",
            "$(round(shadows[orientation_index] * MM_TO_CM^2, digits=1))cm²", 
            "$(round(volumes[orientation_index] * MM_TO_CM^3, digits=1))cm³",
            "$(round(max_crosssections[orientation_index] * MM_TO_CM^2, digits=1))cm²",
            "$(round(up_facing_areas[orientation_index] * MM_TO_CM^2, digits=1))cm²",
            "$(round(down_facing_areas[orientation_index] * MM_TO_CM^2, digits=1))cm²",
            "$(round(gravity_centres[orientation_index], digits=1))mm",
            "$(round(external_volumes[orientation_index] * MM_TO_CM^3, digits=2))cm³",
            "$(round(internal_volumes[orientation_index] * MM_TO_CM^3, digits=2))cm³",
            "$(round(current_scores[][orientation_index], digits=3))",
            "$(round(z_offset[], digits=1))mm",
            "Ext: $(length(external_supports[orientation_index])) | Int: $(length(internal_supports[orientation_index]))"
        ]
        
        for (i, value) in enumerate(metrics_values)
            if i <= length(metrics_labels)
                metrics_labels[i].text = value
            end
        end
        
        for (i, display) in enumerate(weight_value_displays)
            display.label = "$(round(weight_observables[i][], digits=2))"
        end
    end

    function update_3d_visualisation(target_orientation, display_mode)
        selected_orientation_index[] = target_orientation
        current_offset = z_offset[]
        
        target_mesh = if current_offset == 0.0
            oriented_meshes[target_orientation]
        else
            apply_z_offset(original_oriented_meshes[target_orientation], current_offset)
        end
        
        oriented_meshes[target_orientation] = target_mesh
        
        part_mesh_plot[1] = GeometryBasics.Mesh(target_mesh, part_faces)
        
        z_coords = [v[3] for v in target_mesh]
        z_min, z_max = extrema(z_coords)
        slice_height_control.range = z_min:0.1:z_max
        
        current_slice_height = slice_z_position[]
        if current_slice_height < z_min || current_slice_height > z_max
            slice_z_position[] = (z_min + z_max) / 2
            set_close_to!(slice_height_control, slice_z_position[])
        end
        
        if display_mode == "Live Orientation"
            overhang_coloring = [pi - acos(dot(n, Point{3,Float64}(0,0,1))) for n in vertexnormal(part_faces, target_mesh)]
            part_mesh_plot.color = overhang_coloring
        else
            part_mesh_plot.color = fill(0.5, length(target_mesh))
        end
        
        shadow_visible = display_mode in ["Live Orientation", "Best Shadow", "Best Combined"]
        bbox_visible = display_mode in ["Live Orientation", "Best Bounding Box", "Best Combined"]
        
        shadow_mesh_plot.visible = shadow_visible
        if shadow_visible
            shadow_mesh_plot[1] = GeometryBasics.Mesh([Point{3,Float64}(v[1], v[2], 0.0) for v in target_mesh], part_faces)
        end
        
        if bbox_visible
            _, updated_box_edges = calculate_bounding_box_metrics(target_mesh)
            for (i, edge) in enumerate(updated_box_edges)
                if i <= length(bounding_box_lines)
                    bounding_box_lines[i][1] = edge
                    bounding_box_lines[i].visible = true
                end
            end
        else
            for line in bounding_box_lines
                line.visible = false
            end
        end

        update_supports_visualisation(target_orientation, current_offset, display_mode, support_type[])
    
        direction_points = [Point{3,Float64}(0,0,0), orientation_vertices[target_orientation]]
        direction_indicator_line[1] = direction_points
        direction_indicator_point[1] = direction_points

        opposite_direction_points = [Point{3,Float64}(0,0,0), opposite_vertices[target_orientation]]
        opposite_direction_line[1] = opposite_direction_points
        opposite_direction_point[1] = opposite_direction_points
        
        # Update metrics history plot indicator
        current_orientation_line[1] = [target_orientation]
        
        update_slice_visualisation(target_orientation, slice_z_position[], oriented_meshes, part_faces, 
                                 slice_area, slice_plane_visualisation, slice_contour_data, 
                                 show_slice_plane, cross_section_view)

        current_weights = [obs[] for obs in weight_observables]
        update_spider_chart(target_orientation, current_weights)

        update_live_metrics(target_orientation)
        part_view.title = display_mode
    end

    function update_sphere_visualisation(display_mode)
        color_data = if display_mode == "Best Height"
            heights
        elseif display_mode == "Best Shadow"
            shadows
        elseif display_mode == "Best Bounding Box"
            volumes
        elseif display_mode == "Best Max XY Cross-Section"
            max_crosssections
        elseif display_mode == "Best Up-Facing Area"
            maximum(up_facing_areas) .- up_facing_areas .+ minimum(up_facing_areas)
        elseif display_mode == "Best Down-Facing Area"
            down_facing_areas
        elseif display_mode == "Best Gravity Centre"
            gravity_centres
        elseif display_mode == "Best External Support"
            external_volumes
        elseif display_mode == "Best Internal Support"
            internal_volumes
        else
            current_scores[]
        end

        sphere_surface.color = color_data
        sphere_points.color = color_data
        opposite_surface.color = color_data
        opposite_points.color = color_data

        indicator_color = if display_mode == "Best Height"
            :red
        elseif display_mode == "Best Shadow"
            :green
        elseif display_mode == "Best Bounding Box"
            :orange
        elseif display_mode == "Best Max XY Cross-Section"
            :blue
        elseif display_mode == "Best Up-Facing Area"
            :lime
        elseif display_mode == "Best Down-Facing Area"
            :darkred
        elseif display_mode == "Best Gravity Centre"
            :purple
        elseif display_mode == "Best External Support"
            :cyan
        elseif display_mode == "Best Internal Support"
            :magenta
        else
            :purple
        end
        
        direction_indicator_line.color = indicator_color
        direction_indicator_point.color = [indicator_color, indicator_color]
        opposite_direction_line.color = indicator_color
        opposite_direction_point.color = [indicator_color, indicator_color]
    end

    # Event handlers
    on(mode_dropdown.selection) do selected_mode
        target_index = if selected_mode == "Best Height"
            optimal_orientations.height
        elseif selected_mode == "Best Shadow"
            optimal_orientations.shadow
        elseif selected_mode == "Best Bounding Box"
            optimal_orientations.volume
        elseif selected_mode == "Best Max XY Cross-Section"
            optimal_orientations.max_crosssection
        elseif selected_mode == "Best Up-Facing Area"
            optimal_orientations.up_facing_area
        elseif selected_mode == "Best Down-Facing Area"
            optimal_orientations.down_facing_area
        elseif selected_mode == "Best Gravity Centre"
            optimal_orientations.gravity_center
        elseif selected_mode == "Best External Support"
            optimal_orientations.external_support
        elseif selected_mode == "Best Internal Support"
            optimal_orientations.internal_support
        elseif selected_mode == "Best Combined"
            best_combined_index[]
        else
            selected_orientation_index[]
        end
        
        update_sphere_visualisation(selected_mode)
        update_3d_visualisation(target_index, selected_mode)
        
        if selected_mode != "Live Orientation"
            set_close_to!(orientation_control, target_index)
        end
    end

    on(orientation_control.value) do slider_value
        mode_dropdown.selection[] == "Live Orientation" && update_3d_visualisation(slider_value, "Live Orientation")
    end

    on(slice_height_control.value) do slice_height
        slice_z_position[] = slice_height
        update_slice_visualisation(selected_orientation_index[], slice_z_position[], oriented_meshes, part_faces, 
                                 slice_area, slice_plane_visualisation, slice_contour_data, show_slice_plane, cross_section_view)
    end

    on(external_support_toggle.active) do is_active
        show_external_supports[] = is_active
        update_3d_visualisation(selected_orientation_index[], mode_dropdown.selection[])
    end

    on(internal_support_toggle.active) do is_active
        show_internal_supports[] = is_active
        update_3d_visualisation(selected_orientation_index[], mode_dropdown.selection[])
    end

    on(slice_plane_toggle.active) do is_active
        show_slice_plane[] = is_active
        update_slice_visualisation(selected_orientation_index[], slice_z_position[], oriented_meshes, part_faces, 
                                 slice_area, slice_plane_visualisation, slice_contour_data, show_slice_plane, cross_section_view)
    end

    on(bandsaw_plane_toggle.active) do is_active
        show_bandsaw_plane[] = is_active
        update_3d_visualisation(selected_orientation_index[], mode_dropdown.selection[])
    end

    on(z_offset_control.value) do offset_value
        z_offset[] = offset_value
        update_3d_visualisation(selected_orientation_index[], mode_dropdown.selection[])
    end

    on(support_type_dropdown.selection) do selected_support_type
        support_type[] = selected_support_type
        update_3d_visualisation(selected_orientation_index[], mode_dropdown.selection[])
    end

    # Weight balancing system
    weight_update_in_progress = Ref(false)
    
    function rebalance_all_weights()
        weight_update_in_progress[] && return
        weight_update_in_progress[] = true
        
        raw_weight_values = [slider.value[] for slider in weight_sliders]
        lock_states = [weight_locks[i][] for i in 1:9]
        
        locked_total = sum(raw_weight_values[i] for i in 1:9 if lock_states[i]; init=0.0)
        unlocked_indices = findall(.!lock_states)
        
        if locked_total >= 1.0 || isempty(unlocked_indices)
            if locked_total >= 1.0
                locked_indices = findall(lock_states)
                if !isempty(locked_indices)
                    locked_sum = sum(raw_weight_values[i] for i in locked_indices; init=0.0)
                    if locked_sum > 0
                        for i in locked_indices
                            raw_weight_values[i] = raw_weight_values[i] / locked_sum
                        end
                    end
                end
                for i in unlocked_indices
                    raw_weight_values[i] = 0.0
                end
            end
        else
            remaining_weight = 1.0 - locked_total
            unlocked_sum = sum(raw_weight_values[i] for i in unlocked_indices; init=0.0)
            
            if unlocked_sum > 0
                for i in unlocked_indices
                    raw_weight_values[i] = (raw_weight_values[i] / unlocked_sum) * remaining_weight
                end
            else
                equal_share = remaining_weight / length(unlocked_indices)
                for i in unlocked_indices
                    raw_weight_values[i] = equal_share
                end
            end
        end

        for (i, (slider, weight)) in enumerate(zip(weight_sliders, raw_weight_values))
            if !lock_states[i]
                set_close_to!(slider, weight)
            end
            weight_observables[i][] = weight
        end
        
        for (i, display) in enumerate(weight_value_displays)
            display.label = "$(round(raw_weight_values[i], digits=2))"
        end
        
        updated_scores = calculate_weighted_scores(heights, shadows, volumes, max_crosssections,
                                                up_facing_areas, down_facing_areas, gravity_centres,
                                                external_volumes, internal_volumes, raw_weight_values...)
        current_scores[] = updated_scores
        new_best_index = argmin(updated_scores)
        best_combined_index[] = new_best_index

        active_mode = mode_dropdown.selection[]
        if active_mode == "Best Combined"
            update_3d_visualisation(new_best_index, active_mode)
            set_close_to!(orientation_control, new_best_index)
        end
        
        update_sphere_visualisation(active_mode)
        update_spider_chart(selected_orientation_index[], raw_weight_values)
        
        weight_update_in_progress[] = false
    end

    for button in weight_lock_buttons
        button.buttoncolor = :lightgray
    end

    for (i, slider) in enumerate(weight_sliders)
        on(slider.value) do val
            if !weight_locks[i][]
                weight_observables[i][] = val
                weight_value_displays[i].label = "$(round(val, digits=2))"
                rebalance_all_weights()
            end
        end
    end

    for (i, lock_button) in enumerate(weight_lock_buttons)
        on(lock_button.clicks) do n
            current_lock_state = weight_locks[i][]
            weight_locks[i][] = !current_lock_state
            
            if weight_locks[i][]
                lock_button.label = "U"
                lock_button.buttoncolor = :lightcoral
            else
                lock_button.label = "L"
                lock_button.buttoncolor = :lightgray
                rebalance_all_weights()
            end
        end
    end

    z_coords_init = [v[3] for v in oriented_meshes[1]]
    z_min_init, z_max_init = extrema(z_coords_init)
    slice_z_position[] = (z_min_init + z_max_init) / 2
    set_close_to!(slice_height_control, slice_z_position[])
    
    update_slice_visualisation(1, slice_z_position[], oriented_meshes, part_faces, 
                             slice_area, slice_plane_visualisation, slice_contour_data, 
                             show_slice_plane, cross_section_view)
    
    bandsaw_plane_visualisation.visible = false
    
    update_live_metrics(1)

    return main_figure
end

# ============================================================================
# MAIN LAUNCH
# ============================================================================

function run_printing_orientation_analysis(test_case_number=1, window_title="PBF-LB Print Orientation Analysis", resolution=1.0, slice_resolution=50)
    analysis_results = analyse_all_orientations(test_case_number, resolution, slice_resolution)
    println("Creating interface...")
    interface = create_interactive_interface(analysis_results...)
    
    display_window = GLMakie.Screen(title=window_title)
    display(display_window, interface)
    
    return interface, display_window
end

# ============================================================================
# EXAMPLES
# ============================================================================

# figure, window = run_printing_orientation_analysis(4, "Cervical Cage Analysis", 0.5, 30)
figure, window = run_printing_orientation_analysis(1, "Cervical Cage Analysis STL", 0.5, 30)
# figure, window = run_printing_orientation_analysis(3, "Baby Groot 3MF Analysis", 3.0, 20)
# figure, window = run_printing_orientation_analysis(1, "Stanford Bunny", 0.5, 30)
