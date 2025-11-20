using Comodo
using GLMakie
using GeometryBasics
using LinearAlgebra
using Statistics
using FileIO
using Base.Threads
using Rotations

TOOL_PATH = "C:\\Users\\24242721\\Downloads\\Cervical Cage v1.0 v2.stl"
PART_PATH = joinpath(comododir(), "assets", "stl", "stanford_bunny_low.stl")

OVERHANG_ANGLE_DEG = 45.0
MIN_CLEARANCE_MM = 0.0

# Tool Contact Search Region
TOOL_ORIENTATION = "up" # "up" or "down" (to flip tool vertically if needed)
TOOL_THICKNESS_MM = 1 # Thickness of tool region to consider for contact points
DISC_INNER_RADIUS_MM = nothing # Minimum radius (excludes center), set to nothing to disable
DISC_OUTER_RADIUS_MM = nothing  # Maximum radius (excludes far edges), set to nothing to disable

# Rotation Testing (for rotating tool around support points)
ROTATION_SUBDIVISIONS = 0
TEST_BOTH_HEMISPHERES = false

# Ray Testing (for collision detection and inside/outside checks)
RAY_TEST_MODE = "quick"
MINIMUM_RAY_DISTANCE = 1e-7
PROGRESS_UPDATE_INTERVAL = 10

# Visualisation colours
DEFAULT_VERTEX_COLOUR = RGBf(0.7, 0.7, 0.7)
REACHABLE_COLOUR = RGBf(0.2, 0.9, 0.2)
UNREACHABLE_COLOUR = RGBf(0.9, 0.2, 0.2)

# ============================================================================
# Threading
# ============================================================================

"""
    detect_system_threads()

Detect the number of logical and physical CPU threads.
"""
function detect_system_threads()
    total_threads = Sys.CPU_THREADS
    physical_estimate = max(1, total_threads ÷ 2)
    return (total = total_threads, physical = physical_estimate, recommended = total_threads)
end

"""
    print_threading_status()

Print the current threading status and recommendations.
"""
function print_threading_status()
    current_threads = Threads.nthreads()
    sys_info = detect_system_threads()
   
    println("\n" * "="^60)
    println("THREADING STATUS")
    println("="^60)
    println("Logical cores: $(sys_info.total)")
    println("Physical cores (estimated): $(sys_info.physical)")
    println("Active threads: $current_threads")
   
    if current_threads == 1
        println("\nWARNING: Single-threaded mode")
        println("To enable multi-threading, restart Julia with:")
        println("  julia --threads auto")
        println("\nExpected speedup: ~$(sys_info.recommended)x")
    else
        efficiency = round(100 * current_threads / sys_info.recommended, digits=1)
        println("Multi-threading enabled ($efficiency% of recommended)")
    end
    println("="^60)
end

# ============================================================================
# Mesh Processing
# ============================================================================

"""
    load_mesh(filepath)

Load a mesh from the given filepath and return faces and vertices
"""
function load_mesh(filepath::String)
    mesh = load(filepath)
    faces = tofaces(Comodo.faces(mesh))
    vertices = topoints(coordinates(mesh))
    faces, vertices, _ = mergevertices(faces, vertices)
    return faces, vertices
end

"""
    orient_tool_vertical_pca(vertices)

Orient tool so its principal axis (longest direction through the data) aligns with Z axis.
Uses Principal Component Analysis (PCA) to find the axis of maximum variance.

This makes the tool stand upright, regardless of how it's oriented in the STL file.
"""
function orient_tool_vertical_pca(vertices)
    # Convert to matrix for PCA (N x 3)
    n = length(vertices)
    coords = zeros(n, 3)
    for (i, v) in enumerate(vertices)
        coords[i, :] = [v[1], v[2], v[3]]
    end

    # Center the data
    centroid = vec(mean(coords, dims=1))
    centered = coords .- centroid'
   
    # Principal axis = eigenvector with largest eigenvalue
    cov_matrix = (centered' * centered) / n
    eigenvals, eigenvecs = eigen(cov_matrix)
    principal_axis = Vec3f(eigenvecs[:, 3])  # Last column = largest eigenvalue

    # Target to align principal axis with Z-axis [0, 0, 1]
    target_axis = Vec3f(0, 0, 1)
    rotation = rotation_between(principal_axis, target_axis)
   
    # Apply rotation to all vertices
    oriented_vertices = similar(vertices)
    for (i, v) in enumerate(vertices)
        v_centered = Vec3f(v) - Vec3f(centroid)
        v_rotated = rotation * v_centered
        oriented_vertices[i] = v_rotated + Vec3f(centroid)
    end

    z_coords = [v[3] for v in oriented_vertices]
    z_range = maximum(z_coords) - minimum(z_coords)
   
    return oriented_vertices
end

"""
    apply_user_orientation(vertices, orientation_setting)

Apply user-specified orientation: "up" or "down" to flip the tool vertically if needed.
"""
function apply_user_orientation(vertices, orientation_setting::String)
    if orientation_setting == "up"
        println("\n   User setting: TOOL_ORIENTATION = \"up\"")
        return vertices
    elseif orientation_setting == "down"
        println("\n   User setting: TOOL_ORIENTATION = \"down\"")
        flipped_vertices = [Vec3f(v[1], -v[2], -v[3]) for v in vertices]
        return flipped_vertices
    else
        error("TOOL_ORIENTATION must be \"up\" or \"down\"")
    end
end

"""
    select_contact_vertices(vertices, thickness_mm, inner_radius_mm, outer_radius_mm, orientation)

Select contact vertices based on orientation, thickness, and radial filters.
"""
function select_contact_vertices(vertices, thickness_mm, inner_radius_mm, outer_radius_mm, orientation::String)
    z_coords = [v[3] for v in vertices]
    z_max = maximum(z_coords)
    z_min = minimum(z_coords)

    if orientation == "up"
        z_region_min = z_max - thickness_mm
        z_region_max = z_max
        println("     (Top $(thickness_mm) mm of tool)")
    else
        z_region_min = z_min
        z_region_max = z_min + thickness_mm
        println("     (Bottom $(thickness_mm) mm of tool)")
    end
   
    selected_indices = Int[]
   
    for (i, v) in enumerate(vertices)
        if !(z_region_min <= v[3] <= z_region_max)
            continue
        end

        radial_dist = sqrt(v[1]^2 + v[2]^2) # Distance from Z axis
       
        # Inner radius filter (excludes center)
        if !isnothing(inner_radius_mm) && radial_dist < inner_radius_mm
            continue
        end
       
        # Outer radius filter (excludes far edges)
        if !isnothing(outer_radius_mm) && radial_dist > outer_radius_mm
            continue
        end
       
        push!(selected_indices, i)
    end
   
    println("   Selected: $(length(selected_indices)) / $(length(vertices)) contact vertices")
   
    return selected_indices
end

"""
    preview_contact_vertices(tool_faces, tool_vertices, allowed_indices)

Visualise the tool mesh and highlight the selected contact vertices.
"""
function preview_contact_vertices(tool_faces, tool_vertices, allowed_indices)
    println("\n   Opening preview window...")
   
    mesh_colours = fill(RGBf(0.7, 0.7, 0.7), length(tool_vertices))
    vertex_colours = fill(RGBf(0.3, 0.3, 0.3), length(tool_vertices))
    vertex_sizes = fill(2.0, length(tool_vertices))
   
    for idx in allowed_indices
        mesh_colours[idx] = RGBf(0.8, 0.0, 0.0)
        vertex_colours[idx] = RGBf(1.0, 0.0, 0.0)
        vertex_sizes[idx] = 8.0
    end
   
    fig = Figure(size=(1200, 900))
    ax = Axis3(fig[1, 1],
              title="Tool Contact Vertices Preview\n$(length(allowed_indices)) / $(length(tool_vertices)) vertices selected (RED)",
              aspect=:data)

    mesh!(ax, tool_vertices, tool_faces, color=mesh_colours, transparency=true, alpha=0.6)
    points = [Point3f(v[1], v[2], v[3]) for v in tool_vertices]
    scatter!(ax, points, color=vertex_colours, markersize=vertex_sizes)

    display(fig)

    println("\n   Press Enter to continue, or Ctrl+C to cancel")
    readline()
end

# ============================================================================
# Rotation Orientations
# ============================================================================

"""
    create_rotation_orientations(subdivisions, test_both_hemispheres)

Generate rotation orientations based on geodesic sphere subdivisions.
If test_both_hemispheres is false, only return orientations in the upper hemisphere.
"""
function create_rotation_orientations(subdivisions::Int, test_both_hemispheres::Bool)
    sphere_faces, sphere_vertices = geosphere(subdivisions, 1.0)
    orientations = [Vec3f(v[1], v[2], v[3]) for v in sphere_vertices]
   
    if test_both_hemispheres
        return orientations
    else
        upper_orientations = [o for o in orientations if o[3] >= 0]
        return upper_orientations
    end
end

"""
    rotate_tool_around_point(tool_vertices, rotation_axis, center_point)

Rotate the tool vertices around a specified center point using the given rotation axis.
"""
function rotate_tool_around_point(tool_vertices, rotation_axis::Vec3f, center_point::Vec3f)
    default_up = Vec3f(0, 0, 1)
    rotation = rotation_between(default_up, rotation_axis)
   
    rotated_vertices = similar(tool_vertices)
    for (i, v) in enumerate(tool_vertices)
        v_centered = Vec3f(v) - center_point
        v_rotated = rotation * v_centered
        rotated_vertices[i] = v_rotated + center_point
    end
   
    return rotated_vertices
end

# ============================================================================
# Support Point Identification
# ============================================================================

"""
    identify_support_vertices(faces, vertices, overhang_angle_deg)

Identify support vertices based on overhang angle threshold.
"""
function identify_support_vertices(faces, vertices, overhang_angle_deg)
    vertical = Vec3f(0, 0, 1)
    threshold_rad = deg2rad(overhang_angle_deg)
    needs_support = fill(false, length(vertices))
   
    for face in faces
        v1, v2, v3 = vertices[face[1]], vertices[face[2]], vertices[face[3]]
        normal = normalize(cross(v2 - v1, v3 - v1))
        angle_to_vertical = acos(clamp(dot(normal, vertical), -1.0, 1.0))
       
        if angle_to_vertical > (π/2 + threshold_rad)
            needs_support[face[1]] = true
            needs_support[face[2]] = true
            needs_support[face[3]] = true
        end
    end
   
    return findall(needs_support)
end

# ============================================================================
# Ray Testing
# ============================================================================

"""
        generate_ray_test_directions(mode)

Generate ray test directions based on the specified mode.
"""
function generate_ray_test_directions(mode)
    if mode == "quick"
        return [Vec3f(1, 0, 0), Vec3f(0, 1, 0), Vec3f(0, 0, 1)]
    else
        sphere_faces, sphere_vertices = geosphere(parse(Int, mode), 1.0)
        return [Vec3f(v[1], v[2], v[3]) for v in sphere_vertices]
    end
end

"""
    count_ray_intersections(origin, direction, faces, vertices)

Count how many triangle faces a ray intersects.
Ray:      P_ray(t) = Origin + t x Direction [line in 3D]
Triangle: P_tri(u,v) = V0 + (u x E1) + (v x E2) [plane in 3D]
          where E1 = V1 - V0, E2 = V2 - V0
   
Intersection happens when: P_ray(t) = P_tri(u,v):
    Origin + t x Direction = V0 + (u x E1) + (v x E2)
    t x Direction - (u x E1) - (v x E2) = V0 - Origin

    LINEAR SYSTEM: A x [t, u, v]ᵀ = b
   
    Solve for (t, u, v) using Cramer's rule (determinants).
   
    The "root" is the value of t where intersection occurs.
   
    After solving for (t, u, v), check:
   
    1. t > 0 = Intersection is ahead of ray origin (not behind)
   
    2. u ≥ 0 = Inside triangle boundary (barycentric coord 1)
   
    3. v ≥ 0 = Inside triangle boundary (barycentric coord 2)
   
    4. u + v ≤ 1 = Inside triangle (not past opposite edge)


    Barycentric: (u, v, 1-u-v) are weights at vertices
   
    If all 4 pass then ray passed through face

Reference: Möller & Trumbore (1997) "Fast, Minimum Storage Ray-Triangle Intersection"
"""
function count_ray_intersections(origin, direction, faces, vertices)
    intersection_count = 0
   
    for face in faces
        # Returns intersection point if hit, NaN if miss
        hit_point = ray_triangle_intersect(face, vertices, origin, direction; rayType=:ray, triSide=0, tolEps=1e-9)
        if !isnan(hit_point[1])
            distance = dot(hit_point - origin, direction)
            if distance > MINIMUM_RAY_DISTANCE
                intersection_count += 1
            end
        end
    end
   
    return intersection_count
end

"""
    is_point_inside_mesh(point, faces, vertices, ray_directions)

Determine if a point is inside a closed mesh using ray casting.

Cast multiple rays from the point outward in different directions and count how many triangle faces each ray intersects.
If point is inside the ray crosses ODD number of faces (1, 3, 5, ...)
If point is outside the ray crosses EVEN number of faces (0, 2, 4, ...)
"""
function is_point_inside_mesh(point, faces, vertices, ray_directions)
    inside_votes = 0
   
    for ray_dir in ray_directions
        count = count_ray_intersections(point, ray_dir, faces, vertices)
        if isodd(count)
            inside_votes += 1
        end
    end

    return inside_votes > (length(ray_directions) ÷ 2)
end

"""
    point_to_triangle_distance(point, v1, v2, v3)

Calculate the minimum distance from a point to a triangle defined by vertices v1, v2, v3.
This is used for minimum clearance checks.
"""
function point_to_triangle_distance(point, v1, v2, v3)
    point_vec = point - v1
    edge1 = v2 - v1
    edge2 = v3 - v1
   
    d11 = dot(edge1, edge1)
    d12 = dot(edge1, edge2)
    d22 = dot(edge2, edge2)
    dp1 = dot(point_vec, edge1)
    dp2 = dot(point_vec, edge2)
   
    denom = d11 * d22 - d12 * d12
   
    if abs(denom) < 1e-7
        return min(norm(point - v1), norm(point - v2), norm(point - v3))
    end
   
    u = (d22 * dp1 - d12 * dp2) / denom
    v = (d11 * dp2 - d12 * dp1) / denom
   
    if u >= 0 && v >= 0 && (u + v) <= 1
        projected_point = v1 + u * edge1 + v * edge2
        return norm(point - projected_point)
    end
   
    t1 = clamp(dot(point - v1, edge1) / d11, 0.0, 1.0)
    closest1 = v1 + t1 * edge1
    dist1 = norm(point - closest1)
   
    edge3 = v3 - v2
    t2 = clamp(dot(point - v2, edge3) / dot(edge3, edge3), 0.0, 1.0)
    closest2 = v2 + t2 * edge3
    dist2 = norm(point - closest2)
   
    edge4 = v1 - v3
    t3 = clamp(dot(point - v3, edge4) / dot(edge4, edge4), 0.0, 1.0)
    closest3 = v3 + t3 * edge4
    dist3 = norm(point - closest3)
   
    return min(dist1, dist2, dist3)
end

"""
    calculate_distance_to_surface(point, faces, vertices)

Calculate the minimum distance from a point to the surface of a mesh defined by faces and vertices.
This is used for minimum clearance checks.
"""
function calculate_distance_to_surface(point, faces, vertices)
    min_distance = Inf
   
    for face in faces
        v1, v2, v3 = vertices[face[1]], vertices[face[2]], vertices[face[3]]
        distance = point_to_triangle_distance(point, v1, v2, v3)
        min_distance = min(min_distance, distance)
    end
   
    return min_distance
end

# ============================================================================
# Collision Detection
# ============================================================================

"""
    check_tool_collision(tool_vertices, part_faces, part_vertices, contact_vertex_idx, ray_directions)

Check if the tool collides with the part mesh, excluding the contact vertex.
"""
function check_tool_collision(tool_vertices, part_faces, part_vertices, contact_vertex_idx, ray_directions)
    for i in 1:length(tool_vertices)
        if i == contact_vertex_idx
            continue
        end
       
        vertex = Vec3f(tool_vertices[i])
       
        if is_point_inside_mesh(vertex, part_faces, part_vertices, ray_directions)
            return true
        end
       
        if MIN_CLEARANCE_MM > 0.0
            distance = calculate_distance_to_surface(vertex, part_faces, part_vertices)
            if distance < MIN_CLEARANCE_MM
                return true
            end
        end
    end
   
    return false
end

"""
    is_support_point_reachable(support_vertex, tool_vertices, part_faces, part_vertices, contact_indices, rotation_orientations, ray_directions)

Determine if a support point is reachable by the tool with rotation testing.
"""
function is_support_point_reachable(support_vertex, tool_vertices, part_faces, part_vertices, contact_indices, rotation_orientations, ray_directions)
    for contact_idx in contact_indices
        for rotation_axis in rotation_orientations
            support_vec = Vec3f(support_vertex[1], support_vertex[2], support_vertex[3])
            rotated_tool = rotate_tool_around_point(tool_vertices, rotation_axis, support_vec)
           
            rotated_contact_vertex = rotated_tool[contact_idx]
            offset = support_vec - rotated_contact_vertex
            positioned_tool = [v + offset for v in rotated_tool]
           
            has_collision = check_tool_collision(positioned_tool, part_faces, part_vertices, contact_idx, ray_directions)
           
            if !has_collision
                return true
            end
        end
    end
   
    return false
end

# ============================================================================
# Progress Tracking
# ============================================================================

mutable struct ProgressTracker
    counter::Base.Threads.Atomic{Int}
    last_reported::Base.Threads.Atomic{Int}
    start_time::Float64
    total::Int
end

"""
    ProgressTracker(total)

For tracking progress of multi-threaded tasks.
"""
function ProgressTracker(total::Int)
    return ProgressTracker(
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{Int}(0),
        time(),
        total
    )
end

"""
    update_progress!(tracker)

Update and print progress.
"""
function update_progress!(tracker::ProgressTracker)
    current = Base.Threads.atomic_add!(tracker.counter, 1)

    if current % PROGRESS_UPDATE_INTERVAL == 0 || current == tracker.total
        old_val = current == tracker.total ? tracker.last_reported[] : current - PROGRESS_UPDATE_INTERVAL
       
        if Base.Threads.atomic_cas!(tracker.last_reported, old_val, current) == old_val ||
           current == tracker.total
            elapsed = time() - tracker.start_time
            rate = current / elapsed
            eta = (tracker.total - current) / rate
           
            print("\r   Progress: $current / $(tracker.total) | ")
            print("$(round(elapsed, digits=1))s elapsed | ")
            print("ETA: $(round(eta, digits=1))s   ")
            flush(stdout)
        end
    end
end

# ============================================================================
# Visualisation
# ============================================================================

"""
    visualise_results(part_faces, part_vertices, support_indices, reachable, coverage_pct)

Visualise the part mesh with support vertices coloured by reachability.
"""
function visualise_results(part_faces, part_vertices, support_indices, reachable, coverage_pct)
    fig = Figure(size=(1200, 800))
    ax = Axis3(fig[1, 1], title="Support Coverage: $(round(coverage_pct, digits=1))%", aspect=:data)

    vertex_colours = fill(DEFAULT_VERTEX_COLOUR, length(part_vertices))
   
    for (i, support_idx) in enumerate(support_indices)
        vertex_colours[support_idx] = reachable[i] ? REACHABLE_COLOUR : UNREACHABLE_COLOUR
    end

    meshplot!(ax, part_faces, part_vertices, color=vertex_colours, strokewidth=0.5)
    display(fig)
end

# ============================================================================
# Main Analysis
# ============================================================================

"""
    analyse_tool_reachability()

Main function to analyse tool reachability on the part mesh.
"""
function analyse_tool_reachability()
    print_threading_status()
   
    println("\n" * "="^60)
    println("TOOL REACHABILITY ANALYSIS")
    println("="^60)
   
    # Load meshes
    println("\n1) Loading meshes...")
    tool_faces, tool_vertices_raw = load_mesh(TOOL_PATH)
    part_faces, part_vertices = load_mesh(PART_PATH)
   
    println("   Tool: $(length(tool_vertices_raw)) vertices, $(length(tool_faces)) faces")
    println("   Part: $(length(part_vertices)) vertices, $(length(part_faces)) faces")
    tool_vertices_vertical = orient_tool_vertical_pca(tool_vertices_raw)
    tool_vertices = apply_user_orientation(tool_vertices_vertical, TOOL_ORIENTATION)
    z_coords = [v[3] for v in tool_vertices]
    println("   Tool Z range: $(round(minimum(z_coords), digits=1)) to $(round(maximum(z_coords), digits=1)) mm")
   
    # Select contact vertices
    println("\n2) Selecting contact vertices...")
    contact_indices = select_contact_vertices(tool_vertices, TOOL_THICKNESS_MM, DISC_INNER_RADIUS_MM,DISC_OUTER_RADIUS_MM,TOOL_ORIENTATION)
    println("   Contact vertices: $(length(contact_indices))")

    if isempty(contact_indices)
        println("\n   ERROR: No contact vertices found")
        return nothing
    end

    preview_contact_vertices(tool_faces, tool_vertices, contact_indices)
   
    # Rotation orientations
    println("\n3) Generating rotation orientations...")
    rotation_orientations = create_rotation_orientations(ROTATION_SUBDIVISIONS, TEST_BOTH_HEMISPHERES)
    println("   Rotation orientations: $(length(rotation_orientations))")
   
    # Identify support points
    println("\n4) Identifying support vertices (overhang > $(OVERHANG_ANGLE_DEG)°)...")
    support_indices = identify_support_vertices(part_faces, part_vertices, OVERHANG_ANGLE_DEG)
    println("   Support vertices: $(length(support_indices))")
   
    if isempty(support_indices)
        println("\n   No support vertices found")
        return nothing
    end

    # Configuration summary
    total_configurations = length(contact_indices) * length(rotation_orientations)
    total_tests = total_configurations * length(support_indices)
    println("\n5) Configuration summary:")
    println("   Configurations per support point:")
    println("     = $(length(contact_indices)) contact vertices")
    println("       x $(length(rotation_orientations)) orientations")
    println("     = $total_configurations configurations")
    println("   ")
    println("   Total collision tests:")
    println("     = $total_configurations configurations/support")
    println("       x $(length(support_indices)) support points")
    println("     = $(total_tests) tests")
   
    # Ray directions
    println("\n6) Generating ray test directions...")
    ray_directions = generate_ray_test_directions(RAY_TEST_MODE)
    println("   Ray test mode: $RAY_TEST_MODE")
   
    # Analyse reachability
    println("\n7) Analysing reachability...")
    reachable = fill(false, length(support_indices))
    tracker = ProgressTracker(length(support_indices))
   
    @threads for i in 1:length(support_indices)
        support_idx = support_indices[i]
        support_vertex = part_vertices[support_idx]
       
        reachable[i] = is_support_point_reachable(support_vertex, tool_vertices, part_faces, part_vertices, contact_indices, rotation_orientations, ray_directions)
       
        update_progress!(tracker)
    end
   
    total_time = time() - tracker.start_time
    println("\r   Completed in $(round(total_time, digits=2))s" * " "^20)
   
    # Results
    num_reachable = sum(reachable)
    coverage_pct = 100 * num_reachable / length(reachable)
   
    println("\n" * "="^60)
    println("RESULTS")
    println("="^60)
    println("Reachable: $num_reachable / $(length(support_indices))")
    println("Coverage: $(round(coverage_pct, digits=1))%")
    println("Configurations tested: $total_tests")
    println("="^60)
   
    visualise_results(part_faces, part_vertices, support_indices, reachable, coverage_pct)
   
    return support_indices, reachable
end

analyse_tool_reachability()
