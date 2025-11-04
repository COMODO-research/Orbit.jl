using Comodo
using GLMakie
using GeometryBasics
using LinearAlgebra
using FileIO
using Base.Threads

TOOL_PATH = "C:\\Users\\Karl\\Downloads\\1250_polygon_sphere_100mm.STL"
PART_PATH = joinpath(comododir(), "assets", "stl", "stanford_bunny_low.stl")
OVERHANG_ANGLE = 45.0
GEOSPHERE_REFINEMENT = 4        # n=2 gives 162 tool approach directions
RAY_TEST_REFINEMENT = "quick"   # "quick" = 3 hardcoded rays, or geosphere subdivision number (0=12 rays, 1=42 rays)
MIN_CLEARANCE = 0.0             # Safety margin in mm (0.0 to disable)
NUM_SAMPLES = 1520              # Number of tool surface samples (Higher = slower but more accurate) [n > Tool Vertex Count = all points tested]

function detect_system_threads()
    """Detect available CPU threads and give recommendations"""
    total_threads = Sys.CPU_THREADS
    physical_estimate = max(1, total_threads ÷ 2) # Assumes hyperthreading
   
    return (
        total = total_threads,
        physical = physical_estimate,
        recommended = total_threads
    )
end

function print_threading_status()
    """Startup instructions for multi-threading"""
    current = Threads.nthreads()
    sys_info = detect_system_threads()
   
    println("\n" * "="^60)
    println("THREADING STATUS")
    println("="^60)

    println("Total logical cores: $(sys_info.total)")
    println("Estimated physical cores: $(sys_info.physical)")
    println("Currently using: $current thread$(current > 1 ? "s" : "")")
   
    if current == 1
        println("\nPERFORMANCE WARNING: Running on single thread")
        println("\nTo enable multi-threading, exit Julia and restart with:")
        println("  julia --threads auto, (uses $(sys_info.physical) threads)")
        println("  julia --threads $(sys_info.total), (uses all $(sys_info.total) threads)")
        println("\nOr set environment variable before starting Julia:")
        println("  PowerShell:  \$env:JULIA_NUM_THREADS = \"$(sys_info.total)\"")
        println("  CMD:         set JULIA_NUM_THREADS=$(sys_info.total)")
       
        # Estimated speedup
        println("\nExpected speedup with $(sys_info.total) threads: $(min(sys_info.recommended))x faster")
        println("="^60)
    else
        efficiency = round(100 * current / sys_info.recommended, digits=1)
        println("\nMulti-threading enabled")
        println("  Using $current / $(sys_info.recommended) recommended threads ($efficiency%)")
        println("")
    end
    return true
end

function load_mesh(filepath::String)
    M = load(filepath)
    F = tofaces(faces(M))
    V = topoints(coordinates(M))
    F, V, _ = mergevertices(F, V)
    return F, V
end

function identify_support_points(F, V, overhang_deg)
    vertical = Vec3f(0, 0, 1)
    threshold_rad = deg2rad(overhang_deg)
    needs_support = fill(false, length(V))
   
    for face in F
        v1, v2, v3 = V[face[1]], V[face[2]], V[face[3]]
        normal = normalize(cross(v2 - v1, v3 - v1))
        angle = acos(clamp(dot(normal, vertical), -1.0, 1.0))
       
        if angle > (π/2 + threshold_rad)
            needs_support[face[1]] = true
            needs_support[face[2]] = true
            needs_support[face[3]] = true
        end
    end
   
    return findall(needs_support)
end

function generate_test_directions(refinement_level::Int)
    F, V = geosphere(refinement_level, 1.0)
    directions = [Vec3f(v[1], v[2], v[3]) for v in V] # Convert vertices to Vec3f directions (3D Vector locations)
    return directions
end

function find_closest_vertex_in_direction(vertices, direction)
    # Find vertex furthest in the given direction (on tool surface)
    best_idx = 1
    best_proj = dot(vertices[1], direction)
   
    for i in 2:length(vertices) # Start from 2 since 1 is initial best
        proj = dot(vertices[i], direction)
        if proj > best_proj
            best_proj = proj
            best_idx = i
        end
    end
   
    return vertices[best_idx]
end

function count_ray_intersections(origin, direction, part_F, part_V)
    # Count how many times a ray intersects the part mesh
    count = 0
   
    for face in part_F
        hit = ray_triangle_intersect(face, part_V, origin, direction; rayType=:ray, triSide=0, tolEps=1e-9)
       
        if !isnan(hit[1])
            dist = dot(hit - origin, direction)
            if dist > 1e-7
                count += 1
            end
        end
    end
   
    return count
end

function is_point_inside_part(point, part_F, part_V, ray_dirs)
    # Test first two rays for inside/outside
    count1 = count_ray_intersections(point, ray_dirs[1], part_F, part_V)
    inside1 = isodd(count1)
   
    count2 = count_ray_intersections(point, ray_dirs[2], part_F, part_V)
    inside2 = isodd(count2)
   
    # If first two agree, exit early
    if inside1 == inside2
        return inside1
    end
   
    # Disagreement - use 3rd ray as tiebreaker
    if length(ray_dirs) >= 3
        count3 = count_ray_intersections(point, ray_dirs[3], part_F, part_V)
        inside3 = isodd(count3)
       
        # Majority vote
        inside_votes = sum([inside1, inside2, inside3])
        return inside_votes >= 2
    else
        # Only two rays provided - default to inside
        return true
    end
end

function distance_to_mesh_surface(point, part_F, part_V)
    min_dist = Inf
   
    # Sample subset of faces for performance (every 5th face)
    for i in 1:5:length(part_F)
        face = part_F[i]
        v1, v2, v3 = part_V[face[1]], part_V[face[2]], part_V[face[3]]
       
        # Approximate distance using triangle centroid
        centroid = (v1 + v2 + v3) / 3
        dist = norm(point - centroid)
        min_dist = min(min_dist, dist)
    end
   
    return min_dist
end

function calculate_local_exclusion_zone(tool_positioned, contact_point, part_V, support_point)
    """Exclusion zone = contact triangle radius"""
    
    # Tool contact triangle
    tool_distances = [norm(Vec3f(v) - Vec3f(contact_point)) for v in tool_positioned]
    sort!(tool_distances)
    tool_contact_radius = tool_distances[3]  # 3rd vertex = contact triangle boundary
    
    # Part contact triangle
    part_distances = [norm(Vec3f(v) - Vec3f(support_point)) for v in part_V]
    sort!(part_distances)
    part_contact_radius = part_distances[3]

    return max(tool_contact_radius, part_contact_radius)
end

function check_tool_collision(tool_positioned, part_F, part_V, contact_point, ray_dirs, exclusion_zone)
    sample_step = max(1, div(length(tool_positioned), NUM_SAMPLES))
    
    for i in 1:sample_step:length(tool_positioned)
        v = Vec3f(tool_positioned[i])
        
        # Skip vertices in exclusion zone
        if norm(v - Vec3f(contact_point)) < exclusion_zone
            continue
        end
        
        # Test for collision (inside part)
        if is_point_inside_part(v, part_F, part_V, ray_dirs)
            return true
        end
        
        # Test for minimum clearance
        if MIN_CLEARANCE > 0.0
            dist = distance_to_mesh_surface(v, part_F, part_V)
            if dist < MIN_CLEARANCE && norm(v - Vec3f(contact_point)) > exclusion_zone
                return true
            end
        end
    end
    
    return false
end

function check_support_reachable(support_point, tool_V, part_F, part_V, directions, ray_dirs)
    for direction in directions
        tool_contact_point = find_closest_vertex_in_direction(tool_V, direction)
        offset = support_point - tool_contact_point
        tool_positioned = [v + offset for v in tool_V]
       
        exclusion_zone = calculate_local_exclusion_zone(tool_positioned, support_point, part_V, support_point)
       
        if !check_tool_collision(tool_positioned, part_F, part_V, support_point, ray_dirs, exclusion_zone)
            return true
        end
    end
   
    return false
end

function main()
    if !print_threading_status()
        return nothing
    end
   
    println("="^60)
    println("TOOL REACHABILITY CHECK")
    println("="^60)


    println("\n1) Loading meshes")
    tool_F, tool_V = load_mesh(TOOL_PATH)
    part_F, part_V = load_mesh(PART_PATH)
   
    # Centre tool at origin and get radius
    tool_center = sum(tool_V) / length(tool_V)
    tool_V = [v - tool_center for v in tool_V]
    tool_radius = maximum([norm(v) for v in tool_V])
   
    println("   Tool: $(length(tool_V)) vertices, radius=$(round(tool_radius, digits=2))")
    println("   Part: $(length(part_V)) vertices, $(length(part_F)) faces")
   
    println("\n2) Finding support points (overhang > $(OVERHANG_ANGLE)°)...")
    support_indices = identify_support_points(part_F, part_V, OVERHANG_ANGLE)
    println("   Support points: $(length(support_indices))")
   
    if length(support_indices) == 0
        println("\n   No support points found")
        return
    end
   
    # Generate test directions using geosphere
    println("\n3) Generating test directions (geosphere n=$GEOSPHERE_REFINEMENT)...")
    test_directions = generate_test_directions(GEOSPHERE_REFINEMENT)
    println("   Directions: $(length(test_directions))")
   
    # Generate ray directions for inside/outside testing
    if RAY_TEST_REFINEMENT == "quick"
        ray_directions = [Vec3f(1, 0, 0), Vec3f(0, 1, 0), Vec3f(0.707, 0.707, 0)]
        println("   Ray test: quick mode (3 hardcoded rays)")
    else
        ray_directions = generate_test_directions(RAY_TEST_REFINEMENT)
        println("   Ray test directions: $(length(ray_directions)) (geosphere n=$RAY_TEST_REFINEMENT)")
    end
   
    println("\n4) Checking reachability...")
    reachable = fill(false, length(support_indices))
   
    # Thread-safe progress counter
    progress_counter = Base.Threads.Atomic{Int}(0)
    last_reported = Base.Threads.Atomic{Int}(0)
    start_time = time()
   
    # Parallel loop using @threads
    @threads for i in 1:length(support_indices)
        support_idx = support_indices[i]
        support_pt = part_V[support_idx]
       
        reachable[i] = check_support_reachable(support_pt, tool_V, part_F, part_V, test_directions, ray_directions)
       
        # Thread-safe progress reporting
        current = Base.Threads.atomic_add!(progress_counter, 1)
       
        if current % 10 == 0 || current == length(support_indices)
            old_val = current - 10
            if current == length(support_indices)
                old_val = last_reported[]
            end


            if Base.Threads.atomic_cas!(last_reported, old_val, current) == old_val || current == length(support_indices)
                elapsed = time() - start_time
                rate = current / elapsed
                eta = (length(support_indices) - current) / rate
                print("\r   Progress: $current / $(length(support_indices)) | ")
                print("$(round(elapsed, digits=1))s elapsed | ")
                print("ETA: $(round(eta, digits=1))s   ")
                flush(stdout)
            end
        end
    end
   
    total_time = time() - start_time
    println("\r   Progress: $(length(support_indices)) / $(length(support_indices)) - Done in $(round(total_time, digits=2))s   ")
   
    # Results
    num_reachable = sum(reachable)
    coverage = 100 * num_reachable / length(reachable)
   
    println("\n" * "="^60)
    println("RESULTS:")
    println("  Reachable: $num_reachable / $(length(support_indices))")
    println("  Coverage: $(round(coverage, digits=1))%")
    println("="^60)
   
    # Visualise
    fig = Figure(size=(1200, 800))
    ax = AxisGeom(fig[1, 1], title="Support Coverage: $(round(coverage, digits=1))%")
   
    vertex_colours = fill(RGBf(0.7, 0.7, 0.7), length(part_V))
    for (i, support_idx) in enumerate(support_indices)
        vertex_colours[support_idx] = reachable[i] ?
            RGBf(0.2, 0.9, 0.2) : RGBf(0.9, 0.2, 0.2)
    end
   
    meshplot!(ax, part_F, part_V, color=vertex_colours, strokewidth=0.5)
    display(fig)
   
    return support_indices, reachable
end

main()