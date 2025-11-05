
"""
    orbitdir()

Returns the package path 

# Description 
This function simply returns the string for the Orbit path. This is helpful for 
instance to load items, such as meshes, from the `assets` folder. 
"""
function orbitdir()
    pkgdir(@__MODULE__)
end
