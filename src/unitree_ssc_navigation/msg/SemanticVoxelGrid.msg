# Semantic voxel grid from SSC network
Header header

# Grid dimensions
uint32 width    # x dimension
uint32 height   # y dimension
uint32 depth    # z dimension

# Grid origin and resolution
geometry_msgs/Point origin
float32 resolution  # voxel size in meters

# Flattened voxel data (row-major: x * height * depth + y * depth + z)
SemanticVoxel[] voxels
