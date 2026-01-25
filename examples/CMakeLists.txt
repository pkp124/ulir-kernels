# KernelSmith Examples

# Example MLIR files are just for reference and documentation
# No build targets needed

# Copy examples to build directory for easy access
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION ${CMAKE_BINARY_DIR}/examples
  FILES_MATCHING
  PATTERN "*.mlir"
  PATTERN "*.py"
  PATTERN "README.md"
)
