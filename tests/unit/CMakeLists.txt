# KernelSmith Unit Tests (using Google Test)

# Find GTest
find_package(GTest)

if(GTest_FOUND)
  enable_testing()

  add_executable(kernelsmith-unit-tests
    KernelDialectTest.cpp
  )

  target_link_libraries(kernelsmith-unit-tests
    GTest::gtest
    GTest::gtest_main
    KernelSmithDialect
    MLIRIR
    MLIRSupport
  )

  gtest_discover_tests(kernelsmith-unit-tests)

  # Register with CTest
  add_test(NAME kernelsmith-unit
    COMMAND kernelsmith-unit-tests
  )
else()
  message(STATUS "GTest not found, skipping unit tests")
endif()
