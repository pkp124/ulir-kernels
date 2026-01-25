//===----------------------------------------------------------------------===//
// KernelSmith Dialect Unit Tests
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "KernelSmith/Dialect/Kernel/KernelDialect.h"

namespace {

class KernelDialectTest : public ::testing::Test {
protected:
  void SetUp() override {
    context.loadDialect<kernelsmith::ks::KSDialect>();
  }

  mlir::MLIRContext context;
};

TEST_F(KernelDialectTest, DialectLoads) {
  // Verify dialect is registered
  auto *dialect = context.getLoadedDialect<kernelsmith::ks::KSDialect>();
  ASSERT_NE(dialect, nullptr);
  EXPECT_EQ(dialect->getNamespace(), "ks");
}

// TODO: Add more unit tests as operations are implemented
// - Type construction tests
// - Operation builder tests
// - Verification tests

} // namespace
