#include "cudarl/tensor.h"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace cudarl;

// Constants to avoid magic numbers
constexpr float DEFAULT_EPSILON = 1e-6F;
constexpr size_t SMALL_SIZE = 5;
constexpr size_t LARGE_SIZE = 100;
constexpr float SCALAR_MULTIPLIER = 2.0F;

// Test values for tensor operations
constexpr float TEST_VAL_1 = 1.0F;
constexpr float TEST_VAL_2 = 2.0F;
constexpr float TEST_VAL_3 = 3.0F;
constexpr float TEST_VAL_4 = 4.0F;
constexpr float TEST_VAL_5 = 5.0F;
constexpr float TEST_VAL_6 = 6.0F;
constexpr float TEST_VAL_7 = 7.0F;
constexpr float TEST_VAL_9 = 9.0F;

// Helper function to check if two floats are approximately equal
auto approx_equal(float val1, float val2, float epsilon = DEFAULT_EPSILON) -> bool {
    return std::abs(val1 - val2) < epsilon;
}

void test_tensor_creation() {
    std::cout << "Testing tensor creation..." << '\n';

    // Test empty tensor
    Tensor empty;
    assert(empty.size() == 0);
    assert(empty.data() == nullptr);

    // Test sized tensor
    Tensor sized_tensor(SMALL_SIZE);
    assert(sized_tensor.size() == SMALL_SIZE);
    assert(sized_tensor.data() != nullptr);

    // Test vector initialization
    std::vector<float> vals = {TEST_VAL_1, TEST_VAL_2, TEST_VAL_3, TEST_VAL_4};
    Tensor vec_tensor(vals);
    assert(vec_tensor.size() == 4);
    assert(approx_equal(vec_tensor[0], TEST_VAL_1));
    assert(approx_equal(vec_tensor[3], TEST_VAL_4));

    std::cout << "✓ Tensor creation tests passed" << '\n';
}

void test_tensor_factories() {
    std::cout << "Testing tensor factory methods..." << '\n';

    // Test zeros
    Tensor zeros = Tensor::zeros(SMALL_SIZE);
    for (size_t i = 0; i < zeros.size(); ++i) {
        assert(approx_equal(zeros[i], 0.0F));
    }

    // Test ones
    Tensor ones = Tensor::ones(SMALL_SIZE);
    for (size_t i = 0; i < ones.size(); ++i) {
        assert(approx_equal(ones[i], 1.0F));
    }

    // Test random (just check range)
    Tensor rand = Tensor::random(LARGE_SIZE);
    for (size_t i = 0; i < rand.size(); ++i) {
        assert(rand[i] >= 0.0F && rand[i] <= 1.0F);
    }

    std::cout << "✓ Factory method tests passed" << '\n';
}

void test_tensor_operations() {
    std::cout << "Testing tensor operations..." << '\n';

    Tensor tensor_a({TEST_VAL_1, TEST_VAL_2, TEST_VAL_3});
    Tensor tensor_b({TEST_VAL_4, TEST_VAL_5, TEST_VAL_6});

    // Test addition
    Tensor sum = tensor_a + tensor_b;
    assert(approx_equal(sum[0], TEST_VAL_5));
    assert(approx_equal(sum[1], TEST_VAL_7));
    assert(approx_equal(sum[2], TEST_VAL_9));

    // Test subtraction
    Tensor diff = tensor_b - tensor_a;
    assert(approx_equal(diff[0], TEST_VAL_3));
    assert(approx_equal(diff[1], TEST_VAL_3));
    assert(approx_equal(diff[2], TEST_VAL_3));

    // Test scalar multiplication
    Tensor scaled = tensor_a * SCALAR_MULTIPLIER;
    assert(approx_equal(scaled[0], TEST_VAL_2));
    assert(approx_equal(scaled[1], TEST_VAL_4));
    assert(approx_equal(scaled[2], TEST_VAL_6));

    std::cout << "✓ Tensor operation tests passed" << '\n';
}

void test_tensor_copy_move() {
    std::cout << "Testing copy and move semantics..." << '\n';

    Tensor original({TEST_VAL_1, TEST_VAL_2, TEST_VAL_3});

    // Test copy constructor
    Tensor copy_constructed(original);
    assert(copy_constructed.size() == original.size());
    assert(copy_constructed.data() != original.data()); // Different memory
    assert(approx_equal(copy_constructed[0], original[0]));

    // Test copy assignment
    Tensor copy_assigned;
    copy_assigned = original;
    assert(copy_assigned.size() == original.size());
    assert(copy_assigned.data() != original.data()); // Different memory
    assert(approx_equal(copy_assigned[1], original[1]));

    // Test move constructor
    Tensor to_be_moved(original); // Create a copy to move
    Tensor move_constructed(std::move(to_be_moved));
    assert(move_constructed.size() == 3);
    // Note: We don't access to_be_moved after move to avoid use-after-move

    std::cout << "✓ Copy/move tests passed" << '\n';
}

void demo_tensor_usage() {
    std::cout << "\n=== Tensor Usage Demo ===" << '\n';

    // Create some tensors
    Tensor ones_tensor = Tensor::ones(SMALL_SIZE);
    Tensor random_tensor = Tensor::random(SMALL_SIZE);

    std::cout << "ones_tensor = ";
    ones_tensor.print();

    std::cout << "random_tensor = ";
    random_tensor.print();

    // Do some operations
    Tensor sum_result = ones_tensor + random_tensor;
    std::cout << "ones_tensor + random_tensor = ";
    sum_result.print();

    Tensor scaled_result = random_tensor * SCALAR_MULTIPLIER;
    std::cout << "random_tensor * " << SCALAR_MULTIPLIER << " = ";
    scaled_result.print();
}

auto main() -> int {
    std::cout << "Running CudaRL Tensor Tests\n" << '\n';

    try {
        test_tensor_creation();
        test_tensor_factories();
        test_tensor_operations();
        test_tensor_copy_move();
        demo_tensor_usage();

        std::cout << "\n✅ All tests passed!" << '\n';
        return 0;

    } catch (const std::exception &e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << '\n';
        return 1;
    }
}