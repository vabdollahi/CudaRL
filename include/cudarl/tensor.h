#pragma once

#include <vector>
#include <memory>
#include <iostream>

namespace cudarl {

// Forward declarations
class Device;

// This class represents a multi-dimensional array that can live on CPU or GPU.
class Tensor {
private:
    float* data_;          // Raw pointer to data
    size_t size_;          // Total number of elements
    bool owns_memory_;     // Whether this tensor owns its memory
    
public:
    // Constructors
    Tensor() : data_(nullptr), size_(0), owns_memory_(false) {}
    
    // Create a tensor of given size with uninitialized values
    explicit Tensor(size_t size);
    
    // Create a tensor from a vector (CPU only for now)
    explicit Tensor(const std::vector<float>& values);
    
    // Destructor
    ~Tensor();
    
    // Copy constructor and assignment
    Tensor(const Tensor& other);
    auto operator=(const Tensor &other) -> Tensor &;

    // Move constructor and assignment
    Tensor(Tensor&& other) noexcept;
    auto operator=(Tensor &&other) noexcept -> Tensor &;

    // Basic accessors
    [[nodiscard]] auto size() const -> size_t { return size_; }
    auto data() -> float * { return data_; }
    [[nodiscard]] auto data() const -> const float * { return data_; }

    // Element access (CPU only for now)
    auto operator[](size_t idx) -> float &;
    auto operator[](size_t idx) const -> const float &;

    // Basic operations
    auto operator+(const Tensor &other) const -> Tensor;
    auto operator-(const Tensor &other) const -> Tensor;
    auto operator*(float scalar) const -> Tensor;

    // Utility functions
    void fill(float value);
    void print() const;
    [[nodiscard]] auto to_vector() const -> std::vector<float>;

    // Static factory methods
    static auto zeros(size_t size) -> Tensor;
    static auto ones(size_t size) -> Tensor;
    static auto random(size_t size) -> Tensor; // Uniform random [0, 1]
};

} // namespace cudarl