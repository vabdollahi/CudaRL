#include "cudarl/tensor.h"
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <random>
#include <stdexcept>

namespace cudarl {

// Constructor for uninitialized tensor
Tensor::Tensor(size_t size) : size_(size), owns_memory_(true) {
    if (size_ > 0) {
        data_ = new float[size_];
    } else {
        data_ = nullptr;
    }
}

// Constructor from vector
Tensor::Tensor(const std::vector<float> &values) : size_(values.size()), owns_memory_(true) {
    if (size_ > 0) {
        data_ = new float[size_];
        std::copy(values.begin(), values.end(), data_);
    } else {
        data_ = nullptr;
    }
}

// Destructor
Tensor::~Tensor() {
    if (owns_memory_ && (data_ != nullptr)) {
        delete[] data_;
    }
}

// Copy constructor - deep copy
Tensor::Tensor(const Tensor &other) : size_(other.size_), owns_memory_(true) {
    if (size_ > 0) {
        data_ = new float[size_];
        std::memcpy(data_, other.data_, size_ * sizeof(float));
    } else {
        data_ = nullptr;
    }
}

// Copy assignment
auto Tensor::operator=(const Tensor &other) -> Tensor & {
    if (this != &other) {
        // Clean up existing memory
        if (owns_memory_ && (data_ != nullptr)) {
            delete[] data_;
        }

        // Copy new data
        size_ = other.size_;
        owns_memory_ = true;

        if (size_ > 0) {
            data_ = new float[size_];
            std::memcpy(data_, other.data_, size_ * sizeof(float));
        } else {
            data_ = nullptr;
        }
    }
    return *this;
}

// Move constructor
Tensor::Tensor(Tensor &&other) noexcept
    : data_(other.data_), size_(other.size_), owns_memory_(other.owns_memory_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.owns_memory_ = false;
}

// Move assignment
auto Tensor::operator=(Tensor &&other) noexcept -> Tensor & {
    if (this != &other) {
        // Clean up existing memory
        if (owns_memory_ && (data_ != nullptr)) {
            delete[] data_;
        }

        // Steal other's resources
        data_ = other.data_;
        size_ = other.size_;
        owns_memory_ = other.owns_memory_;

        // Leave other in valid but empty state
        other.data_ = nullptr;
        other.size_ = 0;
        other.owns_memory_ = false;
    }
    return *this;
}

// Element access
auto Tensor::operator[](size_t idx) -> float & {
    if (idx >= size_) {
        throw std::out_of_range("Tensor index out of range");
    }
    return data_[idx];
}

auto Tensor::operator[](size_t idx) const -> const float & {
    if (idx >= size_) {
        throw std::out_of_range("Tensor index out of range");
    }
    return data_[idx];
}

// Addition
auto Tensor::operator+(const Tensor &other) const -> Tensor {
    if (size_ != other.size_) {
        throw std::invalid_argument("Tensor sizes must match for addition");
    }

    Tensor result(size_);
    for (size_t i = 0; i < size_; ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

// Subtraction
auto Tensor::operator-(const Tensor &other) const -> Tensor {
    if (size_ != other.size_) {
        throw std::invalid_argument("Tensor sizes must match for subtraction");
    }

    Tensor result(size_);
    for (size_t i = 0; i < size_; ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
}

// Scalar multiplication
auto Tensor::operator*(float scalar) const -> Tensor {
    Tensor result(size_);
    for (size_t i = 0; i < size_; ++i) {
        result.data_[i] = data_[i] * scalar;
    }
    return result;
}

// Fill tensor with a value
void Tensor::fill(float value) { std::fill(data_, data_ + size_, value); }

// Print tensor contents
void Tensor::print() const {
    std::cout << "Tensor([";
    for (size_t i = 0; i < size_; ++i) {
        std::cout << std::fixed << std::setprecision(4) << data_[i];
        if (i < size_ - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "])" << '\n';
}

// Convert to vector
auto Tensor::to_vector() const -> std::vector<float> { return {data_, data_ + size_}; }

// Static factory methods
auto Tensor::zeros(size_t size) -> Tensor {
    Tensor result(size);
    result.fill(0.0F);
    return result;
}

auto Tensor::ones(size_t size) -> Tensor {
    Tensor result(size);
    result.fill(1.0F);
    return result;
}

auto Tensor::random(size_t size) -> Tensor {
    static std::random_device random_device;
    static std::mt19937 generator(random_device());
    static std::uniform_real_distribution<float> distribution(0.0F, 1.0F);

    Tensor result(size);
    for (size_t i = 0; i < size; ++i) {
        result.data_[i] = distribution(generator);
    }
    return result;
}

} // namespace cudarl