# Processing - High-Performance Mathematical Computing Library

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![Rust Version](https://img.shields.io/badge/rust-2024%20edition-orange)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()
[![Documentation](https://img.shields.io/badge/docs-latest-blue)]()

A high-performance, enterprise-grade mathematical computing library for Rust, designed for applications requiring intensive vector and matrix operations such as game engines, scientific computing, computer graphics, and simulation systems.

## 🚀 Features

### Core Mathematical Types

- **Generic Vectors**: N-dimensional vectors with compile-time fixed dimensions
- **Generic Matrices**: ROWS×COLS matrices with compile-time fixed dimensions
- **Stack-Allocated Data**: Zero-overhead abstractions using fixed-size arrays
- **Type Aliases**: Convenient aliases for common types (`Vec2`, `Vec3`, `Mat3`, `Mat4`, etc.)

### Performance Optimizations

- **Compile-Time Dimensions**: All dimensions known at compile time for maximum optimization
- **SIMD-Friendly**: Memory layout optimized for vectorization
- **Cache-Efficient**: Contiguous memory layout for optimal cache performance
- **Zero-Cost Abstractions**: Direct array access with no runtime overhead

### Mathematical Operations

- **Vector Operations**: Addition, subtraction, dot product, cross product, normalization
- **Matrix Operations**: Addition, subtraction, multiplication, transposition, inversion
- **Geometric Functions**: Distance, angle calculation, projection, reflection
- **Operator Overloading**: Intuitive mathematical notation (`+`, `-`, `*`)

### Enterprise-Grade Quality

- **Comprehensive Documentation**: Full API documentation with examples
- **Error Handling**: Result types for fallible operations
- **Type Safety**: Compile-time dimension checking
- **Thread Safety**: Send + Sync for concurrent processing
- **Testing**: Extensive unit tests and integration tests

## 📋 Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
processing = { path = "path/to/processing" }
```

### Basic Usage

```rust
use processing::math::{Vector, Matrix, Vec3, Mat3};
use processing::types::{Vector3D, Matrix3x3};

// Create 3D vectors
let v1 = Vec3::<f32>::new([1.0, 2.0, 3.0]);
let v2 = Vec3::<f32>::new([4.0, 5.0, 6.0]);

// Vector arithmetic using operator overloading
let sum = v1 + v2;                 // [5.0, 7.0, 9.0]
let difference = v1 - v2;          // [-3.0, -3.0, -3.0]
let scaled = v1 * 2.0;             // [2.0, 4.0, 6.0]

// Vector operations using trait methods
let dot_product = v1.dot(&v2);     // 32.0
let cross_product = v1.cross(&v2); // [-3.0, 6.0, -3.0]
let magnitude = v1.magnitude();    // ~3.74

// Matrix operations
let identity = Mat3::<f32>::identity();
let zeros = Mat3::<f32>::zeros();
let custom = Mat3::<f32>::new([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
]);

let sum = identity + zeros;
let product = identity.multiply(&custom);
let transposed = custom.transpose();
```

### Advanced Features

```rust
use processing::math::*;

// Error handling with Result types
let slice = [1.0, 2.0, 3.0];
match Vector::<3, f32>::try_from_slice(&slice) {
    Ok(vector) => println!("Created vector: {:?}", vector),
    Err(e) => println!("Error: {}", e),
}

// Safe element access
let vector = Vec3::<f32>::new([1.0, 2.0, 3.0]);
if let Some(x) = vector.get(0) {
    println!("X component: {}", x);
}

// Geometric operations
let v1 = Vec3::<f32>::new([1.0, 0.0, 0.0]);
let v2 = Vec3::<f32>::new([0.0, 1.0, 0.0]);
let distance = calculate_distance(&v1, &v2);
let angle = calculate_angle(&v1, &v2);
let normalized = calculate_vector_normalization(&v1);
```

## 🏗️ Architecture

### Type System

The library uses a sophisticated type system built around const generics:

```rust
// Core types
Vector<const DIM: usize, T>           // N-dimensional vector
Matrix<const ROWS: usize, const COLS: usize, T>  // ROWS×COLS matrix

// Convenient type aliases
pub type Vec2<T> = Vector<2, T>;      // 2D vector
pub type Vec3<T> = Vector<3, T>;      // 3D vector
pub type Mat3<T> = Matrix<3, 3, T>;   // 3×3 matrix
pub type Mat4<T> = Matrix<4, 4, T>;   // 4×4 matrix

// Specialized aliases
pub type Vector3D<T = f32> = Vec3<T>; // 3D vector with default f32
pub type Matrix3x3<T = f32> = Mat3<T>; // 3×3 matrix with default f32
```

### Trait System

The library provides a comprehensive trait system for extensibility:

```rust
// Core mathematical operations
pub trait VectorOperations<const DIM: usize, T> {
    fn add(&self, other: &Vector<DIM, T>) -> Vector<DIM, T>;
    fn subtract(&self, other: &Vector<DIM, T>) -> Vector<DIM, T>;
    fn dot(&self, other: &Vector<DIM, T>) -> T;
    fn cross(&self, other: &Vector<DIM, T>) -> Vector<DIM, T>;
}

// Transformation operations
pub trait Transformable {
    fn transform(&self, matrix: &[[f32; 4]; 4]) -> Self;
}

pub trait Scalable {
    fn scale(&self, factor: f32) -> Self;
}

pub trait Rotatable {
    fn rotate(&self, angle: f32) -> Self;
}
```

## 🔧 Performance

### Benchmarks

Performance characteristics for common operations:

| Operation | Complexity | Performance Notes |
|-----------|------------|-------------------|
| Vector Creation | O(1) | Direct array initialization |
| Element Access | O(1) | Direct array indexing |
| Vector Addition | O(DIM) | SIMD-vectorizable |
| Matrix Multiplication | O(ROWS × COLS × K) | Cache-efficient implementation |
| Cross Product | O(1) | Specialized 3D implementation |

### Memory Usage

- **Vectors**: Exactly `sizeof(T) * DIM` bytes
- **Matrices**: Exactly `sizeof(T) * ROWS * COLS` bytes
- **Zero Overhead**: No heap allocations, no vtables, no runtime checks

## 🧪 Testing

Run the test suite:

```bash
cargo test
cargo test --doc  # Test documentation examples
```

Check code quality:

```bash
cargo clippy -- -D warnings
cargo fmt --check
```

## 📚 Documentation

Generate and view documentation:

```bash
cargo doc --open
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Code Quality Standards

- All code must pass `cargo clippy -- -D warnings`
- All code must be formatted with `cargo fmt`
- All public APIs must have comprehensive documentation
- All new features must include tests
- All changes must maintain backward compatibility

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Acknowledgments

- Built with Rust's const generics for zero-cost abstractions
- Inspired by modern game engine mathematics libraries
- Designed for high-performance computing applications

---

**Note**: This library requires Rust 2024 edition for full const generics support.
