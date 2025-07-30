//! # Mathematical Operations Library
//!
//! This module provides a comprehensive set of mathematical types and operations
//! for vector and matrix computations, designed for high-performance applications
//! including game engines, scientific computing, and graphics programming.
//!
//! ## Key Features
//!
//! - **Generic Vectors and Matrices**: Parameterized by dimensions and numeric types
//! - **Stack-Allocated Data**: Uses arrays for optimal performance with fixed-size data
//! - **Operator Overloading**: Intuitive mathematical operations using standard operators
//! - **Trait-Based Design**: Extensible and composable mathematical operations
//! - **Enterprise-Grade Quality**: Comprehensive documentation, error handling, and testing
//!
//! ## Core Types
//!
//! - [`Vector<DIM, T>`](Vector): Generic N-dimensional vector
//! - [`Matrix<ROWS, COLS, T>`](Matrix): Generic matrix with compile-time dimensions
//! - Type aliases: `Vec2<T>`, `Vec3<T>`, `Mat2<T>`, `Mat3<T>`, `Mat4<T>`
//!
//! ## Example Usage
//!
//! ```rust
//! use processing::math::{Vector, Matrix, Vec3, Mat3};
//!
//! // Create 3D vectors
//! let v1 = Vec3::<f32>::new([1.0, 2.0, 3.0]);
//! let v2 = Vec3::<f32>::new([4.0, 5.0, 6.0]);
//!
//! // Vector operations
//! let sum = v1 + v2;
//! let dot_product = v1.dot(&v2);
//! let cross_product = v1.cross(&v2);
//!
//! // Matrix operations
//! let m1 = Mat3::<f32>::identity();
//! let m2 = Mat3::<f32>::zeros();
//! let result = m1 + m2;
//! ```

/// Generic transformation trait for applying 4x4 matrix transformations.
///
/// This trait enables objects to be transformed using homogeneous coordinates,
/// commonly used in 3D graphics for translations, rotations, scaling, and projections.
pub trait Transformable {
    /// Apply a 4x4 transformation matrix to the object.
    ///
    /// # Arguments
    /// * `matrix` - A 4x4 transformation matrix in column-major order
    ///
    /// # Returns
    /// A new transformed instance of the object
    fn transform(&self, matrix: &[[f32; 4]; 4]) -> Self;
}

/// Generic scaling trait for uniform scaling operations.
///
/// This trait enables objects to be scaled by a uniform factor,
/// maintaining proportions while changing size.
pub trait Scalable {
    /// Scale the object by a uniform factor.
    ///
    /// # Arguments
    /// * `factor` - The scaling factor (1.0 = no change, >1.0 = larger, <1.0 = smaller)
    ///
    /// # Returns
    /// A new scaled instance of the object
    fn scale(&self, factor: f32) -> Self;
}

/// Generic rotation trait for 2D rotation operations.
///
/// This trait enables objects to be rotated around their origin or a specific axis.
pub trait Rotatable {
    /// Rotate the object by the specified angle.
    ///
    /// # Arguments
    /// * `angle` - The rotation angle in radians
    ///
    /// # Returns
    /// A new rotated instance of the object
    fn rotate(&self, angle: f32) -> Self;
}

/// A generic N-dimensional vector with compile-time fixed dimensions.
///
/// This vector implementation uses stack-allocated arrays for optimal performance
/// and memory locality. The dimension `DIM` is known at compile time, enabling
/// aggressive compiler optimizations and preventing runtime dimension mismatches.
///
/// # Type Parameters
///
/// * `DIM` - The number of dimensions (compile-time constant)
/// * `T` - The numeric type of vector components (f32, f64, i32, etc.)
///
/// # Memory Layout
///
/// The vector data is stored as a contiguous array `[T; DIM]`, ensuring:
/// - Cache-friendly memory access patterns
/// - Zero-cost abstractions over raw arrays
/// - SIMD-friendly alignment for supported types
///
/// # Examples
///
/// ```rust
/// use processing::math::Vector;
///
/// // Create a 3D vector
/// let v1 = Vector::<3, f32>::new([1.0, 2.0, 3.0]);
/// let v2 = Vector::<3, f32>::new([4.0, 5.0, 6.0]);
///
/// // Vector arithmetic using operator overloading
/// let sum = v1 + v2;                 // [5.0, 7.0, 9.0]
/// let scaled = v1 * 2.0;             // [2.0, 4.0, 6.0]
///
/// // Vector operations using trait methods
/// let dot_product = v1.dot(&v2);     // 32.0
/// let cross_product = v1.cross(&v2); // [-3.0, 6.0, -3.0]
///
/// // Access components safely
/// if let Some(x) = v1.get(0) {
///     println!("X component: {}", x);
/// }
///
/// // Convert to/from slices
/// let slice = [1.0, 2.0, 3.0];
/// let vector = Vector::<3, f32>::try_from_slice(&slice)?;
/// ```
///
/// # Performance Characteristics
///
/// - **Creation**: O(1) - direct array initialization
/// - **Component Access**: O(1) - direct array indexing
/// - **Arithmetic Operations**: O(DIM) - vectorizable loops
/// - **Memory Overhead**: Zero - wrapper around plain array
///
/// # Safety
///
/// All operations are bounds-checked at compile time where possible,
/// and runtime bounds checks are used for dynamic operations like slicing.
#[derive(Debug, Clone, PartialEq)]
pub struct Vector<const DIM: usize, T> {
    /// The vector components stored as a fixed-size array
    pub data: [T; DIM],
}

/// A generic ROWS×COLS matrix with compile-time fixed dimensions.
///
/// This matrix implementation uses stack-allocated 2D arrays for optimal performance
/// and memory locality. Both dimensions are known at compile time, enabling
/// aggressive compiler optimizations and preventing runtime dimension mismatches.
///
/// # Type Parameters
///
/// * `ROWS` - The number of rows (compile-time constant)
/// * `COLS` - The number of columns (compile-time constant)  
/// * `T` - The numeric type of matrix elements (f32, f64, i32, etc.)
///
/// # Memory Layout
///
/// The matrix data is stored as a contiguous 2D array `[[T; COLS]; ROWS]` in row-major order:
/// - Excellent cache locality for row-wise operations
/// - SIMD-friendly alignment for supported types
/// - Zero-cost abstractions over raw 2D arrays
///
/// # Mathematical Operations
///
/// The matrix supports standard linear algebra operations:
/// - Addition and subtraction (element-wise)
/// - Scalar multiplication
/// - Matrix multiplication (when dimensions are compatible)
/// - Transposition
/// - Identity and zero matrix construction
///
/// # Examples
///
/// ```rust
/// use processing::math::Matrix;
///
/// // Create matrices
/// let m1 = Matrix::<2, 2, f32>::identity();
/// let m2 = Matrix::<2, 2, f32>::new([
///     [1.0, 2.0],
///     [3.0, 4.0]
/// ]);
///
/// // Matrix arithmetic
/// let sum = m1 + m2;               // Element-wise addition
/// let scaled = m2 * 2.0;           // Scalar multiplication
/// let product = m1.multiply(&m2);  // Matrix multiplication
/// let transposed = m2.transpose(); // Matrix transposition
///
/// // Access elements safely
/// let element = m2[0][1]; // Direct indexing (2.0)
///
/// // Create special matrices
/// let zeros = Matrix::<3, 3, f32>::zeros();
/// let identity = Matrix::<4, 4, f32>::identity();
/// ```
///
/// # Performance Characteristics
///
/// - **Creation**: O(1) - direct array initialization
/// - **Element Access**: O(1) - direct array indexing
/// - **Addition/Subtraction**: O(ROWS × COLS) - vectorizable
/// - **Matrix Multiplication**: O(ROWS × COLS × K) - cache-efficient
/// - **Memory Overhead**: Zero - wrapper around plain 2D array
///
/// # Thread Safety
///
/// Matrix instances are `Send + Sync` when T is `Send + Sync`, making them
/// safe to share between threads for parallel computation.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<const ROWS: usize, const COLS: usize, T> {
    /// The matrix elements stored as a 2D fixed-size array in row-major order
    pub data: [[T; COLS]; ROWS],
}

// Placeholder traits for arithmetic operations
pub trait Multiplicative {}
pub trait Additive {}
pub trait Subtractive {}
pub trait Divisible {}

pub trait GameObject:
    Multiplicative + Additive + Subtractive + Divisible + Transformable + Scalable + Rotatable
{
    fn update(&mut self, delta_time: f32);
    fn render(&self);
    fn get_position(&self) -> [f32; 3];
    fn set_position(&mut self, position: [f32; 3]);
    fn get_rotation(&self) -> f32;
    fn set_rotation(&mut self, rotation: f32);
    fn get_scale(&self) -> f32;
    fn set_scale(&mut self, scale: f32);
}

pub trait PhysicsObject: GameObject {
    fn apply_force(&mut self, force: [f32; 3]);
    fn apply_impulse(&mut self, impulse: [f32; 3]);
    fn get_velocity(&self) -> [f32; 3];
    fn set_velocity(&mut self, velocity: [f32; 3]);
    fn get_mass(&self) -> f32;
    fn set_mass(&mut self, mass: f32);
}

pub trait Renderable {
    fn draw(&self);
    fn set_color(&mut self, color: [f32; 4]);
    fn get_color(&self) -> [f32; 4];
}

pub trait Collidable {
    fn check_collision(&self, other: &Self) -> bool;
    fn resolve_collision(&mut self, other: &mut Self);
    fn get_bounding_box(&self) -> [f32; 6]; // [min_x, min_y, min_z, max_x, max_y, max_z]
}

pub trait Serializable {
    fn serialize(&self) -> String;
    fn deserialize(data: &str) -> Self;
}

pub trait Networkable {
    fn send(&self);
    fn receive(&mut self);
    fn get_network_id(&self) -> u32;
    fn set_network_id(&mut self, id: u32);
}

pub trait Updatable {
    fn update(&mut self, delta_time: f32);
}

pub trait Drawable {
    fn draw(&self);
    fn set_position(&mut self, position: [f32; 3]);
    fn get_position(&self) -> [f32; 3];
    fn set_rotation(&mut self, rotation: f32);
    fn get_rotation(&self) -> f32;
    fn set_scale(&mut self, scale: f32);
    fn get_scale(&self) -> f32;
}

pub trait InputHandler {
    fn handle_input(&mut self, input: &str);
    fn get_input_state(&self) -> String;
}

pub trait MatrixOperations<const ROWS: usize, const COLS: usize, T>
where
    T: Copy + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    fn add(&self, other: &Matrix<ROWS, COLS, T>) -> Matrix<ROWS, COLS, T>;
    fn multiply(&self, other: &Matrix<COLS, ROWS, T>) -> Matrix<ROWS, ROWS, T>;
    fn transpose(&self) -> Matrix<COLS, ROWS, T>;
}

pub trait VectorOperations<const DIM: usize, T>
where
    T: Copy
        + Default
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>,
{
    fn add(&self, other: &Vector<DIM, T>) -> Vector<DIM, T>;
    fn subtract(&self, other: &Vector<DIM, T>) -> Vector<DIM, T>;
    fn dot(&self, other: &Vector<DIM, T>) -> T;
    fn cross(&self, other: &Vector<DIM, T>) -> Vector<DIM, T>;
}

impl<const DIM: usize, T> Vector<DIM, T>
where
    T: Copy,
{
    /// Creates a new vector from an array.
    pub fn new(data: [T; DIM]) -> Self {
        Self { data }
    }

    /// Creates a new vector from a slice, panics if length doesn't match DIM.
    pub fn from_slice(slice: &[T]) -> Self {
        assert_eq!(slice.len(), DIM, "Slice length must match vector dimension");
        let mut data = [slice[0]; DIM]; // This requires T: Copy
        data.copy_from_slice(slice);
        Self { data }
    }

    /// Creates a new vector from a slice, returning an error if length doesn't match DIM.
    pub fn try_from_slice(slice: &[T]) -> Result<Self, &'static str> {
        if slice.len() != DIM {
            return Err("Slice length does not match vector dimension");
        }
        let mut data = [slice[0]; DIM];
        data.copy_from_slice(slice);
        Ok(Self { data })
    }

    /// Get the dimension of the vector.
    pub fn dimension(&self) -> usize {
        DIM
    }

    /// Get a reference to the internal data array.
    pub fn as_array(&self) -> &[T; DIM] {
        &self.data
    }

    /// Get a mutable reference to the internal data array.
    pub fn as_array_mut(&mut self) -> &mut [T; DIM] {
        &mut self.data
    }

    /// Get the element at the given index, returning None if out of bounds.
    pub fn get(&self, index: usize) -> Option<&T> {
        self.data.get(index)
    }

    /// Get a mutable reference to the element at the given index, returning None if out of bounds.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.data.get_mut(index)
    }

    /// Set the element at the given index, returning an error if out of bounds.
    pub fn set(&mut self, index: usize, value: T) -> Result<(), &'static str> {
        if index >= DIM {
            return Err("Index out of bounds");
        }
        self.data[index] = value;
        Ok(())
    }
}

impl<const DIM: usize, T> Vector<DIM, T>
where
    T: Default + Copy,
{
    /// Creates a zero vector.
    pub fn zeros() -> Self {
        Self {
            data: [T::default(); DIM],
        }
    }
}

impl<const ROWS: usize, const COLS: usize, T> Matrix<ROWS, COLS, T>
where
    T: Copy,
{
    /// Creates a new matrix from a 2D array.
    pub fn new(data: [[T; COLS]; ROWS]) -> Self {
        Self { data }
    }
}

impl<const ROWS: usize, const COLS: usize, T> Matrix<ROWS, COLS, T>
where
    T: Default + Copy,
{
    /// Creates a zero matrix.
    pub fn zeros() -> Self {
        Self {
            data: [[T::default(); COLS]; ROWS],
        }
    }
}

impl<const N: usize, T> Matrix<N, N, T>
where
    T: Default + Copy + From<i32>,
{
    /// Creates an identity matrix (only for square matrices).
    pub fn identity() -> Self {
        let mut data = [[T::default(); N]; N];
        for (i, row) in data.iter_mut().enumerate() {
            row[i] = T::from(1);
        }
        Self { data }
    }
}

impl<const ROWS: usize, const COLS: usize, T> MatrixOperations<ROWS, COLS, T>
    for Matrix<ROWS, COLS, T>
where
    T: Copy + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    fn add(&self, other: &Matrix<ROWS, COLS, T>) -> Matrix<ROWS, COLS, T> {
        let mut result = self.data;
        for (row_result, (row_self, row_other)) in result
            .iter_mut()
            .zip(self.data.iter().zip(other.data.iter()))
        {
            for (elem_result, (elem_self, elem_other)) in row_result
                .iter_mut()
                .zip(row_self.iter().zip(row_other.iter()))
            {
                *elem_result = *elem_self + *elem_other;
            }
        }
        Matrix::new(result)
    }

    fn multiply(&self, other: &Matrix<COLS, ROWS, T>) -> Matrix<ROWS, ROWS, T> {
        let mut result = [[T::default(); ROWS]; ROWS];
        for (i, result_row) in result.iter_mut().enumerate() {
            for (j, result_elem) in result_row.iter_mut().enumerate() {
                for k in 0..COLS {
                    *result_elem = *result_elem + (self.data[i][k] * other.data[k][j]);
                }
            }
        }
        Matrix::new(result)
    }

    fn transpose(&self) -> Matrix<COLS, ROWS, T> {
        let mut transposed_data = [[T::default(); ROWS]; COLS];
        for (i, self_row) in self.data.iter().enumerate() {
            for (j, &self_elem) in self_row.iter().enumerate() {
                transposed_data[j][i] = self_elem;
            }
        }
        Matrix::new(transposed_data)
    }
}

impl<const DIM: usize, T> VectorOperations<DIM, T> for Vector<DIM, T>
where
    T: Copy
        + Default
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>,
{
    fn add(&self, other: &Vector<DIM, T>) -> Vector<DIM, T> {
        let mut result_data = self.data;
        for (result_elem, (&self_elem, &other_elem)) in result_data
            .iter_mut()
            .zip(self.data.iter().zip(other.data.iter()))
        {
            *result_elem = self_elem + other_elem;
        }
        Vector::new(result_data)
    }

    fn subtract(&self, other: &Vector<DIM, T>) -> Vector<DIM, T> {
        let mut result_data = self.data;
        for (result_elem, (&self_elem, &other_elem)) in result_data
            .iter_mut()
            .zip(self.data.iter().zip(other.data.iter()))
        {
            *result_elem = self_elem - other_elem;
        }
        Vector::new(result_data)
    }

    fn dot(&self, other: &Vector<DIM, T>) -> T {
        let mut result = T::default();
        for i in 0..DIM {
            result = result + (self.data[i] * other.data[i]);
        }
        result
    }

    fn cross(&self, other: &Vector<DIM, T>) -> Vector<DIM, T> {
        match DIM {
            2 => {
                // 2D cross product returns z-component and 0
                let z = self.data[0] * other.data[1] - self.data[1] * other.data[0];
                let mut result = [T::default(); DIM];
                result[0] = z;
                Vector::new(result)
            }
            3 => {
                if DIM == 3 {
                    let result = [
                        self.data[1] * other.data[2] - self.data[2] * other.data[1],
                        self.data[2] * other.data[0] - self.data[0] * other.data[2],
                        self.data[0] * other.data[1] - self.data[1] * other.data[0],
                    ];
                    // This is a bit of a hack, but we need to convert [T; 3] to [T; DIM]
                    // This should only compile when DIM == 3
                    unsafe { std::mem::transmute_copy(&result) }
                } else {
                    panic!("Cross product is only defined for 2D and 3D vectors.");
                }
            }
            _ => panic!("Cross product is only defined for 2D and 3D vectors."),
        }
    }
}

// Default transformable implementations - these are placeholder implementations
// For specific types, override these with proper transformation logic

impl<const DIM: usize, T> Scalable for Vector<DIM, T>
where
    T: Copy + std::ops::Mul<f32, Output = T>,
{
    fn scale(&self, factor: f32) -> Self {
        let mut scaled_data = self.data;
        for (scaled_elem, &original_elem) in scaled_data.iter_mut().zip(self.data.iter()) {
            *scaled_elem = original_elem * factor;
        }
        Vector::new(scaled_data)
    }
}

impl<const ROWS: usize, const COLS: usize, T> Scalable for Matrix<ROWS, COLS, T>
where
    T: Copy + std::ops::Mul<f32, Output = T>,
{
    fn scale(&self, factor: f32) -> Self {
        let mut scaled_data = self.data;
        for (row_scaled, row_original) in scaled_data.iter_mut().zip(self.data.iter()) {
            for (elem_scaled, &elem_original) in row_scaled.iter_mut().zip(row_original.iter()) {
                *elem_scaled = elem_original * factor;
            }
        }
        Matrix::new(scaled_data)
    }
}

// 2D rotation implementation for Vector<2, f32>
impl Rotatable for Vector<2, f32> {
    fn rotate(&self, angle: f32) -> Self {
        let (sin, cos) = angle.sin_cos();
        let x = self.data[0] * cos - self.data[1] * sin;
        let y = self.data[0] * sin + self.data[1] * cos;
        Vector::new([x, y])
    }
}

// Implement Rotatable for Vector<3, f32> (rotation around Z axis)
impl Rotatable for Vector<3, f32> {
    fn rotate(&self, angle: f32) -> Self {
        let (sin, cos) = angle.sin_cos();
        let x = self.data[0] * cos - self.data[1] * sin;
        let y = self.data[0] * sin + self.data[1] * cos;
        Vector::new([x, y, self.data[2]])
    }
}

// Operator overloading for Vector
impl<const DIM: usize, T> std::ops::Add for Vector<DIM, T>
where
    T: Copy + Default + std::ops::Add<Output = T>,
{
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        calculate_vector_addition(&self, &other)
    }
}

impl<const DIM: usize, T> std::ops::Sub for Vector<DIM, T>
where
    T: Copy + Default + std::ops::Sub<Output = T>,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        calculate_vector_subtraction(&self, &other)
    }
}

impl<const DIM: usize, T> std::ops::Mul<T> for Vector<DIM, T>
where
    T: Copy + Default + std::ops::Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, scalar: T) -> Self::Output {
        let mut result = self.data;
        for (result_elem, &original_elem) in result.iter_mut().zip(self.data.iter()) {
            *result_elem = original_elem * scalar;
        }
        Vector::new(result)
    }
}

// Operator overloading for Matrix
impl<const ROWS: usize, const COLS: usize, T> std::ops::Add for Matrix<ROWS, COLS, T>
where
    T: Copy + Default + std::ops::Add<Output = T>,
{
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        let mut result = self.data;
        for (result_row, (self_row, other_row)) in result
            .iter_mut()
            .zip(self.data.iter().zip(other.data.iter()))
        {
            for (result_elem, (&self_elem, &other_elem)) in result_row
                .iter_mut()
                .zip(self_row.iter().zip(other_row.iter()))
            {
                *result_elem = self_elem + other_elem;
            }
        }
        Matrix::new(result)
    }
}

impl<const ROWS: usize, const COLS: usize, T> std::ops::Sub for Matrix<ROWS, COLS, T>
where
    T: Copy + Default + std::ops::Sub<Output = T>,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        let mut result = self.data;
        for (result_row, (self_row, other_row)) in result
            .iter_mut()
            .zip(self.data.iter().zip(other.data.iter()))
        {
            for (result_elem, (&self_elem, &other_elem)) in result_row
                .iter_mut()
                .zip(self_row.iter().zip(other_row.iter()))
            {
                *result_elem = self_elem - other_elem;
            }
        }
        Matrix::new(result)
    }
}

impl<const ROWS: usize, const COLS: usize, T> std::ops::Mul<T> for Matrix<ROWS, COLS, T>
where
    T: Copy + Default + std::ops::Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, scalar: T) -> Self::Output {
        let mut result = self.data;
        for result_row in result.iter_mut() {
            for result_elem in result_row.iter_mut() {
                *result_elem = *result_elem * scalar;
            }
        }
        Matrix::new(result)
    }
}

// Multiplicative, Additive, Subtractive, Divisible for Vector
impl<const DIM: usize, T> Multiplicative for Vector<DIM, T> {}
impl<const DIM: usize, T> Additive for Vector<DIM, T> {}
impl<const DIM: usize, T> Subtractive for Vector<DIM, T> {}
impl<const DIM: usize, T> Divisible for Vector<DIM, T> {}

// Multiplicative, Additive, Subtractive, Divisible for Matrix
impl<const ROWS: usize, const COLS: usize, T> Multiplicative for Matrix<ROWS, COLS, T> {}
impl<const ROWS: usize, const COLS: usize, T> Additive for Matrix<ROWS, COLS, T> {}
impl<const ROWS: usize, const COLS: usize, T> Subtractive for Matrix<ROWS, COLS, T> {}
impl<const ROWS: usize, const COLS: usize, T> Divisible for Matrix<ROWS, COLS, T> {}

pub type Vec2<T> = Vector<2, T>;
pub type Vec3<T> = Vector<3, T>;

pub type Mat4<T> = Matrix<4, 4, T>;
pub type Mat3<T> = Matrix<3, 3, T>;
pub type Mat2<T> = Matrix<2, 2, T>;

// Remove duplicate and redundant transform traits and impls

// Remove duplicate calculate_vector_cross_product and keep only calculate_cross_product
// Remove duplicate calculate_matrix_product and calculate_matrix_transpose

/// Calculates the distance between two vectors.
pub fn calculate_distance<const DIM: usize, T>(a: &Vector<DIM, T>, b: &Vector<DIM, T>) -> f32
where
    T: Copy + std::ops::Sub<Output = T> + Into<f32>,
{
    let mut sum = 0.0;
    for i in 0..DIM {
        let diff = a.data[i] - b.data[i];
        sum += diff.into() * diff.into();
    }
    sum.sqrt()
}

/// Calculates the angle (in radians) between two vectors.
pub fn calculate_angle<const DIM: usize, T>(a: &Vector<DIM, T>, b: &Vector<DIM, T>) -> f32
where
    T: Copy
        + std::ops::Sub<Output = T>
        + Into<f32>
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + Default,
{
    let dot_product = a.dot(b);
    let magnitude_a = a
        .data
        .iter()
        .map(|x| (*x).into() * (*x).into())
        .sum::<f32>()
        .sqrt();
    let magnitude_b = b
        .data
        .iter()
        .map(|x| (*x).into() * (*x).into())
        .sum::<f32>()
        .sqrt();

    (dot_product.into() / (magnitude_a * magnitude_b)).acos()
}

/// Calculates the cross product of two 3D vectors.
pub fn calculate_cross_product<T>(a: &Vector<3, T>, b: &Vector<3, T>) -> Vector<3, T>
where
    T: Copy
        + Default
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>,
{
    a.cross(b)
}

/// Calculates the product of two matrices.
pub fn calculate_matrix_product<const ROWS: usize, const COLS: usize, T>(
    a: &Matrix<ROWS, COLS, T>,
    b: &Matrix<COLS, ROWS, T>,
) -> Matrix<ROWS, ROWS, T>
where
    T: Copy + Default + std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
{
    let mut result = [[T::default(); ROWS]; ROWS];
    for (i, result_row) in result.iter_mut().enumerate() {
        for (j, result_elem) in result_row.iter_mut().enumerate() {
            for k in 0..COLS {
                *result_elem = *result_elem + (a.data[i][k] * b.data[k][j]);
            }
        }
    }
    Matrix::new(result)
}

/// Calculates the transpose of a matrix.
pub fn calculate_matrix_transpose<const ROWS: usize, const COLS: usize, T>(
    matrix: &Matrix<ROWS, COLS, T>,
) -> Matrix<COLS, ROWS, T>
where
    T: Copy + Default,
{
    let mut transposed_data = [[T::default(); ROWS]; COLS];
    for (i, matrix_row) in matrix.data.iter().enumerate() {
        for (j, &matrix_elem) in matrix_row.iter().enumerate() {
            transposed_data[j][i] = matrix_elem;
        }
    }
    Matrix::new(transposed_data)
}

/// Adds two vectors element-wise.
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// A new vector containing the element-wise sum of `a` and `b`
///
/// # Example
/// ```
/// use processing::math::{Vector, calculate_vector_addition};
/// let v1 = Vector::<3, f32>::new([1.0, 2.0, 3.0]);
/// let v2 = Vector::<3, f32>::new([4.0, 5.0, 6.0]);
/// let result = calculate_vector_addition(&v1, &v2);
/// assert_eq!(result.data, [5.0, 7.0, 9.0]);
/// ```
pub fn calculate_vector_addition<const DIM: usize, T>(
    a: &Vector<DIM, T>,
    b: &Vector<DIM, T>,
) -> Vector<DIM, T>
where
    T: Copy + std::ops::Add<Output = T>,
{
    let mut result_data = a.data;
    for (result_elem, &b_elem) in result_data.iter_mut().zip(b.data.iter()) {
        *result_elem = *result_elem + b_elem;
    }
    Vector::new(result_data)
}

/// Subtracts two vectors element-wise.
///
/// # Arguments
/// * `a` - First vector (minuend)
/// * `b` - Second vector (subtrahend)
///
/// # Returns
/// A new vector containing the element-wise difference of `a` and `b` (a - b)
///
/// # Example
/// ```
/// use processing::math::{Vector, calculate_vector_subtraction};
/// let v1 = Vector::<3, f32>::new([4.0, 5.0, 6.0]);
/// let v2 = Vector::<3, f32>::new([1.0, 2.0, 3.0]);
/// let result = calculate_vector_subtraction(&v1, &v2);
/// assert_eq!(result.data, [3.0, 3.0, 3.0]);
/// ```
pub fn calculate_vector_subtraction<const DIM: usize, T>(
    a: &Vector<DIM, T>,
    b: &Vector<DIM, T>,
) -> Vector<DIM, T>
where
    T: Copy + std::ops::Sub<Output = T>,
{
    let mut result_data = a.data;
    for (result_elem, &b_elem) in result_data.iter_mut().zip(b.data.iter()) {
        *result_elem = *result_elem - b_elem;
    }
    Vector::new(result_data)
}

/// Calculates the dot product of two vectors.
pub fn calculate_vector_dot_product<const DIM: usize, T>(
    a: &Vector<DIM, T>,
    b: &Vector<DIM, T>,
) -> f32
where
    T: Copy + std::ops::Mul<Output = T> + Into<f32>,
{
    let mut result = 0.0;
    for i in 0..DIM {
        result += a.data[i].into() * b.data[i].into();
    }
    result
}

/// Scales a vector by a scalar factor.
///
/// # Arguments
/// * `vector` - The vector to scale
/// * `factor` - The scaling factor
///
/// # Returns
/// A new vector with each element multiplied by the factor
///
/// # Example
/// ```
/// use processing::math::{Vector, calculate_vector_scale};
/// let v = Vector::<3, f32>::new([1.0, 2.0, 3.0]);
/// let result = calculate_vector_scale(&v, 2.0);
/// assert_eq!(result.data, [2.0, 4.0, 6.0]);
/// ```
pub fn calculate_vector_scale<const DIM: usize, T>(
    vector: &Vector<DIM, T>,
    factor: f32,
) -> Vector<DIM, T>
where
    T: Copy + std::ops::Mul<f32, Output = T>,
{
    let mut scaled_data = vector.data;
    for (scaled_elem, &original_elem) in scaled_data.iter_mut().zip(vector.data.iter()) {
        *scaled_elem = original_elem * factor;
    }
    Vector::new(scaled_data)
}

/// Calculates the magnitude of a vector.
pub fn calculate_vector_magnitude<const DIM: usize, T>(vector: &Vector<DIM, T>) -> f32
where
    T: Copy + std::ops::Mul<Output = T> + Into<f32>,
{
    let mut sum = 0.0;
    for i in 0..DIM {
        sum += vector.data[i].into() * vector.data[i].into();
    }
    sum.sqrt()
}

/// Normalizes a vector to unit length.
///
/// # Arguments
/// * `vector` - The vector to normalize
///
/// # Returns
/// A new vector with the same direction but unit magnitude (length = 1.0)
///
/// # Example
/// ```
/// use processing::math::{Vector, calculate_vector_normalization};
/// let v = Vector::<3, f32>::new([3.0, 4.0, 0.0]);
/// let result = calculate_vector_normalization(&v);
/// // Result should be approximately [0.6, 0.8, 0.0] with magnitude 1.0
/// ```
///
/// # Note
/// Returns a zero vector if the input vector has zero magnitude
pub fn calculate_vector_normalization<const DIM: usize, T>(
    vector: &Vector<DIM, T>,
) -> Vector<DIM, T>
where
    T: Copy + Default + std::ops::Mul<f32, Output = T> + std::ops::Mul<Output = T> + Into<f32>,
{
    let magnitude = calculate_vector_magnitude(vector);
    if magnitude == 0.0 {
        return Vector::zeros();
    }
    let mut normalized_data = vector.data;
    let inv_magnitude = 1.0 / magnitude;
    for (normalized_elem, &original_elem) in normalized_data.iter_mut().zip(vector.data.iter()) {
        *normalized_elem = original_elem * inv_magnitude;
    }
    Vector::new(normalized_data)
}

/// Calculates the projection of vector a onto vector b.
pub fn calculate_vector_projection<const DIM: usize, T>(
    a: &Vector<DIM, T>,
    b: &Vector<DIM, T>,
) -> Vector<DIM, T>
where
    T: Copy + Default + std::ops::Mul<Output = T> + std::ops::Mul<f32, Output = T> + Into<f32>,
{
    let dot_product = calculate_vector_dot_product(a, b);
    let magnitude_b_squared = calculate_vector_dot_product(b, b);
    if magnitude_b_squared == 0.0 {
        return Vector::zeros();
    }
    let scale = dot_product / magnitude_b_squared;
    calculate_vector_scale(b, scale)
}

/// Calculates the reflection of a vector off a surface normal.
pub fn calculate_vector_reflection<const DIM: usize, T>(
    vector: &Vector<DIM, T>,
    normal: &Vector<DIM, T>,
) -> Vector<DIM, T>
where
    T: Copy
        + std::ops::Mul<Output = T>
        + std::ops::Mul<f32, Output = T>
        + std::ops::Sub<Output = T>
        + Into<f32>,
{
    let dot_product = calculate_vector_dot_product(vector, normal);
    let scaled_normal = calculate_vector_scale(normal, 2.0 * dot_product);
    calculate_vector_subtraction(vector, &scaled_normal)
}

/// Calculates the angle between two vectors.
pub fn calculate_vector_angle<const DIM: usize, T>(a: &Vector<DIM, T>, b: &Vector<DIM, T>) -> f32
where
    T: Copy + std::ops::Mul<Output = T> + Into<f32>,
{
    let dot_product = calculate_vector_dot_product(a, b);
    let magnitude_a = calculate_vector_magnitude(a);
    let magnitude_b = calculate_vector_magnitude(b);

    if magnitude_a == 0.0 || magnitude_b == 0.0 {
        return 0.0;
    }

    (dot_product / (magnitude_a * magnitude_b)).acos()
}

// Index traits for convenient element access
impl<const DIM: usize, T> std::ops::Index<usize> for Vector<DIM, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<const DIM: usize, T> std::ops::IndexMut<usize> for Vector<DIM, T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<const ROWS: usize, const COLS: usize, T> std::ops::Index<usize> for Matrix<ROWS, COLS, T> {
    type Output = [T; COLS];

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<const ROWS: usize, const COLS: usize, T> std::ops::IndexMut<usize> for Matrix<ROWS, COLS, T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}
