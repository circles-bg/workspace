// Re-export common types from math module for convenience
pub use crate::math::{Mat2, Mat3, Mat4, Matrix, Vec2, Vec3, Vector};

// Additional common type aliases for backwards compatibility and convenience
pub type Vector3D<T = f32> = Vec3<T>;
pub type Matrix3x3<T = f32> = Mat3<T>;
pub type Point2D = Vec2<f32>;
pub type Point3D = Vec3<f32>;
pub type Color = [f32; 4]; // RGBA
pub type Transform2D = Mat3<f32>;
pub type Transform3D = Mat4<f32>;

// Common geometric types
#[derive(Debug, Clone, PartialEq)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Circle {
    pub center: Point2D,
    pub radius: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sphere {
    pub center: Point3D,
    pub radius: f32,
}

impl Rect {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub fn contains_point(&self, point: &Point2D) -> bool {
        point.data[0] >= self.x
            && point.data[0] <= self.x + self.width
            && point.data[1] >= self.y
            && point.data[1] <= self.y + self.height
    }
}

impl Circle {
    pub fn new(center: Point2D, radius: f32) -> Self {
        Self { center, radius }
    }

    pub fn contains_point(&self, point: &Point2D) -> bool {
        let dx = point.data[0] - self.center.data[0];
        let dy = point.data[1] - self.center.data[1];
        (dx * dx + dy * dy) <= (self.radius * self.radius)
    }
}

impl Sphere {
    pub fn new(center: Point3D, radius: f32) -> Self {
        Self { center, radius }
    }

    pub fn contains_point(&self, point: &Point3D) -> bool {
        let dx = point.data[0] - self.center.data[0];
        let dy = point.data[1] - self.center.data[1];
        let dz = point.data[2] - self.center.data[2];
        (dx * dx + dy * dy + dz * dz) <= (self.radius * self.radius)
    }
}
