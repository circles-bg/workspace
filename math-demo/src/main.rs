use processing::math::*;
use processing::types::*;

pub trait Observable<Item, Err, Observer>
where
    Observer: GameObject<Item = Item, Err = Err>, 
{
    type CancellableToken;

    fn watch(&mut self, entity: Observer) -> Self::CancellableToken;
}

pub trait Publisher<Item, Err> {
    fn p_next(&mut self, value: Item);
    fn p_error(self: Box<Self>, err: Err);
    fn p_complete(self: Box<Self>);
    fn p_unsubscribe(self: Box<Self>);
    fn p_is_closed(&self) -> bool;
}

pub struct Subscriber<O>(std::rc::Rc<std::cell::RefCell<Option<O>>>);
pub struct SubscriberThreads<O>(std::sync::Arc<std::sync::Mutex<Option<O>>>);

fn main() {
    println!("🎯 Math Library Demo - All Improvements Implemented!");
    println!("============================================\n");

    // 1. Stack-allocated arrays for performance
    println!("1. Stack-allocated Vector and Matrix:");
    let v1 = Vector ::new([4.0, 5.0, 6.0]);
    println!("   v1 = {:?}", v1.as_array());
    println!("   v2 = {:?}", v1.as_array());

    // 2. Operator overloading for intuitive math
    println!("\n2. Operator Overloading:");
    let v3 = v1 + v1;
    let v4 = v1 * 2.0;
    println!("   v1 + v2 = {:?}", v3.as_array());
    println!("   v1 * 2.0 = {:?}", v4.as_array());

    // 3. Index access for convenience
    println!("\n3. Index Access:");
    println!("   v1[0] = {}", v1[0]);
    println!("   v1[1] = {}", v1[1]);
    println!("   v1[2] = {}", v1[2]);

    // 4. Safe error handling with Result types
    println!("\n4. Safe Error Handling:");
    let slice = [1.0, 2.0, 3.0];
    match Vector3D::try_from_slice(&slice) {
        Ok(vec) => println!("   Vector from slice: {:?}", vec.as_array()),
        Err(e) => println!("   Error: {}", e),
    }

    let bad_slice = [1.0, 2.0]; // Wrong size
    match Vector3D::try_from_slice(&bad_slice) {
        Ok(vec) => println!("   Vector from slice: {:?}", vec.as_array()),
        Err(e) => println!("   Expected error: {}", e),
    }

    // 5. Additional mathematical operations
    println!("\n5. Mathematical Operations:");
    let magnitude = calculate_vector_magnitude(&v1);
    let normalized = calculate_vector_normalization(&v1);
    let dot_product = calculate_vector_dot_product(&v1, &v2);
    println!("   |v1| = {:.2}", magnitude);
    println!("   normalized v1 = {:?}", normalized.as_array().map(|x| (x * 100.0).round() / 100.0));
    println!("   v1 · v2 = {:.2}", dot_product);

    // 6. Matrix operations
    println!("\n6. Matrix Operations:");
    let m1 = Matrix3x3::identity();
    let m2 = Matrix3x3::new([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]);
    println!("   Identity matrix:");
    for i in 0..3 {
        println!("     {:?}", m1[i]);
    }
    println!("   Custom matrix:");
    for i in 0..3 {
        println!("     {:?}", m2[i]);
    }

    // 7. Geometric primitives from types.rs
    println!("\n7. Geometric Primitives:");
    let point = Point2D::new([10.0, 20.0]);
    let rect = Rect::new(0.0, 0.0, 100.0, 50.0);
    let circle = Circle::new(Point2D::new([50.0, 25.0]), 15.0);
    
    println!("   Point: ({}, {})", point[0], point[1]);
    println!("   Rectangle: {}x{} at ({}, {})", rect.width, rect.height, rect.x, rect.y);
    println!("   Circle: radius {} at ({}, {})", circle.radius, circle.center[0], circle.center[1]);
    println!("   Point in rectangle: {}", rect.contains_point(&point));
    println!("   Point in circle: {}", circle.contains_point(&point));

    println!("\n✅ All improvements successfully implemented!");
    println!("   • Stack-allocated arrays for performance");
    println!("   • Operator overloading (+, -, *, indexing)");
    println!("   • Safe error handling with Result types");
    println!("   • Rich mathematical operations");
    println!("   • Geometric primitives and utilities");
    println!("   • Clean, modern Rust 2024 const generics API");
}
