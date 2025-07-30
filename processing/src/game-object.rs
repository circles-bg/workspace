pub trait GameObject {
    fn get_id(&self) -> u32;
    fn get_name(&self) -> &str;
    fn get_position(&self) -> (f32, f32, f32);
    fn set_position(&mut self, x: f32, y: f32, z: f32);
    fn get_rotation(&self) -> (f32, f32, f32);
    fn set_rotation(&mut self, x: f32, y: f32, z: f32);
    fn get_scale(&self) -> (f32, f32, f32);
    fn set_scale(&mut self, x: f32, y: f32, z: f32);
}