use std::{fmt, mem};
use std::fs::File;
use std::io::{BufReader, Read, BufWriter};
use std::f64::{INFINITY, NEG_INFINITY, NAN};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::io::Write;
use rayon::prelude::*;

/// OFFSET defines the offset for generating slices. First slice is below the minimum z-value of the
/// model and last slice is above the maximum z-value of the model.
static OFFSET: f64 = 1e-3;

/// ROUND defines the rounding of f64 to u64 for Point struct. This is necessary to implement
/// PartialEq and Hash traits. These traits are necessary to implement HashKey and find unique
/// intersection points and how they are connected.
static ROUND: u32 = 1000000;

/// Point struct stores x, y, and z value of the Point
#[derive(Debug, Copy, Clone)]
struct Point{
    x:f64,
    y:f64,
    z:f64,
}
impl Point{
    /// isValid checks if a point is valid. If either x, y, or z is NAN, NEG_INFINITY, or INFINITY,
    /// isValid returns false, else returns true
    pub fn isValid(&self)->bool{
        return self.x.is_finite() & self.y.is_finite() & self.z.is_finite();
    }
}

/// implements fmt::Display for Point
impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{},{},{}", self.x, self.y, self.z)
    }
}

fn half_down(x:f64)->i64{
    let x2 = (x*(ROUND as f64)*10.0).trunc() as i64;
    let x3 = (x*(ROUND)as f64).trunc() as i64 *10;
    let out;
    if (x2-x3)>5{ out = (x2+5)/10}
    else {out = x2/10}
    return out;
}
/// implements PartialEq for Point. x, y, and z are rounded to nearest OFFSET and compared.
/// Necessary for implementing HashMap.
impl PartialEq for Point {
    fn eq(&self, other: &Self) -> bool {
        return (half_down(self.x) == half_down(other.x))&
            (half_down(self.y) == half_down(other.y)) &
            (half_down(self.z) == half_down(other.z));
        /*  let spx = (math::round::half_down(self.x,ROUND)*(10.0)).powi(ROUND as i32) as u64;
          let spy = (math::round::half_down(self.y,ROUND)*(10.0)).powi(ROUND as i32) as u64;
          let spz = (math::round::half_down(self.z,ROUND)*(10.0)).powi(ROUND as i32) as u64;

          let opx = (math::round::half_down(other.x,ROUND)*(10.0)).powi(ROUND as i32) as u64;
          let opy = (math::round::half_down(other.y,ROUND)*(10.0)).powi(ROUND as i32) as u64;
          let opz = (math::round::half_down(other.z,ROUND)*(10.0)).powi(ROUND as i32) as u64;
          return (spx == opx) & (spy == opy) & (spz == opz);*/
    }
}
/// implements Eq for Point
impl Eq for Point {}

/// implements Hash for Point. Rounds x, y, and z to nearest ROUND, and takes 128 * x + 32 *y + z.
/// Multiplies the value by 10^ROUND and converts it to u64
impl Hash for Point {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Edge::max(self.pt1, self.pt2).hash(state);
        //Edge::min(self.pt1, self.pt2).hash(state);
        // let x = (math::round::half_down(self.x,ROUND)*(10.0)).powi(ROUND as i32) as u64;
        // let y = (math::round::half_down(self.y,ROUND)*(10.0)).powi(ROUND as i32) as u64;
        //let z = (math::round::half_down(self.z,ROUND)*(10.0)).powi(ROUND as i32) as u64;
        let x = half_down(self.x);
        let y = half_down(self.y);
        let z = half_down(self.z);
        let pp = (x*64+y) as u64;
        //let pp = x*100.0+y*10.0+z;
        pp.hash(state);
    }
}

/// Triangle struct contains minimum z-value, maximum z-value, and the values array with first three
/// values corresponding to the normal, second, third, and fourth three values corresponding to the
/// second, third, and fourth points. values[0..3] -> normals, values[3..6] -> Point1, values[6..9]
/// -> Point2, values[9..12] -> Point3
#[derive(Debug, Copy, Clone)]
struct Triangle{
    minz:f64,
    maxz:f64,
    values:[f64;12],
}

/// StlFile contains minimum z-value and maximum z-value of the model, 80 character info string,
/// number of triangles and Vec<Triangle>
struct StlFile{
    minz:f64,
    maxz:f64,
    info:String,
    num_tri:usize,
    trivals:Vec<Triangle>,
}

///implements functions for StlFile
impl StlFile {
    /// read a binary stl file and store the data in StlFile struct.
    /// let new_stl_file = read_binary_stl_file("c:\\stlFiles\\sample_stl_file.stl");
    pub fn read_binary_stl_file(filename: &str) -> StlFile {
        let mut file_buffer = File::open(filename).expect("no such file found");
        let mut input_file = BufReader::with_capacity(10000000, file_buffer);
        let mut info_buffer = [0; 80];
        let mut num_tri_buffer = [0; 4];
        let mut data_buffer = [0; 4];
        let mut attribute_buffer = [0; 2];
        input_file.read(&mut info_buffer).expect("cannot read info buffer");
        input_file.read(&mut num_tri_buffer).expect("cannot read number of triangles");
        let info = std::str::from_utf8(&info_buffer).expect("cant convert to string");
        //let info1:String = info as String;
        let num_tri = u32::from_le_bytes(num_tri_buffer);
        let mut all_triangles: Vec<Triangle> = Vec::new();
        all_triangles.reserve(num_tri as usize);

        let mut global_min_z: f64 = INFINITY;
        let mut global_max_z: f64 = NEG_INFINITY;

        for i in 0..num_tri as usize {
            // input_file.read(&mut data_buffer).expect("cant read data");
            let mut temp_f32: [f32; 12] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ];

            //println!("{}",temp_f32);

            for j in 0..12{
                input_file.read(&mut data_buffer).expect("cant read data");
                unsafe { temp_f32[j] = mem::transmute(data_buffer) };
            }
            let mut minz = temp_f32[5];
            let mut maxz = temp_f32[5];
            if temp_f32[8] < minz { minz = temp_f32[8] };
            if temp_f32[11] < minz { minz = temp_f32[11] };
            if temp_f32[8] > maxz { maxz = temp_f32[8] };
            if temp_f32[11] > maxz { maxz = temp_f32[11] };

            if (minz as f64) < global_min_z { global_min_z = (minz as f64) };
            if (maxz as f64) > global_max_z { global_max_z = (maxz as f64)};
            let mut temp_f64 :[f64;12] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ];
            for i in 0..12{
                temp_f64[i] = temp_f32[i] as f64;
            }
            let mut tri = Triangle {
                minz: minz as f64,
                maxz: maxz as f64,
                values: temp_f64,
            };
            all_triangles.push(tri);
            input_file.read(&mut attribute_buffer).expect("cant read attrib buffer");
        }
        let out_file: StlFile = StlFile {
            minz: global_min_z,
            maxz: global_max_z,
            info: String::from(info),
            num_tri: num_tri as usize,
            trivals: all_triangles,
        };
        return out_file;
    }
}

/// Struct for slicing stl file. Stores data from stl file as file and the z-values of the slicing
/// planes as slices.
struct StlFileSlicer{
    file:StlFile,
    slices:Vec<f64>,
}

/// implements StlFileSlicer
impl StlFileSlicer {
    /// generates a StlFileSlicer class from a stl_file and slice height
    /// let new_stl_file_slicer = StlFileSlicer::new(stl_file, 0.36);
    pub fn new(stl_file: StlFile, slice_height: f64) -> StlFileSlicer {
        let num_layers = ((stl_file.maxz - stl_file.minz) / slice_height).round() as u32;

        let mut slices = Vec::new();
        for i in 0..num_layers + 2 {
            slices.push(stl_file.minz - OFFSET + (i as f64) * (slice_height + (6.0 * OFFSET) / (num_layers as f64)));
        }
        let out = StlFileSlicer { file: stl_file, slices };
        return out;
    }

    /// finds the triangles that intersect with a given plane using binary search.
    /// returns a vector of vector of indices of triangle stored in StlFile struct.
    pub fn find_intersecting_triangles(&self) -> Vec<Vec<usize>> {
        let mut intersecting_triangles_in_layer: Vec<Vec<usize>> = Vec::new();

        for i in 0..self.slices.len() {
            let tri: Vec<usize> = Vec::new();
            intersecting_triangles_in_layer.push(tri);
        }

        for i in 0..self.file.num_tri {
            let hival = self.binary_search_hi(self.file.trivals[i].maxz);
            let loval = self.binary_search_lo(self.file.trivals[i].minz);
            for j in loval as usize..(hival + 1) as usize {
                intersecting_triangles_in_layer[j].push(i);
            }
        }
        return intersecting_triangles_in_layer;
    }

    /// finds the lowest plane in slices vector that intersects a given triangle. Find the index of
    /// such a plane that is lower than the minz value of the triangle. The plane above it is higher
    /// than the minz value of the triangle.
    fn binary_search_lo(&self, key: f64) -> isize {
        //println!("in binary search, minz or maxz key {}",&key);
        //if (key  > self.slices[self.slices.len()-1]) | (key  < self.slices[0]  ) {return -1}
        let mut lo = 0;
        let mut hi = self.slices.len() - 1;
        let mut mid;
        while lo <= hi {
            mid = lo + (hi - lo) / 2;
            if mid + 1 > self.slices.len() - 1 { return -1 }
            if (key > self.slices[mid] as f64) & (key < self.slices[mid + 1] as f64) { return (mid + 1) as isize } else if (key < self.slices[mid] as f64) { hi = mid - 1 } else { lo = mid + 1 }
        }
        return -1;
    }
    /// finds the highest plane in slices vector that intersects a given triangle. Find the index of
    /// such a plane that is higher than the maxz value of the triangle. The plane below it is
    /// lower than the max value of the triangle.
    fn binary_search_hi(&self, key: f64) -> isize {
        //println!("in binary search, minz or maxz key {}",&key);
        //if (key  > self.slices[self.slices.len()-1]) | (key  < self.slices[0]  ) {return -1}
        let mut lo = 0;
        let mut hi = self.slices.len() - 1;
        let mut mid;
        while lo <= hi {
            mid = lo + (hi - lo) / 2;
            if mid < 1 { return -1 } else if (key > self.slices[mid - 1] as f64) & (key < self.slices[mid] as f64) { return (mid - 1) as isize } else if (key < self.slices[mid] as f64) { hi = mid - 1 } else { lo = mid + 1 }
        }
        return -1;
    }

    /// calculate the intersection point of a line defined by (x1,y1,z1) and (x2,y2,z2), and a plane
    /// defined by z = c. If intersection point is above or below the line, return NAN for all
    /// values of x, y, and z.
    fn calc_intersection_line_plane(x1: f64, y1: f64, z1: f64, x2: f64, y2: f64, z2: f64, c: f64) -> Point {
        let mut t = (c - z1) / (z2 - z1);
        if ((z1 > c) & (z2 > c)) | ((z1 < c) & (z2 < c)) {
            t = NAN;
        }
        return Point {
            x: x1 + t * (x2 - x1),
            y: y1 + t * (y2 - y1),
            z: c,
        }
    }

    /// calculate intersection edges for triangles and a given plane. Return a vector of two points
    /// that gives the edge of intersection between the triangles and the plane.
    pub fn calc_intersection_line_plane_layer(&self, triangles_in_layer: &Vec<usize>, zvalue: f64) -> Vec<[Point; 2]> {
        let mut intersection_points: Vec<[Point; 2]> = Vec::new();
        intersection_points.reserve(8000);

        for i in triangles_in_layer {
            let tdata = self.file.trivals[i.clone()].values;
            let p1 = [tdata[3], tdata[4], tdata[5]];
            let p2 = [tdata[6], tdata[7], tdata[8]];
            let p3 = [tdata[9], tdata[10], tdata[11]];

            let ip1 = StlFileSlicer::calc_intersection_line_plane(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], zvalue);
            let ip2 = StlFileSlicer::calc_intersection_line_plane(p3[0], p3[1], p3[2], p2[0], p2[1], p2[2], zvalue);
            let ip3 = StlFileSlicer::calc_intersection_line_plane(p1[0], p1[1], p1[2], p3[0], p3[1], p3[2], zvalue);

            if (ip1.isValid() & ip2.isValid()) { intersection_points.push([ip1, ip2]) };
            if (ip3.isValid() & ip2.isValid()) { intersection_points.push([ip3, ip2]) };
            if (ip1.isValid() & ip3.isValid()) { intersection_points.push([ip1, ip3]) };
        }
        return intersection_points;
    }

    /// find unique points and edges created by intersection of triangles and a plane. Unique
    /// vertices are stored as a HashMap and edges refer to the index of the vertex.
    pub fn find_unique_points_and_edges(edges_input: Vec<[Point; 2]>) -> (HashMap<usize, Point>, Vec<[usize; 2]>) {
        let mut points: HashMap<usize, Point> = HashMap::with_capacity(40000);
        let mut reverse_points: HashMap<Point, usize> = HashMap::with_capacity(40000);
        let mut edges = Vec::with_capacity(40000);
        let mut points_counter: usize = 0;
        let mut edges_counter: usize = 0;
        for i in edges_input {
            if !reverse_points.contains_key(&i[0]) {
                reverse_points.insert(i[0], points_counter);
                points.insert(points_counter, i[0]);
                points_counter = points_counter + 1;
            }
            if !reverse_points.contains_key(&i[1]) {
                reverse_points.insert(i[1], points_counter);
                points.insert(points_counter, i[1]);
                points_counter = points_counter + 1;
            }
            let e1 = reverse_points.get(&i[0]).expect("no such key e1");
            let e2 = reverse_points.get(&i[1]).expect("no such key e2");
            edges.push([e1.clone(), e2.clone()]);
        }
        return (points, edges);
    }

    /// generates movepath for a given layer using breadth first search for the edges. Assumes each
    /// point is connected to two other points. Panics if a point is connected to only one point.
    /// Gives spurious results if a point is connected to more than two points. Output is Vec<Vec<
    /// Points>> -> list of vectors for each loop in a plane.
    pub fn generate_path_for_layer(start_pt: &u32, points_and_edges: (HashMap<usize, Point>, Vec<[usize; 2]>)) -> Vec<Vec<Point>> {
        let mut collector = Vec::with_capacity(40000);
        let mut vertices = Vec::with_capacity(40000);
        vertices.reserve(points_and_edges.0.len());
        for i in 0..points_and_edges.0.len() {
            //let mut m = [0,0];
            vertices.push(Vec::new());
        }
        for i in points_and_edges.1.clone() {
            vertices[i[0]].push(i[1]);
            vertices[i[1]].push(i[0]);
        }
       /// This is written for bad stl files that have a non-closing loop
        for i in 0..vertices.len(){
            if vertices[i].len()==1{
                let selfpoint = vertices[i][0].clone();
                vertices[i].push(selfpoint);
            }
        }
        let mut marked = Vec::new();
        for i in 0..vertices.len() {
            marked.push(false);
        }
        for i in 0..marked.len() {
            if !marked[i] {
                let mut little_collector: Vec<Point> = Vec::with_capacity(5000);
                little_collector.push(points_and_edges.0.get(&(i)).expect("no such key").clone());
                marked[i] = true;
                let mut next = i;
                while (!marked[vertices[next][0] as usize] | !marked[vertices[next][1] as usize]) {
                    if !marked[vertices[next][0] as usize] {
                        marked[vertices[next][0] as usize] = true;
                        little_collector.push(points_and_edges.0.get(&(vertices[next][0])).expect("no such key").clone());
                        next = vertices[next][0] as usize;
                    } else if !marked[vertices[next][1] as usize] {
                        marked[vertices[next][1] as usize] = true;
                        little_collector.push(points_and_edges.0.get(&(vertices[next][1])).expect("no such key").clone());
                        next = vertices[next][1] as usize;
                    } else {
                        println!("something weird has just happened. check stl file or repair");
                    }
                }
                little_collector.push(points_and_edges.0.get(&(i)).expect("no such key").clone());
                collector.push(little_collector);
            }
        }
        return collector;
    }

    /// generates path for all layers in the StlFileSlicer. Output is Vec<Vec<Vec<Points>>> ->
    /// a vector of (vector of ( vector of (for each point in loop) for each closed loop) for each
    /// layer). Parallel version. Serial version is below
    pub fn generate_path_for_all_par(&self) -> Vec<Vec<Vec<Point>>>{
        let find_layers = self.find_intersecting_triangles();
        let mut all_collector: Vec<Vec<Vec<Point>>> = Vec::with_capacity(self.slices.len().clone());
        let mut total = self.slices.len().clone() - 1;
        let mut counter = 0;
        let iterator = (0..self.slices.len()).map(|i| i).collect::<Vec<usize>>();
        let mut all_collector = Vec::with_capacity(self.slices.len().clone());
         iterator.par_iter().map(|&i| self.calc_ips_upe_mpth(&find_layers,i)).collect_into_vec(&mut all_collector);
        return all_collector;
    }

    /// regular serial version of generate_path_for_all
    pub fn generate_path_for_all(&self) -> Vec<Vec<Vec<Point>>>{
        let find_layers = self.find_intersecting_triangles();
        let iterator = (0..self.slices.len()).map(|i| i).collect::<Vec<usize>>();
        let all_collector = iterator.iter().map(|&i| self.calc_ips_upe_mpth(&find_layers,i)).collect::<Vec<Vec<Vec<Point>>>>();
        return all_collector;
    }

    /// convenience function to implement iter.map on the data to implement parallel processing
    fn calc_ips_upe_mpth(&self, find_layers:&Vec<Vec<usize>>,kk:usize)->Vec<Vec<Point>>{
        println!("{} out of {}",kk,self.slices.len().clone()-1);
        let ips = self.calc_intersection_line_plane_layer(&find_layers[kk], self.slices[kk]);
        let upe = StlFileSlicer::find_unique_points_and_edges(ips);
        let mpth = StlFileSlicer::generate_path_for_layer(&(0), upe);
        return mpth;
    }
    /// write the movepath for the model. The continuous loops are separated by NaN,NaN,NaN, and the
    /// layers are separed by NaN,NaN,NaN.
    pub fn write_movepath_to_file(movepath:Vec<Vec<Vec<Point>>>, filename:&str){
        let mut file = File::create(filename).expect("can't create file");
        let mut file3 = std::io::BufWriter::with_capacity(1000000,file);
        for i in movepath{
            for j in i{
                for k in j{
                write!(file3, "{}\n", k);
                }
            write!(file3, "NaN,NaN,NaN\n");
            }
            write!(file3, "NaN,NaN,NaN\n");
        }
    }
}

fn main() {
    println!("Hello, world!");
    //  let mut filet = File::create("c:\\rustFiles\\trisinga.csv").expect("cant create file");
    // let mut file = File::create("c:\\rustFiles\\pointsinga.csv").expect("cant create file");
    // let mut file2 = File::create("c:\\rustFiles\\pointsinga2.csv").expect("cant create file");

    let new_stl_file = StlFile::read_binary_stl_file("c:\\rustFiles\\10.bear.stl");
   // let new_stl_file = StlFile::read_binary_stl_file("/mnt/c/rustFiles/coneb.stl");
    let stl_slicer = StlFileSlicer::new(new_stl_file,0.1);
    let movepath = stl_slicer.generate_path_for_all();
    //StlFileSlicer::write_movepath_to_file(movepath, "c:\\rustFiles\\movepath.csv");
    //StlFileSlicer::write_movepath_to_file(movepath, "/mnt/c/rustFiles/movepath.csv");

}
