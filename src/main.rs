extern crate minidom;

use std::{fmt, mem};
use std::fs::{File, read_to_string};
use std::io::{BufReader, Read, BufWriter};
use std::f64::{INFINITY, NEG_INFINITY, NAN};
use std::collections::{HashMap, BTreeMap};
use std::hash::{Hash, Hasher};
use std::io::Write;
use rayon::prelude::*;
use rayon::ThreadPool;
use std::time::{Duration, Instant};
use std::cmp::{Ord, PartialOrd, Ordering};
use rustc_hash::FxHashMap;
use std::env;


use minidom::Element;
use std::str::FromStr;
//use rand::Rng;


/// OFFSET defines the offset for generating slices. First slice is below the minimum z-value of the
/// model and last slice is above the maximum z-value of the model.
static OFFSET: f64 = 1e-3;

/// ROUND defines the rounding of f64 to u64 for Point struct. This is necessary to implement
/// PartialEq and Hash traits. These traits are necessary to implement HashKey and find unique
/// intersection points and how they are connected.
static ROUND: u32 = 4294967295;
static EPSILON:f64 = 1e-7;

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
        return self.x.is_finite() && self.y.is_finite() && self.z.is_finite();
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
        return ((self.x-other.x).abs() < EPSILON && (self.y-other.y).abs() < EPSILON);//&& (self.z-other.z).abs() < EPSILON)
        /*return (half_down(self.x) == half_down(other.x))&
            (half_down(self.y) == half_down(other.y)) &
            (half_down(self.z) == half_down(other.z));*/
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

fn iseq(a:f64,b:f64)->bool{
    (a-b).abs()<EPSILON
}
/// implements Ord for Point
impl Ord for Point {
    fn cmp(&self, other: &Self) -> Ordering {
        return if iseq(self.x, other.x) {
            if iseq(self.y, other.y) {
                Ordering::Equal
            } else if self.y > other.y {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        } else if self.x > other.x {
            Ordering::Greater
        } else {
            Ordering::Less
        }
    }
}

impl PartialOrd for Point {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}


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
        //let z = half_down(self.z);
        let pp = (x*1024+y) as u64;
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
///

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

    pub fn new(points:Vec<f64>)->StlFile{
        let num_tri = points.len()/9;
        let mut trivals = Vec::with_capacity(num_tri);
        let mut minz = INFINITY;
        let mut maxz = NEG_INFINITY;


        for i in 0..num_tri{
            let mut mintri = INFINITY;
            let mut maxtri = NEG_INFINITY;
            let mut values = [0.0;12];
            for j in 0..9{
                //println!("{},{},{}",i*12+j, i,j);
                values[j+3] = points[i*9+j];
            }
            if values[5] > maxtri {maxtri = values[5]};
            if values[8] > maxtri {maxtri = values[8]};
            if values[11] > maxtri {maxtri = values[11]};
            if values[5] < mintri {mintri = values[5]};
            if values[8] < mintri {mintri = values[8]};
            if values[11] < mintri {mintri = values[11]};
            let triangle = Triangle{
                minz: mintri,
                maxz: maxtri,
                values: values,
            };

            minz = f64::min(mintri, minz);
            maxz = f64::max(maxtri, maxz);

            trivals.push(triangle);
        }

        return StlFile{
            minz,
            maxz,
            info:"created with array input".to_string(),
            num_tri: num_tri,
            trivals: trivals,
        }
    }

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
        let mut all_triangles: Vec<Triangle> = Vec::with_capacity(num_tri as usize);


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
        if ((z1 > c) && (z2 > c)) || ((z1 < c) && (z2 < c)) {
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
    pub fn calc_intersection_line_plane_layer(&self, triangles_in_layer: &Vec<usize>, zvalue: f64 , intersection_points: &mut Vec<[Point;2]>)  {
        //let mut intersection_points: Vec<[Point; 2]> = Vec::with_capacity(80000);
        intersection_points.clear();
        //intersection_points.reserve(80000);

        for i in triangles_in_layer {
            let tdata = self.file.trivals[*i].values;
            let p1 = [tdata[3], tdata[4], tdata[5]];
            let p2 = [tdata[6], tdata[7], tdata[8]];
            let p3 = [tdata[9], tdata[10], tdata[11]];

            let ip1 = StlFileSlicer::calc_intersection_line_plane(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], zvalue);
            let ip2 = StlFileSlicer::calc_intersection_line_plane(p3[0], p3[1], p3[2], p2[0], p2[1], p2[2], zvalue);
            let ip3 = StlFileSlicer::calc_intersection_line_plane(p1[0], p1[1], p1[2], p3[0], p3[1], p3[2], zvalue);

            if (ip1.isValid() && ip2.isValid()) { intersection_points.push([ip1, ip2]) };
            if (ip3.isValid() && ip2.isValid()) { intersection_points.push([ip3, ip2]) };
            if (ip3.isValid() && ip1.isValid()) { intersection_points.push([ip3, ip1]) };
        }
    }

    /// find unique points and edges created by intersection of triangles and a plane. Unique
    /// vertices are stored as a HashMap and edges refer to the index of the vertex.
    pub fn find_unique_points_and_edges(edges_input: &Vec<[Point; 2]>, points: &mut FxHashMap<usize, Point>, reverse_points: &mut FxHashMap<Point,usize> , edges: &mut Vec<[usize;2]>) {
        //let mut points: HashMap<usize, Point> = HashMap::with_capacity(100000);
        //let mut reverse_points: HashMap<Point, usize> = HashMap::with_capacity(100000);
        points.clear();
        reverse_points.clear();

        //let mut edges = Vec::with_capacity(40000);
        edges.clear();
        let mut points_counter: usize = 0;
        //let mut edges_counter: usize = 0;
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
    }

    /// generates movepath for a given layer using breadth first search for the edges. Assumes each
    /// point is connected to two other points. Panics if a point is connected to only one point.
    /// Gives spurious results if a point is connected to more than two points. Output is Vec<Vec<
    /// Points>> -> list of vectors for each loop in a plane.
    pub fn generate_path_for_layer(start_pt: &u32, points: &FxHashMap<usize, Point>, edges:&Vec<[usize; 2]>,
                                   collector: &mut Vec<Point>, vertices: &mut Vec<Vec<usize>>,
                                   marked:&mut Vec<bool>, vertex_filled: &mut Vec<bool>,
                                   start_pts:&mut Vec<usize>, end_pts:&mut Vec<usize>) {
        //let mut collector = Vec::with_capacity(points_and_edges.0.len()+4000);
        //let mut vertices = Vec::with_capacity(points_and_edges.0.len()+4000);
        //let separation_pts = Vec<usize>;
        collector.clear();
        //vertices.clear();
        marked.clear();
        //vertex_filled.clear();
        //vertices.reserve();
        // vertex can be connected to a maximum of two other vertices for a closed loop

        /* for i in 0..points.len(){
             vertices.push(Vec::with_capacity(2));
         }*/
        //et vertex_filled:Vec<bool> = vec![false; points.len()];
        //vertices.clear();
        // let mut vertices = Vec::with_capacity(10000);
        // find first point
        let first_point = 0;


        start_pts.push(0);
        let mut start_end_pos = 0;

        for i in 0..points.len() {
            //let mut m = [0,0];
            vertices[i].clear();
            //vertices.push(Vec::new());
        }
        for i in edges {
            vertices[i[0]].push(i[1]);
            vertices[i[1]].push(i[0]);
        }


        // This is written for bad stl files that have a non-closing loop
        for i in 0..points.len(){
            if !vertex_filled[i]{
                let selfpoint = vertices[i][0].clone();
                vertices[i].push(selfpoint);
                //vertices[i][1]= vertices[i][0];
            }
        }
        //let mut marked = Vec::with_capacity(vertices.len());
        if vertices.len()>0{
            for i in 0..points.len() {
                marked.push(false);
            }
            for i in 0..marked.len() {

                if !marked[i] {
                    //collector: Vec<Point> = Vec::with_capacity(2000);
                    // println!("error at {}",i);
                    collector.push(points.get(&(i)).expect("no such key").clone()); start_end_pos += 1;

                    marked[i] = true;
                    let mut next = i;
                    while (!marked[vertices[next][0]] || !marked[vertices[next][1]]) {
                        if !marked[vertices[next][0]] {
                            marked[vertices[next][0]] = true;
                            collector.push(points.get(&(vertices[next][0])).expect("no such key").clone()); start_end_pos += 1;
                            next = vertices[next][0] ;
                        } else if !marked[vertices[next][1] ] {
                            marked[vertices[next][1]] = true;
                            collector.push(points.get(&(vertices[next][1])).expect("no such key").clone()); start_end_pos += 1;
                            next = vertices[next][1] ;
                        } else {
                            println!("something weird has just happened. check stl file or repair");
                        }

                    }
                    //loop closing if loop //bad stl files with unclosed geometry will have unclosed loops
                    if vertices[next][1] == i {
                        collector.push(points.get(&(vertices[next][1])).expect("no such key").clone()); start_end_pos += 1;
                    }
                    else if vertices[next][0] == i {
                        collector.push(points.get(&(vertices[next][0])).expect("no such key").clone()); start_end_pos += 1;
                    }
                    //collector.push(Point{x:f64::NAN,y:f64::NAN,z:f64::NAN});
                    start_pts.push(start_end_pos);
                    end_pts.push(start_end_pos);
                    //collector.push(Point{x:f64::NAN,y:f64::NAN,z:f64::NAN});
                }




            }

        }
        // println!("collector len {}",collector.len());
    }

    pub fn generate_path_for_layer2(start_pt: &u32, points: &FxHashMap<usize, Point>, edges:&Vec<[usize; 2]>, collector: &mut Vec<Point>, vertices: &mut Vec<usize>, start_end: &mut Vec<[usize;2]>, marked: &mut Vec<bool>) {
        collector.clear();






    }


    /// generates path for all layers in the StlFileSlicer. Output is Vec<Vec<Vec<Points>>> ->
    /// a vector of (vector of ( vector of (for each point in loop) for each closed loop) for each
    /// layer). Parallel version. Serial version is below
    ///
    ///generates path in serial
    pub fn generate_path_for_all_serial(&self) -> (Vec<Vec<Point>>,Vec<Vec<usize>>, Vec<Vec<usize>>) {
        println!("find intersecting triangles start");
        let find_layers = self.find_intersecting_triangles();
        println!("find intersecting triangles end");
        let mut all_collector: Vec<Vec<Vec<Point>>> = Vec::with_capacity(self.slices.len().clone());
        let mut total = self.slices.len().clone() - 1;
        let mut counter = 0;
        //let iterator = (0..self.slices.len()).map(|i| i).collect::<Vec<usize>>();
        let mut start_pts = Vec::with_capacity(self.slices.len());
        let mut end_pts = Vec::with_capacity(self.slices.len());
        let mut all_collector = Vec::with_capacity(self.slices.len().clone());
        for i in 0..self.slices.len() {
            all_collector.push(Vec::with_capacity(20000));
            start_pts.push(Vec::with_capacity(50));
            end_pts.push(Vec::with_capacity(50));

        }
        let mut ips_temp = Vec::with_capacity(100000);
        let mut vertices = Vec::with_capacity(100000);
        let mut vertex_filled = vec![false; 100000];
        for i in 0..100000{
            vertices.push(Vec::with_capacity(2));
        }
        let mut points = FxHashMap::default();//with_capacity(10000);
        let mut reverse_points = FxHashMap::default();//with_capacity(10000);
        let mut edges = Vec::with_capacity(40000);
        let mut marked = Vec::with_capacity(10000);

        println!("total layers, {}", all_collector.len());
        println!("find movepath layers start");
        for i in 0..find_layers.len() {
            self.calc_intersection_line_plane_layer(&find_layers[i], self.slices[i], &mut ips_temp);
            StlFileSlicer::find_unique_points_and_edges(&ips_temp, &mut points, &mut reverse_points, &mut edges);
            StlFileSlicer::generate_path_for_layer(&(0), &points, &edges, &mut all_collector[i], &mut vertices, &mut marked, &mut vertex_filled, &mut start_pts[i], &mut end_pts[i]);
        }
        println!("find movepath layers end");
        return (all_collector, start_pts, end_pts)
    }

    /// parallel version of generate_path_for_all
    pub fn generate_path_for_all(&self) -> (Vec<Vec<Point>>, Vec<Vec<usize>>, Vec<Vec<usize>>){
        let maxthreads = num_cpus::get_physical()+1;
        println!("using all {} cpus", maxthreads-1);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(maxthreads)
            .build()
            .expect("can't build the threadpool");
        println!("group layers start");
        let find_layers = self.find_intersecting_triangles();
        println!("group layers end");
        //let iterator = (0..self.slices.len()).map(|i| i).collect::<Vec<usize>>();
        let mut all_collector:Vec<Vec<Point>> =Vec::with_capacity(self.slices.len());
        let mut start_pts = Vec::with_capacity(self.slices.len());
        let mut end_pts = Vec::with_capacity(self.slices.len());
        for i in 0..self.slices.len(){
            //let mut loops = Vec::with_capacity(15);
            let mut single_loop:Vec<Point> = Vec::with_capacity(20000);
            //loops.push(single_loop);
            all_collector.push(single_loop);
            start_pts.push(Vec::with_capacity(50));
            end_pts.push(Vec::with_capacity(50));

        }
        /* {
             let (all_collector1, all_collector2) = all_collector.split_at_mut(self.slices.len() / 2);
             pool.install(||rayon::join(
                 || {
                     for i in 0..self.slices.len() / 2 {
                         all_collector1[i] = self.calc_ips_upe_mpth(&find_layers, i);
                     }
                 },
                 || {
                     for i in self.slices.len() / 2..self.slices.len() {
                         all_collector2[i-self.slices.len()/2] = self.calc_ips_upe_mpth(&find_layers,i);
                     }
                 }
             ));
         }

         //let all_collector = iterator.iter().map(|&i| self.calc_ips_upe_mpth(&find_layers,i)).collect::<Vec<Vec<Vec<Point>>>>();*/
        let mut layerno = Vec::with_capacity(self.slices.len());
        for i in 0..self.slices.len(){
            layerno.push(i);
        }
        println!("generate mpath parallel start");
        self.generate_mpth_parallel(all_collector.as_mut_slice(), 1, maxthreads, &pool,&find_layers,layerno.as_slice() , start_pts.as_mut_slice(), end_pts.as_mut_slice());
        println!("generate mpath parallel end");
        return (all_collector, start_pts, end_pts);
    }
    /// helper function to generate parallel path
    fn generate_mpth_parallel(&self, ac:&mut [Vec<Point>], cpus:usize, max_cpus:usize,pool:&ThreadPool, find_layers:&[Vec<usize>],layerno:&[usize],start_pts:&mut [Vec<usize>], end_pts:&mut [Vec<usize>]){
        if cpus<max_cpus{
            let aclen = ac.len();
            let(ac1,ac2) = ac.split_at_mut(aclen/2);
            let (sp1,sp2) = start_pts.split_at_mut(aclen/2);
            let (ep1,ep2) = end_pts.split_at_mut(aclen/2);
            //let(fl1, fl2) = find_layers.split_at(aclen/2);
            let(ln1,ln2) = layerno.split_at(aclen/2);
            pool.install(||rayon::join(||self.generate_mpth_parallel( ac1, cpus*2, max_cpus, pool, find_layers,ln1, sp1, ep1),
                                       ||self.generate_mpth_parallel(ac2,cpus*2,max_cpus,pool, find_layers,ln2, sp2, ep2)
            ));
        }
        else{
            let mut ips_temp = Vec::with_capacity(100000);
            let mut vertices = Vec::with_capacity(100000);
            let mut vertex_filled = vec![false; 100000];
            for i in 0..100000{
                vertices.push(Vec::with_capacity(2));
            }
            let mut points = FxHashMap::default();//with_capacity(10000);
            let mut reverse_points = FxHashMap::default();//with_capacity(10000);
            let mut edges = Vec::with_capacity(40000);
            let mut marked = Vec::with_capacity(10000);
            for i in 0..ac.len(){
                let kk = layerno[i];
                //ac[i] = self.calc_ips_upe_mpth(find_layers,layerno[i]);
                self.calc_intersection_line_plane_layer(&find_layers[kk], self.slices[kk], &mut ips_temp);
                StlFileSlicer::find_unique_points_and_edges(&ips_temp,&mut points, &mut reverse_points, &mut edges);
                StlFileSlicer::generate_path_for_layer(&(0), &points, &edges,&mut ac[i], &mut vertices,&mut marked,&mut vertex_filled, &mut start_pts[i], &mut end_pts[i]);
            }

        }
    }

    /// convenience function to implement iter.map on the data to implement parallel processing
    /* fn calc_ips_upe_mpth(&self, find_layers:&[Vec<usize>],kk:usize)->Vec<Vec<Point>>{
         // println!("{} out of {}",kk,self.slices.len().clone()-1);
         self.calc_intersection_line_plane_layer(&find_layers[kk], self.slices[kk]);
         let upe = StlFileSlicer::find_unique_points_and_edges(ips);
         let mpth = StlFileSlicer::generate_path_for_layer(&(0), upe);
         return mpth;
     }*/

    pub fn orient_movepath2(&self,  movepath: &mut(Vec<Vec<Point>>, Vec<Vec<usize>>, Vec<Vec<usize>>), clockwise: bool){
        //let mut file = File::create(filename).expect("can't create file");
        //let mut file3 = std::io::BufWriter::with_capacity(1000000,file);
        for i in 0..movepath.0.len() {
            let path = &mut movepath.0[i];
            let start_pts = &movepath.1[i];
            //let end_pts = &movepath.2[i];
            //println!("start {:?}", start_pts);
            //println!("end {:?}", end_pts);
            //println!("end pts len", end_pts.len());
            //println

            if start_pts.len() > 0 {
                for j in 0..start_pts.len() - 1 {
                    let mut minpt = 0;
                    let mut minx = f64::INFINITY;
                    let mut miny = f64::INFINITY;
                    let mut centerx = 0.0;
                    let mut centery = 0.0;
                    let mut count = 0;
                    for k in start_pts[j]..start_pts[j + 1] {
                        centerx = centerx + path[k].x;
                        centery = centery + path[k].y;
                        count = count + 1;
                        if (path[k].x - minx).abs() > 1e-7 {
                            if path[k].x < minx {
                                minx = path[k].x;
                                miny = path[k].y;
                                minpt = k;
                            }
                        } else {
                            if (path[k].y - miny).abs() > 1e-7 {
                                if path[k].y < miny {
                                    miny = path[k].y;
                                    minpt = k;
                                }
                            }
                            //write!(file3, "{}\n", path[k]);
                        }
                        //write!(file3, "NaN,NaN,NaN\n");
                    }
                    /* if i == 1 {
                        println!("minx miny {} {} {}",minx, miny, minpt);
                    }*/
                    centerx = centerx/count as f64;
                    centery = centery/count as f64;
                    let a;
                    let b;
                    let c;
                    if minpt == 0 {
                        a = Point{x:centerx, y:centery, z:0.0};
                        b = path[minpt];
                        c = path[minpt + 1];
                    } else if minpt == path.len() - 1 {
                        a = Point{x:centerx, y:centery,z:0.0};
                        b = path[minpt-1];
                        c = path[minpt];
                    } else {
                        a = Point{x:centerx, y:centery,z:0.0};
                        b = path[minpt-1];
                        c = path[minpt];
                    }


                    let orientation = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
                    let mvo = (a.x - b.x).abs() * 1e-3;

                    if (orientation.abs() > mvo) {
                        //println!("layer {} loop {} orientation > 0.0 {} and clockwise {}", i,j,orientation>0.0, clockwise);
                        //println!("orientation >0.0 and clockwise {} , {} , {}", orientation>0.0, clockwise, (orientation>0.0 && clockwise));
                        //println!("orientation <0.0 and anticlockwise {} , {}, {}", orientation<0.0, !clockwise, (orientation<0.0 && !clockwise));

                        if (orientation > 0.0 && clockwise) || (orientation < 0.0 && !clockwise) {
                            //swap from begin to end
                             //println!{"orientation changed for layer {} loop {}",i,j }
                            path[start_pts[j]..start_pts[j + 1]].reverse();
                        }
                    } else {
                        //println!("err : could not orient. layer{}, loop, {}, determinant of three consecutive points too small ,{}, mvo {}", i, j, orientation, mvo);
                    }
                }


            }
            //write!(file3, "NaN,NaN,NaN\n");
        }



    }
    /// calculate centroid and check orientation with the first point
    pub fn orient_movepath(&self,  movepath: &mut(Vec<Vec<Point>>, Vec<Vec<usize>>, Vec<Vec<usize>>), clockwise: bool){
        //let mut file = File::create(filename).expect("can't create file");
        //let mut file3 = std::io::BufWriter::with_capacity(1000000,file);
        for i in 0..movepath.0.len() {
            let path = &mut movepath.0[i];
            let start_pts = &movepath.1[i];
            //let end_pts = &movepath.2[i];
            //println!("start {:?}", start_pts);
            //println!("end {:?}", end_pts);
            //println!("end pts len", end_pts.len());
            //println

            if start_pts.len() > 0 {

                for j in 0..start_pts.len() - 1 {

                    let mut minpt = 0;
                    let mut minx = f64::INFINITY;
                    let mut miny = f64::INFINITY;

                    let mut centerx = 0.0;
                    let mut centery = 0.0;
                    let mut count = 0;

                    for k in start_pts[j]..start_pts[j + 1] {
                        centerx += path[k].x;
                        centery += path[k].y;
                        count += 1;


                    }
                    /* if i == 1 {
                        println!("minx miny {} {} {}",minx, miny, minpt);
                    }*/
                    centerx = centerx/count as f64;
                    centery = centery/count as f64;
                    let a = Point {x:centerx, y:centery, z:0.0};
                    let b = path[start_pts[j]];
                    let c = path[start_pts[j]+1];



                    let orientation = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
                    let mvo = (a.x - b.x).abs() * 1e-3;

                    if (orientation.abs() > mvo) {
                        //println!("layer {} loop {} orientation > 0.0 {} and clockwise {}", i,j,orientation>0.0, clockwise);
                        //println!("orientation >0.0 and clockwise {} , {} , {}", orientation>0.0, clockwise, (orientation>0.0 && clockwise));
                        //println!("orientation <0.0 and anticlockwise {} , {}, {}", orientation<0.0, !clockwise, (orientation<0.0 && !clockwise));

                        if (orientation > 0.0 && clockwise) || (orientation < 0.0 && !clockwise) {
                            //swap from begin to end
                            //println!{"orientation changed for layer {} loop {}",i,j }
                            path[start_pts[j]..start_pts[j + 1]].reverse();
                        }
                    } else {
                        //println!("err : could not orient. layer{}, loop, {}, determinant of three consecutive points too small ,{}, mvo {}", i, j, orientation, mvo);
                    }
                }


            }
            //write!(file3, "NaN,NaN,NaN\n");
        }



    }


    /// write the movepath for the model. The continuous loops are separated by NaN,NaN,NaN, and the
    /// layers are separed by NaN,NaN,NaN.
    pub fn write_movepath_to_file(movepath:(Vec<Vec<Point>>, Vec<Vec<usize>>, Vec<Vec<usize>>), filename:&str){
        let mut file = File::create(filename).expect("can't create file");
        let mut file3 = std::io::BufWriter::with_capacity(1000000,file);
        let st_pt = Point{x:0.0,y:0.0,z:0.0};

        for i in 0..movepath.0.len(){
            let path = &movepath.0[i];
            let start_pts = &movepath.1[i];
            //let end_pts = &movepath.2[i];
            //println!("start {:?}", start_pts);
            //println!("end {:?}", end_pts);
            //println!("end pts len", end_pts.len());
            //println
            if start_pts.len() >0 {
                for j in 0..start_pts.len() - 1 {
                    let mut min_pt_pos = 0;
                    let mut mindist = INFINITY;
                    for k in start_pts[j]..start_pts[j + 1]{
                        let dx = path[k].x-st_pt.x;
                        let dy = path[k].y-st_pt.y;
                        let dist = dx*dx + dy*dy;
                        if dist<mindist{
                            min_pt_pos = k;
                            mindist = dist;
                        }
                    }
                    for k in (min_pt_pos..start_pts[j + 1]).chain(start_pts[j]..min_pt_pos+1) {
                        write!(file3, "{}\n", path[k]);
                    }
                    write!(file3, "NaN,NaN,NaN\n");
                }
            }
        }
        write!(file3, "NaN,NaN,NaN\n");
    }

}


/*pub fn stl_to_movepath(points:Vec<f64>, sliceheight:f64)-> Vec<f64>{
    let stlfile = StlFile::new(points);
    let stlslicer = StlFileSlicer::new(stlfile, sliceheight);
    let movepath = stlslicer.generate_path_for_all_serial();
    let mut mpthvec = Vec::with_capacity(movepath.len());
    //let mut layerchangepos = Vec::with_capacity(movepath.len());
    for i in movepath{
        //let mut loopchangepos = Vec::with_capacity(i.len());
        //layerchangepos.push(i.len());
        for j in i{
            // loopchangepos.push(j.len());
            for k in j{
                mpthvec.push(k.x);
                mpthvec.push(k.y);
                mpthvec.push(k.z);
            }
            mpthvec.push(NAN);
            mpthvec.push(NAN);
            mpthvec.push(NAN);
        }
        mpthvec.push(NAN);
        mpthvec.push(NAN);
        mpthvec.push(NAN);

    }
    return mpthvec;
}*/


///Triangle from AMF file has three numbered vertices
#[derive(Debug, Copy, Clone)]
struct AmfTriangle{
    minz:f64,
    maxz:f64,
    v0:usize,
    v1:usize,
    v2:usize,
}

///AmfFile
struct AmfFile{
    vertices: Vec<Point>,
    triangles: Vec<AmfTriangle>,
    num_tri:usize
}

impl AmfFile{
    fn new(filename: &str)->AmfFile{
        let mut file = File::open(filename).expect("can't open the file");
        let mut filebuffer = BufReader::with_capacity(10000,file);
        let mut vertices = Vec::with_capacity(10000);
        let mut triangles = Vec::with_capacity(10000);
        let input_string = read_to_string(filename).expect("cant read to string");
        let root: minidom::Element = input_string.parse().expect("can't parse the data");
        for i in root.children(){
            if i.name() == "object" {
                for j in i.children(){
                    if j.name() == "mesh" {
                        for k in j.children() {
                            if k.name() == "vertices"{
                                for l in k.children(){
                                        for m in l.children(){
                                            let mut newpt = Point{x:0.0,y:0.0,z:0.0};
                                            for n in m.children(){
                                                if n.name() == "x"{ newpt.x = f64::from_str(n.text().as_str()).expect("can't parse to f64")}
                                                if n.name() == "y"{ newpt.y = f64::from_str(n.text().as_str()).expect("can't parse to f64")}
                                                if n.name() == "z"{ newpt.z = f64::from_str(n.text().as_str()).expect("can't parse to f64")}
                                            }
                                            vertices.push(newpt);
                                        }
                                }
                            }

                            if k.name() == "volume"{
                                for l in k.children(){
                                    if l.name() == "triangle"{
                                        let mut atri = AmfTriangle{ minz: 0.0, maxz: 0.0, v0:0, v1:0, v2:0};
                                        for m in l.children(){
                                            if m.name() == "v1" {atri.v0 = usize::from_str(m.text().as_str()).expect("can't parse to usize")};
                                            if m.name() == "v2" {atri.v1 = usize::from_str(m.text().as_str()).expect("can't parse to usize")};
                                            if m.name() == "v3" {atri.v2 = usize::from_str(m.text().as_str()).expect("can't parse to usize")};
                                        }
                                        // find minz and maxz for atri
                                        let z1 = vertices[atri.v0].z; let z2 = vertices[atri.v1].z; let z3 = vertices[atri.v2].z;
                                        let mut minz = z1; let mut maxz=z1;
                                        if z2<minz{minz=z2}; if z3<minz{minz=z3};
                                        if z2>maxz{maxz=z2}; if z3>maxz{maxz=z3};
                                        atri.minz = minz; atri.maxz = maxz;
                                        triangles.push(atri);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let mut num_tri = triangles.len();
    AmfFile{vertices,triangles,num_tri}
    }
}

/// how to run this program
/// program.exe [input file] [slice height] [parallel or serial] [write or nowrite] [if write : output filename]
/// untitled22.exe c:\rustfiles\all_shapesb.stl 0.1 parallel write c:\rustfiles\movepath.csv
/// c:\rustfiles\timecmd target\release\untitled22 c:\rustfiles\all_shapesb.stl 0.1 parallel write c:\rustfiles\movepath.csv
fn main() {
    println!("Hello, world!");
    let args:Vec<String> = env::args().collect();
    //  let mut filet = File::create("c:\\rustFiles\\trisinga.csv").expect("cant create file");
    // let mut file = File::create("c:\\rustFiles\\pointsinga.csv").expect("cant create file");
    // let mut file2 = File::create("c:\\rustFiles\\pointsinga2.csv").expect("cant create file");

    println!("{:?}",args.clone());
    println!("read file start");
    let tic = Instant::now();
    let new_stl_file = StlFile::read_binary_stl_file(&*args[1]);//"c:\\rustfiles\\07.tesla.stl");
    //let new_stl_file = StlFile::read_binary_stl_file("/mnt/c/rustfiles/10.bear.stl");
    let toc = tic.elapsed();
    println!("read file end \n time taken to read file, {:?}", toc);
    //let mut data = vec![-100.0, 100.0, 0.0, 0.0, 0.0, 214.45069885253906, 100.0, 100.0, 0.0, 100.0, 100.0, 0.0, 0.0, 0.0, 214.45069885253906, 100.0, -100.0, 0.0, 100.0, -100.0, 0.0, 0.0, 0.0, 214.45069885253906, -100., -100., 0., -100., -100., 0., 0., 0., 214.45069885253906, -100., 100., 0., 100., 100., 0., -100., 100., 0., 100., -100.0, 0.0, 100.0, -100.0, 0.0, -100.0, 100.0, 0.0, -100.0, -100.0,101.0, 0.0];

    //let new_stl_file = StlFile::new(data);
    //println!("{},{},{},{:?},{}",new_stl_file.num_tri, new_stl_file.minz, new_stl_file.maxz,new_stl_file.trivals, new_stl_file.info);
    // let new_stl_file = StlFile::read_binary_stl_file("/mnt/c/rustFiles/coneb.stl");
    //let stl_slicer = StlFileSlicer::new(new_stl_file,1.0);

    let stl_slicer = StlFileSlicer::new(new_stl_file,args[2].parse::<f64>().expect("not working"));

    println!("slicing start");

    let mut movepath;
    if args[3] == "parallel"{
        let tic = Instant::now();
        movepath = stl_slicer.generate_path_for_all();
        let toc = tic.elapsed();
        println!("time taken to slice total, {:?}", toc);
    } else{
        let tic = Instant::now();
        movepath = stl_slicer.generate_path_for_all_serial();
        let toc = tic.elapsed();
        println!("time taken to slice total, {:?}", toc);
    }
    //println!("args len {}",args.len());
    if args.len()>7 {
        //println!("this code is run YAAAY");
       // println!("args 5 and 6 {} {}", &args[6], &args[7]);
        if args[6] == "orient" {

            if args[7] == "clockwise" {
                let tic = Instant::now();
                stl_slicer.orient_movepath(&mut movepath, true);
                let toc = tic.elapsed();
                println!("time taken to orient the movepath {} , {:?}", args[7], toc);
            } else if args[7] == "anticlockwise" {
                let tic = Instant::now();
                stl_slicer.orient_movepath(&mut movepath, false);
                let toc = tic.elapsed();
                println!("time taken to orient the movepath {} , {:?}", args[7], toc);
            } else {
                println!("invalid orientation, use clockwise or anticlockwise, movepath will not be oriented");
            }
        }
    }

    if args[4] == "write" {
    println!("writing file");
        let tic = Instant::now();
        StlFileSlicer::write_movepath_to_file(movepath, &*args[5]);
        let toc = tic.elapsed();
        println!("time taken to write file {:?}",toc);
    }
    //StlFileSlicer::write_movepath_to_file(movepath, "c:\\rustFiles\\movepath.csv");
    //StlFileSlicer::write_movepath_to_file(movepath, "/mnt/c/rustFiles/movepath.csv");
}


