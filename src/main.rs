//#![feature(option_result_unwrap_unchecked)]

use std::{fmt, mem};
use std::fs::{File, read_to_string};
use std::io::{BufReader, Read, BufWriter};
use std::hash::{Hash, Hasher};
use std::io::Write;
use rayon::prelude::*;
use rayon::ThreadPool;
use std::time::{Duration, Instant};
use rustc_hash::FxHashMap;
use std::env;



use std::str::FromStr;
use std::mem::transmute;
use fltk::{app, prelude::*, button::*, frame::*, group::*, window::Window, dialog, text::*};
use fltk::dialog::FileDialogType;
//use rand::Rng;


/// OFFSET defines the offset for generating slices. First slice is below the minimum z-value of the
/// model and last slice is above the maximum z-value of the model.
static OFFSET: f64 = 1e-3;

///EPSILON defines the rounding error
static EPSILON:f64 = 1e-7;

/// Point struct stores x, y, and z value of the Point
#[derive(Debug, Copy, Clone)]
struct Point{
    x:f64,
    y:f64,
    z:f64,
}
impl Point{
    /// isValid checks if a point is valid. If either x, y, or z is f64::NAN, f64::NEG_INFINITY, or f64::INFINITY,
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
/// implements PartialEq for Point. x, y, and z are rounded to nearest OFFSET and compared.
/// Necessary for implementing HashMap.
impl PartialEq for Point {
    fn eq(&self, other: &Self) -> bool {
        return ((self.x-other.x).abs() < EPSILON && (self.y-other.y).abs() < EPSILON);//&& (self.z-other.z).abs() < EPSILON)
    }
}
/// implements Eq for Point
impl Eq for Point {}


/// implements Hash for Point. Rounds x, y, and z to nearest ROUND, and takes 128 * x + 32 *y + z.
/// Multiplies the value by 10^ROUND and converts it to u64
impl Hash for Point {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let x = ((self.x) as f32).to_le_bytes();
        let y = ((self.y) as f32).to_le_bytes();
        let pp;
        unsafe{
            //pp = transmute::<[u8;8],u64>([x[0],x[1],x[2],x[3],y[0],y[1],y[2],y[3]]);
            pp = transmute::<[u8;8],u64>([x[2],x[1],x[0],x[3],y[1],y[2],y[0],y[3]]);
        }
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
        let mut minz = f64::INFINITY;
        let mut maxz = f64::NEG_INFINITY;


        for i in 0..num_tri{
            let mut mintri = f64::INFINITY;
            let mut maxtri = f64::NEG_INFINITY;
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


        let mut global_min_z: f64 = f64::INFINITY;
        let mut global_max_z: f64 = f64::NEG_INFINITY;

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
    /// defined by z = c. If intersection point is above or below the line, return f64::NAN for all
    /// values of x, y, and z.
    fn calc_intersection_line_plane(x1: f64, y1: f64, z1: f64, x2: f64, y2: f64, z2: f64, c: f64) -> Point {
        let mut t = (c - z1) / (z2 - z1);
        if ((z1 > c) && (z2 > c)) || ((z1 < c) && (z2 < c)) {
            t = f64::NAN;
        }
        return Point {
            x: x1 + t * (x2 - x1),
            y: y1 + t * (y2 - y1),
            z: c,
        }
    }
    /// calculate intersection edges for triangles and a given plane. Return a vector of two points
    /// that gives the edge of intersection between the triangles and the plane. Uses big array, has lower
    /// precision than the HashMap version, but is faster
    pub fn calc_intersection_line_plane_layer_mem_hungry(&self, triangles_in_layer: &Vec<usize>, zvalue: f64, intersection_points: &mut Vec<Point>,
                                                         edges:&mut Vec<[usize;2]>) {
        intersection_points.clear();
        //points_map.clear();
        edges.clear();
        let mut ip_temp = Vec::with_capacity(100000);
        //let mut e_temp = Vec::with_capacity(100000);
        let mut points_check = vec![false; u32::MAX as usize];
        let mut points_map = vec![0; u32::MAX as usize];

        for i in triangles_in_layer {
            let tdata = self.file.trivals[*i].values;
            let p1 = [tdata[3], tdata[4], tdata[5]];
            let p2 = [tdata[6], tdata[7], tdata[8]];
            let p3 = [tdata[9], tdata[10], tdata[11]];

            let ip1 = StlFileSlicer::calc_intersection_line_plane(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], zvalue);
            let ip2 = StlFileSlicer::calc_intersection_line_plane(p3[0], p3[1], p3[2], p2[0], p2[1], p2[2], zvalue);
            let ip3 = StlFileSlicer::calc_intersection_line_plane(p1[0], p1[1], p1[2], p3[0], p3[1], p3[2], zvalue);

            if (ip1.isValid() && ip2.isValid()) { StlFileSlicer::help_push_into_list(ip1, ip2,  edges, &mut ip_temp)};
            if (ip3.isValid() && ip2.isValid()) { StlFileSlicer::help_push_into_list(ip2, ip3,  edges, &mut ip_temp)};
            if (ip3.isValid() && ip1.isValid()) { StlFileSlicer::help_push_into_list(ip3, ip1,  edges, &mut ip_temp)};
        }


        StlFileSlicer::help_remove_duplicates(&mut ip_temp, edges,&mut points_map, &mut points_check, intersection_points);

    }

    fn help_push_into_list(ip1:Point, ip2:Point, edges: &mut Vec<[usize;2]>, intersection_points: &mut Vec<Point>){
        let iplen = intersection_points.len();
        edges.push([iplen, iplen+1]);
        intersection_points.push(ip1);
        intersection_points.push(ip2);
    }

    fn help_remove_duplicates(intersection_points: &mut Vec<Point>, edges: &mut Vec<[usize;2]>, points_map: &mut Vec<usize>, points_check: &mut Vec<bool>, unique_points: &mut Vec<Point>){
        let mut minx = f64::INFINITY; let mut miny = f64::INFINITY;
        let mut maxx = f64::NEG_INFINITY; let mut maxy = f64::NEG_INFINITY;

        for i in 0..intersection_points.len(){
            if intersection_points[i].x < minx {minx = intersection_points[i].x}
            if intersection_points[i].y < miny {miny = intersection_points[i].y}
            if intersection_points[i].x > maxx {maxx = intersection_points[i].x}
            if intersection_points[i].y > maxy {maxy = intersection_points[i].y}
        }

        for i in edges{
            let pt1x = (((intersection_points[i[0]].x-minx)/(maxx-minx)) as f32).to_le_bytes();
            let pt1y = (((intersection_points[i[0]].y-miny)/(maxy-miny)) as f32).to_le_bytes();
            let pt2x = (((intersection_points[i[1]].x-minx)/(maxx-minx))as f32).to_le_bytes();
            let pt2y = (((intersection_points[i[1]].y-miny)/(maxy-miny)) as f32).to_le_bytes();
            let index1;
            let index2;
            unsafe{
                index1 = mem::transmute::<[u8;8],usize>([pt1x[3],pt1x[2],pt1y[3],pt1y[2],0,0,0,0]);
                index2 = mem::transmute::<[u8;8],usize>([pt2x[3],pt2x[2],pt2y[3],pt2y[2],0,0,0,0]);
            }
            if points_check[index1] {
                i[0] = points_map[index1]
            }
            else{
                unique_points.push(intersection_points[i[0]]);
                points_check[index1] = true;
                points_map[index1] = unique_points.len()-1;
                i[0] = unique_points.len()-1;
            }
            if points_check[index2]{
                i[1] = points_map[index2]
            }
            else
            {
                unique_points.push(intersection_points[i[1]]);
                points_check[index2] = true;
                points_map[index2] = unique_points.len()-1;
                i[1] = unique_points.len()-1;
            }

        }

        //println!("unique points {}",unique_points.len());

    }

    /// calculate intersection edges for triangles and a given plane. Return a vector of two points
    /// that gives the edge of intersection between the triangles and the plane.
    pub fn calc_intersection_line_plane_layer(&self, triangles_in_layer: &Vec<usize>, zvalue: f64 , intersection_points: &mut Vec<Point>,
                                              points_map: &mut FxHashMap<Point,usize>, edges:&mut Vec<[usize;2]>)  {
        //let mut intersection_points: Vec<[Point; 2]> = Vec::with_capacity(80000);
        intersection_points.clear();
        points_map.clear();
        edges.clear();
        //intersection_points.reserve(80000);



        for i in triangles_in_layer {
            let tdata = self.file.trivals[*i].values;
            let p1 = [tdata[3], tdata[4], tdata[5]];
            let p2 = [tdata[6], tdata[7], tdata[8]];
            let p3 = [tdata[9], tdata[10], tdata[11]];

            let ip1 = StlFileSlicer::calc_intersection_line_plane(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], zvalue);
            let ip2 = StlFileSlicer::calc_intersection_line_plane(p3[0], p3[1], p3[2], p2[0], p2[1], p2[2], zvalue);
            let ip3 = StlFileSlicer::calc_intersection_line_plane(p1[0], p1[1], p1[2], p3[0], p3[1], p3[2], zvalue);

            if (ip1.isValid() && ip2.isValid()) { StlFileSlicer::help_push_into_hashmap(ip1, ip2, points_map, edges, intersection_points)};
            if (ip3.isValid() && ip2.isValid()) { StlFileSlicer::help_push_into_hashmap(ip2, ip3, points_map, edges, intersection_points)};
            if (ip3.isValid() && ip1.isValid()) { StlFileSlicer::help_push_into_hashmap(ip3, ip1, points_map, edges, intersection_points)};
        }
       //println!("points len {}, points_map.len() {}, edges.len() {}", intersection_points.len(), points_map.len(), edges.len());
    }

    /// this helper function checks if the point is already in the list of points. If the point is already in the list, adds the index of the
    /// point to the edges, if the point is new, it adds the point to the list
    fn help_push_into_hashmap(ip1:Point, ip2:Point, points_map: &mut FxHashMap<Point,usize>, edges:&mut Vec<[usize;2]>, points: &mut Vec<Point>){
        //superluminal_perf::begin_event("hash");
        let mut e12 = [0,0];
        let tt = points_map.get_key_value(&ip1);
        if tt.is_some(){
            unsafe{e12[0] = *tt.unwrap().1};
        }
        else{
            e12[0] = points_map.len();
            points_map.insert(ip1, points_map.len());
            points.push(ip1);
        }

        let tt2 = points_map.get_key_value(&ip2);
        if tt2.is_some(){
            unsafe{e12[1] = *tt2.unwrap().1};
        }

        else{
            e12[1] = points_map.len();
            points_map.insert(ip2, points_map.len());
            points.push(ip2);
        }
        edges.push(e12);
    }




    /// generates movepath for a given layer using breadth first search for the edges. Assumes each
    /// point is connected to two other points. Panics if a point is connected to only one point.
    /// Gives spurious results if a point is connected to more than two points. Output is Vec<Vec<
    /// Points>> -> list of vectors for each loop in a plane.
    pub fn generate_path_for_layer(start_pt: &u32, points: &Vec<Point>, edges:&Vec<[usize; 2]>,
                                   collector: &mut Vec<Point>, vertices: &mut Vec<Vec<usize>>,
                                   marked:&mut Vec<bool>, vertex_filled: &mut Vec<bool>,
                                   start_pts:&mut Vec<usize>, end_pts:&mut Vec<usize>) {
        //let mut collector = Vec::with_capacity(points_and_edges.0.len()+4000);
        //let mut vertices = Vec::with_capacity(points_and_edges.0.len()+4000);
        //let separation_pts = Vec<usize>;
        collector.clear();
        //vertices.clear();
        marked.clear();

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
                    collector.push(points[i]); start_end_pos += 1;

                    marked[i] = true;
                    let mut next = i;
                    while (!marked[vertices[next][0]] || !marked[vertices[next][1]]) {
                        if !marked[vertices[next][0]] {
                            marked[vertices[next][0]] = true;
                            collector.push(points[vertices[next][0]]); start_end_pos += 1;
                            next = vertices[next][0] ;
                        } else if !marked[vertices[next][1] ] {
                            marked[vertices[next][1]] = true;
                            collector.push(points[vertices[next][1]]); start_end_pos += 1;
                            next = vertices[next][1] ;
                        } else {
                            println!("something weird has just happened. check stl file or repair");
                        }

                    }
                    //loop closing if loop //bad stl files with unclosed geometry will have unclosed loops
                    if vertices[next][1] == i {
                        collector.push(points[vertices[next][1]]); start_end_pos += 1;
                    }
                    else if vertices[next][0] == i {
                        collector.push(points[vertices[next][0]]); start_end_pos += 1;
                    }
                    //collector.push(Point{x:f64::f64::NAN,y:f64::f64::NAN,z:f64::f64::NAN});
                    start_pts.push(start_end_pos);
                    end_pts.push(start_end_pos);
                    //collector.push(Point{x:f64::f64::NAN,y:f64::f64::NAN,z:f64::f64::NAN});
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
        let mut start_pts = Vec::with_capacity(self.slices.len());
        let mut end_pts = Vec::with_capacity(self.slices.len());
        let mut all_collector = Vec::with_capacity(self.slices.len().clone());
        for i in 0..self.slices.len() {
            all_collector.push(Vec::with_capacity(20000));
            start_pts.push(Vec::with_capacity(50));
            end_pts.push(Vec::with_capacity(50));
        }
        let mut vertices = Vec::with_capacity(100000);
        let mut vertex_filled = vec![false; 100000];
        for i in 0..100000{
            vertices.push(Vec::with_capacity(2));
        }
        //let mut reverse_points = FxHashMap::default();//with_capacity(10000);
        let mut edges = Vec::with_capacity(40000);
        let mut marked = Vec::with_capacity(10000);
        let mut points_array:Vec<Point> = Vec::with_capacity(10000);
        println!("total layers, {}", all_collector.len());
        let mut tic = Instant::now();
        let mut tocunique = tic.elapsed();
        let mut tocintersect = tocunique.clone();
        let mut tocgenpath = tocunique.clone();
        println!("find movepath layers start");
        for i in 0..find_layers.len() {
            let mut tic = Instant::now();
            //self.calc_intersection_line_plane_layer(&find_layers[i], self.slices[i], &mut points_array, &mut reverse_points, &mut edges);
            self.calc_intersection_line_plane_layer_mem_hungry(&find_layers[i], self.slices[i], &mut points_array,  &mut edges);
            tocintersect = tocintersect+tic.elapsed();

            let mut tic = Instant::now();
            StlFileSlicer::generate_path_for_layer(&(0), &points_array, &edges, &mut all_collector[i], &mut vertices, &mut marked, &mut vertex_filled, &mut start_pts[i], &mut end_pts[i]);
            tocgenpath = tocgenpath + tic.elapsed();
        }
        println!("total time to find intersection pts and unique {:?}",tocintersect);
        println!("total time to generate path {:?}",tocgenpath);
        println!("find movepath layers end");
        return (all_collector, start_pts, end_pts)
    }

    /// parallel version of generate_path_for_all
    pub fn generate_path_for_all(&self) -> (Vec<Vec<Point>>, Vec<Vec<usize>>, Vec<Vec<usize>>){
        let maxthreads = num_cpus::get_physical()+2;
        println!("using all {} cpus", num_cpus::get_physical());
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
            let mut vertices = Vec::with_capacity(10000);
            let mut points_array:Vec<Point> = Vec::with_capacity(10000);
            let mut vertex_filled = vec![false; 100000];
            for i in 0..100000{
                vertices.push(Vec::with_capacity(2));
            }
            let mut reverse_points = FxHashMap::default();
            let mut edges = Vec::with_capacity(40000);
            let mut marked = Vec::with_capacity(10000);
            for i in 0..ac.len(){
                let kk = layerno[i];
                //ac[i] = self.calc_ips_upe_mpth(find_layers,layerno[i]);
                self.calc_intersection_line_plane_layer(&find_layers[kk], self.slices[kk], &mut points_array, &mut reverse_points, &mut edges);
               // StlFileSlicer::find_unique_points_and_edges(&ips_temp,&mut points, &mut reverse_points, &mut edges,&mut points_array);
                StlFileSlicer::generate_path_for_layer(&(0), &points_array, &edges,&mut ac[i], &mut vertices,&mut marked,&mut vertex_filled, &mut start_pts[i], &mut end_pts[i]);
            }

        }
    }


    pub fn orient_movepath(&self,  movepath: &mut(Vec<Vec<Point>>, Vec<Vec<usize>>, Vec<Vec<usize>>), clockwise: bool){
        for i in 0..movepath.0.len() {
            let path = &mut movepath.0[i];
            let start_pts = &movepath.1[i];
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
                        }
                    }
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

                        if (orientation > 0.0 && clockwise) || (orientation < 0.0 && !clockwise) {
                            path[start_pts[j]..start_pts[j + 1]].reverse();
                        }
                    } else {
                    }
                }
            }
        }
    }



    /// write the movepath for the model. The continuous loops are separated by f64::NAN,f64::NAN,f64::NAN, and the
    /// layers are separed by f64::NAN,f64::NAN,f64::NAN.
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
                    let mut mindist = f64::INFINITY;
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
                    write!(file3, "NAN,NAN,NAN\n");
                }
            }
        }
        write!(file3, "NAN,NAN,NAN\n");
    }

}



/// how to run this program
/// program.exe [input file] [slice height] [parallel or serial] [write or nowrite] [if write : output filename] [orient or noorient] [if orient: clockwise or anticlockwise]
/// untitled22.exe c:\rustfiles\all_shapesb.stl 0.1 parallel write c:\rustfiles\movepath.csv
/// c:\rustfiles\timecmd target\release\untitled22 c:\rustfiles\all_shapesb.stl 0.1 parallel write c:\rustfiles\movepath.csv
fn main() {
    println!("Hello, world!");
    let app = app::App::default();
    let mut wind = Window::new(100,100,400,300,"StlSlicer");
    wind.end();
    wind.show();
    let mut button = Button::new(160,200,80,40,"Slice!");
    wind.add(&button);

   /* let mut filechooser = fltk::dialog::FileChooser::new(".","*.stl",dialog::FileChooserType::Single, "Input file",);
    filechooser.show();
    filechooser.window().set_pos(300,300);

    while filechooser.shown(){
        app::wait();
    }
    if filechooser.value(0).is_none(){
        println!("user cancelled");
    }
    else {
        println!("{}", filechooser.value(0).unwrap());
        println!("{}", filechooser.directory().unwrap());
    }*/

    let mut file_dialog = fltk::dialog::FileDialog::new(FileDialogType::BrowseFile);
    file_dialog.set_directory("c:\\rustfiles");
    file_dialog.set_filter("*.stl");
    file_dialog.set_title("Input STL File");

    





    file_dialog.show();
   // let mut file_directory = ;//file_dialog.directory().as_path().to_str().clone();
    println!("{:?}", file_dialog.filename().to_str().unwrap());
    //println!("{:?}",file_directory );
    //let file_name = file_dialog.filename().file_name().unwrap().to_str().clone();
   // println!("{}", file_name.unwrap());







    app.run().expect("cant run app");


   /* let args:Vec<String> = env::args().collect();
    //  let mut filet = File::create("c:\\rustFiles\\trisinga.csv").expect("cant create file");
    // let mut file = File::create("c:\\rustFiles\\pointsinga.csv").expect("cant create file");
    // let mut file2 = File::create("c:\\rustFiles\\pointsinga2.csv").expect("cant create file");

    println!("{:?}",args.clone());
    println!("read file start");
    let tic = Instant::now();
    let new_stl_file = StlFile::read_binary_stl_file(&*args[1]);
    let toc = tic.elapsed();
    println!("read file end \n time taken to read file, {:?}", toc);


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

    if args.len()>7 {

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
    }*/
}


