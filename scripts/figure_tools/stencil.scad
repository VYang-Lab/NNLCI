// Conceptual 3 x 3 sample locations above a domain tile; dimensions are arbitrary.
flat = false;
$fn = 24;
module scene() {
    color([.85,.91,.95]) translate([0,0,-2]) cube([32,32,2],center=true);
    for (x=[-10:10:10], y=[-10:10:10])
        color([.05,.5,.55]) translate([x,y,1]) sphere(r=(x==0 && y==0)?2.8:1.6);
}
if (flat) projection(cut=false) scene(); else scene();
