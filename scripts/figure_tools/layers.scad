// Editable conceptual layers. Set flat=true for a vector top projection.
flat = false;
$fn = 24;
module scene() {
    for (z=[0:2]) color([.08+.08*z, .45+.12*z, .65])
        translate([0,0,z*7]) cube([30,22,2],center=true);
    translate([23,0,0]) {
        cylinder(h=16,r=1.2);
        translate([0,0,16]) cylinder(h=5,r1=3,r2=0);
    }
}
if (flat) projection(cut=false) scene(); else scene();
