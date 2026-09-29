// Production geometry for the NNLCI POI animation. Units are display units.
// Both LF planes use this SAME 100 x 100 slab and 20 x 20 patch frame.
// Native node spacing is applied by Python: LF1=5 units, LF2=2.5 units.
// Export each part as OFF; the CPU compositor projects the actual CAD meshes.
part = "plane";
$fn = 20;

if (part == "plane")
    translate([-50, -50, -2]) cube([100, 100, 2]);
else if (part == "patch")
    translate([0, 0, 0.35]) linear_extrude(height=0.5)
        difference() {
            square([21, 21], center=true);
            square([20, 20], center=true);
        }
else if (part == "sample")
    cylinder(h=0.55, r=0.65, $fn=12);
else if (part == "pin")
    union() {
        cylinder(h=9, r1=0, r2=3.4);
        translate([0, 0, 10]) sphere(r=4.4);
    }
else
    assert(false, "Unknown part");
