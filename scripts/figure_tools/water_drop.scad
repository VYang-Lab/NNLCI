// A closed 3D teardrop. Display geometry only: this marks a CFD sample point.
// Broad rounded body, tapered tip; 32 sides keep the CPU lesson inexpensive.
$fn = 32;
rotate_extrude(convexity=4)
    polygon([[0, 0], [2.1, .5], [3.6, 1.5], [4.5, 3], [4.8, 4.5],
             [4.5, 6], [3.6, 7.8], [2.5, 9.6], [1.4, 11.5],
             [.55, 13.2], [0, 15]]);
