//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {0, 0.1, 0, 1.0};
//+
Point(3) = {0, 0.1, 0.1, 1.0};
//+
Point(4) = {0, 0.0, 0.1, 1.0};
//+
Point(5) = {0.5, 0.0, 0., 1.0};
//+
Point(6) = {0, 0.1, 0, 1.0};
//+
Point(7) = {0, 0, 0, 1.0};
//+
Point(8) = {0.5, 0.1, 0, 1.0};
//+
Point(9) = {0.5, 0.1, 0.1, 1.0};
//+
Point(10) = {0.5, 0., 0.1, 1.0};
//+
Point(11) = {1, 0., 0, 1.0};
//+
Point(12) = {1, 0.1, 0, 1.0};
//+
Point(13) = {1, 0.1, 0.1, 1.0};
//+
Point(14) = {1, 0., 0.1, 1.0};
//+
Line(1) = {2, 3};
Transfinite Curve {1} = 5;
//+
Line(2) = {9, 3};
Transfinite Curve {2} = 25;
//+
Line(3) = {2, 8};
Transfinite Curve {3} = 25;
//+
Line(4) = {8, 9};
Transfinite Curve {4} = 5;
//+
Line(5) = {8, 5};
Transfinite Curve {5} = 5;
//+
Line(6) = {5, 10};
Transfinite Curve {6} = 5;
//+
Line(7) = {10, 9};
Transfinite Curve {7} = 5;
//+
Line(8) = {2, 1};
Transfinite Curve {8} = 5;
//+
Line(9) = {4, 1};
Transfinite Curve {9} = 5;
//+
Line(10) = {4, 3};
Transfinite Curve {10} = 5;
//+
Line(11) = {1, 5};
Transfinite Curve {11} = 25;
//+
Line(12) = {4, 10};
Transfinite Curve {12} = 25;
//+
Line(13) = {13, 12};
//+
Line(14) = {12, 11};
//+
Line(15) = {11, 14};
//+
Line(16) = {14, 13};
//+
Line(17) = {13, 9};
//+
Line(18) = {8, 12};
//+
Line(19) = {11, 5};
//+
Line(20) = {10, 14};
//+
Curve Loop(1) = {3, 4, 2, -1};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {4, -17, 13, -18};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {8, 11, -5, -3};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {8, -9, 10, -1};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {11, 6, -12, 9};
//+
Plane Surface(5) = {5};
//+
Curve Loop(6) = {10, -2, -7, -12};
//+
Plane Surface(6) = {6};
//+
Curve Loop(7) = {5, 6, 7, -4};
//+
Plane Surface(7) = {7};
//+
Curve Loop(8) = {5, -19, -14, -18};
//+
Plane Surface(8) = {8};
//+
Curve Loop(9) = {20, -15, 19, 6};
//+
Plane Surface(9) = {9};
//+
Curve Loop(10) = {17, -7, 20, 16};
//+
Plane Surface(10) = {10};
//+
Curve Loop(11) = {14, 15, 16, 13};
//+
Plane Surface(11) = {11};
//+
Surface Loop(1) = {1, 3, 4, 5, 6, 7};
//+
Volume(1) = {1};
//+
Surface Loop(2) = {2, 10, 9, 11, 8, 7};
//+
Volume(2) = {2};
//+
Characteristic Length {2, 3, 1, 4, 8, 9, 5, 10, 12, 13, 14, 11} = 0.02;
//+
Physical Volume("ZG") = {1};
//+
Physical Volume("ZD") = {2};
//+
Physical Surface("XMIN") = {4};
//+
Physical Surface("XMAX") = {11};
//+
Physical Surface("YMAX") = {2, 1};
//+
Physical Surface("YMIN") = {5, 9};
//+
Physical Surface("ZMAX") = {10, 6};
//+
Physical Surface("ZMIN") = {3, 8};

Transfinite Surface {1};
Recombine Surface {1};

Transfinite Surface {3};
Recombine Surface {3};

Transfinite Surface {4};
Recombine Surface {4};

Transfinite Surface {5};
Recombine Surface {5};

Transfinite Surface {6};
Recombine Surface {6};

Transfinite Surface {7};
Recombine Surface {7};

Transfinite Volume {1};