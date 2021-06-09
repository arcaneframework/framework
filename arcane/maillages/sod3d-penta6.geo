//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {0.5, 0, 0, 1.0};
//+
Point(3) = {1, 0, 0, 1.0};
//+
Point(4) = {0, 0.1, 0, 1.0};
//+
Point(5) = {0.5, 0.1, 0, 1.0};
//+
Point(6) = {1, 0.1, 0, 1.0};
//+
Line(1) = {4, 1};
//+
Line(2) = {5, 2};
//+
Line(3) = {6, 3};
//+
Line(4) = {1, 2};
//+
Line(5) = {2, 3};
//+
Line(6) = {4, 5};
//+
Line(7) = {5, 6};
//+
Curve Loop(1) = {1, 4, -2, -6};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {2, 5, -3, -7};
//+
Plane Surface(2) = {2};
//+
Transfinite Surface {1};
//+
//Surface {2};
//+
Transfinite Curve {1, 2, 3} = 8 Using Progression 1;
//+
Transfinite Curve {6, 4, 7, 5} = 24 Using Progression 1;
//+
Recombine Surface {1};
//+
Extrude {0, 0, 0.1} {
  Surface{1}; Surface{2}; Curve{7}; Curve{2}; Curve{5}; Curve{3}; Curve{6}; Curve{1}; Curve{4}; Point{4}; Point{1}; Point{2}; Point{5}; Point{6}; Point{3}; Layers{5}; Recombine;
}

//+
Physical Surface("XMIN") = {16};
//+
Physical Surface("XMAX") = {46};
//+
Physical Surface("YMAX") = {50, 28};
//+
Physical Surface("YMIN") = {20, 42};
//+
Physical Surface("ZMAX") = {29, 51};
//+
Physical Surface("ZMIN") = {1, 2};

//+
Physical Volume("ZD") = {2};
//+
Physical Volume("ZG") = {1};
