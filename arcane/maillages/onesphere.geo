meshSize = 0; // WARNING : Default value. Correct value is at the end of the script.
//BEGIN OF SPHERE 
Point(22)={0.59028242000247755,0.55159101398321897,0.8758215455650612,meshSize};
Point(23)={0.93393242000247756,0.55159101398321897,0.8758215455650612,meshSize};
Point(24)={0.59028242000247755,0.89524101398321898,0.8758215455650612,meshSize};
Point(25)={0.59028242000247755,0.55159101398321897,1.2194715455650611,meshSize};
Point(26)={0.24663242000247754,0.55159101398321897,0.8758215455650612,meshSize};
Point(27)={0.59028242000247755,0.20794101398321896,0.8758215455650612,meshSize};
Point(28)={0.59028242000247755,0.55159101398321897,0.53217154556506119,meshSize};
Circle(22)={23,22,24};
Circle(23)={24,22,26};
Circle(24)={26,22,27};
Circle(25)={27,22,23};
Circle(26)={23,22,28};
Circle(27)={28,22,26};
Circle(28)={26,22,25};
Circle(29)={25,22,23};
Circle(30)={27,22,28};
Circle(31)={28,22,24};
Circle(32)={24,22,25};
Circle(33)={25,22,27};
Line Loop(22) = {22,32,29};
Line Loop(23) = {23,28,-32};
Line Loop(24) = {24,-33,-28};
Line Loop(25) = {25,-29,33};
Line Loop(26) = {26,31,-22};
Line Loop(27) = {-23,-31,27};
Line Loop(28) = {-24,-27,-30};
Line Loop(29) = {-25,30,-26};
Surface(22)={22};
Surface(23)={23};
Surface(24)={24};
Surface(25)={25};
Surface(26)={26};
Surface(27)={27};
Surface(28)={28};
Surface(29)={29};
Surface Loop(21)={22,23,24,25,26,27,28,29};
Volume(21)={21};
//END OF SPHERE 

 // BEGIN  Points of the enveloppe
Point(2)={1.0902824200024774,0.05159101398321897,0.3758215455650612,meshSize};
Point(3)={0.090282420002477548,1.051591013983219,0.3758215455650612,meshSize};
Point(4)={1.0902824200024774,1.051591013983219,0.3758215455650612,meshSize};
Point(5)={0.090282420002477548,0.05159101398321897,1.3758215455650613,meshSize};
Point(6)={1.0902824200024774,0.05159101398321897,1.3758215455650613,meshSize};
Point(7)={0.090282420002477548,1.051591013983219,1.3758215455650613,meshSize};
Point(8)={1.0902824200024774,1.051591013983219,1.3758215455650613,meshSize};
Point(1)={0.090282420002477548,0.05159101398321897,0.3758215455650612,meshSize};
// END  Points of the enveloppe


 // BEGIN  Edges of the enveloppe
Line(2) = {2,4};
Line(3) = {4,3};
Line(4) = {3,1};
Line(5) = {1,2};
Line(6) = {2,6};
Line(7) = {6,8};
Line(8) = {8,4};
Line(11) = {1,5};
Line(12) = {5,6};
Line(14) = {3,7};
Line(15) = {7,5};
Line(20) = {8,7};
// END  Edges of the enveloppe


 // BEGIN  CurveLoops of the enveloppe
Line Loop(2) = {2,3,4,5};
Line Loop(3) = {6,7,8,-2};
Line Loop(4) = {-5,11,12,-6};
Line Loop(5) = {14,15,-11,-4};
Line Loop(6) = {-3,-8,20,-14};
Line Loop(7) = {-15,-20,-7,-12};
// END  CurveLoops of the enveloppe


 // BEGIN  Surfaces of the enveloppe
Plane Surface(4) = {4};
Plane Surface(6) = {6};
Plane Surface(3) = {3};
Plane Surface(5) = {5};
Plane Surface(2) = {2};
Plane Surface(7) = {7};
// END  Surfaces of the enveloppe


 // BEGIN  SurfaceLoop of the enveloppe
Surface Loop(1) = {2,3,4,5,6,7};
// END  SurfaceLoop of the enveloppe


 // BEGIN  Matrix volume
Volume(1) = {1,21};
// END  Matrix volume


 // BEGIN  Periodic surfaces of the enveloppe
Periodic Surface {2 }={ 7 } Translate {0,0,-1};
Periodic Surface {5 }={ 3 } Translate {-0.99999999999999989,0,0};
Periodic Surface {4 }={ 6 } Translate {0,-1,0};
// END  Periodic surfaces of the enveloppe

Physical Surface(1) = {2,3,4,5,6,7};
Physical Volume(1) = {1};
Physical Volume(2) = {21};

MeshSize {:} =0.50000000000000003;

Mesh.ElementOrder =2;
Mesh 2;
Mesh 3;
Mesh.MshFileVersion = 4.1;
Save "onesphere.msh";
