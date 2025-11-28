// Définition de la géométrie
Point(1) = {0, 0, 0, 1.0};
Point(2) = {1, 0, 0, 1.0};
Point(3) = {0, 1, 0, 1.0};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 1};
Line Loop(1) = {1, 2, 3};
Surface(1) = {1};

Physical Surface("all") = {1};

// Paramètres de maillage
Mesh.CharacteristicLengthMin = 0.1;
Mesh.CharacteristicLengthMax = 0.2;
Mesh.SecondOrderLinear = 1;
Mesh.ElementOrder = 3;

// Génération du maillage
Mesh 2;
Mesh.MshFileVersion = 4.1;
Save "triangles10_v41.msh";
