SetFactory("OpenCASCADE");

DefineConstant[ R = {1.0, Min 0.1, Max 10, Step 0.1, Name "Rayon"} ];
DefineConstant[ h = {0.2, Min 0.01, Max 2, Step 0.01, Name "Taille de maille"} ];

Sphere(1) = {0, 0, 0, R};

// raffiner = baisser h (~2*Pi*R/h éléments sur un grand cercle)
Mesh.MeshSizeMin = h;
Mesh.MeshSizeMax = h;

// seul le volume est dans un Physical -> le maillage exporté ne contient que les tétras
Physical Volume("sphere") = {1};
