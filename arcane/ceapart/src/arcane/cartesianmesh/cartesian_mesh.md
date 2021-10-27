Gestion des maillages cartésiens {#arcanedoc_cartesianmesh}
================================

\tableofcontents

Cette page décrit la gestion des maillages cartésiens dans %Arcane.

\note Pour l'instant, %Arcane ne gère pas le recalcul des infos
de structuration lorsque le maillage change.

\warning Les classes gérant la structuration ne sont
finalisées et l'API pourra donc évoluer si besoin.

Initialisation {#arcanedoc_cartesianmesh_init}
-----------------------------------

Pour avoir les infos sur un maillage cartésien, il est nécessaire
d'avoir une instance de la classe Arcane::ICartesianMesh. Pour
l'instant, une telle instance est créée via la méthode
Arcane::arcaneCreateCartesianMesh():

~~~~~~~~~~~~~~~~~~~~~{.cpp}
using namespace Arcane;
IMesh* mesh = ...;
ICartesianMesh* cartesian_mesh = arcaneCreateCartesianMesh(mesh);
~~~~~~~~~~~~~~~~~~~~~

Une fois l'instance créée et avant de pouvoir l'utiliser, il
est nécessaire de calculer les infos de direction via la méthode
Arcane::ICartesianMesh::computeDirections(). Cet appel ne doit être fait
qu'une seule fois, par exemple lors de l'initialisation du code.

~~~~~~~~~~~~~~~~~~~~~{.cpp}
cartesian_mesh->computeDirections();
~~~~~~~~~~~~~~~~~~~~~

Utilisation des infos par direction {#arcanedoc_cartesianmesh_direction}
-----------------------------------

Une fois ceci fait, il est possible d'avoir des infos sur les entités
pour une direction donnée. Les directions possibles sont données par
le type #eMeshDirection. Il est aussi possible d'utiliser un entier
pour spécifier la direction, 0 correspondant à la direction X, 1 à la
direction Y et 2 à la direction Z. Pour des raisons de lisibilité, il
est conseillé d'utiliser le type énuméré si possible.
Par exemple, pour récupérer les infos
sur les mailles de la direction Y:

~~~~~~~~~~~~~~~~~~~~~{.cpp}
using namespace Arcane;
CellDirectionMng cell_dm(cartesian_mesh->cellDirection(MD_DirY));
CellDirectionMng cell_dm(cartesian_mesh->cellDirection(1));
~~~~~~~~~~~~~~~~~~~~~

\warning Les objets gérant les entités par direction sont des objets
temporaires qui ne doivent pas être conservés notamment d'une
itération à l'autre ou lorsque le maillage change.

Une fois une direction récupérée, il est possible d'itérer sur toutes
les entités de la direction et pour les mailles par exemple d'avoir
la maille avant et après:

~~~~~~~~~~~~~~~~~~~~~{.cpp}
using namespace Arcane;
ENUMERATE_CELL(icell,cell_dm.allCells()){
  Cell cell = *icell;
  DirCell dir_cell(cell_dm[icell]); // Infos de direction pour cell
  Cell prev_cell = dir_cell.previous(); // Maille avant
  Cell next_cell = dir_cell.next(); // Maille après.
}
~~~~~~~~~~~~~~~~~~~~~

Pour les mailles de bord, il est possible que \a prev_cell ou \a
next_cell soit nulle. Cela peut se tester via la méthode Arcane::Cell::null().

La récupération des noeuds d'une direction se fait de la même manière.

~~~~~~~~~~~~~~~~~~~~~{.cpp}
using namespace Arcane;
NodeDirectionMng node_dm(cartesian_mesh->nodeDirection(MD_DirX));
ENUMERATE_NODE(inode,node_dm.allNodes()){
  Node node = *inode;
  DirNode dir_node(node_dm[inode]); // Infos de direction pour node
  Node prev_cell = dir_node.previous(); // Noeud avant
  Node next_cell = dir_node.next(); // Noeud après
}
~~~~~~~~~~~~~~~~~~~~~

Pour les faces, l'écriture est similaire mais au lieu de récupérer
la face avant et après la face courante, on peut récupérer la maille
avant et après:

~~~~~~~~~~~~~~~~~~~~~{.cpp}
using namespace Arcane;
FaceDirectionMng face_dm(cartesian_mesh->faceDirection(MD_DirX));
ENUMERATE_FACE(iface,face_dm.allFaces()){
  Face face = *iface;
  DirFace dir_face(face_dm[iface]);
  Cell prev_cell = dir_face.previousCell(); // Maille avant la face
  Cell next_cell = dir_face.nextCell(); // Maille après la face
}
~~~~~~~~~~~~~~~~~~~~~

Enfin, pour les mailles, il est possible de récupérer des infos
directionnelles sur les noeuds d'une maille suivant une direction,
via la classe DirCellNode.
\note Cela n'est pour l'instant implémenté que pour les maillages 2D.

~~~~~~~~~~~~~~~~~~~~~{.cpp}
using namespace Arcane;
CellDirectionMng cell_dm(cartesian_mesh->cellDirection(MD_DirY));
ENUMERATE_CELL(icell,cell_dm.allCells()){
  Cell cell = *icell;
  DirCellNode cn(cell_dm.cellNode(cell));
  Node next_left = cn.nextLeft(); // Noeud à gauche vers la maille d'après.
  Node next_right = cn.nextRight(); // Noeud à droite vers la maille d'après.
  Node prev_right = cn.previousRight(); // Noeud à droite vers la maille d'avant .
  Node prev_left = cn.previousLeft(); // Noeud à gauche vers la maille d'avant .
}
~~~~~~~~~~~~~~~~~~~~~

De la même manière, il est aussi possible de connaitre la face devant et derrière la
maille dans une direction donnée (cela fonctionne aussi en 3D):

~~~~~~~~~~~~~~~~~~~~~{.cpp}
using namespace Arcane;
CellDirectionMng cell_dm(cartesian_mesh->cellDirection(MD_DirY));
ENUMERATE_CELL(icell,cell_dm.allCells()){
  Cell cell = *icell;
  DirCellFace cf(cell_dm.cellFace(cell));
  Face next_left = cf.next(); // Face connectée à la maille d'après.
  Face prev_right = cf.previous(); // Face connectée à la maille d'avant.
}
~~~~~~~~~~~~~~~~~~~~~

Pour itérer sur toutes les directions d'un maillage, il est
possible de boucler comme suit:

~~~~~~~~~~~~~~~~~~~~~{.cpp}
using namespace Arcane;
Integer nb_dir = mesh->dimension();
for( Integer idir=0; idir<nb_dir; ++idir){
  CellDirectionMng cdm(cartesian_mesh->cellDirection(idir));
  ENUMERATE_CELL(icell,cdm.allCells()){
    ...
  }
}
~~~~~~~~~~~~~~~~~~~~~

Depuis la version 1.23.0 de %Arcane, il est possible de connaître le
nombre global de maille dans une direction donnée via
CellDirectionMng::globalNbCell(). De même, il est possible, en
supposant que le découpage en sous-domaine peut se représenter sous
forme d'une grille, de connaître la numérotation dans cette
grille via CellDirectionMng::subDomainOffset(). Cette numérotation
commence à 0.

Depuis la version 1.23.1, il est aussi possible de connaître le
nombre de maille propre du sous-domaine dans une direction donnée via
CellDirectionMng::ownNbCell(). Il est aussi possible de connaître
l'offset dans la grile de la première maille propre via
CellDirectionMng::ownCellOffset().

\warning Ces informations ne sont accessibles que si le maillage a
été généré via le générateur spécifique cartésien. En particulier,
elles ne sont pas accessible si le maillage est issu d'un
fichier. Pour plus d'informations, se reporter à la description de
ces méthodes.

Utilisation des connectivités cartésiennnes {#arcanedoc_cartesianmesh_cartesian_connectivity}
-----------------------------------

Depuis la version 1.20.0 de %Arcane, il est possible en 2D d'avoir
accès aux mailles autour d'un noeud et aux noeuds de la maille sans
passer par les connectivités directionnelles. Cela se fait via
l'objet CartesianConnectivity qui est retourné par l'appel à
ICartesianMesh::connectivity(). Par exemple:

\snippet CartesianMeshTesterModule.cc SampleNodeToCell

Et de la même manière pour les mailles:

~~~~~~~~~~~~~~~~~~~~~{.cpp}
using namespace Arcane;
CartesianConnectivity cc = cartesian_mesh->connectivity();
ENUMERATE_CELL(icell,allCells()){
  Cell c = *icell;
  Node n1 = cc.upperLeft(c); // Noeud en haut à gauche
  Node n2 = cc.upperRight(c); // Noeud en haut à droite
  Node n3 = cc.lowerRight(c); // Noeud en bas à droite
  Node n4 = cc.lowerLeft(c); // Noeud en bas à gauche
}
~~~~~~~~~~~~~~~~~~~~~

Depuis la version 1.22.1, ces connectivités sont aussi accessible en
3D. La nomemclature est la même que pour les connectivités 2D. Le
préfixe topZ est utilisé pour les noeuds du dessus de la même suivant
la direction Z. Pour ceux du dessous, il n'y a pas de préfixe et donc
le nom de la méthode est le même qu'en 2D. Cela permet éventuellement
d'utiliser le même code en 2D et en 3D.

~~~~~~~~~~~~~~~~~~~~~{.cpp}
using namespace Arcane;
CartesianConnectivity cc = cartesian_mesh->connectivity();
ENUMERATE_CELL(icell,allCells()){
  Cell c = *icell;
  Node n1 = cc.upperLeft(c); // Noeud en dessous en Z, en haut à gauche
  Node n2 = cc.upperRight(c); // Noeud en dessous en Z, en haut à droite
  Node n3 = cc.lowerRight(c); // Noeud en dessous en Z, en bas à droite
  Node n4 = cc.lowerLeft(c); // Noeud en dessous en Z, en bas à gauche
  Node n5 = cc.topZUpperLeft(c); // Noeud au dessus en Z, en haut à gauche
  Node n6 = cc.topZUpperRight(c); // Noeud au dessus en Z, en haut à droite
  Node n7 = cc.topZLowerRight(c); // Noeud au dessus en Z,en bas à droite
  Node n8 = cc.topZLowerLeft(c); // Noeud au dessus en Z,en bas à gauche
}
~~~~~~~~~~~~~~~~~~~~~
