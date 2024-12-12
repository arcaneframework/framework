// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneBasicMeshSubdividerService.cc                         (C) 2000-2024 */
/*                                                                           */
/* Service Arcane gérant un maillage du jeu de données.                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IMeshSubdivider.h"
#include "arcane/impl/ArcaneBasicMeshSubdividerService_axl.h"

#include "arcane/core/ItemGroup.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IGhostLayerMng.h"
#include "arcane/core/MeshUtils.h"
#include "arcane/core/IMeshModifier.h"

#include "arcane/core/SimpleSVGMeshExporter.h" // Write au format svg pour le 2D
// Write variables

#include "arcane/core/ServiceBuilder.h"
#include "arcane/core/Directory.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/BasicService.h"
#include "arcane/core/IPrimaryMesh.h"
#include "arcane/core/Item.h"
// Post processor
#include "arcane/core/PostProcessorWriterBase.h"
// get parameter
#include "arcane/utils/ApplicationInfo.h"
#include "arcane/utils/CommandLineArguments.h"

// Ajouter des variables
#include "arcane/core/VariableBuildInfo.h"
// utils
#include <unordered_set>
#include <algorithm>
#include <iterator>

#include "arcane/core/IMeshUtilities.h"

#include <arcane/utils/List.h>
//
#include "arcane/core/ISubDomain.h"

#include <arcane/utils/MDDim.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

typedef UniqueArray<UniqueArray<Int64>> StorageRefine;


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe Pattern qui permet de manipuler un motif (pattern en anglais) de raffinement.
 */
class Pattern{
  public:
    Int16 type;          // Type de l'élément a rafiner
    Int16 face_type;     // Type de la face de l'élément a rafiner
    Int16 cell_type;     // Type des cellules enfants
    StorageRefine nodes; // Matrice pour la génération des nouveaux noeuds
    StorageRefine faces; // idem pour les faces
    StorageRefine cells; // idem pour les cellules
    StorageRefine child_faces; // Lien entre faces de la cellule mère et les faces de la cellule fille
  public:
  Pattern():type(0),face_type(0),cell_type(0){}

  Pattern(Int16 type, Int16 face_type, Int16 cell_type, StorageRefine nodes,StorageRefine faces,StorageRefine cells,StorageRefine child_faces){
    this->type = type;
    this->face_type = face_type;
    this->cell_type = cell_type;
    this->nodes = nodes;
    this->faces = faces;
    this->cells = cells;
    this->child_faces = child_faces;
  }

  Pattern(Pattern&& other) noexcept
        : type(other.type),
          face_type(other.face_type),
          cell_type(other.cell_type),
          nodes(other.nodes),
          faces(other.faces),
          cells(other.cells),
          child_faces(other.child_faces){
        std::cout << "Constructeur par déplacement appelé\n";
    }
  Pattern& operator=(const Pattern& other) {
    if (this != &other) {
      type = other.type;
      face_type = other.face_type;
      cell_type = other.cell_type;
      nodes = other.nodes; // Référence partagée
      faces = other.faces; // Référence partagée
      cells = other.cells; // Référence partagée
      child_faces = other.child_faces ;
    }
    return *this;
  }

  Pattern& operator=(Pattern&& other) noexcept {
    if (this != &other) {
        type = other.type;
        face_type = other.face_type;
        cell_type = other.cell_type;
        nodes = other.nodes;
        faces = other.faces;
        cells = other.cells;
        child_faces = other.child_faces ;
    }
    return *this;
  }
  Pattern& operator=(Pattern& other) noexcept {
    if (this != &other) {
        type = other.type;
        face_type = other.face_type;
        cell_type = other.cell_type;
        nodes = other.nodes;
        faces = other.faces;
        cells = other.cells;
        child_faces = other.child_faces;
    }
    return *this;
  }

};

class PatternBuilder{
  public:
  // N'utilise pas la numérotation des faces de arcane pour l'instant
  static Pattern hextohex(){
    StorageRefine nodes = {
      {0, 1}, // 8  // Sur arêtes
      {0, 3}, // 9
      {0, 4}, // 10
      {1, 2}, // 11
      {1, 5}, // 12
      {2, 3}, // 13
      {2, 6}, // 14
      {3, 7}, // 15
      {4, 5}, // 16
      {4, 7}, // 17
      {5, 6}, // 18
      {6, 7}, // 19
      { 0, 1, 2, 3 }, // 20 // Sur faces
      { 0, 1, 5, 4 }, // 21
      { 0, 4, 7, 3 }, // 22
      { 1, 5, 6, 2 }, // 23
      { 2, 3, 7, 6 }, // 24
      { 4, 5, 6, 7 }, // 25
      {0, 1, 5, 4, 3, 2, 7, 6} // 26 // Centroid
    };
    StorageRefine faces = {
        // Externes
        {0, 8, 20, 9},    // Derrière // 0 1 2 3  // 0
        {9, 20, 13, 3},
        {8, 1, 11, 20},
        {20, 11, 2, 13},
        {0, 10, 22, 9},   // Gauche // 0 3 7 4 // 1
        {9, 22, 15, 3},
        {10, 4, 17, 22},
        {22, 17, 7, 15},
        {4, 16, 21, 10},  // Bas // 4 5 0 1 // 2
        {10, 21, 8, 0},
        {16, 5, 12, 21},
        {21, 12, 1, 8},
        {4 ,16, 25 ,17}, // Devant // 4 5 6 7 // 3
        {17, 25, 19, 7},
        {16, 5, 18, 25},
        {25, 18, 6, 19},
        {1, 12, 23, 11},  // Droite // 1 2 5 6 // 4
        {11, 23, 14, 2},
        {12, 5, 18, 23},
        {23, 18, 6, 14},
        {7, 19 ,24, 15},  // Haut // 7 6 2 3 // 5
        {19, 6 ,14, 24},
        {15, 24, 13, 3},
        {24, 14, 2, 13},
        // Internes
        {8, 20, 26, 21},
        {20, 13, 24, 26},
        {9, 22, 26, 20},
        {20, 26, 23, 11},
        {21, 16, 25, 26},
        {26, 25, 19, 24},
        {22, 17, 25, 26},
        {26, 25, 18, 23},
        {10, 21, 26, 22},
        {21, 12, 23, 26},
        {22, 26, 24, 15},
        {26, 23, 14, 24},

    };
    StorageRefine child_faces= {
        {0,1,2,3},
      {4,5,6,7},
      {8,9,10,11},
      {12,13,14,15},
      {16,17,18,19},
      {20,21,22,23}
    };
    StorageRefine cells = {  
        {0, 8, 20, 9, 10, 21, 26, 22 },
        {10, 21, 26, 22, 4, 16, 25, 17 },
        {8, 1, 11, 20, 21, 12, 23, 26 },
        {21, 12, 23, 26, 16, 5, 18, 25 },
        {9, 20, 13, 3, 22, 26, 24, 15 },
        {22, 26, 24, 15, 17, 25, 19, 7 },
        {20, 11, 2, 13, 26, 23, 14, 24 },
        {26, 23, 14, 24, 25, 18, 6, 19 }
    };
    return Pattern(IT_Hexaedron8,IT_Quad4,IT_Hexaedron8,nodes,faces,cells,child_faces);
  }

  static Pattern tettotet(){
    StorageRefine nodes = {
      {0,1}, // 4
      {1,2}, // 5
      {0,2}, // 6
      {0,3}, // 7
      {2,3}, // 8
      {1,3}, // 9
    };
    StorageRefine faces = {
      {0, 4, 6},
      {0, 6, 7},
      {0, 4, 7},
      {4, 6, 7},
      {1, 4, 5},
      {4, 5, 9},
      {1, 4, 9},
      {1, 5, 9},
      {2, 5, 6},
      {2, 6, 8},
      {5, 6, 8},
      {2, 5, 8},
      {7, 8, 9},
      {3, 7, 8},
      {3, 7, 9},
      {3, 8, 9},
      {4, 7, 9},
      {4, 6, 9},
      {6, 7, 9},
      {4, 5, 6},
      {5, 6, 9},
      {6, 8, 9},
      {6, 7, 8},
      {5, 8, 9}
    };
    StorageRefine child_faces = {
      {0,19,4,8},
      {1,22,19,9},
      {2,16,6,14},
        {11,23,7,15}
    };
    StorageRefine cells = {
      {0, 4, 6, 7},
      {4, 1, 5, 9},
      {6, 5, 2, 8},
      {7, 9, 8, 3},
      {4, 6, 7, 9},
      {4, 9, 5, 6},
      {6, 7, 9, 8},
      {6, 8, 9, 5}
    };
    return {IT_Tetraedron4,IT_Triangle3,IT_Tetraedron4,nodes,faces,cells,child_faces};
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service Arcane un maillage du jeu de données.
 */
class ArcaneBasicMeshSubdividerService
: public ArcaneArcaneBasicMeshSubdividerServiceObject
{
 public:

  explicit ArcaneBasicMeshSubdividerService(const ServiceBuildInfo& sbi);

 public:
  void _init();
  static UniqueArray<Int64> _computeNodeUid(UniqueArray<Int64> node_uid ,const StorageRefine & node_pattern );
  void _computeNodeCoord();
  /*
  void _computeNodeCoord();
  void _computeNodeUid();
  void _computeFaceUid();
  void _computeCellUid();
  void _processOwner();
  void _setOwner();
  void _processOwnerCell();
  void _processOwnerFace();
  void _processOwnerNode();
  void _getRefinePattern(Int16 type);
  void _execute();
    */
  /*Creer un tetra et test la subdivision*/
  void _testTetra(IPrimaryMesh* mesh);

  void _generateOneTetra(IPrimaryMesh* mesh);

  void _refineOneTetra(IPrimaryMesh* mesh);

  void _generateOneHexa(IPrimaryMesh* mesh);

  void _generateOneTri(IPrimaryMesh * mesh);

  void _generateOneQuad(IPrimaryMesh * mesh);

  void _uniqueArrayTest();

  void subdivideMesh([[maybe_unused]] IPrimaryMesh* mesh) override;

  void _refineOnce([[maybe_unused]] IPrimaryMesh* mesh);
};


void ArcaneBasicMeshSubdividerService::_refineOnce(IPrimaryMesh* mesh)
{
  // options()->nbSubdivision();
  info() << "#subdivide mesh";
  // Est-ce qu'on est en 3D ?
  // Si non on fait rien;
  bool is_implemented=false;
  ENUMERATE_CELL(icell,mesh->ownCells()){
    const Cell & cell = *icell;
    if( cell.itemTypeId() == IT_Hexaedron8 || cell.itemTypeId() == IT_Tetraedron4 ){
      is_implemented = true;
    }else {
      info() << "Not implemented item type" << cell.itemTypeId() ;
      return ;
    }
  }

  std::unordered_map<Int16, Pattern> pattern_manager;
  // Génération des patterns
  pattern_manager[IT_Tetraedron4] = PatternBuilder::tettotet();
  pattern_manager[IT_Hexaedron8] = PatternBuilder::hextohex();
  Pattern & a = pattern_manager[IT_Tetraedron4];

  Int32 my_rank = mesh->parallelMng()->commRank();
  IMeshModifier* mesh_modifier = mesh->modifier();
  IGhostLayerMng* gm = mesh->ghostLayerMng();
  debug() << "PART 3D nb ghostlayer" << gm->nbGhostLayer();
  mesh->utilities()->writeToFile("3D_last_input"+std::to_string(my_rank)+".vtk", "VtkLegacyMeshWriter");
  Integer nb_ghost_layer_init = gm->nbGhostLayer(); // On garde le nombre de ghost layer initial pour le remettre à la fin
  Int32 version = gm->builderVersion();
  if (version < 3)
    gm->setBuilderVersion(3);

  gm->setNbGhostLayer(0);
  mesh_modifier->setDynamic(true);
  mesh_modifier->updateGhostLayers();

  // Uniquement pour les vérifications asserts à la fin
  Integer nb_cell_init = mesh->nbCell();
  Integer nb_face_init = mesh->nbFace();
  Integer nb_edge_init = mesh->nbEdge();
  Integer nb_node_init = mesh->nbNode();

  // Compter les arêtes
  // On cherche un moyen de compter les arêtes pour faire un test facile sur le nombre de noeud inséré.
  // ARCANE_ASSERT((nb_edge_init+ nb_cell_init + nb_face_init)== nb_node_added,("Mauvais nombre de noeuds insérés"));
  //debug() << "#NOMBRE INITS " << nb_node_init << " " << mesh->allEdges().size() << " " << edg.size() << " " << nb_face_init << " " << nb_cell_init ;

  // VARIABLES
  // Items à ajouter avec connectivités pour E F et C
  UniqueArray<Int64> nodes_to_add;
  UniqueArray<Int64> edges_to_add;
  UniqueArray<Int64> faces_to_add;
  UniqueArray<Int64> cells_to_add;

  Integer nb_cell_to_add = 0;
  Integer nb_face_to_add = 0;

  VariableNodeReal3& nodes_coords = mesh->nodesCoordinates();
  std::unordered_map<Int64, Real3> nodes_to_add_coords;
  debug() << "ARRAY SIZE " << nodes_coords.arraySize() ;
  // Noeuds sur les entités
  std::set<Int64> new_nodes; // Utiliser une map permet s'assurer qu'on ajoute une seule fois un noeud avec un uniqueId()
  std::set<Int64> new_faces; // ^--- Pareil pour les faces
  // Maps pour la gestions des propriétaires
  std::unordered_map<Int64, Int32> node_uid_to_owner;
  std::unordered_map<Int64, Int32> edge_uid_to_owner; // pas utilisé
  std::unordered_map<Int64, Int32> face_uid_to_owner;
  std::unordered_map<Int64, Int32> child_cell_owner;   // pas utilisé
  std::unordered_map<Int32, Int32> old_face_lid_to_owner; // pas utilisé

  UniqueArray<Int32> cells_to_detach; // Cellules à détacher
  UniqueArray<Int64> faces_uids; // Contient uniquement les uids pas les connectivités
  UniqueArray<Int64> edges_uids; // Idem

  // Calcul nombre de noeuds à insérer
  const Integer nb_node_to_add_total = mesh->nbCell()+mesh->nbFace()+mesh->nbEdge(); // Attention pattern dépendant
  //nodes_to_add.reserve(nb_node_to_add_total);
  //nodes_to_add_coords.reserve(nb_node_to_add_total);

  Integer ind_new_cell = 0 ;

  ARCANE_ASSERT((mesh->nbEdge() == 0 ),("Wrong number of edge"));

  UniqueArray<Int64> parent_faces(mesh->ownFaces().size());
  UniqueArray<Int64> parent_cells(mesh->ownCells().size());
  UniqueArray<Int64> child_cells; // Toutes les nouvelles cells
  UniqueArray<Int64> child_faces; // Au bord de l'élément uniquement (pas de faces interne à l'élément)

  // Permet de récupere les entités enfants à partir d'une cellule parent
  std::unordered_map<Int64, std::pair<Int64,Int64>> parents_to_childs_cell; // A partir d'un uid on récupère le premier enfant (pair<index,number of child>)
  std::unordered_map<Int64,std::pair<Int64,Int64>> parents_to_childs_faces; // A partir d'un uid on récupère le premier enfant (pair<index,number of child>)
  // ^--- uniquement pour les faces "externes"
  Int64 childs_count=0; // A chaque nouvelle cellule on se décale du nombre d'enfant (  +4 si quad, +3 ou +4 pour les tri selon le pattern )

  // Groups
  std::unordered_map<Int64,std::pair<Int64,Int64>> parents_to_childs_faces_groups; // Map face parente -> face enfant externe dans array face_external_uid
  UniqueArray<Int64> face_external_uid; // Toutes les faces externes du proc uid

  // ATTENTION DEBUG NODE #TAG
  /*ENUMERATE_NODE(inode,mesh->ownNodes())
  {
    const Node & node = *inode;
    nodes_to_add_coords[node.uniqueId().asInt64()] = nodes_coords[node];
  }*/
  // Traitement pour une cellule
  ENUMERATE_CELL(icell,mesh->ownCells())
  {
      debug() << "Refining element" ;
      // POUR UN ELEMNT:
      // Détacher cellules parente
      // Génération des nouveaux noeuds (uid et coordonnées)
      // Sur Arêtes
      // Sur Faces
      // Sur Cellule
      // Nouveaux noeuds,coordonnées

      // Génération des Faces (uid et composants (Noeuds)) utilises nouveaux noeuds
      // Internes
      // Externes

      // Génération des Cellules (uid et composants (Noeuds))
      // Gestion des groupes
      // FIN UN ELEMENT

      // Détachement des cellules
      // Ajout des noeuds enfants
      // Ajout des faces enfants
      // Ajout des cellules enfants (et assignation propriétaire)

      // Ajout d'une couche fantome
      // Calcul des propriétaires des noeuds
      // Calcul des propriétaires des faces
      // Supression de la couche fantome
      // ?? Calcul des groupes F C

      // Assignation des noeuds au propriétaire
      // Assignation des faces au propriétaire

      const Cell& cell = *icell;

      Pattern & p = pattern_manager[cell.type()]; // Pattern Manager
      StorageRefine & node_pattern = p.nodes;

      UniqueArray<Int64> face_in_cell;      // Toutes les faces de la cellule uid
      StorageRefine & child_faces = p.child_faces;

      cells_to_detach.add(cell.localId());
      // Génération des noeud
      UniqueArray<Int64> node_in_cell;
      node_in_cell.resize(node_pattern.size()+cell.nbNode()); // pattern dépendant
      UniqueArray<Real3> coords_in_cell;
      debug() << "Noeuds initiaux" ;
      // Noeuds initiaux
      for(Int32 i = 0 ; i < cell.nbNode() ; i++ ) {
        node_in_cell[i] = cell.node(static_cast<Int32>(i)).uniqueId().asInt64();
        //debug() << i << " " << node_in_cell[i] << " size " << node_in_cell.size() ;
      }

      Integer index_27 = cell.nbNode();

      // - Génération des uid noeuds
      debug() << "Génération des uid noeuds" ;
      debug() << "Génération des uid noeudsbkbkbkbk" ;
      for( Integer i = 0 ; i < node_pattern.size() ; i++ ){
        // uid
        UniqueArray<Int64> tmp;
        for( Integer j = 0 ; j < node_pattern[i].size() ; j++  ) {
          tmp.add(node_in_cell[node_pattern[i][j]]);
        }
        std::sort(tmp.begin(),tmp.end());
        Int64 uid = MeshUtils::generateHashUniqueId(tmp.constView());
        node_in_cell[index_27+i] = uid;


        if( new_nodes.find(node_in_cell[i+index_27]) == new_nodes.end() ) {
          // Coords
          Arcane::Real3 middle_coord(0.0,0.0,0.0);
          for( Integer j = 0 ; j < node_pattern[i].size() ; j++  ) {
            info() << "loop" << cell.node(static_cast<Integer>(node_pattern[i][j])) << " " << nodes_coords[cell.node(static_cast<Integer>(node_pattern[i][j]))];
            middle_coord += nodes_coords[cell.node(static_cast<Integer>(node_pattern[i][j]))];
          }
          ARCANE_ASSERT((node_pattern[i].size() != 0),("Wrong size"));
          middle_coord /= node_pattern[i].size();
          new_nodes.insert(node_in_cell[i+index_27]);
          // Insertion dans map uid -> coord
          nodes_to_add_coords[node_in_cell[i+index_27]] = middle_coord;
          // Insertion dans Uarray uid
          nodes_to_add.add(uid);
          info() << node_pattern[i] ;
          ARCANE_ASSERT((middle_coord != Real3(0.0,0.0,0.0)),("Wrong size"));
          debug() << i << " " << uid << " sizenic " << tmp << middle_coord ;
        }
      }
      // Est-ce qu'on doit ajouter les anciens noeuds ? Normalement non
      // #TAG
      /*for(Integer i = 0 ; i < cell.nbNode() ; i++ ) {
        nodes_to_add.add(cell.node(i).uniqueId().asInt64());
        new_nodes.insert(cell.node(i).uniqueId().asInt64());
      }*/

      debug() << "nodetoadd size " << nodes_to_add.size() << " " << nodes_to_add_coords.size();
      debug() << "Node coord & uid TRUC" << nodes_to_add_coords.size() << " " << nodes_to_add.size() ;
      //ARCANE_ASSERT((nodes_to_add_coords.size() == static_cast<size_t>(nodes_to_add.size())),("Has to be same"));
      //ARCANE_ASSERT((nodes_to_add_coords.size() == new_nodes.size()),("Has to be same"));

      // Génération des Faces
      StorageRefine & face_refine = p.faces;
      debug() << "face_refine.size() " << face_refine.size() ;
      //ARCANE_ASSERT((face_refine.size() == 36), ("WRONG NUMBER OF CELL ADDED")); // One cube assert
      debug() << "Refine face" ;
      for(Integer i = 0 ; i < face_refine.size() ; i++ ) {
        // Generation du hash de la face
        UniqueArray<Int64> tmp;
        //tmp.resize(face_refine[i].size());
        for(Integer j = 0 ; j < face_refine[i].size() ; j++ ) {
          tmp.add(node_in_cell[face_refine[i][j]]);
        }
        std::sort(tmp.begin(),tmp.end());
        //ARCANE_ASSERT(( tmp.size() == 4 ),("Wrong size of UniqueArray")); // one cube assert
        Int64 uid = Arcane::MeshUtils::generateHashUniqueId(tmp.constView());
        face_in_cell.add(uid); // Pour groups
        // Vérifier si on l'a déja crée
        if( new_faces.find(uid) == new_faces.end()){
          // Ajout
          faces_to_add.add(p.face_type);
          faces_to_add.add(uid);
          //debug() << "Face " << uid << " " << tmp ;
          for(Integer j = 0 ; j < face_refine[i].size() ; j++ ) {
            //debug() << node_in_cell[face_refine[i][j]] ;
            faces_to_add.add(node_in_cell[face_refine[i][j]]);
          }
          // Ajouter dans tableau uids faces
          faces_uids.add(uid);
          nb_face_to_add++;
          new_faces.insert(uid);
        }
      }

      // Nouvelle Gestion des groupes
      // Pour chaque faces
          // Générer hash
          // associer hash uid face
      // Parcours des faces parentes
      debug() << "Gestion face groupe" ;
      for( Integer i = 0 ; i < child_faces.size() ; i++ ) {
        parents_to_childs_faces_groups[cell.face(i).uniqueId()] = std::pair<Int64,Int64>(face_external_uid.size(),child_faces[i].size());
        for(Integer j = 0 ; j < child_faces[i].size() ; j++ ){
           face_external_uid.add(face_in_cell[child_faces[i][j]]); // start c'est l'index des face_in_cell // en fait on fait un array pour chaque élément aussi // face_in_cell (sinon les index seront faux donc on ajoute plusieurs fois les faces)
        }
      }

      // Génération des cells
      StorageRefine & cells_refine = p.cells;
      // Génération des cellules enfants
      debug() << "Génération des cellules enfants" ;
      // L'uid est généré à partir du hash de chaque noeuds triés par ordre croissant
      for( Integer i = 0 ; i < cells_refine.size() ; i++ ){
        // Le nouvel uid est généré avec le hash des nouveaux noeuds qui composent la nouvelle cellule
        UniqueArray<Int64> tmp;
        for( Integer j = 0 ; j < cells_refine[i].size() ; j++){
          tmp.add(node_in_cell[cells_refine[i][j]]);
        }
        std::sort(tmp.begin(),tmp.end());
        Int64 cell_uid = Arcane::MeshUtils::generateHashUniqueId(tmp.constView());//max_cell_uid+ind_new_cell;

        cells_to_add.add(p.cell_type); // Type
        cells_to_add.add(cell_uid);    // TODO CHANGER par max_uid + cell_uid * max_nb_node
        for(Integer j = 0 ; j < cells_refine[i].size() ; j++) {
          cells_to_add.add(node_in_cell[cells_refine[i][j]]);
        }
        child_cell_owner[cell_uid] = cell.owner();
        parent_cells.add(cell.uniqueId());
        child_cells.add(cell_uid); // groups doublons d'informations avec cells_to_add
        nb_cell_to_add++;
        ind_new_cell++;
      }
      // groups
      parents_to_childs_cell[cell.uniqueId()] = std::pair<Int64,Int64>(childs_count,cells_refine.size());
      childs_count += cells_refine.size(); // à modifier selon le nombre d'enfant associé au motif de rafinement !
    }
    // Ajout des nouveaux Noeuds
    Integer nb_node_added = nodes_to_add.size();
    UniqueArray<Int32> nodes_lid(nb_node_added);

    info() << "JustBeforeAdd " << nodes_to_add ;
    mesh->modifier()->addNodes(nodes_to_add, nodes_lid.view());

    // Edges: Pas de génération d'arrête

    debug() << "Faces_uids " <<faces_uids << " faces_to_add " << faces_to_add.size() << " faces_to_add/6 " << faces_to_add.size()/6 ;

    ARCANE_ASSERT((nodes_to_add.size() != 0),("End"));
    //ARCANE_ASSERT((nb_face_to_add == 68),("WRONG NUMBER OF FACES")); // two hex
    // Ajout des Faces enfants
    UniqueArray<Int32> face_lid(faces_uids.size());

    mesh->modifier()->addFaces(MeshModifierAddFacesArgs(nb_face_to_add, faces_to_add.constView(),face_lid.view()));
    debug() << "addOneFac" << nb_face_to_add ;
    mesh->faceFamily()->itemsUniqueIdToLocalId(face_lid,faces_uids,true);
    debug() << "NB_FACE_ADDED AFTER " << face_lid.size() << " " << new_faces.size()  ;

    //ARCANE_ASSERT((nb_face_to_add == (faces_to_add.size()/6)),("non consistant number of faces")); // Pour hex

    // Ajout des cellules enfants
    UniqueArray<Int32> cells_lid(nb_cell_to_add);
    mesh->modifier()->addCells(nb_cell_to_add, cells_to_add.constView(),cells_lid);

    // Pour tout les itemgroups
    UniqueArray<Int32> child_cells_lid(child_cells.size());
    mesh->cellFamily()->itemsUniqueIdToLocalId(child_cells_lid,child_cells,true);


    // Gestion des itemgroups ici (différents matériaux par exemple)
    // - On cherche à ajouter les enfants dans les mêmes groupes que les parents pour :
    //   - Faces
    //   - Cells
    // - Pas de déduction automatique pour :
    //   - Noeuds
    //   - Arêtes
    // Algo
    // Pour chaque group
    //   Pour chaque cellules de ce group
    //     ajouter cellules filles de ce group


    // Traiter les groupes pour les faces
    //
    // En fait on ne peut traiter que les faces externes. Est-ce qu'on doit/peut déduire les groupes des faces internes ?
    // Dans le cas du test microhydro on peut car on a que les faces externes aux éléments: XYZ min max
    // A ce moment nous n'avons pas fait de lien face_parent_externe -> face_enfant_externe
    // Pour le faire nous allons parcourir les faces internes parentes, trier les ids  et trier les éléménts

    // Problème des groupes.
    // Tableau faces_externals
    // Pour chaque face parent
    //   - Ajouter dans une nouvelle variable faces_externals l'uid de chaque nouvelle face
    //   - dans map sauvegarder uid, et index

    IItemFamily* face_family = mesh->faceFamily();
    IItemFamily* cell_family = mesh->cellFamily();

    UniqueArray<Int32> face_external_lid(face_external_uid.size());
    mesh->faceFamily()->itemsUniqueIdToLocalId(face_external_lid,face_external_uid);
    // Traiter les groups pour les faces
    info() << "#mygroupname face " << face_family->groups().count();
    for( ItemGroupCollection::Enumerator igroup(face_family->groups()); ++igroup; ){
      ItemGroup group = *igroup;
      info() << "#mygroupname face " << group.fullName();
      if (group.isOwn() && mesh->parallelMng()->isParallel() ){
        info() << "#groups: OWN";
        continue;
      }
      if (group.isAllItems()  ){ // besoin de ça pour seq et //
        info() << "#groups: ALLITEMS";
        continue;
      }
      info() << "#groups: Added ";
      UniqueArray<Int32> to_add_to_group;

      ENUMERATE_(Item,iitem,group){ // Pour chaque cellule du groupe on ajoute ses 8 enfants ( ou n )
        Int64 step = parents_to_childs_faces_groups[iitem->uniqueId().asInt64()].first;
        Int64 n_childs = parents_to_childs_faces_groups[iitem->uniqueId().asInt64()].second;
        auto subview = face_external_lid.subView(step,static_cast<Integer>(n_childs));
        ARCANE_ASSERT((subview.size() == 4 ), ("SUBVIEW"));
        to_add_to_group.addRange(subview);
      }
      group.addItems(to_add_to_group,true);
    }

    // Traiter les groupes pour les cellules
    for( ItemGroupCollection::Enumerator igroup(cell_family->groups()); ++igroup; ){
      CellGroup group = *igroup;
      info() << "#mygroupname" << group.fullName();
      info() << "#mygroupname " << cell_family->nbItem();

      if (group.isOwn() && mesh->parallelMng()->isParallel() ){
        info() << "#groups: OWN";
        continue;
      }
      if (group.isAllItems() ){ // besoin de ça pour seq et //
        info() << "#groups: ALLITEMS";
        continue;
      }

      info() << "#groups: Added ";
      UniqueArray<Int32> to_add_to_group;

      ENUMERATE_(Item,iitem,group){ // Pour chaque cellule du groupe on ajoute ses 8 enfants ( ou n )
        ARCANE_ASSERT(( static_cast<Integer>(parents_to_childs_cell.size()) == child_cells_lid.size()/8 ),("Wrong number of childs"));
        Int64 step = parents_to_childs_cell[iitem->uniqueId().asInt64()].first;
        Int64 n_childs = parents_to_childs_cell[iitem->uniqueId().asInt64()].second;
        auto subview = child_cells_lid.subView(step,static_cast<Integer>(n_childs));


        ARCANE_ASSERT((subview.size() == 8 ), ("SUBVIEW"));
        to_add_to_group.addRange(subview);
      }
      info() << "#Added " << to_add_to_group.size() << " to group " << group.fullName();
      group.addItems(to_add_to_group,true);
    }

    // fin gestion itemgroups
    mesh->modifier()->removeCells(cells_to_detach.constView());
    mesh->modifier()->endUpdate();

    // DEBUG
    debug() << "Debug faces " << faces_to_add;
    /*for(Integer i = 0 ; i < faces_to_add.size() ; i++){
        debug() << new_faces[i] ;
    }*/
    // ENDEBUG


    // Gestion et assignation du propriétaire pour chaque cellule
    // Le propriétaire est simplement le sous domaine qui a générer les nouvelles cellules
    ENUMERATE_ (Cell, icell, mesh->allCells()){
        Cell cell = *icell;
        cell.mutableItemBase().setOwner(my_rank, my_rank);
    }
    mesh->cellFamily()->notifyItemsOwnerChanged();

    ARCANE_ASSERT((nodes_lid.size() != 0),("End"));
    ARCANE_ASSERT((nodes_lid.size() == nodes_to_add.size()),("End"));
    // Assignation des coords aux noeuds
    ENUMERATE_(Node, inode, mesh->nodeFamily()->view(nodes_lid)){
      Node node = *inode;
      debug() << node.uniqueId().asInt64() ;
      //ARCANE_ASSERT((new_nodes.find(node.uniqueId().asInt64()) != new_nodes.end()),("Not found in set !"))
      //ARCANE_ASSERT((nodes_to_add_coords.find(node.uniqueId().asInt64()) != nodes_to_add_coords.end()),("Not found in map coord!"))
      // si il est pas dans la map c'est qu'il existe déja !

      nodes_coords[node] = nodes_to_add_coords[node.uniqueId().asInt64()];
      info() << "InSBD" << node.uniqueId().asInt64() << " " << nodes_to_add_coords[node.uniqueId().asInt64()] ;
    }

    //info() << "#NODECOORDS" << nodes_coords.asArray() ;
    // Ajout d'une couche fantôme
    Arcane::IGhostLayerMng * gm2 = mesh->ghostLayerMng();
    gm2->setNbGhostLayer(1);
    mesh->updateGhostLayers(true);

    // Gestion des propriétaires de noeuds
    // Le propriétaire est la cellule incidente au noeud avec le plus petit uniqueID()
    ENUMERATE_(Node, inode, mesh->allNodes()){
      Node node = *inode;
      auto it = std::min_element(node.cells().begin(),node.cells().end());
      Cell cell = node.cell(static_cast<Int32>(std::distance(node.cells().begin(),it)));
      node_uid_to_owner[node.uniqueId().asInt64()] = cell.owner();
    }

    // Gestion des propriétaires des faces
    // Le propriétaire est la cellule incidente à la face avec le plus petit uniqueID()
    ENUMERATE_(Face, iface, mesh->allFaces()){
      Face face = *iface;
      auto it = std::min_element(face.cells().begin(),face.cells().end());
      Cell cell = face.cell(static_cast<Int32>(std::distance(face.cells().begin(),it)));
      face_uid_to_owner[face.uniqueId().asInt64()] = cell.owner();
    }

    // Utiliser les couches fantôme est couteux (construction destruction)
    // - Optim: pour les noeuds partager avoir une variable all to all (gather) qui permet de récuper le rank de l'owner
    // - Déduction possible des owners des faces enfants avec la face parent directement
    // - Les cellules enfants ont le même owner que la cellule parent
    // Supression de la couche fantôme
    gm2->setNbGhostLayer(0);
    mesh->updateGhostLayers(true);


    // DEBUG
    debug() << "#Faces mesh" ;
    ENUMERATE_(Face, iface, mesh->allFaces()){
      Face face = *iface;
      debug() << face.uniqueId().asInt64() ;
    }


    // Quelques sur le nombres d'entités insérés
    ARCANE_ASSERT((mesh->nbCell() == nb_cell_init*8 ),("Wrong number of cell added"));
    debug() <<"nbface " << mesh->nbFace() << " " << nb_face_to_add << " Attendu " << nb_face_init*4 + 12 * nb_cell_init;
    ARCANE_ASSERT((mesh->nbFace() <= nb_face_init*4 + 12 * nb_cell_init ),("Wrong number of face added"));
    // A ajouter pour vérifier le nombre de noeud si les arêtes sont crées
    // ARCANE_ASSERT((mesh->nbNode() == nb_edge_init + nb_face_init + nb_cell_init ),("Wrong number of node added"))

    // Assignation du nouveau propriétaire pour chaque noeud
    ENUMERATE_ (Node, inode, mesh->allNodes()){
      Node node = *inode;
      node.mutableItemBase().setOwner(node_uid_to_owner[node.uniqueId().asInt64()], my_rank);
    }
    mesh->nodeFamily()->notifyItemsOwnerChanged();

    // Assignation du nouveaux propriétaires pour chaque face
    ENUMERATE_ (Face, iface, mesh->allFaces()){
      Face face = *iface;
      face.mutableItemBase().setOwner(face_uid_to_owner[face.uniqueId().asInt64()], my_rank);
    }
    mesh->faceFamily()->notifyItemsOwnerChanged();

    // On met de nouveau le ghost layer pour une future simulation
    gm2->setNbGhostLayer(nb_ghost_layer_init);
    mesh->updateGhostLayers(true);

    // Ecriture au format VTK
    mesh->utilities()->writeToFile("3Drefined"+std::to_string(my_rank)+".vtk", "VtkLegacyMeshWriter");
    info() << "Writing VTK 3Drefine" ;
    debug() << "END 3D fun" ;
    debug() << "NB CELL " << mesh->nbCell() << " " << nb_cell_init*8 ;
    debug() << mesh->nbNode() << " " << nb_node_init << " " << nb_edge_init << " " << nb_face_init << " " << nb_cell_init;
    debug() << mesh->nbFace() << "nb_face_init " << nb_face_init <<  " " <<  nb_face_init << " " << nb_cell_init ;
    debug() << "Faces: " << mesh->nbFace() << " theorical nb_face_to add: " << nb_face_init*4 + nb_cell_init*12 <<  " nb_face_init " <<  nb_face_init << " nb_cell_init " << nb_cell_init ;
    info() << "#NODES_CHECK #all" << mesh->allNodes().size() << " #own " << mesh->ownNodes().size() ;

}

// TODO test
void ArcaneBasicMeshSubdividerService::_refineOneTetra(IPrimaryMesh* mesh){
  IMeshModifier* modifier = mesh->modifier();
  Int64UniqueArray cells_infos;
  Int64UniqueArray faces_infos;
  Int64UniqueArray nodes_uid;
  std::unordered_map<Int64, Real3> nodes_to_add_coords;
  VariableNodeReal3& nodes_coords = mesh->nodesCoordinates();


  info() << "#_refineOneTetra" ;
  ENUMERATE_CELL(icell,mesh->allCells()) {
    const Cell & cell = *icell;
    info() << "cell[" << icell->localId() << "," << icell->uniqueId() << "] type="
	   << icell->type() << ", nb nodes=" << icell->nbNode();

    for(Face face :cell.faces() ){
      info() << "Face " << face.uniqueId() << " Noeuds " ;
      for( Node node : face.nodes() ){
        info() << node.uniqueId() << " " ;
      }
    }
  }
  mesh->utilities()->writeToFile("subdivider_one_tetra_output.vtk", "VtkLegacyMeshWriter");

  Integer cellcount=0;
  UniqueArray<Int32> cells_to_detach; // Cellules à détacher
  UniqueArray<Int64>  node_in_cell;
  Int64 face_count = 0;
  ENUMERATE_CELL(icell,mesh->allCells())
  {

    const Cell & cell = *icell;
    info() << "Detach"  ;
    cells_to_detach.add(cell.localId());
    info() << "Get Pattern" ;

    Pattern p = PatternBuilder::tettotet();
    StorageRefine & node_pattern = p.nodes;
    StorageRefine & cells = p.cells;
    StorageRefine & faces = p.faces;

    info() << "Get Nodes" ;
    for( Integer i = 0; i < cell.nbNode(); i++ ) {
      node_in_cell.add(cell.node(i).uniqueId().asInt64());
    }

    _computeNodeUid(node_in_cell,node_pattern);
    // New nodes and coords
    for( Integer i = 0 ; i < node_pattern.size() ; i++ ){
      info() << "test " << i ;
      UniqueArray<Int64> tmp = node_pattern[i];

      // uid
      std::sort(tmp.begin(),tmp.end());
      node_in_cell.add(4+i); //= 4+i;// = 4+i; //Arcane::MeshUtils::generateHashUniqueId(tmp.constView());
      nodes_uid.add(node_in_cell[4+i]);
      // Coords
      Arcane::Real3 middle_coord(0.0,0.0,0.0);
      for( Integer j = 0 ; j < node_pattern[i].size() ; j++  ) {
        middle_coord += nodes_coords[cell.node(node_pattern[i][j])];
      }
      middle_coord /= node_pattern[i].size();
      nodes_to_add_coords[node_in_cell[4+i]] = middle_coord;
      info() << "NodeX " << 4+i << " " << node_in_cell[4+i]  ;
    }

    info() << "test2 ";
    // Génération nouvelles faces et cells
    // New faces
    for( Integer i = 0 ; i < faces.size() ; i++ ){
      // Header
      faces_infos.add(IT_Triangle3);            // type  // Dépendant pattern
      faces_infos.add(i);                    // face uid
      for( Integer j = 0 ; j < faces[i].size() ; j++ ) {
        faces_infos.add(node_in_cell[faces[i][j]]);  // node 0
      }
      // Face_info
      info() << "face " << face_count << " " << node_in_cell[faces[i][0]] << " " << node_in_cell[faces[i][1]] << " " << node_in_cell[faces[i][2]];
      face_count++;
    }
    // New cells
    for( Integer i = 0 ; i < cells.size() ; i++ ){
      // Header
      cells_infos.add(IT_Tetraedron4);          // type  // Dépendant pattern
      cells_infos.add(i);                    // cell uid
      // Cell_info
      for(Integer j = 0 ; j < cells[i].size() ; j++) {
        cells_infos.add(node_in_cell[cells[i][j]]);
      }
      cellcount++;
    }
    for(Integer i = 0 ; i < node_in_cell.size() ; i++ ) {
      info() << "node_in_cell[i] " << node_in_cell[i] ;
    }
    info() << "test3 ";

  }
  UniqueArray<Int32> nodes_lid(nodes_uid.size());
  // Debug ici
  info() << "test3 " << nodes_uid.size() << " " << nodes_lid.size() ;
  nodes_lid.clear();
  nodes_lid.reserve(nodes_uid.size());

  modifier->addNodes(nodes_uid,nodes_lid);
  info() << "After nodes" ;
  UniqueArray<Int32> faces_lid(face_count);
  modifier->addFaces(face_count, faces_infos, faces_lid);
  info() << "After faces" ;
  UniqueArray<Int32> cells_lid(cellcount);
  modifier->addCells(cellcount, cells_infos, cells_lid);
  modifier->removeCells(cells_to_detach.constView());
  info() << "cellsize " << cells_infos.size() << " " << cellcount ;

  modifier->endUpdate();
  // Assignation coords aux nouveaux noeuds
  VariableNodeReal3 coords_bis = mesh->nodesCoordinates();

  info() << nodes_lid.size();
  ENUMERATE_(Node, inode, mesh->nodeFamily()->view().subView(4,nodes_uid.size())){
    Node node = *inode;
    coords_bis[node] = nodes_to_add_coords[node.uniqueId()];
    info() << node.uniqueId() << " " << nodes_coords[node] ;
  }

  info() << "#My mesh ";
  // Affichage maillage
  ENUMERATE_CELL(icell,mesh->allCells()) {
    const Cell & cell = *icell;
    info() << "Cell " << cell.uniqueId() << " " << cell.nodeIds() ;

    for( Face face : cell.faces()){
      UniqueArray<Int64> stuff;
      for(Node node : face.nodes() ) {
        stuff.add(node.uniqueId());
      }
      info() << "Faces " << face.uniqueId() << " node " << stuff ;
    }
  }
  info() << "#ENUM Faces" ;
  ENUMERATE_FACE(iface,mesh->allFaces()) {
    const Face & face = *iface;
    UniqueArray<Int64> stuff;
    for(Node node : face.nodes() ) {
      stuff.add(node.uniqueId());
    }
    info() << "Faces " << face.uniqueId() << " node " << stuff ;
  }

  Arcane::VariableScalarInteger m_temperature(Arcane::VariableBuildInfo(mesh, "ArcaneCheckpointNextIteration" ));

  VariableCellInt64* arcane_cell_uid = nullptr;
  VariableFaceInt64* arcane_face_uid = nullptr;
  VariableNodeInt64* arcane_node_uid = nullptr;
  arcane_cell_uid = new VariableCellInt64(Arcane::VariableBuildInfo(mesh, "arcane_cell_uid", mesh->cellFamily()->name()));
  arcane_face_uid = new VariableFaceInt64(Arcane::VariableBuildInfo(mesh, "arcane_face_uid", mesh->faceFamily()->name()));
  arcane_node_uid = new VariableNodeInt64(Arcane::VariableBuildInfo(mesh, "arcane_node_uid", mesh->nodeFamily()->name()));

  ENUMERATE_CELL(icell,mesh->allCells()){
      (*arcane_cell_uid)[icell] = icell->uniqueId().asInt64();

  }
  ENUMERATE_FACE(iface,mesh->allFaces()){
    (*arcane_face_uid)[iface] = iface->uniqueId().asInt64();
  }
  info() << "#INODE" ;
  ENUMERATE_NODE(inode,mesh->allNodes()){
    (*arcane_node_uid)[inode] = inode->uniqueId().asInt64();
    info() << inode->uniqueId().asInt64() ;
  }
  ENUMERATE_(Node, inode, mesh->nodeFamily()->view().subView(4,nodes_uid.size())){
    Node node = *inode;
    nodes_coords[node] = nodes_to_add_coords[node.uniqueId()];
    info() << node.uniqueId() << " " << nodes_coords[node] ;
  }
  //
  // On va chercher le service directement sans utiliser dans le .arc
  Directory d = mesh->subDomain()->exportDirectory();
  info() << "Writing at " << d.path() ;
  ServiceBuilder<IPostProcessorWriter> spp(mesh->handle());

  Ref<IPostProcessorWriter> post_processor = spp.createReference("Ensight7PostProcessor"); // vtkHdf5PostProcessor
  //Ref<IPostProcessorWriter> post_processor = spp.createReference("VtkLegacyMeshWriter"); // (valid values = UCDPostProcessor, UCDPostProcessor, Ensight7PostProcessor, Ensight7PostProcessor)
  // Path de base
  // <fichier-binaire>false</fichier-binaire>
  post_processor->setBaseDirectoryName(d.path());

  post_processor->setTimes(UniqueArray<Real>{0.1}); // Juste pour fixer le pas de temps

  VariableList variables;
  variables.add(mesh->nodesCoordinates().variable());
  variables.add(*arcane_node_uid);
  variables.add(*arcane_face_uid);
  variables.add(*arcane_cell_uid);
  post_processor->setVariables(variables);

  ItemGroupList groups;
  groups.add(mesh->allNodes());
  groups.add(mesh->allFaces());
  groups.add(mesh->allCells());
  post_processor->setGroups(groups);

  IVariableMng* vm = mesh->subDomain()->variableMng();

  vm->writePostProcessing(post_processor.get());
  mesh->utilities()->writeToFile("subdivider_one_tetra_refine_output.vtk", "VtkLegacyMeshWriter");
  info() << "#ENDSUBDV " ;
}


// TODO test
void ArcaneBasicMeshSubdividerService::_generateOneTri(IPrimaryMesh * mesh)
{
  mesh->utilities()->writeToFile("subdivider_one_hexa_input.vtk", "VtkLegacyMeshWriter");
  // On supprime l'ancien maillage
  Int32UniqueArray lids(mesh->allCells().size());
  ENUMERATE_CELL(icell,mesh->allCells()) {
    info() << "cell[" << icell->localId() << "," << icell->uniqueId() << "] type="
	   << icell->type() << ", nb nodes=" << icell->nbNode();
    lids[icell.index()] = icell->localId();
  }
  IMeshModifier* modifier = mesh->modifier();
  modifier->removeCells(lids);
  modifier->endUpdate();
  // On creer notre Hexa
  Int64UniqueArray nodes_uid(3);
  for(Integer i = 0; i < 3; i++)
    nodes_uid[i] = i;

  UniqueArray<Int32> nodes_lid(nodes_uid.size());
  modifier->addNodes(nodes_uid,nodes_lid.view());
  mesh->nodeFamily()->endUpdate();
  VariableNodeReal3& nodes_coords = mesh->nodesCoordinates();
  NodeInfoListView new_nodes(mesh->nodeFamily());
  nodes_coords[new_nodes[nodes_lid[0]]] = Arcane::Real3(0.0,0.0,0.0);
  nodes_coords[new_nodes[nodes_lid[1]]] = Arcane::Real3(10.0,0.0,0.0);
  nodes_coords[new_nodes[nodes_lid[2]]] = Arcane::Real3(10.0,10.0,0.0);

  Int64UniqueArray cells_infos(10);
  Int64UniqueArray faces_infos;
  cells_infos[0] = IT_Triangle3; // type
  cells_infos[1] = 44;            // cell uid
  cells_infos[2] = nodes_uid[0];  // node 0
  cells_infos[3] = nodes_uid[1];  // ...  1
  cells_infos[4] = nodes_uid[2];  // ...  2

  IntegerUniqueArray cells_lid;
  modifier->addCells(1, cells_infos, cells_lid);
  mesh->utilities()->writeToFile("subdivider_one_tri.vtk", "VtkLegacyMeshWriter");
}

UniqueArray<Int64> ArcaneBasicMeshSubdividerService::_computeNodeUid(UniqueArray<Int64> node_in_cell , const StorageRefine & node_pattern )
{
  UniqueArray<Int64> new_node_uid;
  Integer init_size = node_in_cell.size();
  for( Integer i = 0 ; i < node_pattern.size() ; i++ ){
    //info() << "test " << i ;
    UniqueArray<Int64> tmp = node_pattern[i];
    tmp.resize(node_pattern[i].size());
    for( Integer j = 0 ; j < node_pattern[i].size() ; j++  ) {
      tmp.add(node_in_cell[node_pattern[i][j]]);
    }
    // uid
    std::sort(tmp.begin(),tmp.end());
    node_in_cell.add(init_size+i); //= 4+i;// = 4+i; //Arcane::MeshUtils::generateHashUniqueId(tmp.constView());
    new_node_uid.add(node_in_cell[init_size+i]);
  }
  return new_node_uid;
}



/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


void ArcaneBasicMeshSubdividerService::_uniqueArrayTest(){
  UniqueArray<Int64> a = {1,8,4};
  UniqueArray<Int64> b = a;
  std::sort(a.begin(),a.end());
  Arcane::MeshUtils::generateHashUniqueId(a.constView());
  ARCANE_ASSERT((Arcane::MeshUtils::generateHashUniqueId(a.constView()) != Arcane::MeshUtils::generateHashUniqueId(b.constView())),("a==b")); // On modifie a
}

ArcaneBasicMeshSubdividerService::
ArcaneBasicMeshSubdividerService(const ServiceBuildInfo& sbi)
: ArcaneArcaneBasicMeshSubdividerServiceObject(sbi)
{
}

void ArcaneBasicMeshSubdividerService::_generateOneHexa(IPrimaryMesh* mesh){
  mesh->utilities()->writeToFile("subdivider_one_hexa_input.vtk", "VtkLegacyMeshWriter");
  // On supprime l'ancien maillage
  Int32UniqueArray lids(mesh->allCells().size());
  ENUMERATE_CELL(icell,mesh->allCells()) {
    info() << "cell[" << icell->localId() << "," << icell->uniqueId() << "] type=" 
	   << icell->type() << ", nb nodes=" << icell->nbNode();
    lids[icell.index()] = icell->localId();
  }
  IMeshModifier* modifier = mesh->modifier();
  modifier->removeCells(lids);
  modifier->endUpdate();
  // On creer notre Hexa
  Int64UniqueArray nodes_uid(8);
  for(Integer i = 0; i < 8; i++)
    nodes_uid[i] = i;
  
  UniqueArray<Int32> nodes_lid(nodes_uid.size());
  modifier->addNodes(nodes_uid,nodes_lid.view());
  mesh->nodeFamily()->endUpdate();
  VariableNodeReal3& nodes_coords = mesh->nodesCoordinates();
  NodeInfoListView new_nodes(mesh->nodeFamily());
  nodes_coords[new_nodes[nodes_lid[0]]] = Arcane::Real3(0.0,0.0,0.0);
  nodes_coords[new_nodes[nodes_lid[1]]] = Arcane::Real3(10.0,0.0,0.0);
  nodes_coords[new_nodes[nodes_lid[2]]] = Arcane::Real3(10.0,10.0,0.0);
  nodes_coords[new_nodes[nodes_lid[3]]] = Arcane::Real3(0.0,10.0,0.0);
  nodes_coords[new_nodes[nodes_lid[4]]] = Arcane::Real3(0.0,0.0,10.0);
  nodes_coords[new_nodes[nodes_lid[5]]] = Arcane::Real3(10.0,0.0,10.0);
  nodes_coords[new_nodes[nodes_lid[6]]] = Arcane::Real3(10.0,10.0,10.0);
  nodes_coords[new_nodes[nodes_lid[7]]] = Arcane::Real3(0.0,10.0,10.0);

  Int64UniqueArray cells_infos(10);
  Int64UniqueArray faces_infos;
  cells_infos[0] = IT_Hexaedron8; // type
  cells_infos[1] = 44;            // cell uid
  cells_infos[2] = nodes_uid[0];  // node 0
  cells_infos[3] = nodes_uid[1];  // ...  1
  cells_infos[4] = nodes_uid[2];  // ...  2
  cells_infos[5] = nodes_uid[3];  // ...  3
  cells_infos[6] = nodes_uid[4];  // ...  4
  cells_infos[7] = nodes_uid[5];  // ...  5
  cells_infos[8] = nodes_uid[6];  // ...  6
  cells_infos[9] = nodes_uid[7];  // ...  7

  IntegerUniqueArray cells_lid;
  modifier->addCells(1, cells_infos, cells_lid);
  mesh->utilities()->writeToFile("subdivider_one_hexa_ouput.vtk", "VtkLegacyMeshWriter");
}

void ArcaneBasicMeshSubdividerService::_generateOneTetra(IPrimaryMesh* mesh){

  mesh->utilities()->writeToFile("subdivider_one_tetra_input.vtk", "VtkLegacyMeshWriter");
  Int32UniqueArray lids(mesh->allCells().size());
  VariableNodeReal3& nodes_coords = mesh->nodesCoordinates();
  ENUMERATE_CELL(icell,mesh->allCells()) {
    info() << "cell[" << icell->localId() << "," << icell->uniqueId() << "] type="
     << icell->type() << ", nb nodes=" << icell->nbNode();
    lids[icell.index()] = icell->localId();
  }

  IMeshModifier* modifier = mesh->modifier();
  modifier->removeCells(lids);
  modifier->endUpdate();

  // Maillage vide, on créer notre tetra

  info() << "===================== THE MESH IS EMPTY";

  // On ajoute des noeuds
  Int64UniqueArray nodes_uid(4);
  for(Integer i = 0; i < 4; i++)
    nodes_uid[i] = i;

  UniqueArray<Int32> nodes_lid(nodes_uid.size());
  modifier->addNodes(nodes_uid,nodes_lid.view());
  mesh->nodeFamily()->endUpdate();
  info() << "===================== THE MESH IS EMPTY";

  NodeInfoListView new_nodes(mesh->nodeFamily());

  nodes_coords[new_nodes[nodes_lid[0]]] = Arcane::Real3(0.0,0.0,0.0);
  nodes_coords[new_nodes[nodes_lid[1]]] = Arcane::Real3(10.0,0.0,0.0);
  nodes_coords[new_nodes[nodes_lid[2]]] = Arcane::Real3(5.0,5.0/3.0,10.0);
  nodes_coords[new_nodes[nodes_lid[3]]] = Arcane::Real3(5.0,5.0,0.0);

  Int64UniqueArray cells_infos(1*6);
  Int64UniqueArray faces_infos;


  cells_infos[0] = IT_Tetraedron4;// type
  cells_infos[1] = 44;            // cell uid
  cells_infos[2] = nodes_uid[0];  // node 0
  cells_infos[3] = nodes_uid[1];  // ...  1
  cells_infos[4] = nodes_uid[2];  // ...  2
  cells_infos[5] = nodes_uid[3];  // ...  3

  IntegerUniqueArray cells_lid;
  modifier->addCells(1, cells_infos, cells_lid);

  info() << "===================== THE CELLS ARE ADDED";
  modifier->endUpdate();
}

/* Le but est simplement d'avoir l'ordre des faces dans un maillage tetra */
void ArcaneBasicMeshSubdividerService::_testTetra(IPrimaryMesh* mesh){
  mesh->utilities()->writeToFile("3D_last_input_seq.vtk", "VtkLegacyMeshWriter");
  ENUMERATE_CELL(icell,mesh->ownCells()){
    
    const Cell & cell = *icell;
    for( Face face : cell.faces()){
      info() << face.uniqueId();
    }
    exit(0);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


void ArcaneBasicMeshSubdividerService::
subdivideMesh([[maybe_unused]] IPrimaryMesh* mesh)
{
  info() << "subdivideMesh" ;
  //_generateOneTetra(mesh);
  for(Integer i = 0 ; i < options()->nbSubdivision ; i++) {
    _refineOnce(mesh);
    VariableNodeReal3 vrc = mesh->nodesCoordinates();
    ENUMERATE_NODE(inode,mesh->ownNodes())
    {
      const Node & node = *inode;
      info() << node.uniqueId() << " " << vrc[node];
    }
    debug() << i << "refine done";
  }
  /*
  VariableNodeReal3 vrc = mesh->nodesCoordinates();
  ENUMERATE_NODE(inode,mesh->ownNodes())
  {
    const Node & node = *inode;
    info() << node.uniqueId() << " " << vrc[node];
  }
  */
  //_refineOnce(mesh);

  /*
  info() << "subdivideMesh2";
  ARCANE_ASSERT((mesh->nbNode() == 27),("Wrong nb node"));
  info() << "after subdivide" ;
  VariableNodeReal3 vrc = mesh->nodesCoordinates();
  ENUMERATE_NODE(inode,mesh->ownNodes())
  {
    const Node & node = *inode;
    info() << node.uniqueId() << " " << vrc[node];
  }
  info() << "#beforesd";
  _refineOnce(mesh);
  info() << "subdivideMesh3";
  ARCANE_ASSERT((mesh->nbNode() == 125),("Wrong nb node"));
  ENUMERATE_NODE(inode,mesh->ownNodes())
  {
    const Node & node = *inode;
    info() << node.uniqueId() << " " << mesh->nodesCoordinates()[node];
  }
  */
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_ARCANEBASICMESHSUBDIVIDERSERVICE(ArcaneBasicMeshSubdivider,
                                                         ArcaneBasicMeshSubdividerService);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
