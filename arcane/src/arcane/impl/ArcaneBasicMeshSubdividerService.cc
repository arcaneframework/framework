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

// get parameter
#include "arcane/utils/ApplicationInfo.h"
#include "arcane/utils/CommandLineArguments.h"

// utils
#include <unordered_set>
#include <algorithm>
#include <iterator>

#include "arcane/core/IMeshUtilities.h"


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

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

  void subdivideMesh([[maybe_unused]] IPrimaryMesh* mesh) override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArcaneBasicMeshSubdividerService::
ArcaneBasicMeshSubdividerService(const ServiceBuildInfo& sbi)
: ArcaneArcaneBasicMeshSubdividerServiceObject(sbi)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneBasicMeshSubdividerService::
subdivideMesh([[maybe_unused]] IPrimaryMesh* mesh)
{
  info() << "#subdivide mesh";
  // Est-ce qu'on est en 3D ?
  // Si non on fait rien;
  bool is_hex=true;
  ENUMERATE_CELL(icell,mesh->ownCells()){
    const Cell & cell = *icell;
    if( cell.itemTypeId() != IT_Hexaedron8)
      is_hex = false;
  }
  if( !is_hex )
    return ;
   
  Int32 my_rank = mesh->parallelMng()->commRank();
  IMeshModifier* mesh_modifier = mesh->modifier();
  IGhostLayerMng* gm = mesh->ghostLayerMng();
  debug() << "PART 3D nb ghostlayer" << gm->nbGhostLayer();
  mesh->utilities()->writeToFile("3D_last_input"+std::to_string(my_rank)+".vtk", "VtkLegacyMeshWriter");
  // PAS DE GHOST LAYER
  
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

  Integer nb_cell_to_add=0;
  Integer nb_face_to_add=0;

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
  nodes_to_add.reserve(nb_node_to_add_total);
  nodes_to_add_coords.reserve(nb_node_to_add_total);

  Arcane::VariableNodeReal3& nodes_coordinates = mesh->nodesCoordinates();
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

  // Traitement pour une cellule
  ENUMERATE_CELL(icell,mesh->ownCells())
  {
      // Génération des nouveaux noeuds (uid et coordonnées)
      // Sur Arêtes
      // Sur Faces
      // Sur Cellule

      // Génération des Faces (uid et composants (Noeuds))
      // Internes
      // Externes

      // Génération des Cellules (uid et composants (Noeuds))
      // Détachement des cellules
      // Ajout des noeuds enfants
      // Ajout des faces enfants
      // Ajout des cellules enfants (et assignation propriétaire)

      // Ajout d'une couche fantome
      // Calcul des propriétaires des noeuds
      // Calcul des propriétaires des faces
      // Supression de la couche fantome

      // Assignation des noeuds au propriétaire
      // Assignation des faces au propriétaire

      const Cell& cell = *icell;
      //debug() << "CELL_OWNER " << cell.owner() ;
      cells_to_detach.add(cell.localId());
      // Génération des noeuds
      Int64  node_in_cell[27];
      // Noeuds initiaux
      node_in_cell[0] = cell.node(0).uniqueId().asInt64();
      node_in_cell[1] = cell.node(1).uniqueId().asInt64();
      node_in_cell[2] = cell.node(2).uniqueId().asInt64();
      node_in_cell[3] = cell.node(3).uniqueId().asInt64();
      node_in_cell[4] = cell.node(4).uniqueId().asInt64();
      node_in_cell[5] = cell.node(5).uniqueId().asInt64();
      node_in_cell[6] = cell.node(6).uniqueId().asInt64();
      node_in_cell[7] = cell.node(7).uniqueId().asInt64();
      Integer index_27 = 8;
      // Génération des nouveaux noeuds sur arêtes
      Integer new_nodes_on_edges_couple[][2] = {{0, 1},{0, 3},{0, 4},{1, 2},{1, 5},{2, 3},{2, 6},{3, 7},{4, 5},{4, 7},{5, 6},{6, 7}};
      // ^--- Tableau d'arretes
      for( Integer i = 0 ; i < 12 ; i++ ){
        // uid
        UniqueArray<Int64> tmp = {
          node_in_cell[new_nodes_on_edges_couple[i][0]],
          node_in_cell[new_nodes_on_edges_couple[i][1]],
        };
        if( tmp[0] > tmp[1]   ){
            std::swap(tmp[0],tmp[1]);
        }
        debug() << "#TMP " << "cell" << cell.uniqueId() << ' ' << tmp ;
        node_in_cell[index_27] = Arcane::MeshUtils::generateHashUniqueId(tmp.constView());
        // Coord on edge
        Arcane::Real3 middle_coord(0.0,0.0,0.0);
        middle_coord = (nodes_coordinates[cell.node(new_nodes_on_edges_couple[i][0])] + nodes_coordinates[cell.node(new_nodes_on_edges_couple[i][1])] ) / 2.0;
        nodes_to_add_coords[node_in_cell[index_27]] = middle_coord;

        index_27++;
      }
      ARCANE_ASSERT((index_27 == 20),("wrong number"));
      // Noeuds sur faces
      Integer new_nodes_on_faces_quatuor[][4] ={
        { 0, 1, 2, 3 },
        { 0, 1, 5, 4 },
        { 0, 4, 7, 3 },
        { 1, 5, 6, 2 },
        { 2, 3, 7, 6 },
        { 4, 5, 6, 7 }
      };

      for( Integer i = 0 ; i < 6 ; i++ ){
        UniqueArray<Int64> tmp = {
          node_in_cell[new_nodes_on_faces_quatuor[i][0]],
          node_in_cell[new_nodes_on_faces_quatuor[i][1]],
          node_in_cell[new_nodes_on_faces_quatuor[i][2]],
          node_in_cell[new_nodes_on_faces_quatuor[i][3]]
        };
        std::sort(tmp.begin(),tmp.end());
        Int64 nuid = Arcane::MeshUtils::generateHashUniqueId(tmp.constView());
        node_in_cell[index_27] = nuid;
        // Coord on face
        Arcane::Real3 middle_coord(0.0,0.0,0.0);
        middle_coord = ( nodes_coordinates[cell.node(new_nodes_on_faces_quatuor[i][0])] + nodes_coordinates[cell.node(new_nodes_on_faces_quatuor[i][1])] + nodes_coordinates[cell.node(new_nodes_on_faces_quatuor[i][2])] + nodes_coordinates[cell.node(new_nodes_on_faces_quatuor[i][3])] ) / 4.0;
        nodes_to_add_coords[node_in_cell[index_27]] = middle_coord;
        index_27++;
      }

      ARCANE_ASSERT((index_27 == 26),("wrong number"));

      Integer new_nodes_on_cell_oct[8] = {0, 1, 5, 4, 3, 2, 7, 6};
      // Noeud sur cell
      UniqueArray<Int64> tmp = {
        node_in_cell[new_nodes_on_cell_oct[0]],
        node_in_cell[new_nodes_on_cell_oct[1]],
        node_in_cell[new_nodes_on_cell_oct[2]],
        node_in_cell[new_nodes_on_cell_oct[3]],
        node_in_cell[new_nodes_on_cell_oct[4]],
        node_in_cell[new_nodes_on_cell_oct[5]],
        node_in_cell[new_nodes_on_cell_oct[6]],
        node_in_cell[new_nodes_on_cell_oct[7]]
      };

      // Le noeud central à son uid généré a partir des uid de la cellule parent
      node_in_cell[index_27] = Arcane::MeshUtils::generateHashUniqueId(tmp.constView());

      // Calcul des coordonnées du noeud central
      Arcane::Real3 middle_coord(0.0,0.0,0.0);
      for( Integer i = 0 ; i < 8 ; i++ ){
        middle_coord += nodes_coordinates[cell.node(new_nodes_on_cell_oct[i])];
      }
      middle_coord /= 8.0;
      nodes_to_add_coords[node_in_cell[index_27]] = middle_coord;

      // Génération des Edges
      // Pas obligatoire
      for( Integer i = 8 ; i < 27 ; i++){
        // Nous calculons plusieurs fois les noeuds pour chaque Cellule (TODO améliorer ça)
        // Si un noeud n'est pas dans la map, on l'ajoute
        if( new_nodes.find(node_in_cell[i]) == new_nodes.end()  ){
          nodes_to_add.add(node_in_cell[i]);
          new_nodes.insert(node_in_cell[i]);
        }
      }
      debug() << nodes_to_add_coords.size() << " " << nodes_to_add.size() ;

      ARCANE_ASSERT((nodes_to_add_coords.size() == static_cast<size_t>(nodes_to_add.size())),("Has to be same"));
      // Génération des Faces
      // - Internes 12
      Int64 internal_faces[][4]=
      {
        {8, 20, 26, 21},  //  6
        {20, 13, 24, 26}, //  7
        {9, 22, 26, 20},  //  8
        {20, 26, 23, 11}, //  9
        {21, 16, 25, 26}, //  10
        {26, 25, 19, 24}, //  11
        {22, 17, 25, 26}, //  12
        {26, 25, 18, 23}, //  13
        {10, 21, 26, 22}, //  22, 26, 21, 10}, //  14
        {21, 12, 23, 26}, //  15 :21 12 23 26 ? 26, 23, 12, 21
        {22, 26, 24, 15}, //  16 :22 26 24 15 ? // ici BUG
        {26, 23, 14, 24}, //  17 : 26 23 14 24 // ici BUG
      };

      // Génération des faces enfants
      // L'uid d'une nouvelle face est généré avec un hash utilisant les uid des noeuds triés
      for( Integer i = 0 ; i < 12 ; i++ ){
        UniqueArray<Int64> tmp = {node_in_cell[internal_faces[i][0]],node_in_cell[internal_faces[i][1]],node_in_cell[internal_faces[i][2]],node_in_cell[internal_faces[i][3]]};
        std::sort(tmp.begin(),tmp.end());
        Int64 uidface =  Arcane::MeshUtils::generateHashUniqueId(tmp.constView());

        if( new_faces.find(uidface) == new_faces.end() ){ // Not already in map pas utile ici normalement
          // Ajouter connectivités
          faces_to_add.add(IT_Quad4);
          faces_to_add.add(uidface);
          faces_to_add.add(node_in_cell[internal_faces[i][0]]);
          faces_to_add.add(node_in_cell[internal_faces[i][1]]);
          faces_to_add.add(node_in_cell[internal_faces[i][2]]);
          faces_to_add.add(node_in_cell[internal_faces[i][3]]);
          // Ajouter dans tableau uids faces
          faces_uids.add(uidface);
          debug() << 6+ nb_face_to_add << " " << uidface ;
          nb_face_to_add++;
          new_faces.insert(uidface);
        }
      }

      // - Externes 6*4
      const Int64 faces[][4] = {
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
        {24, 14, 2, 13} 
      };

      // Pour chaque faces 
          // Générer hash 
          // associer hash uid face
      //node_in_cell[parent_faces_nodes[i][j]] // i < 6 , j < 4  
      // Parcours des faces parentes
      for(Integer i  = 0 ; i < 6 ; i++){
        parents_to_childs_faces[cell.face(i).uniqueId()] = std::pair<Int64,Int64>(faces_uids.size(),4);
        // Parcours des nouvelles faces
        for(Integer j = 0 ; j < 4 ; j++){
          Int64 step = i*4+j;
          UniqueArray<Int64> tmp = {node_in_cell[faces[step][0]],node_in_cell[faces[step][1]],node_in_cell[faces[step][2]],node_in_cell[faces[step][3]]};
          std::sort(tmp.begin(),tmp.end());
          Int64 uidface =  Arcane::MeshUtils::generateHashUniqueId(tmp.constView());
          
          if( new_faces.find(uidface) == new_faces.end() ){
            faces_to_add.add(IT_Quad4);
            faces_to_add.add(uidface);
            faces_to_add.add(node_in_cell[faces[step][0]]);
            faces_to_add.add(node_in_cell[faces[step][1]]);
            faces_to_add.add(node_in_cell[faces[step][2]]);
            faces_to_add.add(node_in_cell[faces[step][3]]);
            // Ajouter dans tableau uids faces
            faces_uids.add(uidface);
            new_faces.insert(uidface);
            nb_face_to_add++;          
          }
        }
      
      }
      
      // Génération des Hexs
      const Integer new_hex_nodes_index[][8] = {
        {0, 8, 20, 9, 10, 21, 26, 22 },
        {10, 21, 26, 22, 4, 16, 25, 17 },
        {8, 1, 11, 20, 21, 12, 23, 26 },
        {21, 12, 23, 26, 16, 5, 18, 25 },
        {9, 20, 13, 3, 22, 26, 24, 15 },
        {22, 26, 24, 15, 17, 25, 19, 7 },
        {20, 11, 2, 13, 26, 23, 14, 24 },
        {26, 23, 14, 24, 25, 18, 6, 19 }
      };

      // Génération des cellules enfants
      // L'uid est généré à partir du hash de chaque noeuds triés par ordre croissant
      for( Integer i = 0 ; i < 8 ; i++ ){
        // Le nouvel uid est généré avec le hash des nouveaux noeuds qui composent la nouvelle cellule
        UniqueArray<Int64> tmp;
        tmp.reserve(8);
        for( Integer j = 0 ; j < 8 ; j++){
          tmp.add(node_in_cell[new_hex_nodes_index[i][j]]);
        }
        std::sort(tmp.begin(),tmp.end());
        Int64 cell_uid = Arcane::MeshUtils::generateHashUniqueId(tmp.constView());//max_cell_uid+ind_new_cell;

        //Int64 cell_uid = max_cell_uid+ind_new_cell;
        cells_to_add.add(IT_Hexaedron8);
        cells_to_add.add(cell_uid);// uid hex // TODO CHANGER par max_uid + cell_uid * max_nb_node
        cells_to_add.add(node_in_cell[new_hex_nodes_index[i][0]]);
        cells_to_add.add(node_in_cell[new_hex_nodes_index[i][1]]);
        cells_to_add.add(node_in_cell[new_hex_nodes_index[i][2]]);
        cells_to_add.add(node_in_cell[new_hex_nodes_index[i][3]]);
        cells_to_add.add(node_in_cell[new_hex_nodes_index[i][4]]);
        cells_to_add.add(node_in_cell[new_hex_nodes_index[i][5]]);
        cells_to_add.add(node_in_cell[new_hex_nodes_index[i][6]]);
        cells_to_add.add(node_in_cell[new_hex_nodes_index[i][7]]);
        child_cell_owner[cell_uid] = cell.owner();
        parent_cells.add(cell.uniqueId());
        child_cells.add(cell_uid); // groups doublons d'informations avec cells_to_add mais accès plus rapide
        nb_cell_to_add++;
        ind_new_cell++;
        
      }
      // groups
      parents_to_childs_cell[cell.uniqueId()] = std::pair<Int64,Int64>(childs_count,8);
      childs_count += 8; // à modifier selon le nombre d'enfant associé au motif de rafinement !
    }
    // Ajout des nouveaux Noeuds
    Integer nb_node_added = nodes_to_add.size();
    UniqueArray<Int32> nodes_lid(nb_node_added);

    mesh->modifier()->addNodes(nodes_to_add, nodes_lid.view());

    // Edges: Pas de génération d'arrête

    // Ajout des Faces enfants
    UniqueArray<Int32> face_lid(faces_uids.size());

    mesh->modifier()->addFaces(MeshModifierAddFacesArgs(nb_face_to_add, faces_to_add.constView(),face_lid.view()));
    debug() << "addOneFac" << nb_face_to_add ;
    mesh->faceFamily()->itemsUniqueIdToLocalId(face_lid,faces_uids,true);
    debug() << "NB_FACE_ADDED AFTER " << face_lid.size() << " " << new_faces.size()  ;

    ARCANE_ASSERT((nb_face_to_add == (faces_to_add.size()/6)),("non consistant number of faces"));

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

    // Face
    IItemFamily* face_family = mesh->faceFamily();
    // Cell
    IItemFamily* cell_family = mesh->cellFamily();
    

    // Traiter les groupes pour les faces
    // En fait on ne peut traiter que les faces externes. Est-ce qu'on doit/peut déduire les propriétés des faces internes ?
    // Dans le cas du test microhydro on peut car on a que les faces externes aux éléments: XYZ min max
    // A ce moment nous n'avons pas fait de lien face_parent_externe -> face_enfant_externe 
    // Pour le faire nous allons parcourir les faces internes parentes, trier les ids  et trier les éléménts 
    
    
    //IItemFamily* face_family = mesh->faceFamily();
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
      ENUMERATE_ITEM(iitem,group){ // Pour chaque cellule du groupe on ajoute ses 8 enfants ( ou n )
        Int64 step = parents_to_childs_faces[iitem._internalItemBase().uniqueId().asInt64()].first; 
        Int64 n_childs = parents_to_childs_faces[iitem._internalItemBase().uniqueId().asInt64()].second; 
        auto subview = face_lid.subView(step,static_cast<Integer>(n_childs));
        ARCANE_ASSERT((subview.size() == 4 ), ("SUBVIEW"));
        to_add_to_group.addRange(subview);
      }
      group.addItems(to_add_to_group,true);
    }
  
    // Traiter les groupes pour les cellules
    for( ItemGroupCollection::Enumerator igroup(cell_family->groups()); ++igroup; ){
      ItemGroup group = *igroup;
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
      ENUMERATE_ITEM(iitem,group){ // Pour chaque cellule du groupe on ajoute ses 8 enfants ( ou n )
        ARCANE_ASSERT(( static_cast<Integer>(parents_to_childs_cell.size()) == child_cells_lid.size()/8 ),("Wrong number of childs"));
        Int64 step = parents_to_childs_cell[iitem._internalItemBase().uniqueId().asInt64()].first; 
        Int64 n_childs = parents_to_childs_cell[iitem._internalItemBase().uniqueId().asInt64()].second; 
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

    // Gestion et assignation du propriétaire pour chaque cellule
    // Le propriétaire est simplement le sous domaine qui a générer les nouvelles cellules
    ENUMERATE_ (Cell, icell, mesh->allCells()){
        Cell cell = *icell;
        cell.mutableItemBase().setOwner(my_rank, my_rank);
    }
    mesh->cellFamily()->notifyItemsOwnerChanged();

    // Assignation des coords aux noeuds
    ENUMERATE_(Node, inode, mesh->nodeFamily()->view(nodes_lid)){
      Node node = *inode;
      nodes_coords[node] = nodes_to_add_coords[node.uniqueId()];
    }

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

    // Supression de la couche fantôme
    gm2->setNbGhostLayer(0);
    mesh->updateGhostLayers(true);

    // Quelques sur le nombres d'entités insérés
    ARCANE_ASSERT((mesh->nbCell() == nb_cell_init*8 ),("Wrong number of cell added"));
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
    gm2->setNbGhostLayer(1);
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_ARCANEBASICMESHSUBDIVIDERSERVICE(ArcaneBasicMeshSubdivider,
                                                         ArcaneBasicMeshSubdividerService);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
