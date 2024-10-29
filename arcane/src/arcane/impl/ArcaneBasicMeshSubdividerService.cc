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



#include "arcane/ItemGroup.h"
#include "arcane/ItemPrinter.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/IGhostLayerMng.h"
#include "arcane/MeshUtils.h"
#include "arcane/IMesh.h"
#include "arcane/IMeshModifier.h"

#include "arcane/core/SimpleSVGMeshExporter.h" // Write au format svg pour le 2D
// Write variables
#include "arcane/core/IPostProcessorWriter.h"
#include "arcane/core/ServiceBuilder.h"
#include "arcane/core/Directory.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/IParallelMng.h"
#include "arcane/BasicService.h"
#include<arcane/IPrimaryMesh.h>

// get parameter
#include "arcane/utils/ApplicationInfo.h"
#include "arcane/utils/CommandLineArguments.h"

// utils
#include <map>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>

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
  warning() << "SubdivideMesh: Function not implemented";

  //Arcane::ParameterList cmd = subDomain()->applicationInfo().commandLineArguments().parameters();
  //String m_mesh_file_name = StringVariableReplace::replaceWithCmdLineArgs(subDomain()->applicationInfo().commandLineArguments().parameters(), options()->filename);
  // getParameter("nb_refine")
  //applicationInfo().commandLineArguments();
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

  debug() << "PART 3D";
  mesh->utilities()->writeToFile("3D_last_input"+std::to_string(my_rank)+".vtk", "VtkLegacyMeshWriter");

  // PAS DE GHOST LAYER
  IGhostLayerMng* gm = mesh->ghostLayerMng();
  Int32 version = gm->builderVersion();
  if (version < 3)
    gm->setBuilderVersion(3);
  Int32 nb_ghost_layer = gm->nbGhostLayer();
  gm->setNbGhostLayer(0);
  mesh_modifier->setDynamic(true);
  mesh_modifier->updateGhostLayers();

  // Uniquement pour les asserts à la fin
  Integer nb_cell_init = mesh->nbCell();
  Integer nb_face_init = mesh->nbFace();
  Integer nb_edge_init = mesh->nbEdge();
  Integer nb_node_init = mesh->nbNode();

  // Compter les arrêtes
  // On cherche un moyen de compter les arrêtes pour faire un test facile sur le nombre de noeud inséré.
  // ARCANE_ASSERT((nb_edge_init+ nb_cell_init + nb_face_init)== nb_node_added,("Mauvais nombre de noeuds insérés"));
  //std::cout << "#NOMBRE INITS " << nb_node_init << " " << mesh->allEdges().size() << " " << edg.size() << " " << nb_face_init << " " << nb_cell_init << std::endl;

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
  std::cout << "ARRAY SIZE " << nodes_coords.arraySize() << std::endl;
  // Nodes on entities
  std::set<Int64> new_nodes; // Using a map to ensure that we don't add nodes multiple times (maybe consider doing a different way)
  std::set<Int64> new_faces; // ^--- same for faces
  // Maps owners
  std::unordered_map<Int64, Int32> node_uid_to_owner;
  std::unordered_map<Int64, Int32> edge_uid_to_owner;
  std::unordered_map<Int64, Int32> face_uid_to_owner;
  std::unordered_map<Int64,Int32> child_cell_owner;
  std::unordered_map<Int32, Int32> old_face_lid_to_owner;


  UniqueArray<Int32> cells_to_detach;

  UniqueArray<Int64> faces_uids; // Contient uniquement les uids pas les connectivités
  UniqueArray<Int64> edges_uids; // Idem

  // Calcul nombre de noeuds à insérer
  const Integer nb_node_to_add_total = mesh->nbCell()+mesh->nbFace()+mesh->nbEdge(); // Attention pattern dépendant
  nodes_to_add.reserve(nb_node_to_add_total);
  nodes_to_add_coords.reserve(nb_node_to_add_total);

  IParallelMng* pm = mesh->parallelMng();
  // Calcul max_node_uid

  Int64 max_node_uid = 0;
  Arcane::VariableNodeReal3& nodes_coordinates = mesh->nodesCoordinates();

  ENUMERATE_NODE(inode,mesh->allNodes())
  {
    const Node& node = *inode;
    const Int64 uid = node.uniqueId();
    if (uid>max_node_uid)
    max_node_uid = uid;
  }
  Integer m_max_node_uid = 0, m_next_node_uid;
  if (pm->commSize() > 1)
    m_max_node_uid = pm->reduce(Parallel::ReduceMax, max_node_uid);
  else
    m_max_node_uid = max_node_uid;
  info() << "NODE_UID_INFO: MY_MAX_UID=" << max_node_uid << " GLOBAL=" << m_max_node_uid;
  m_next_node_uid = m_max_node_uid + 1 + my_rank;

  // Calcul max_cell_uid
  Int64 max_cell_uid = 0;
  ENUMERATE_CELL(icell,mesh->allCells())
  {
    const Cell& cell = *icell;
    const Int64 uid = cell.uniqueId();
    if (uid>max_cell_uid)
    max_cell_uid = uid;
  }
  Integer m_max_cell_uid = 0;
  if (pm->commSize() > 1)
    m_max_cell_uid = pm->reduce(Parallel::ReduceMax, max_cell_uid);
  else
    m_max_cell_uid = max_cell_uid;

  std::cout << "MAXCELLUID " << max_cell_uid << " " << m_max_cell_uid << std::endl;
  max_cell_uid ++;
  Integer ind_new_cell = 0 ;

  ARCANE_ASSERT((mesh->nbEdge() == 0 ),("Wrong number of edge"));

  // Saving owner for new nodes on edges
  ENUMERATE_FACE(iface,mesh->ownFaces()){
    const Face& face = *iface;
    old_face_lid_to_owner[face.localId()] = face.owner();
  }

  // Traitement pour une cellule
  ENUMERATE_CELL(icell,mesh->ownCells())
  {
      const Cell& cell = *icell;



      debug() << "CELL_OWNER " << cell.owner() ;
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
      // Noeuds sur arrêtes
      Integer new_nodes_on_edges_couple[][2] = {{0, 1},{0, 3},{0, 4},{1, 2},{1, 5},{2, 3},{2, 6},{3, 7},{4, 5},{4, 7},{5, 6},{6, 7}};

      for( Integer i = 0 ; i < 12 ; i++ ){
        // uid
        UniqueArray<Int64> tmp = {
          node_in_cell[new_nodes_on_edges_couple[i][0]],
          node_in_cell[new_nodes_on_edges_couple[i][1]],
        };
        Int32 min_index=0; // tmp[0]
        if( tmp[0] > tmp[1]   ){
            std::swap(tmp[0],tmp[1]);
            min_index=1;
        }
        std::cout << "#TMP " << "cell" << cell.uniqueId() << ' ' << tmp << std::endl;
        node_in_cell[index_27] = Arcane::MeshUtils::generateHashUniqueId(tmp.constView());
        // Coord on edge
        Arcane::Real3 middle_coord(0.0,0.0,0.0);
        middle_coord = (nodes_coordinates[cell.node(new_nodes_on_edges_couple[i][0])] + nodes_coordinates[cell.node(new_nodes_on_edges_couple[i][1])] ) / 2.0;
        nodes_to_add_coords[node_in_cell[index_27]] = middle_coord;
        // Now we use the owner of the lower id as owner

        node_uid_to_owner[node_in_cell[index_27]] = cell.node(new_nodes_on_edges_couple[i][min_index]).owner();
        /*
        FaceLocalIdView fa = cell.node(new_nodes_on_edges_couple[i][0]).faceIds() ;
        FaceLocalIdView fb = cell.node(new_nodes_on_edges_couple[i][1]).faceIds() ;
        std::set<Int32> sta( fa.begin(), fa.end() );
        std::set<Int32> stb( fb.begin(), fb.end() );
        std::vector<Int32> res;

        std::set_intersection(sta.begin(), sta.end(), stb.begin(), stb.end(), std::back_inserter(res.begin()));
        Int32 min_rank = old_face_lid_to_owner[res[0]];

        for( auto it = res.begin()+1 ; it != res.end() ; ++it ){
            min_rank = std::min(min_rank ,old_face_lid_to_owner[*it]);
        }
        node_uid_to_owner[node_in_cell[index_27]] = min_rank;*/
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
        // Is the node already compute ? compute coord : don't compute coord ;
        //

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
      //std::sort(tmp.begin(),tmp.end());
      node_in_cell[index_27] = Arcane::MeshUtils::generateHashUniqueId(tmp.constView());

      node_uid_to_owner[node_in_cell[index_27]] = cell.owner();

      // Coord on cell
      Arcane::Real3 middle_coord(0.0,0.0,0.0);
      for( Integer i = 0 ; i < 8 ; i++ ){
        middle_coord += nodes_coordinates[cell.node(new_nodes_on_cell_oct[i])];
      }
      middle_coord /= 8.0;
      nodes_to_add_coords[node_in_cell[index_27]] = middle_coord;


      // Génération des Edges
      // not mandatory

      // !here
      for( Integer i = 8 ; i < 27 ; i++){
        // We compute multiple times coordinates of a node TODO improve that
        // We add new node if it is not already in node
        if( new_nodes.find(node_in_cell[i]) == new_nodes.end()  ){
          nodes_to_add.add(node_in_cell[i]);
          new_nodes.insert(node_in_cell[i]);
        }
      }
      std::cout << nodes_to_add_coords.size() << " " << nodes_to_add.size() << std::endl;

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
          std::cout << 6+ nb_face_to_add << " " << uidface << std::endl;
          nb_face_to_add++;
          new_faces.insert(uidface);
        }
      }

      // - Externes 6*4
      const Int64 faces[][4] = {
        {0, 8, 20, 9},    //  Arr
        {9, 20, 13, 3},
        {8, 1, 11, 20},
        {20, 11, 2, 13},
        {0, 10, 22, 9},   // Gauche
        {9, 22, 15, 3},
        {10, 4, 17, 22},
        {22, 17, 7, 15},
        {1, 12, 23, 11},  // Droite
        {11, 23, 14, 2},
        {12, 5, 18, 23},
        {23, 18, 6, 14},
        {7, 19 ,24, 15},  // haut
        {19, 6 ,14, 24},
        {15, 24, 13, 3},
        {24, 14, 2, 13},
        {4, 16, 21, 10},  // bas
        {10, 21, 8, 0},
        {16, 5, 12, 21},
        {21, 12, 1, 8},
        {4 ,16, 25 ,17 }, //devant
        {17, 25, 19, 7},
        {16, 5, 18, 25 },
        {25, 18, 6, 19}
        };

        for(Integer i = 0 ; i < 24 ; i++){
          UniqueArray<Int64> tmp = {node_in_cell[faces[i][0]],node_in_cell[faces[i][1]],node_in_cell[faces[i][2]],node_in_cell[faces[i][3]]};
          std::sort(tmp.begin(),tmp.end());// maybe to generate the same uid we will need to sort ids ?
          Int64 uidface =  Arcane::MeshUtils::generateHashUniqueId(tmp.constView());
          if( new_faces.find(uidface) == new_faces.end() ){ // Not already in map pas utile ici normalement
            faces_to_add.add(IT_Quad4);
            faces_to_add.add(uidface);
            faces_to_add.add(node_in_cell[faces[i][0]]);
            faces_to_add.add(node_in_cell[faces[i][1]]);
            faces_to_add.add(node_in_cell[faces[i][2]]);
            faces_to_add.add(node_in_cell[faces[i][3]]);
            // Ajouter dans tableau uids faces
            faces_uids.add(uidface);
            new_faces.insert(uidface);
            nb_face_to_add++;
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

      for( Integer i = 0 ; i < 8 ; i++ ){
        // An other policy to generate celluid
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
        nb_cell_to_add++;
        ind_new_cell++;
      }
    }
    // Add nodes
    Integer nb_node_added = nodes_to_add.size();
    UniqueArray<Int32> nodes_lid(nb_node_added);

    mesh->modifier()->addNodes(nodes_to_add, nodes_lid.view());
    // Gestion ownership noeuds ici
    // - Assign to the new nodes on faces the owner of the cell (recompute hash)

    std::cout << "NOEUDS" << std::endl;

    ENUMERATE_(Face, iface, mesh->allFaces()){
        Face face = *iface;
        UniqueArray<Int64> tmp;
        tmp.reserve(4);
        for(Node node : face.nodes()){
          tmp.add(node.uniqueId());
        }
        std::sort(tmp.begin(),tmp.end());
        Int64 uidface =  Arcane::MeshUtils::generateHashUniqueId(tmp.constView());
        node_uid_to_owner[uidface] = face.owner();
    }
    ENUMERATE_ (Node, inode, mesh->nodeFamily()->view(nodes_lid)) {
        Node node = *inode;
        std::cout << node.uniqueId() << " " ;
        node.mutableItemBase().setOwner( node_uid_to_owner[node.uniqueId()] ,my_rank);   //
    }

    mesh->nodeFamily()->notifyItemsOwnerChanged();
    // Edges: Pas de génération d'arrête

    debug() <<  "#NBNODE " <<mesh->nbNode() <<  " nb_node_init " << nb_node_init << " nb_node_added "  << nb_node_added <<  " " ;
    debug() << "#SIZES" << nodes_to_add.size() << " new_nodes " << new_nodes.size() << " " << edges_to_add.size() << " " << faces_to_add.size() << " " << cells_to_add.size() << " " << nodes_to_add_coords.size() << " " << nodes_lid.size() << " " << nodes_coordinates.arraySize() ;
    for( auto x=cells_to_add.begin() ; x != cells_to_add.end(); ++x ){
      if( std::distance(cells_to_add.begin(),x) % 10 == 0 )
        std::cout << std::endl;
      std::cout << *x << std::endl;
    }

    // Faces
    UniqueArray<Int32> face_lid(faces_uids.size());

    std::cout << "NB_FACE " << mesh->nbFace() ;

    std::cout << "addOneFac" << nb_face_to_add << std::endl;
    mesh->modifier()->addFaces(MeshModifierAddFacesArgs(nb_face_to_add, faces_to_add.constView(),face_lid.view()));
    std::cout << "addOneFac" << nb_face_to_add << std::endl;
    mesh->faceFamily()->itemsUniqueIdToLocalId(face_lid,faces_uids,true);
    std::cout << "NB_FACE_ADDED AFTER " << face_lid.size() << " " << new_faces.size() << std::endl ;

    // Gestion ownership faces ici
    ENUMERATE_ (Face, iface, mesh->faceFamily()->view(face_lid)){
        Face face = *iface;
        face.mutableItemBase().setOwner(face_uid_to_owner[face.uniqueId()],my_rank);
        std::cout << ItemPrinter(face) << std::endl;
    }
    std::cout << "Debug" << std::endl;
    // Debug

    ENUMERATE_ (Face, iface, mesh->faceFamily()->view(face_lid)){
        Face face = *iface;
        std::cout << ItemPrinter(face) << "\nNodes:" << std::endl;
        for(Node node : face.nodes()){
          std::cout << node.uniqueId() << std::endl;
        }
    }

    // Faces supprimées quand toute les cells sont supprimées
    mesh->faceFamily()->notifyItemsOwnerChanged();
    ARCANE_ASSERT((nb_face_to_add == (faces_to_add.size()/6)),("non consistant number of faces"));
    // Cells
    UniqueArray<Int32> cells_lid(nb_cell_to_add);
    mesh->modifier()->addCells(nb_cell_to_add, cells_to_add.constView(),cells_lid);
    mesh->modifier()->removeCells(cells_to_detach.constView());
    std::cout << "FinalNbfaces" << mesh->nbFace() << std::endl ;
    // Gestion ownership cells ici
    mesh->modifier()->endUpdate();

    ENUMERATE_ (Cell, icell, mesh->allCells()){
        Cell cell = *icell;
        cell.mutableItemBase().setOwner(my_rank, my_rank);
    }
    mesh->cellFamily()->notifyItemsOwnerChanged();

    // On peut laisser la gestion des owners au primary mesh
    //mesh->endAllocate();
    /*Arcane::IGhostLayerMng * gm2 = mesh->ghostLayerMng();
    gm2->setNbGhostLayer(1);
    mesh->updateGhostLayers(true);
    Arcane::IPrimaryMesh * pmesh = mesh->toPrimarymesh;
    pmesh->setOwnersFromCells();
    */

    //mesh->utilities()->changeOwnersFromCells();
    // Assignement des coords aux noeuds
    ENUMERATE_(Node, inode, mesh->nodeFamily()->view(nodes_lid)){
      Node node = *inode;
      nodes_coords[node] = nodes_to_add_coords[node.uniqueId()];
    }


    mesh->utilities()->writeToFile("3Drefined"+std::to_string(my_rank)+".vtk", "VtkLegacyMeshWriter");


    debug() << "END 3D fun" ;
    debug() << "NB CELL " << mesh->nbCell() << " " << nb_cell_init*8 ;
    debug() << mesh->nbNode() << " " << nb_node_init << " " << nb_edge_init << " " << nb_face_init << " " << nb_cell_init;
    debug() << mesh->nbFace() << "nb_face_init " << nb_face_init <<  " " <<  nb_face_init << " " << nb_cell_init ;
    debug() << "Faces: " << mesh->nbFace() << " theorical nb_face_to add: " << nb_face_init*4 + nb_cell_init*12 <<  " nb_face_init " <<  nb_face_init << " nb_cell_init " << nb_cell_init ;
    /* Debug
    Integer cpt2 = 0;
    std::cout << "Node_coordinate: " << mesh->nbNode() << std::endl ;
    ENUMERATE_ (Node, inode, mesh->allNodes()) {
        Node node = *inode;
        std::cout << nodes_coordinates[node] << " " << cpt2++ << " N:" << node.uniqueId() << std::endl;
    }
    */
    // ARCANE_ASSERT((mesh->nbNode() == nb_edge_init + nb_face_init + nb_cell_init ),("Wrong number of node added")) // Ajouter en debug uniquement pour savoir combien de noeud on à la fin
    ARCANE_ASSERT((mesh->nbCell() == nb_cell_init*8 ),("Wrong number of cell added"));
    ARCANE_ASSERT((mesh->nbFace() <= nb_face_init*4 + 12 * nb_cell_init ),("Wrong number of face added"));
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
