// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FaceUniqueIdBuilder.cc                                      (C) 2000-2025 */
/*                                                                           */
/* Construction des identifiants uniques des faces.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/CheckedConvert.h"

#include "arcane/core/IMeshUniqueIdMng.h"
#include "arcane/core/IParallelExchanger.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ISerializeMessage.h"
#include "arcane/core/ISerializer.h"
#include "arcane/core/ParallelMngUtils.h"

#include "arcane/mesh/DynamicMesh.h"
#include "arcane/mesh/OneMeshItemAdder.h"
#include "arcane/mesh/GhostLayerBuilder.h"
#include "arcane/mesh/FaceUniqueIdBuilder.h"
#include "arcane/mesh/ItemTools.h"
#include "arcane/mesh/ItemsOwnerBuilder.h"

#include <unordered_set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" void
_computeFaceUniqueIdVersion3(DynamicMesh* mesh);
extern "C++" void
_computeFaceUniqueIdVersion5(DynamicMesh* mesh);
extern "C++" void
arcaneComputeCartesianFaceUniqueId(DynamicMesh* mesh);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FaceUniqueIdBuilder::
FaceUniqueIdBuilder(DynamicMeshIncrementalBuilder* mesh_builder)
: TraceAccessor(mesh_builder->mesh()->traceMng())
, m_mesh(mesh_builder->mesh())
, m_mesh_builder(mesh_builder)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceUniqueIdBuilder::
computeFacesUniqueIds()
{
  IParallelMng* pm = m_mesh->parallelMng();
  Real begin_time = platform::getRealTime();
  Integer face_version = m_mesh->meshUniqueIdMng()->faceBuilderVersion();
  bool is_parallel = pm->isParallel();
  info() << "Using version=" << face_version << " to compute faces unique ids"
         << " mesh=" << m_mesh->name() << " is_parallel=" << is_parallel;

  if (face_version > 5 || face_version < 0)
    ARCANE_FATAL("Invalid value '{0}' for compute face unique ids versions: v>=0 && v<=6", face_version);

  if (face_version == 5)
    _computeFaceUniqueIdVersion5(m_mesh);
  else if (face_version == 4)
    arcaneComputeCartesianFaceUniqueId(m_mesh);
  else if (face_version == 3)
    _computeFaceUniqueIdVersion3(m_mesh);
  else if (face_version == 0) {
    info() << "No face renumbering";
  }
  else {
    // Version 1 ou 2
    if (is_parallel) {
      if (face_version == 2) {
        //PAS ENCORE PAR DEFAUT
        info() << "Use new mesh init in FaceUniqueIdBuilder";
        _computeFacesUniqueIdsParallelV2();
      }
      else {
        // Version par défaut.
        _computeFacesUniqueIdsParallelV1();
      }
    }
    else {
      _computeFacesUniqueIdsSequential();
    }
  }

  Real end_time = platform::getRealTime();
  Real diff = (Real)(end_time - begin_time);
  info() << "TIME to compute face unique ids=" << diff;

  if (arcaneIsCheck())
    _checkNoDuplicate();

  ItemInternalMap& faces_map = m_mesh->facesMap();

  // Il faut ranger à nouveau #m_faces_map car les uniqueId() des
  // faces ont été modifiés
  if (face_version != 0)
    m_mesh->faceFamily()->notifyItemsUniqueIdChanged();

  bool is_verbose = m_mesh_builder->isVerbose();
  if (is_verbose) {
    info() << "NEW FACES_MAP after re-indexing";
    faces_map.eachItem([&](Item face) {
      info() << "Face uid=" << face.uniqueId() << " lid=" << face.localId();
    });
  }
  // Avec la version 0 ou 5, les propriétaires ne sont pas positionnées
  // Il faut le faire maintenant
  if (face_version == 0 || face_version == 5) {
    ItemsOwnerBuilder owner_builder(m_mesh);
    owner_builder.computeFacesOwner();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérifie qu'on n'a pas deux fois le même uniqueId().
 */
void FaceUniqueIdBuilder::
_checkNoDuplicate()
{
  info() << "Check no duplicate face uniqueId";
  ItemInternalMap& faces_map = m_mesh->facesMap();
  std::unordered_set<Int64> checked_faces_map;
  faces_map.eachItem([&](Item face) {
    ItemUniqueId uid = face.uniqueId();
    auto p = checked_faces_map.find(uid);
    if (p!=checked_faces_map.end()){
      pwarning() << "Duplicate Face UniqueId=" << uid;
      ARCANE_FATAL("Duplicate Face uniqueId={0}",uid);
    }
    checked_faces_map.insert(uid);
  });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe d'aide pour la détermination en parallèle
 * des unique_id des faces.
 *
 * \note Tous les champs de cette classe doivent être de type Int64
 * car elle est sérialisée par cast en Int64*.
 */
class T_CellFaceInfo
{
 public:

  T_CellFaceInfo(Int64 uid,Integer nb_back_face,Integer nb_true_boundary_face)
  : m_unique_id(uid), m_nb_back_face(nb_back_face), m_nb_true_boundary_face(nb_true_boundary_face)
  {
  }

  T_CellFaceInfo()
  : m_unique_id(NULL_ITEM_ID), m_nb_back_face(0), m_nb_true_boundary_face(0)
  {
  }

 public:

  bool operator<(const T_CellFaceInfo& ci) const
    {
      return m_unique_id<ci.m_unique_id;
    }

 public:

  Int64 m_unique_id;
  Int64 m_nb_back_face;
  Int64 m_nb_true_boundary_face;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \brief Calcul les numéros uniques de chaque face en parallèle.

  En plus de la numérotation, détermine le sous-domaine propriétaire de
  chaque face, en considérant qu'une face appartient au même sous-domaine
  que sa maille qui est derrière (ou au sous-domaine courant pour une
  maille frontière).
  
  \todo optimiser l'algorithme pour ne pas avoir à créer un tableau
  dimensionné au nombre total de mailles du maillage (en procédent en
  plusieurs étapes)

  \todo trouver un algorithme plus optimum la recherche du propriétaire de
  chaque face et ce pour équilibrer les coms.
*/  
void FaceUniqueIdBuilder::
_computeFacesUniqueIdsParallelV1()
{
  IParallelMng* pm = m_mesh->parallelMng();
  Integer my_rank = pm->commRank();
  Integer nb_rank = pm->commSize();
  
  Integer nb_local_face = m_mesh_builder->oneMeshItemAdder()->nbFace();
  Integer nb_local_cell = m_mesh_builder->oneMeshItemAdder()->nbCell();
  bool is_verbose = m_mesh_builder->isVerbose();

  UniqueArray<Int64> faces_opposite_cell_uid(nb_local_face);
  faces_opposite_cell_uid.fill(NULL_ITEM_ID);
  UniqueArray<Integer> faces_opposite_cell_index(nb_local_face);
  UniqueArray<Integer> faces_opposite_cell_owner(nb_local_face);

  // Pour vérification, s'assure que tous les éléments de ce tableau
  // sont valides, ce qui signifie que toutes les faces ont bien été
  // renumérotés
  UniqueArray<Int64> faces_new_uid(nb_local_face);
  faces_new_uid.fill(NULL_ITEM_ID);
  
  Integer nb_recv_sub_domain_boundary_face = 0;

  Int64UniqueArray faces_infos;
  faces_infos.reserve(10000);
  ItemInternalMap& cells_map = m_mesh->cellsMap();
  ItemInternalMap& faces_map = m_mesh->facesMap();
  ItemInternalMap& nodes_map = m_mesh->nodesMap();


  // NOTE: ce tableau n'est pas utile sur toutes les mailles. Il
  // suffit qu'il contienne les mailles dont on a besoin, c'est à dire
  // les notres + celles connectées à une de nos faces. Une table
  // de hashage sera plus appropriée.
  HashTableMapT<Int64,Int64> cells_first_face_uid(m_mesh_builder->oneMeshItemAdder()->nbCell()*2,true);
  
  // Rassemble les données des autres processeurs dans recv_cells;
  // Pour éviter que les tableaux ne soient trop gros, on procède en plusieurs
  // étapes.
  // Chaque sous-domaine construit sa liste de faces frontières, avec pour
  // chaque face:
  // - son type
  // - la liste de ses noeuds,
  // - le numéro unique de sa maille
  // - le propriétaire de sa maille
  // - son indice dans sa maille
  // Cette liste sera ensuite envoyée à tous les sous-domaines.
  {
    ItemTypeMng* itm = m_mesh->itemTypeMng();

    UniqueArray<Int32> faces_local_id;

    faces_map.eachItem([&](Face face) {
      bool boundary_val = face.isSubDomainBoundary();
      if (boundary_val)
        faces_local_id.add(face.localId());
    });

    Integer nb_sub_domain_boundary_face = faces_local_id.size();
    Int64 global_nb_boundary_face = pm->reduce(Parallel::ReduceSum,(Int64)nb_sub_domain_boundary_face);
    debug() << "NB BOUNDARY FACE=" << nb_sub_domain_boundary_face
            << " NB_FACE=" << nb_local_face
            << " GLOBAL_NB_BOUNDARY_FACE=" << global_nb_boundary_face;
    

    Int64UniqueArray faces_infos2;
    Int64UniqueArray recv_faces_infos;

    // Calcule la taille d'un bloc d'envoi
    // La mémoire nécessaire pour une face est égale à nb_node + 4.
    // Si on suppose qu'on a des quadrangles en général, cela
    // fait 8 Int64 par face, soit 64 octets par face.
    // La mémoire nécessaire pour tout envoyer est donc 64 * global_nb_boundary_face.
    // step_size * nb_proc * 64 octets.
    // Le step_size par défaut est calculé pour que la mémoire nécessaire
    // soit de l'ordre de 100 Mo pour chaque message.
    Int64 step_size = 1500000;
    Integer nb_phase = CheckedConvert::toInteger((global_nb_boundary_face / step_size) + 1);
    FaceInfoListView faces(m_mesh->faceFamily());
    for( Integer i_phase=0; i_phase<nb_phase; ++i_phase ){
      Integer nb_face_to_send = nb_sub_domain_boundary_face / nb_phase;
      Integer first_face_to_send = nb_face_to_send * i_phase;
      Integer last_face_to_send = first_face_to_send + nb_face_to_send;
      if (i_phase+1==nb_phase)
        last_face_to_send = nb_sub_domain_boundary_face;
      Integer real_nb_face_to_send = last_face_to_send - first_face_to_send;

      faces_infos2.clear();
      for( Integer i_face=first_face_to_send; i_face<first_face_to_send+real_nb_face_to_send; ++i_face ){
        Face face(faces[faces_local_id[i_face]]);
        face.mutableItemBase().addFlags(ItemFlags::II_Shared | ItemFlags::II_SubDomainBoundary);
        bool has_back_cell = face.itemBase().flags() & ItemFlags::II_HasBackCell;
        faces_infos2.add(face.type());
        for( Node node : face.nodes() )
          faces_infos2.add(node.uniqueId().asInt64());
        
        //info() << " ADD FACE lid=" << face->localId();
        
        Cell cell = face.cell(0);
        faces_infos2.add(cell.uniqueId().asInt64());
        faces_infos2.add(cell.owner());

        Integer face_index_in_cell = 0;
        if (has_back_cell){
          for( Face current_face_in_cell : cell.faces() ){
            if (current_face_in_cell==face)
              break;
            if (current_face_in_cell.backCell()==cell)
              ++face_index_in_cell;
          }
        }
        else
          face_index_in_cell = NULL_ITEM_ID;
        faces_infos2.add(face_index_in_cell);
      }
      faces_infos2.add(IT_NullType); // Pour dire que la liste s'arête

      pm->allGatherVariable(faces_infos2,recv_faces_infos);

      info() << "Number of face bytes received: " << recv_faces_infos.size()
             << " phase=" << i_phase << "/" << nb_phase
             << " first_face=" << first_face_to_send
             << " last_face=" << last_face_to_send
             << " nb=" << real_nb_face_to_send;
  
      Integer recv_faces_infos_index = 0;

      for( Integer i_sub_domain=0; i_sub_domain<nb_rank; ++i_sub_domain ){
        bool is_end = false;
        // Il ne faut pas lire les infos qui viennent de moi
        if (i_sub_domain==my_rank){
          recv_faces_infos_index += faces_infos2.size();
          continue;
        }
        // Désérialise les infos de chaque sous-domaine
        while (!is_end){
          Integer face_type = CheckedConvert::toInteger(recv_faces_infos[recv_faces_infos_index]);
          ++recv_faces_infos_index;
          if (face_type==IT_NullType){
            is_end = true;
            break;
          }
          ItemTypeInfo* itt = itm->typeFromId(face_type);
          Integer face_nb_node = itt->nbLocalNode();
          ConstArrayView<Int64> faces_nodes_uid(face_nb_node,&recv_faces_infos[recv_faces_infos_index]);

          recv_faces_infos_index += face_nb_node;
          Int64 cell_uid = recv_faces_infos[recv_faces_infos_index];
          ++recv_faces_infos_index;
          Integer cell_owner = CheckedConvert::toInteger(recv_faces_infos[recv_faces_infos_index]);
          ++recv_faces_infos_index;
          Integer cell_face_index = CheckedConvert::toInteger(recv_faces_infos[recv_faces_infos_index]);
          ++recv_faces_infos_index;

          Node node = nodes_map.tryFind(faces_nodes_uid[0]);
          // Si la face n'existe pas dans mon sous-domaine, elle ne m'intéresse pas
          if (node.null())
            continue;
          Face face = ItemTools::findFaceInNode2(node, face_type, faces_nodes_uid);
          if (face.null())
            continue;
          ++nb_recv_sub_domain_boundary_face;
          faces_opposite_cell_uid[face.localId()] = cell_uid;
          faces_opposite_cell_index[face.localId()] = cell_face_index;
          faces_opposite_cell_owner[face.localId()] = cell_owner;
          cells_first_face_uid.add(cell_uid,-1);
        }
      }
    }
    info() << "Number of faces on the subdomain interface: "
           << nb_sub_domain_boundary_face << ' ' << nb_recv_sub_domain_boundary_face;
  }

  // Cherche le uniqueId max des mailles sur tous les sous-domaines.
  Int64 max_cell_uid = 0;
  Int32 max_cell_local_id = 0;
  cells_map.eachItem([&](Item cell) {
    Int64 cell_uid = cell.uniqueId().asInt64();
    Int32 cell_local_id = cell.localId();
    if (cell_uid>max_cell_uid)
      max_cell_uid = cell_uid;
    if (cell_local_id>max_cell_local_id)
      max_cell_local_id = cell_local_id;
  });
  Int64 global_max_cell_uid = pm->reduce(Parallel::ReduceMax,max_cell_uid);
  debug() << "GLOBAL MAX CELL UID=" << global_max_cell_uid;


  UniqueArray<T_CellFaceInfo> my_cells_faces_info;
  my_cells_faces_info.reserve(nb_local_cell);
  IntegerUniqueArray my_cells_nb_back_face(max_cell_local_id+1);
  my_cells_nb_back_face.fill(0);

  cells_map.eachItem([&](Cell cell) {
    Int64 cell_uid = cell.uniqueId().asInt64();
    Int32 cell_local_id = cell.localId();
    Integer nb_back_face = 0;
    Integer nb_true_boundary_face = 0;
    for( Face face : cell.faces() ){
      Int64 opposite_cell_uid = faces_opposite_cell_uid[face.localId()];
      if (face.backCell()==cell)
        ++nb_back_face;
      else if (face.nbCell()==1 && opposite_cell_uid==NULL_ITEM_ID){
        ++nb_true_boundary_face;
      }
    }
    my_cells_nb_back_face[cell_local_id] = nb_back_face;
    my_cells_faces_info.add(T_CellFaceInfo(cell_uid,nb_back_face,nb_true_boundary_face));
  });
  std::sort(std::begin(my_cells_faces_info),std::end(my_cells_faces_info));

  {
    Integer nb_phase = 16;
    Integer first_cell_index_to_send = 0;
    Int64 current_face_uid = 0;
    
    for( Integer i_phase=0; i_phase<nb_phase; ++i_phase ){
      Integer nb_uid_to_send = CheckedConvert::toInteger(global_max_cell_uid / nb_phase);
      Integer first_uid_to_send = nb_uid_to_send * i_phase;
      Int64 last_uid_to_send = first_uid_to_send + nb_uid_to_send;
      if (i_phase+1==nb_phase)
        last_uid_to_send = global_max_cell_uid;

      //Integer last_cell_index_to_send = first_cell_index_to_send;
      Integer nb_cell_to_send = 0;
      for( Integer zz=first_cell_index_to_send, zs=my_cells_faces_info.size(); zz<zs; ++zz ){
        if (my_cells_faces_info[zz].m_unique_id<=last_uid_to_send)
          ++nb_cell_to_send;
        else
          break;
      }
      debug() << "FIRST TO SEND=" << first_cell_index_to_send
              << " NB=" << nb_cell_to_send
              << " first_uid=" << first_uid_to_send
              << " last_uid=" << last_uid_to_send;
      
      T_CellFaceInfo* begin_cell_array = my_cells_faces_info.data()+first_cell_index_to_send;
      Int64* begin_array = reinterpret_cast<Int64*>(begin_cell_array);
      Integer begin_size = nb_cell_to_send*3;
      
      Int64ConstArrayView cells_faces_infos(begin_size,begin_array);
      
      Int64UniqueArray recv_cells_faces_infos;
      pm->allGatherVariable(cells_faces_infos,recv_cells_faces_infos);
      first_cell_index_to_send += nb_cell_to_send;

      info() << "Infos faces (received) nb_int64=" << recv_cells_faces_infos.size();
      Integer recv_nb_cell = recv_cells_faces_infos.size() / 3;

      // NOTE: comme on connait le uid min et max possible, on peut optimiser en
      // creant un tableau dimensionne avec comme borne ce min et ce max et
      // remplir directement les elements de ce tableau. Comme cela, on
      // n'a pas besoin de tri.
      UniqueArray<T_CellFaceInfo> global_cells_faces_info(recv_nb_cell);
      for( Integer i=0; i<recv_nb_cell; ++i ){
        Int64 cell_uid = recv_cells_faces_infos[i*3];
        global_cells_faces_info[i].m_unique_id = cell_uid;
        global_cells_faces_info[i].m_nb_back_face = recv_cells_faces_infos[(i*3) +1];
        global_cells_faces_info[i].m_nb_true_boundary_face = recv_cells_faces_infos[(i*3) +2];
      }
      info() << "Sorting the faces nb=" << global_cells_faces_info.size();
      std::sort(std::begin(global_cells_faces_info),std::end(global_cells_faces_info));

      for( Integer i=0; i<recv_nb_cell; ++i ){
        Int64 cell_uid = global_cells_faces_info[i].m_unique_id;
        if (cells_map.hasKey(cell_uid) || cells_first_face_uid.hasKey(cell_uid))
          cells_first_face_uid.add(cell_uid,current_face_uid);
        current_face_uid += global_cells_faces_info[i].m_nb_back_face + global_cells_faces_info[i].m_nb_true_boundary_face;
      }
    }
  }

  cells_map.eachItem([&](Cell cell) {
    Int64 cell_uid = cell.uniqueId();
    Int32 cell_local_id = cell.localId();
    Integer num_local_face = 0;
    Integer num_true_boundary_face = 0;
    for( Face face : cell.faces() ){
      Int64 opposite_cell_uid = faces_opposite_cell_uid[face.localId()];
      Int64 face_new_uid = NULL_ITEM_UNIQUE_ID;
      if (face.backCell()==cell){
        if (!cells_first_face_uid.hasKey(cell_uid))
          fatal() << "NO KEY 0 for cell_uid=" << cell_uid;
        face_new_uid = cells_first_face_uid[cell_uid]+num_local_face;
        face.mutableItemBase().setOwner(my_rank,my_rank);
        ++num_local_face;
      }
      else if (face.nbCell()==1){
        if (opposite_cell_uid==NULL_ITEM_UNIQUE_ID){
          // Il s'agit d'une face frontière du domaine initial
          if (!cells_first_face_uid.hasKey(cell_uid))
            fatal() << "NO KEY 1 for cell_uid=" << cell_uid;
          face_new_uid = cells_first_face_uid[cell_uid] + my_cells_nb_back_face[cell_local_id] + num_true_boundary_face;
          ++num_true_boundary_face;
          face.mutableItemBase().setOwner(my_rank,my_rank);
        }
        else{
          if (!cells_first_face_uid.hasKey(opposite_cell_uid))
            fatal() << "NO KEY 1 for cell_uid=" << cell_uid << " opoosite=" << opposite_cell_uid;
          face_new_uid = cells_first_face_uid[opposite_cell_uid]+faces_opposite_cell_index[face.localId()];
          face.mutableItemBase().setOwner(faces_opposite_cell_owner[face.localId()],my_rank);
        }
      }
      if (face_new_uid!=NULL_ITEM_UNIQUE_ID){
        faces_new_uid[face.localId()] = face_new_uid;
        face.mutableItemBase().setUniqueId(face_new_uid);
      }
    }
  });

  // Vérifie que toutes les faces ont été réindéxées
  {
    Integer nb_error = 0;
    for( Integer i=0, is=nb_local_face; i<is; ++i ){
      if (faces_new_uid[i]==NULL_ITEM_UNIQUE_ID){
        ++nb_error;
        if (nb_error<10)
          error() << "The face lid=" << i << " has not been re-indexed.";
      }
    }
    if (nb_error!=0)
      ARCANE_FATAL("Some ({0}) faces have not been reindexed",nb_error);
  }

  if (is_verbose){
    OStringStream ostr;
    cells_map.eachItem([&](Cell cell) {
      Int64 cell_uid = cell.uniqueId().asInt64();
      Integer face_index = 0;
      for( Face face : cell.faces() ){
        Int64 opposite_cell_uid = faces_opposite_cell_uid[face.localId()];
        bool shared = false;
        bool true_boundary = false;
        bool internal_other = false;
        if (face.backCell()==cell){
        }
        else if (face.nbCell()==1){
          if (opposite_cell_uid==NULL_ITEM_ID)
            true_boundary = true;
          else
            shared = true;
        }
        else{
          internal_other = true;
          opposite_cell_uid = face.backCell().uniqueId().asInt64();
        }
        ostr() << "NEW UNIQUE ID FOR FACE"
               << " lid=" << face.localId()
               << " cell=" << cell_uid
               << " face=" << face.uniqueId()
               << " nbcell=" << face.nbCell()
               << " cellindex=" << face_index << " (";
        for( Node node : face.nodes() )
          ostr() << ' ' << node.uniqueId();
        ostr() << ")";
        if (internal_other)
          ostr() << " internal-other";
        if (true_boundary)
          ostr() << " true-boundary";
        if (opposite_cell_uid!=NULL_ITEM_ID){
          ostr() << " opposite " << opposite_cell_uid;
        }
        if (shared)
          ostr() << " (shared)";
        ostr() << "\n";
        ++face_index;
      }
    });
    info() << ostr.str();
    String file_name("faces_uid.");
    file_name = file_name + my_rank;    
    std::ofstream ofile(file_name.localstr());
    ofile << ostr.str();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * //COPIE DEPUIS GhostLayerBuilder.
 * Faire une classe unique.
 */
void FaceUniqueIdBuilder::
_exchangeData(IParallelExchanger* exchanger,BoundaryInfosMap& boundary_infos_to_send)
{
  for( BoundaryInfosMapEnumerator i_map(boundary_infos_to_send); ++i_map; ){
    Int32 sd = i_map.data()->key();
    exchanger->addSender(sd);
  }
  exchanger->initializeCommunicationsMessages();
  Integer nb_sender = exchanger->nbSender();
  Integer nb_receiver = exchanger->nbReceiver();
  info() << "NB_SEND=" << nb_sender << " NB_RECV=" << nb_receiver;
  Integer total = nb_sender+nb_receiver;
  Integer global_total = exchanger->parallelMng()->reduce(Parallel::ReduceSum,total);
  info() << "GLOBAL_NB_MESSAGE=" << global_total;

  {
    for( Integer i=0, ns=exchanger->nbSender(); i<ns; ++i ){
      ISerializeMessage* sm = exchanger->messageToSend(i);
      Int32 rank = sm->destination().value();
      ISerializer* s = sm->serializer();
      Int64ConstArrayView infos  = boundary_infos_to_send[rank];
      Integer nb_info = infos.size();
      s->setMode(ISerializer::ModeReserve);
      s->reserveInt64(1); // Pour le nombre d'elements
      s->reserveSpan(eBasicDataType::Int64,nb_info); // Pour les elements
      s->allocateBuffer();
      s->setMode(ISerializer::ModePut);
      s->putInt64(nb_info);
      s->putSpan(infos);
    }
  }
  exchanger->processExchange();
  debug() << "END EXCHANGE";
}

template<typename DataType>
class ItemInfoMultiList
{
 public:
 private:

  class MyInfo
  {
   public:
    MyInfo(const DataType& d,Integer n) : data(d), next_index(n) {}
   public:
    DataType data;
    Integer next_index;
  };

 public:
  ItemInfoMultiList() : m_last_index(5000,true) {}

 public:

  void add(Int64 node_uid,const DataType& data)
  {
    Integer current_index = m_values.size();

    bool is_add = false;
    HashTableMapT<Int64,Int32>::Data* d = m_last_index.lookupAdd(node_uid,-1,is_add);

    m_values.add(MyInfo(data,d->value()));
    d->value() = current_index;
  }

 public:
  UniqueArray<MyInfo> m_values;
  HashTableMapT<Int64,Int32> m_last_index;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \brief Calcul les numéros uniques de chaque face en parallèle V2.
  

  \note Cette version est utilisée pour test mais n'a jamais
  été mise en service et est maintenant remplacée par ...
*/ 
void FaceUniqueIdBuilder::
_computeFacesUniqueIdsParallelV2()
{
  IParallelMng* pm = m_mesh->parallelMng();
  Integer my_rank = pm->commRank();
  Integer nb_rank = pm->commSize();

  Integer nb_local_face = m_mesh_builder->oneMeshItemAdder()->nbFace();
  //Integer nb_local_cell = m_mesh_builder->nbCell();
  //bool is_verbose = m_mesh_builder->isVerbose();

  Int64UniqueArray faces_opposite_cell_uid(nb_local_face);
  faces_opposite_cell_uid.fill(NULL_ITEM_ID);
  IntegerUniqueArray faces_opposite_cell_index(nb_local_face);
  IntegerUniqueArray faces_opposite_cell_owner(nb_local_face);

  // Pour vérification, s'assure que tous les éléments de ce tableau
  // sont valides, ce qui signifie que toutes les faces ont bien été
  // renumérotés
  Int64UniqueArray faces_new_uid(nb_local_face);
  faces_new_uid.fill(NULL_ITEM_ID);

  Int64UniqueArray faces_infos;
  faces_infos.reserve(10000);
  ItemInternalMap& faces_map = m_mesh->facesMap();
  ItemInternalMap& nodes_map = m_mesh->nodesMap();


  // NOTE: ce tableau n'est pas utile sur toutes les mailles. Il
  // suffit qu'il contienne les mailles dont on a besoin, c'est à dire
  // les notres + celles connectées à une de nos faces.
  HashTableMapT<Int32,Int32> cell_first_face_uid(m_mesh_builder->oneMeshItemAdder()->nbCell()*2,true);
  
  // Rassemble les données des autres processeurs dans recv_cells;
  // Pour éviter que les tableaux ne soient trop gros, on procède en plusieurs
  // étapes.
  // Chaque sous-domaine construit sa liste de faces frontières, avec pour
  // chaque face:
  // - son type
  // - la liste de ses noeuds,
  // - le numéro unique de sa maille
  // - le propriétaire de sa maille
  // - son indice dans sa maille
  // Cette liste sera ensuite envoyée à tous les sous-domaines.
  ItemTypeMng* itm = m_mesh->itemTypeMng();

  // Détermine le unique id max des noeuds
  Int64 my_max_node_uid = NULL_ITEM_UNIQUE_ID;
  nodes_map.eachItem([&](Item node) {
    Int64 node_uid = node.uniqueId();
    if (node_uid>my_max_node_uid)
      my_max_node_uid = node_uid;
  });
  Int64 global_max_node_uid = pm->reduce(Parallel::ReduceMax,my_max_node_uid);
  debug() << "NODE_UID_INFO: MY_MAX_UID=" << my_max_node_uid
         << " GLOBAL=" << global_max_node_uid;
 
  //TODO: choisir bonne valeur pour initialiser la table
  BoundaryInfosMap boundary_infos_to_send(nb_rank,true);
  NodeUidToSubDomain uid_to_subdomain_converter(global_max_node_uid,nb_rank);
  info() << "NB_CORE modulo=" << uid_to_subdomain_converter.modulo();
  HashTableMapT<Int64,SharedArray<Int64> > nodes_info(100000,true);
  IItemFamily* node_family = m_mesh->nodeFamily();
  UniqueArray<bool> is_boundary_nodes(node_family->maxLocalId(),false);

  // Marque tous les noeuds frontieres car ce sont ceux qu'il faudra envoyer
  faces_map.eachItem([&](Face face) {
    Integer face_nb_cell = face.nbCell();
    if (face_nb_cell==1){
      for( Node node : face.nodes() )
        is_boundary_nodes[node.localId()] = true;
    }
  });

  // Détermine la liste des faces frontières
  faces_map.eachItem([&](Face face) {
    Node first_node = face.node(0);
    Int64 first_node_uid = first_node.uniqueId();
    SharedArray<Int64> v;
    Int32 dest_rank = -1;
    if (!is_boundary_nodes[first_node.localId()]){
      v = nodes_info.lookupAdd(first_node_uid)->value();
    }
    else{
      dest_rank = uid_to_subdomain_converter.uidToRank(first_node_uid);
      v = boundary_infos_to_send.lookupAdd(dest_rank)->value();
    }
    v.add(first_node_uid);     // 0
    v.add(my_rank);            // 1
    v.add(face.uniqueId());   // 2
    v.add(face.type());     // 3
    Cell back_cell = face.backCell();
    Cell front_cell = face.frontCell();
    if (back_cell.null())     // 4 : only used for debug
      v.add(NULL_ITEM_UNIQUE_ID);
    else
      v.add(back_cell.uniqueId());
    if (front_cell.null())    // 5 : only used for debug
      v.add(NULL_ITEM_UNIQUE_ID);
    else
      v.add(front_cell.uniqueId());
    for( Integer z=0, zs=face.nbNode(); z<zs; ++z )
      v.add(face.node(z).uniqueId());
  });

  // Positionne la liste des envoies
  Ref<IParallelExchanger> exchanger{ParallelMngUtils::createExchangerRef(pm)};
  _exchangeData(exchanger.get(),boundary_infos_to_send);

  {
    Integer nb_receiver = exchanger->nbReceiver();
    debug() << "NB RECEIVER=" << nb_receiver;
    Int64UniqueArray received_infos;
    for( Integer i=0; i<nb_receiver; ++i ){
      ISerializeMessage* sm = exchanger->messageToReceive(i);
      //Int32 orig_rank = sm->destSubDomain();
      ISerializer* s = sm->serializer();
      s->setMode(ISerializer::ModeGet);
      Int64 nb_info = s->getInt64();
      //info() << "RECEIVE NB_INFO=" << nb_info << " from=" << orig_rank;
      received_infos.resize(nb_info);
      s->getSpan(received_infos);
      //if ((nb_info % 3)!=0)
      //fatal() << "info size can not be divided by 3";
      Integer z =0; 
      while(z<nb_info){
        Int64 node_uid = received_infos[z+0];
        Int32 face_type = (Int32)received_infos[z+3];
        ItemTypeInfo* itt = itm->typeFromId(face_type);
        Integer face_nb_node = itt->nbLocalNode();
        Int64Array& a = nodes_info.lookupAdd(node_uid)->value();
        a.addRange(Int64ConstArrayView(6+face_nb_node,&received_infos[z]));
        z += 6;
        z += face_nb_node;
      }
    }
    Integer my_max_face_node = 0;
    for( HashTableMapT<Int64,SharedArray<Int64> >::Enumerator inode(nodes_info); ++inode; ){
      Int64ConstArrayView a = *inode;
      Integer nb_info = a.size();
      Integer z = 0;
      Integer node_nb_face = 0;
      while(z<nb_info){
        ++node_nb_face;
        Int32 face_type = (Int32)a[z+3];
        ItemTypeInfo* itt = itm->typeFromId(face_type);
        Integer face_nb_node = itt->nbLocalNode();
        z += 6;
        z += face_nb_node;
      }
      my_max_face_node = math::max(node_nb_face,my_max_face_node);
    }
    Integer global_max_face_node = pm->reduce(Parallel::ReduceMax,my_max_face_node);
    debug() << "GLOBAL MAX FACE NODE=" << global_max_face_node;
    // OK, maintenant donne comme uid de la face (node_uid * global_max_face_node + index)
    IntegerUniqueArray indexes;
    boundary_infos_to_send = BoundaryInfosMap(nb_rank,true);

    for( HashTableMapT<Int64,SharedArray<Int64> >::Enumerator inode(nodes_info); ++inode; ){
      Int64ConstArrayView a = *inode;
      Integer nb_info = a.size();
      Integer z = 0;
      Integer node_nb_face = 0;
      indexes.clear();
      while(z<nb_info){
        Int64 node_uid = a[z+0];
        Int32 sender_rank = (Int32)a[z+1];
        Int64 face_uid = a[z+2];
        Int32 face_type = (Int32)a[z+3];
        ItemTypeInfo* itt = itm->typeFromId(face_type);
        Integer face_nb_node = itt->nbLocalNode();

        // Regarde si la face est déjà dans la liste:
        Integer face_index = node_nb_face;
        Int32 face_new_owner = sender_rank;
        for( Integer y=0; y<node_nb_face; ++y ){
          if (memcmp(&a[indexes[y]+6],&a[z+6],sizeof(Int64)*face_nb_node)==0){
            face_index = y;
            face_new_owner = (Int32)a[indexes[y]+1];
          }
        }
        Int64 face_new_uid = (node_uid * global_max_face_node) + face_index;
        Int64Array& v = boundary_infos_to_send.lookupAdd(sender_rank)->value();
        // Indique au propriétaire de cette face son nouvel uid
        v.add(face_uid);
        v.add(face_new_uid);
        v.add(face_new_owner);
        indexes.add(z);
        z += 6;
        z += face_nb_node;
        ++node_nb_face;
      }
      my_max_face_node = math::max(node_nb_face,my_max_face_node);
    }
  }
  exchanger = ParallelMngUtils::createExchangerRef(pm);

  _exchangeData(exchanger.get(),boundary_infos_to_send);
  {
    Integer nb_receiver = exchanger->nbReceiver();
    debug() << "NB RECEIVER=" << nb_receiver;
    Int64UniqueArray received_infos;
    for( Integer i=0; i<nb_receiver; ++i ){
      ISerializeMessage* sm = exchanger->messageToReceive(i);
      ISerializer* s = sm->serializer();
      s->setMode(ISerializer::ModeGet);
      Int64 nb_info = s->getInt64();
      received_infos.resize(nb_info);
      s->getSpan(received_infos);
      if ((nb_info % 3)!=0)
        ARCANE_FATAL("info size can not be divided by 3 v={0}",nb_info);;
      Int64 nb_item = nb_info / 3;
      for (Int64 z=0; z<nb_item; ++z ){
        Int64 old_uid = received_infos[(z*3)];
        Int64 new_uid = received_infos[(z*3)+1];
        Int32 new_owner = (Int32)received_infos[(z*3)+2];
        impl::MutableItemBase face(faces_map.tryFind(old_uid));
        if (face.null())
          ARCANE_FATAL("Can not find own face uid={0}", old_uid);
        face.setUniqueId(new_uid);
        face.setOwner(new_owner, my_rank);
      }
    }
  }

  traceMng()->flush();
  pm->barrier();
  debug() << "END OF TEST NEW FACE COMPUTE";
  return;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \brief Calcul les numéros uniques de chaque face en séquentiel.
  
  \sa computeFacesUniqueIds()
*/  
void FaceUniqueIdBuilder::
_computeFacesUniqueIdsSequential()
{
  bool is_verbose = m_mesh_builder->isVerbose();
  
  ItemInternalMap& cells_map = m_mesh->cellsMap();

  // En séquentiel, les uniqueId() des mailles ne peuvent dépasser la
  // taille des Integers même en 32bits.
  Int32 max_uid = 0;
  cells_map.eachItem([&](Item cell) {
    Int32 cell_uid = cell.uniqueId().asInt32();
    if (cell_uid>max_uid)
      max_uid = cell_uid;
  });
  info() << "Max uid=" << max_uid;
  Integer nb_computed = max_uid + 1;
  Int32UniqueArray cell_first_face_uid(nb_computed,0);
  Int32UniqueArray cell_nb_num_back_face(nb_computed,0);
  Int32UniqueArray cell_true_boundary_face(nb_computed,0);

  cells_map.eachItem([&](Cell cell) {
    Int32 cell_uid = cell.uniqueId().asInt32();
    Integer nb_num_back_face = 0;
    Integer nb_true_boundary_face = 0;
    for( Face face : cell.faces() ){
      if (face.backCell()==cell)
        ++nb_num_back_face;
      else if (face.nbCell()==1){
        ++nb_true_boundary_face;
      }
    }
    cell_nb_num_back_face[cell_uid] = nb_num_back_face;
    cell_true_boundary_face[cell_uid] = nb_true_boundary_face;
  });

  Integer current_face_uid = 0;
  for( Integer i=0; i<nb_computed; ++i ){
    cell_first_face_uid[i] = current_face_uid;
    current_face_uid += cell_nb_num_back_face[i] + cell_true_boundary_face[i];
  }
  
  if (is_verbose){
    cells_map.eachItem([&](Item cell) {
      Int32 i = cell.uniqueId().asInt32();
      info() << "Recv: Cell FaceInfo celluid=" << i
             << " firstfaceuid=" << cell_first_face_uid[i]
             << " nbback=" << cell_nb_num_back_face[i]
             << " nbbound=" << cell_true_boundary_face[i];
    });
  }

  cells_map.eachItem([&](Cell cell) {
    Int32 cell_uid = cell.uniqueId().asInt32();
    Integer nb_num_back_face = 0;
    Integer nb_true_boundary_face = 0;
    for( Face face : cell.faces() ){
      Int64 face_new_uid = NULL_ITEM_UNIQUE_ID;
      if (face.backCell()==cell){
        face_new_uid = cell_first_face_uid[cell_uid] + nb_num_back_face;
        ++nb_num_back_face;
      }
      else if (face.nbCell()==1){
        face_new_uid = cell_first_face_uid[cell_uid] + cell_nb_num_back_face[cell_uid] + nb_true_boundary_face;
        ++nb_true_boundary_face;
      }
      if (face_new_uid!=NULL_ITEM_UNIQUE_ID){
        face.mutableItemBase().setUniqueId(face_new_uid);
      }
    }
  });

  if (is_verbose){
    OStringStream ostr;
    cells_map.eachItem([&](Cell cell) {
      Integer face_index = 0;
      for( Face face : cell.faces() ){
        Int64 opposite_cell_uid = NULL_ITEM_UNIQUE_ID;
        bool true_boundary = false;
        bool internal_other = false;
        if (face.backCell()==cell){
        }
        else if (face.nbCell()==1){
          true_boundary = true;
        }
        else{
          internal_other = true;
          opposite_cell_uid = face.backCell().uniqueId().asInt64();
        }
        ostr() << "NEW LOCAL ID FOR CELLFACE cell_uid=" << cell.uniqueId() << ' '
               << face_index << " uid=" << face.uniqueId() << " (";
        for( Node node : face.nodes() )
          ostr() << ' ' << node.uniqueId();
        ostr() << ")";
        if (internal_other)
          ostr() << " internal-other";
        if (true_boundary)
          ostr() << " true-boundary";
        if (opposite_cell_uid!=NULL_ITEM_ID)
          ostr() << " opposite " << opposite_cell_uid;
        ostr() << '\n';
        ++face_index;
      }
    });
    info() << ostr.str();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
