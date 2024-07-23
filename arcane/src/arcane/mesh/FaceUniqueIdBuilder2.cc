// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FaceUniqueIdBuilder2.cc                                     (C) 2000-2024 */
/*                                                                           */
/* Construction des identifiants uniques des faces.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/CheckedConvert.h"

#include "arcane/mesh/DynamicMesh.h"
#include "arcane/mesh/FaceUniqueIdBuilder.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/Timer.h"

#include "arcane/parallel/BitonicSortT.H"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Construction des uniqueId() des faces.
 *
 * Cette classe permet de calculer les uniqueId() des faces.
 * Après appel à computeFacesUniqueId(), les champs uniqueId() et owner()
 * de chaque face sont positionnés.
 *
 * Cette algorithme garanti que la numérotation est la même
 * indépendamment du découpage et du nombre de processeurs.
 * En séquentiel, l'algorithme peut s'ecrire comme suit:
 \code
 * Int64 face_unique_id_counter = 0;
 * // Parcours les mailles en supposant les uniqueId() croissants.
 * ENUMERATE_CELL(icell,allCells()){
 *   Cell cell = *icell;
 *   ENUMERATE_FACE(iface,cell.faces()){
 *    Face face = *iface;
 *    // Si je n'ai pas déjà un uniqueId(), en affecte un et incrémente le compteur
 *    if (face.uniqueId()==NULL_ITEM_UNIQUE_ID){
 *      face.setUniqueId(face_unique_id_counter);
 *      ++face_unique_id_counter;
 *    }
 *   }
 * }
 \endcode
 * L'algorithme séquentiel suppose qu'on parcourt les mailles dans l'ordre
 * croissant des uniqueId(). Pour une face donnée, c'est donc la maille
 * de plus petit uniqueId() qui va donner le uniqueId() de la face et
 * par la même le propriétaire de cette face.
 *
 * Cette version utilise un tri parallèle pour garantir que
 * le nombre de messages augmente en log2(N), avec N le nombre de processeurs.
 * Cela évite d'avoir potentiellement un grand nombre de messages, ce qui
 * n'est pas supporté par certaines implémentations MPI (par exemple MPC).
 */
class FaceUniqueIdBuilder2
: public TraceAccessor
{
 public:

  class NarrowCellFaceInfo;
  class WideCellFaceInfo;
  class AnyFaceInfo;
  class BoundaryFaceInfo;
  class ResendCellInfo;
  class AnyFaceBitonicSortTraits;
  class BoundaryFaceBitonicSortTraits;
  class UniqueIdSorter;

  // Choisir le bon typedef suivant le choix de la structure d'échange.
  typedef NarrowCellFaceInfo CellFaceInfo;

 public:

  using ItemInternalMap = DynamicMeshKindInfos::ItemInternalMap;
  using ItemInternalMapData = ItemInternalMap::Data;

 public:

  //! Construit une instance pour le maillage \a mesh
  explicit FaceUniqueIdBuilder2(DynamicMesh* mesh);

 public:

  void computeFacesUniqueIdAndOwnerVersion3();
  void computeFacesUniqueIdAndOwnerVersion5();

 private:

  DynamicMesh* m_mesh = nullptr;
  IParallelMng* m_parallel_mng = nullptr;
  bool m_is_verbose = false;

 private:

  void _resendCellsAndComputeFacesUniqueId(ConstArrayView<AnyFaceInfo> all_csi);
  void _checkFacesUniqueId();
  void _unsetFacesUniqueId();
  void _computeAndSortBoundaryFaces(Array<BoundaryFaceInfo>& boundary_faces_info);
  void _computeParallel();
  void _computeSequential();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Stocke les infos sur une face d'une maille.

 * Cette structure est utilisée lors du tri des faces. Comme le tri
 * est parallèle et afin de limiter la taille des messages envoyés,
 * il faut que la taille de cette structure soit la plus petite possible.
 * Pour cela, on suppose que les valeurs max ne sont pas atteignables.
 * Normalement, on a:
 * - cell_uid -> Int64
 * - rank -> Int32
 * - local_face_index -> Int32
 * On suppose en pratique les limites suivantes:
 * - cell_uid         -> 39 bits soit 250 milliards de mailles
 * - rank             -> 20 bits soit 1048576 PE
 * - local_face_index -> 5 bits soit 32 faces par maille
 *
 * Logiquement, ces limites ne seront pas atteintes avant un moment (on est
 * en 2012). Et lorsque ce sera le cas, il suffira d'utiliser la structure
 * WideCellFaceInfo en changeant le typedef qui va bien.
 *
 * On utilise donc un seul Int64, avec les 39 premiers bits pour le uid,
 * les 20 suivantes pour le rang et les 5 derniers pour le local_index.
 * A noter que pour éviter des problèmes de signe, on stocke la valeur donnée
 * plus 1.
 */
class FaceUniqueIdBuilder2::NarrowCellFaceInfo
{
 public:
  static const Int64 BITS_CELL_UID = 39;
  static const Int64 BITS_RANK = 20;
  static const Int64 BITS_INDEX = 5;
  static const Int64 ONE_INT64 = 1;
  static const Int64 MASK_CELL_UID = (ONE_INT64 << BITS_CELL_UID) - 1;
  static const Int64 MASK_RANK = ((ONE_INT64 << BITS_RANK) - 1) << BITS_CELL_UID;
  static const Int64 MASK_INDEX = ((ONE_INT64 << BITS_INDEX) - 1) << (BITS_CELL_UID+BITS_RANK);
  
 public:
  NarrowCellFaceInfo()
  {
    setValue(NULL_ITEM_UNIQUE_ID,-1,-1);
  }
 public:

  bool isMaxValue() const
  {
    Int64 max_id = (MASK_CELL_UID - 1);
    return cellUid()==max_id;
  }

  void setMaxValue()
  {
    Int64 max_id = (MASK_CELL_UID - 1);
    setValue(max_id,-1,-1);
  }

  void setValue(Int64 cell_uid,Int32 _rank,Int32 face_local_index)
  {
    Int64 v_fli = face_local_index+1;
    Int64 v_rank = _rank+1;
    Int64 v_uid = cell_uid+1;
    m_value = v_fli << (BITS_CELL_UID+BITS_RANK);
    m_value += v_rank << (BITS_CELL_UID);
    m_value += v_uid;
    if (cellUid()!=cell_uid)
      ARCANE_FATAL("Bad uid expected='{0}' computed='{1}' v={2}",cell_uid,cellUid(),m_value);
    if (rank()!=_rank)
      ARCANE_FATAL("Bad rank expected='{0}' computed='{1}'",_rank,rank());
    if (faceLocalIndex()!=face_local_index)
      ARCANE_FATAL("Bad local_index expected='{0}' computed='{1}'",face_local_index,faceLocalIndex());
  }
  Int64 cellUid() const { return (m_value & MASK_CELL_UID) - 1; }
  Int32 rank() const { return CheckedConvert::toInt32( ((m_value & MASK_RANK) >> BITS_CELL_UID) - 1 ); }
  Int32 faceLocalIndex() const { return CheckedConvert::toInt32( ((m_value & MASK_INDEX) >> (BITS_CELL_UID+BITS_RANK)) - 1 ); }

  bool isValid() const { return cellUid()!=NULL_ITEM_UNIQUE_ID; }
    
 private:

  Int64 m_value = -1;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Stocke les infos sur une face d'une maille.
 */
class FaceUniqueIdBuilder2::WideCellFaceInfo
{
 public:
  WideCellFaceInfo() : m_cell_uid(NULL_ITEM_UNIQUE_ID), m_rank(-1), m_face_local_index(-1){}
 public:
  bool isMaxValue() const
  {
    Int64 max_id = INT64_MAX;
    return cellUid()==max_id;
  }

  void setMaxValue()
  {
    Int64 max_id = INT64_MAX;
    setValue(max_id,-1,-1);
  }
  void setValue(Int64 cell_uid,Int32 rank,Int32 face_local_index)
  {
    m_cell_uid = cell_uid;
    m_rank = rank;
    m_face_local_index = face_local_index;
  }
  Int64 cellUid() const { return m_cell_uid; }
  Int32 rank() const { return m_rank; }
  Int32 faceLocalIndex() const { return m_face_local_index; }
  bool isValid() const { return m_cell_uid!=NULL_ITEM_UNIQUE_ID; }
    
 private:

  Int64 m_cell_uid;
  Int32 m_rank;
  Int32 m_face_local_index;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Infos pour gérer les faces frontières des sous-domaines.
 *
 * Quel que soit le type de la face, une faces peut être déterminée
 * de manière unique par ses trois premiers noeuds.
 */
class FaceUniqueIdBuilder2::BoundaryFaceInfo
{
 public:
  BoundaryFaceInfo() 
  : m_node0_uid(NULL_ITEM_UNIQUE_ID), m_node1_uid(NULL_ITEM_UNIQUE_ID),
    m_node2_uid(NULL_ITEM_UNIQUE_ID), m_cell_uid(NULL_ITEM_UNIQUE_ID),
    m_rank(-1), m_face_local_index(-1)
  {}
  bool hasSameNodes(const BoundaryFaceInfo& fsi) const
  {
    return fsi.m_node0_uid==m_node0_uid && fsi.m_node1_uid==m_node1_uid
    && fsi.m_node2_uid==m_node2_uid;
  }
  void setNodes(Face face)
  {
    Integer nb_node = face.nbNode();
    if (nb_node>=1)
      m_node0_uid = face.node(0).uniqueId();
    if (nb_node>=2)
      m_node1_uid = face.node(1).uniqueId();
    if (nb_node>=3)
      m_node2_uid = face.node(2).uniqueId();
  }
 public:
  Int64 m_node0_uid;
  Int64 m_node1_uid;
  Int64 m_node2_uid;
  Int64 m_cell_uid;
  Int32 m_rank;
  Int32 m_face_local_index;
 public:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Infos pour gérer les faces des sous-domaines.
 *
 * Une instance de cette classe contient pour une face du maillage
 * les infos sur ces deux mailles attachées. Pour chaque maille, on
 * stocke le uniqueId(), le propriétaire et le numéro local de la face
 */
class FaceUniqueIdBuilder2::AnyFaceInfo
{
 public:

  AnyFaceInfo() = default;

 public:
  void setCell0(Int64 uid,Int32 rank,Int32 face_local_index)
  {
    m_cell0.setValue(uid,rank,face_local_index);
  }
  void setCell1(Int64 uid,Int32 rank,Int32 face_local_index)
  {
    m_cell1.setValue(uid,rank,face_local_index);
  }
 public:
  CellFaceInfo m_cell0;
  CellFaceInfo m_cell1;
 public:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Attention, cette classe doit avoir une taille multiple de Int64
class FaceUniqueIdBuilder2::ResendCellInfo
{
 public:
  Int64 m_cell_uid;
  // Ce champs contient à la fois le numéro local de la face dans la maille
  // et le rang du propriétaire de la maille.
  // m_face_local_index_and_owner_rank / nb_rank -> face_index
  // m_face_local_index_and_owner_rank % nb_rank -> owner_rank
  Int32 m_face_local_index_and_owner_rank;
  Int32 m_index_in_rank_list;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonctor pour trier les BoundaryFaceInfo via le tri bitonic.
 */
class FaceUniqueIdBuilder2::BoundaryFaceBitonicSortTraits
{
 public:
  static bool compareLess(const BoundaryFaceInfo& k1,const BoundaryFaceInfo& k2)
  {
    if (k1.m_node0_uid<k2.m_node0_uid)
      return true;
    if (k1.m_node0_uid>k2.m_node0_uid)
      return false;

    // ke.node0_uid == k2.node0_uid
    if (k1.m_node1_uid<k2.m_node1_uid)
      return true;
    if (k1.m_node1_uid>k2.m_node1_uid)
      return false;

    // ke.node1_uid == k2.node1_uid
    if (k1.m_node2_uid<k2.m_node2_uid)
      return true;
    if (k1.m_node2_uid>k2.m_node2_uid)
      return false;

    // ke.node2_uid == k2.node2_uid
    return (k1.m_cell_uid<k2.m_cell_uid);
  }

  static Parallel::Request send(IParallelMng* pm,Int32 rank,ConstArrayView<BoundaryFaceInfo> values)
  {
    const BoundaryFaceInfo* fsi_base = values.data();
    return pm->send(ByteConstArrayView(messageSize(values),(const Byte*)fsi_base),rank,false);
  }
  static Parallel::Request recv(IParallelMng* pm,Int32 rank,ArrayView<BoundaryFaceInfo> values)
  {
    BoundaryFaceInfo* fsi_base = values.data();
    return pm->recv(ByteArrayView(messageSize(values),(Byte*)fsi_base),rank,false);
  }
  static Integer messageSize(ConstArrayView<BoundaryFaceInfo> values)
  {
    return CheckedConvert::toInteger( values.size()*sizeof(BoundaryFaceInfo) );
  }
  static BoundaryFaceInfo maxValue()
  {
    BoundaryFaceInfo fsi;
    fsi.m_cell_uid = INT64_MAX;
    fsi.m_rank = INT32_MAX;
    fsi.m_node0_uid = INT64_MAX;
    fsi.m_node1_uid = INT64_MAX;
    fsi.m_node2_uid = INT64_MAX;
    return fsi;
  }
  static bool isValid(const BoundaryFaceInfo& fsi)
  {
    return fsi.m_cell_uid!=INT64_MAX;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonctor pour trier les AnyFaceInfo via le tri bitonic.
 *
 * Le but est de trier la liste pour que les mailles de plus petit
 * uniqueId() soient en premier et pour une même maille, le plus
 * petit \a face_local_index en premier.
 *
 * Seul les infos de la première maille de AnyFaceInfo est utilisée
 * pour le tri (la seconde maille sert uniquement à renvoyer les infos
 * au processeurs concerné).
 */
class FaceUniqueIdBuilder2::AnyFaceBitonicSortTraits
{
 public:
  static bool compareLess(const AnyFaceInfo& k1,const AnyFaceInfo& k2)
  {
    Int64 k1_cell0_uid = k1.m_cell0.cellUid();
    Int64 k2_cell0_uid = k2.m_cell0.cellUid();
    if (k1_cell0_uid<k2_cell0_uid)
      return true;
    if (k1_cell0_uid>k2_cell0_uid)
      return false;

    Int64 k1_face0_local_index = k1.m_cell0.faceLocalIndex();
    Int64 k2_face0_local_index = k2.m_cell0.faceLocalIndex();
    if (k1_face0_local_index<k2_face0_local_index)
      return true;
    if (k1_face0_local_index>k2_face0_local_index)
      return false;

    return (k1.m_cell1.cellUid()<k2.m_cell1.cellUid());
  }

  static Parallel::Request send(IParallelMng* pm,Int32 rank,ConstArrayView<AnyFaceInfo> values)
  {
    const AnyFaceInfo* fsi_base = values.data();
    Integer message_size = CheckedConvert::toInteger(values.size()*sizeof(AnyFaceInfo));
    return pm->send(ByteConstArrayView(message_size,(const Byte*)fsi_base),rank,false);
  }

  static Parallel::Request recv(IParallelMng* pm,Int32 rank,ArrayView<AnyFaceInfo> values)
  {
    AnyFaceInfo* fsi_base = values.data();
    Integer message_size = CheckedConvert::toInteger(values.size()*sizeof(AnyFaceInfo));
    return pm->recv(ByteArrayView(message_size,(Byte*)fsi_base),rank,false);
  }

  static AnyFaceInfo maxValue()
  {
    AnyFaceInfo csi;
    csi.m_cell0.setMaxValue();
    csi.m_cell1.setMaxValue();
    return csi;
  }

  static bool isValid(const AnyFaceInfo& csi)
  {
    return !csi.m_cell0.isMaxValue();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class FaceUniqueIdBuilder2::UniqueIdSorter
{
 public:
  bool operator()(ItemInternal* i1,ItemInternal* i2) const
  {
    return i1->uniqueId() < i2->uniqueId();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FaceUniqueIdBuilder2::
FaceUniqueIdBuilder2(DynamicMesh* mesh)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
, m_parallel_mng(mesh->parallelMng())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 *\brief Calcul les numéros uniques de chaque face en parallèle.
 */  
void FaceUniqueIdBuilder2::
computeFacesUniqueIdAndOwnerVersion3()
{
  if (m_parallel_mng->isParallel())
    _computeParallel();
  else
    _computeSequential();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 *\brief Calcul les numéros uniques de chaque face en sequentiel.
 */  
void FaceUniqueIdBuilder2::
_computeSequential()
{
  info() << "Compute FacesUniqueId() Sequential V3";

  //TODO: permettre de ne pas commencer a zero.
  Int64 face_unique_id_counter = 0;

  ItemInternalMap& cells_map = m_mesh->cellsMap();
  Integer nb_cell = cells_map.count();
  UniqueArray<ItemInternal*> cells;
  cells.reserve(nb_cell);
  // D'abord, il faut trier les mailles par leur uniqueId()
  // en ordre croissant
  ENUMERATE_ITEM_INTERNAL_MAP_DATA(iid,cells_map){
    cells.add(iid->value());
  }
  std::sort(std::begin(cells),std::end(cells),UniqueIdSorter());

  // Invalide les uid pour être certain qu'ils seront tous positionnés.
  _unsetFacesUniqueId();

  for( Integer i=0; i<nb_cell; ++i ){    
    Cell cell = cells[i];
    for( Face face : cell.faces()){
      if (face.uniqueId()==NULL_ITEM_UNIQUE_ID){
        face.mutableItemBase().setUniqueId(face_unique_id_counter);
        ++face_unique_id_counter;
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 *\brief Calcul les numéros uniques de chaque face en parallèle.
 */  
void FaceUniqueIdBuilder2::
_computeParallel()
{
  IParallelMng* pm = m_parallel_mng;
  Int32 my_rank = pm->commRank();

  bool is_verbose = m_is_verbose;

  ItemInternalMap& cells_map = m_mesh->cellsMap();

  info() << "Compute FacesUniqueId() V3 using parallel sort";

  // Calcule et trie pour les faces frontières
  UniqueArray<BoundaryFaceInfo> boundary_faces_info;
  _computeAndSortBoundaryFaces(boundary_faces_info);

  // Ici, les faces de bord sont triées en fonction de leur noeuds.
  // Normalement, dans cette liste, 2 éléments consécutifs BoundaryFaceInfo qui
  // ont les même noeuds représentent la même face. Dans ce cas, on génère
  // un AnyFaceInfo avec les infos des deux mailles issus de ces deux éléments de
  // la liste en faisant bien attention de mettre en premier la maille
  // de plus petit uniqueId().
  // Si deux éléments consécutifs de la liste n'ont pas les mêmes noeuds, cela
  // signifie que la face est au bord du domaine global.
  // Il faut tout de même traiter le cas des deux éléments consécutifs de la liste
  // qui se trouvent sur des processeurs différents. Pour gérer ce cas, chaque
  // processeur envoie au suivant le dernier élément de sa liste s'il ne peut pas
  // être fusionné avec l'avant dernier, dans l'espoir qu'il pourra l'être avec le
  // premier de la liste du processeur suivant.
  UniqueArray<AnyFaceInfo> all_face_list;
  {
    ConstArrayView<BoundaryFaceInfo> all_fsi = boundary_faces_info;
    Integer n = all_fsi.size();
    bool is_last_already_done = false;
    for( Integer i=0; i<n; ++i ){
      const BoundaryFaceInfo& fsi = all_fsi[i];
      Int64 cell_uid0 = fsi.m_cell_uid;
      bool is_inside = false;
      //TODO: traiter le cas si la maille d'avant est sur un autre proc
      // Pour cela, il faut recupérer la dernière valeur du proc précédent
      // et regarder si elle correspond à notre première valeur
      is_inside = ((i+1)!=n && fsi.hasSameNodes(all_fsi[i+1]));
      if (is_last_already_done){
        is_last_already_done = false;
      }
      else{
        AnyFaceInfo csi;
        if (is_inside){
          const BoundaryFaceInfo& next_fsi = all_fsi[i+1];
          Int64 cell_uid1 = next_fsi.m_cell_uid;
          if (cell_uid0<cell_uid1){
            csi.setCell0(cell_uid0,fsi.m_rank,fsi.m_face_local_index);
            csi.setCell1(cell_uid1,next_fsi.m_rank,next_fsi.m_face_local_index);
          }
          else{
            csi.setCell0(cell_uid1,next_fsi.m_rank,next_fsi.m_face_local_index);
            csi.setCell1(cell_uid0,fsi.m_rank,fsi.m_face_local_index);
          }
          is_last_already_done = true;
        }
        else{
          csi.setCell0(cell_uid0,fsi.m_rank,fsi.m_face_local_index);
        }
        all_face_list.add(csi);
      }
      if (is_verbose)
        info() << "FACES_KEY i=" << i
               << " n0=" << fsi.m_node0_uid
               << " n1=" << fsi.m_node1_uid
               << " n2=" << fsi.m_node2_uid
               << " cell=" << fsi.m_cell_uid
               << " rank=" << fsi.m_rank
               << " li=" << fsi.m_face_local_index
               << " in=" << is_inside;
    }
  }

  // Ajoute les faces propres a notre sous-domaine.
  // Il s'agit de toutes les faces qui ont 2 mailles connectées.
  ENUMERATE_ITEM_INTERNAL_MAP_DATA(iid,cells_map){
    Cell cell(iid->value());
    Integer cell_nb_face = cell.nbFace();
    Int64 cell_uid = cell.uniqueId();
    for( Integer z=0; z<cell_nb_face; ++z ){
      Face face = cell.face(z);
      if (face.nbCell()!=2)
        continue;
      Cell cell0 = face.cell(0);
      Cell cell1 = face.cell(1);
      Cell next_cell = (cell0==cell) ? cell1 : cell0;
      Int64 next_cell_uid = next_cell.uniqueId();
      // N'enregistre que si je suis la maille de plus petit uid
      if (cell_uid<next_cell_uid){
        AnyFaceInfo csi;
        csi.m_cell0.setValue(cell_uid,my_rank,z);
        // Le face_local_index de la maille 1 ne sera pas utilisé
        csi.m_cell1.setValue(next_cell_uid,my_rank,-1);
        all_face_list.add(csi);
      }
    }
  }

  if (is_verbose){
    Integer n = all_face_list.size();
    for( Integer i=0; i<n; ++i ){
      const AnyFaceInfo& csi = all_face_list[i];
      info() << "CELL_TO_SORT i=" << i
             << " cell0=" << csi.m_cell0.cellUid()
             << " lidx0=" << csi.m_cell0.faceLocalIndex()
             << " cell1=" << csi.m_cell1.cellUid();
    }
  }

  info() << "ALL_FACE_LIST memorysize=" << sizeof(AnyFaceInfo)*all_face_list.size();
  Parallel::BitonicSort<AnyFaceInfo,AnyFaceBitonicSortTraits> all_face_sorter(pm);
  all_face_sorter.setNeedIndexAndRank(false);
  Real sort_begin_time = platform::getRealTime();
  all_face_sorter.sort(all_face_list);
  Real sort_end_time = platform::getRealTime();
  info() << "END_ALL_FACE_SORTER time=" << (Real)(sort_end_time - sort_begin_time);

  _resendCellsAndComputeFacesUniqueId(all_face_sorter.keys());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Détermine la liste des faces frontières de chaque sous-domaine
 * et les trie sur tous les procs.
 */
void FaceUniqueIdBuilder2::
_computeAndSortBoundaryFaces(Array<BoundaryFaceInfo>& boundary_faces_info)
{
  IParallelMng* pm = m_parallel_mng;
  Int32 my_rank = pm->commRank();
  bool is_verbose = m_is_verbose;
  ItemInternalMap& faces_map = m_mesh->facesMap();

  Parallel::BitonicSort<BoundaryFaceInfo,BoundaryFaceBitonicSortTraits> boundary_face_sorter(pm);
  boundary_face_sorter.setNeedIndexAndRank(false);

  //UniqueArray<BoundaryFaceInfo> boundary_face_list;
  boundary_faces_info.clear();
  ENUMERATE_ITEM_INTERNAL_MAP_DATA(iid,faces_map){
    Face face(iid->value());
    BoundaryFaceInfo fsi;
    Integer nb_cell = face.nbCell();
    if (nb_cell==2)
      continue;

    fsi.m_rank = my_rank;
    fsi.setNodes(face);
    Cell cell = face.cell(0);
    fsi.m_cell_uid = cell.uniqueId();
    Integer face_local_index = 0;
    for( Integer z=0, zs=cell.nbFace(); z<zs; ++z )
      if (cell.face(z)==face){
        face_local_index = z;
        break;
      }
    fsi.m_face_local_index = face_local_index;
    boundary_faces_info.add(fsi);
  }

  if (is_verbose){
    ConstArrayView<BoundaryFaceInfo> all_fsi = boundary_faces_info;
    Integer n = all_fsi.size();
    for( Integer i=0; i<n; ++i ){
      const BoundaryFaceInfo& fsi = all_fsi[i];
      info() << "KEY i=" << i
             << " n0=" << fsi.m_node0_uid
             << " n1=" << fsi.m_node1_uid
             << " n2=" << fsi.m_node2_uid
             << " cell=" << fsi.m_cell_uid
             << " rank=" << fsi.m_rank
             << " li=" << fsi.m_face_local_index;
    }
  }

  Real sort_begin_time = platform::getRealTime();
  boundary_face_sorter.sort(boundary_faces_info);
  Real sort_end_time = platform::getRealTime();
  info() << "END_BOUNDARY_FACE_SORT time=" << (Real)(sort_end_time - sort_begin_time);

  {
    ConstArrayView<BoundaryFaceInfo> all_bfi = boundary_face_sorter.keys();
    Integer n = all_bfi.size();
    if (is_verbose){
      for( Integer i=0; i<n; ++i ){
        const BoundaryFaceInfo& bfi = all_bfi[i];
        info() << " AFTER KEY i=" << i
               << " n0=" << bfi.m_node0_uid
               << " n1=" << bfi.m_node1_uid
               << " n2=" << bfi.m_node2_uid
               << " cell=" << bfi.m_cell_uid
               << " rank=" << bfi.m_rank
               << " li=" << bfi.m_face_local_index;
      }
    }

    // Comme un même noeud peut être présent dans la liste du proc précédent, chaque PE
    // (sauf le 0) envoie au proc précédent le début sa liste qui contient les même noeuds.
    
    // TODO: fusionner ce code avec celui de GhostLayerBuilder2
    UniqueArray<BoundaryFaceInfo> end_face_list;
    Integer begin_own_list_index = 0;
    if (n!=0 && my_rank!=0){
      if (BoundaryFaceBitonicSortTraits::isValid(all_bfi[0])){
        Int64 node0_uid = all_bfi[0].m_node0_uid;
        for( Integer i=0; i<n; ++i ){
          if (all_bfi[i].m_node0_uid!=node0_uid){
            begin_own_list_index = i;
            break;
          }
          else
            end_face_list.add(all_bfi[i]);
        }
      }
    }
    info() << "BEGIN_OWN_LIST_INDEX=" << begin_own_list_index;
    if (is_verbose){
      for( Integer k=0, kn=end_face_list.size(); k<kn; ++k )
        info() << " SEND n0=" << end_face_list[k].m_node0_uid
               << " n1=" << end_face_list[k].m_node1_uid
               << " n2=" << end_face_list[k].m_node2_uid;
    }

    UniqueArray<BoundaryFaceInfo> end_face_list_recv;

    UniqueArray<Parallel::Request> requests;
    Integer recv_message_size = 0;
    Integer send_message_size = BoundaryFaceBitonicSortTraits::messageSize(end_face_list);

    Int32 nb_rank = pm->commSize();

    // Envoie et réceptionne d'abord les tailles.
    if (my_rank!=(nb_rank-1)){
      requests.add(pm->recv(IntegerArrayView(1,&recv_message_size),my_rank+1,false));
    }
    if (my_rank!=0){
      requests.add(pm->send(IntegerConstArrayView(1,&send_message_size),my_rank-1,false));
    }
    
    pm->waitAllRequests(requests);
    requests.clear();
    
    if (recv_message_size!=0){
      Integer message_size = CheckedConvert::toInteger(recv_message_size/sizeof(BoundaryFaceInfo));
      end_face_list_recv.resize(message_size);
      requests.add(BoundaryFaceBitonicSortTraits::recv(pm,my_rank+1,end_face_list_recv));
    }
    if (send_message_size!=0)
      requests.add(BoundaryFaceBitonicSortTraits::send(pm,my_rank-1,end_face_list));

    pm->waitAllRequests(requests);

    boundary_faces_info.clear();
    boundary_faces_info.addRange(all_bfi.subConstView(begin_own_list_index,n-begin_own_list_index));
    boundary_faces_info.addRange(end_face_list_recv);
  }

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceUniqueIdBuilder2::
_resendCellsAndComputeFacesUniqueId(ConstArrayView<AnyFaceInfo> all_csi)
{
  IParallelMng* pm = m_parallel_mng;
  Int32 nb_rank = pm->commSize();
  Int32 my_rank = pm->commRank();
  bool is_verbose = m_is_verbose;

  ItemInternalMap& cells_map = m_mesh->cellsMap();

  Int64 nb_computed_face = all_csi.size();

  if (is_verbose){
    for( Integer i=0; i<nb_computed_face; ++i ){
      const AnyFaceInfo& csi = all_csi[i];
      info() << "CELLS_KEY i=" << i
             << " cell0=" << csi.m_cell0.cellUid()
             << " lidx0=" << csi.m_cell0.faceLocalIndex()
             << " cell1=" << csi.m_cell1.cellUid()
             << " lidx1=" << csi.m_cell1.faceLocalIndex()
             << " rank0=" << csi.m_cell0.rank()
             << " rank1=" << csi.m_cell1.rank();
    }
  }

  // Calcul pour chaque rang le nombre de valeurs à envoyer
  // et le stocke dans nb_info_to_send;
  IntegerUniqueArray nb_info_to_send(nb_rank,0);
  {
    for( Integer i=0; i<nb_computed_face; ++i ){
      const AnyFaceInfo& csi = all_csi[i];
      Int32 rank0 = csi.m_cell0.rank();
      Int32 rank1 = csi.m_cell1.rank();

      ++nb_info_to_send[rank0];

      // Il ne faut l'envoyer que si le rang est différent de m_rank0
      if (csi.m_cell1.isValid() && rank1!=rank0)
        ++nb_info_to_send[rank1];
    }
  }

  // Tableau pour chaque proc indiquant le uniqueId() de la première
  // face de ce proc.
  Int64UniqueArray all_first_face_uid(nb_rank);
  {
    // Chaque proc récupère le nombre de mailles dans la liste.
    // Comme cette liste sera triée, cela correspond aux uid de la première
    // face de ce proc.
    Int64 nb_cell_to_sort = all_csi.size();
    pm->allGather(Int64ConstArrayView(1,&nb_cell_to_sort),all_first_face_uid);

    Int64 to_add = 0;
    for( Integer i=0; i<nb_rank; ++i ){
      Int64 next = all_first_face_uid[i];
      all_first_face_uid[i] = to_add;
      to_add += next;
    }
  }

  Integer total_nb_to_send = 0;
  IntegerUniqueArray nb_info_to_send_indexes(nb_rank,0);
  for( Integer i=0; i<nb_rank; ++i ){
    nb_info_to_send_indexes[i] = total_nb_to_send;
    total_nb_to_send += nb_info_to_send[i];
  }
  info() << "TOTAL_NB_TO_SEND=" << total_nb_to_send;

  UniqueArray<ResendCellInfo> resend_infos(total_nb_to_send);
  {
    for( Integer i=0; i<nb_computed_face; ++i ){
      const AnyFaceInfo& csi = all_csi[i];
      Int32 rank0 = csi.m_cell0.rank();
      Int32 rank1 = csi.m_cell1.rank();

      ResendCellInfo& rci0 = resend_infos[nb_info_to_send_indexes[rank0]];
      rci0.m_cell_uid = csi.m_cell0.cellUid();
      rci0.m_face_local_index_and_owner_rank = (csi.m_cell0.faceLocalIndex() * nb_rank) + rank0;
      rci0.m_index_in_rank_list = i;
      ++nb_info_to_send_indexes[rank0];

      if (csi.m_cell1.isValid() && rank1!=rank0){
        ResendCellInfo& rci1 = resend_infos[nb_info_to_send_indexes[rank1]];
        rci1.m_cell_uid = csi.m_cell1.cellUid();
        // Meme si je suis la maille 1, le proprietaire de la face sera la maille 0.
        rci1.m_face_local_index_and_owner_rank = (csi.m_cell1.faceLocalIndex() * nb_rank) + rank0;
        rci1.m_index_in_rank_list = i;
        ++nb_info_to_send_indexes[rank1];
      }

    }
  }

  // Faire un seul reduce
  Int64 total_nb_computed_face = pm->reduce(Parallel::ReduceSum,nb_computed_face);
  info() << "TOTAL_NB_COMPUTED_FACE=" << total_nb_computed_face;
  
  // Indique a chaque PE combien d'infos je vais lui envoyer
  if (is_verbose)
    for( Integer i=0; i<nb_rank; ++i )
      info() << "NB_TO_SEND: I=" << i << " n=" << nb_info_to_send[i];

  IntegerUniqueArray nb_info_to_recv(nb_rank,0);
  {
    Timer::SimplePrinter sp(traceMng(),"Sending size with AllToAll");
    pm->allToAll(nb_info_to_send,nb_info_to_recv,1);
  }

  if (is_verbose)
    for( Integer i=0; i<nb_rank; ++i )
      info() << "NB_TO_RECV: I=" << i << " n=" << nb_info_to_recv[i];

  Integer total_nb_to_recv = 0;
  for( Integer i=0; i<nb_rank; ++i )
    total_nb_to_recv +=  nb_info_to_recv[i];

  // Il y a de fortes chances que cela ne marche pas si le tableau est trop grand,
  // il faut proceder avec des tableaux qui ne depassent pas 2Go a cause des
  // Int32 de MPI.
  // TODO: Faire le AllToAll en plusieurs fois si besoin.
  UniqueArray<ResendCellInfo> recv_infos;
  {
    Int32 vsize = sizeof(ResendCellInfo) / sizeof(Int64);
    Int32UniqueArray send_counts(nb_rank);
    Int32UniqueArray send_indexes(nb_rank);
    Int32UniqueArray recv_counts(nb_rank);
    Int32UniqueArray recv_indexes(nb_rank);
    Int32 total_send = 0;
    Int32 total_recv = 0;
    for( Integer i=0; i<nb_rank; ++i ){
      send_counts[i] = (Int32)(nb_info_to_send[i] * vsize);
      recv_counts[i] = (Int32)(nb_info_to_recv[i] * vsize);
      send_indexes[i] = total_send;
      recv_indexes[i] = total_recv;
      total_send += send_counts[i];
      total_recv += recv_counts[i];
    }
    recv_infos.resize(total_nb_to_recv);

    Int64ConstArrayView send_buf(total_nb_to_send*vsize,(Int64*)resend_infos.data());
    Int64ArrayView recv_buf(total_nb_to_recv*vsize,(Int64*)recv_infos.data());

    info() << "BUF_SIZES: send=" << send_buf.size() << " recv=" << recv_buf.size();
    {
      Timer::SimplePrinter sp(traceMng(),"Send values with AllToAll");
      pm->allToAllVariable(send_buf,send_counts,send_indexes,recv_buf,recv_counts,recv_indexes);
    }
  }

  // Invalide les uid pour être certain qu'ils seront tous positionnés.
  _unsetFacesUniqueId();

  if (is_verbose){
    Integer index = 0;
    for( Int32 rank=0; rank<nb_rank; ++rank ){
      for( Integer z=0, zs=nb_info_to_recv[rank]; z<zs; ++z ){
        const ResendCellInfo& rci = recv_infos[index];
        ++index;

        Int64 cell_uid = rci.m_cell_uid;
        Int32 full_local_index = rci.m_face_local_index_and_owner_rank;
        Int32 face_local_index = full_local_index / nb_rank;
        Int32 owner_rank = full_local_index % nb_rank;
        Int64 face_uid = all_first_face_uid[rank] + rci.m_index_in_rank_list;
        info() << "RECV index=" << index << " uid=" << cell_uid
               << " local_idx=" << full_local_index
               << " face_local_idx=" << face_local_index
               << " owner_rank=" << owner_rank
               << " rank_idx=" << rci.m_index_in_rank_list
               << " rank="<< rank
               << " first_face_uid=" << all_first_face_uid[rank]
               << " computed_uid=" << face_uid;
      }
    }
  }

  // Positionne les uniqueId() et les owner() des faces.
  {
    Integer index = 0;
    for( Int32 i=0; i<nb_rank; ++i ){
      Int32 rank = i;
      for( Integer z=0, zs=nb_info_to_recv[i]; z<zs; ++z ){
        const ResendCellInfo& rci = recv_infos[index];
        ++index;

        Int64 cell_uid = rci.m_cell_uid;
        Int32 full_local_index = rci.m_face_local_index_and_owner_rank;
        Int32 face_local_index = full_local_index / nb_rank;
        Int32 owner_rank = full_local_index % nb_rank;
        
        ItemInternalMapData* cell_data = cells_map.lookup(cell_uid);
        if (!cell_data)
          ARCANE_FATAL("Can not find cell data for '{0}'",cell_uid);
        Cell cell(cell_data->value());
        Face face = cell.face(face_local_index);
        Int64 face_uid = all_first_face_uid[rank] + rci.m_index_in_rank_list;
        face.mutableItemBase().setUniqueId(face_uid);
        face.mutableItemBase().setOwner(owner_rank,my_rank);
      }
    }
  }

  // Vérifie que toutes les faces ont un uid valide
  _checkFacesUniqueId();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 *\brief Calcule les uniqueId() via un hash généré par les uniqueId() des noeuds.
 */
void FaceUniqueIdBuilder2::
computeFacesUniqueIdAndOwnerVersion5()
{
  info() << "Compute FacesUniqueId() V5 (experimental)";

  IParallelMng* pm = m_parallel_mng;
  Int32 my_rank = pm->commRank();
  bool is_parallel = pm->isParallel();

  ItemInternalMap& faces_map = m_mesh->facesMap();
  UniqueArray<Int64> nodes_uid;
  ENUMERATE_ITEM_INTERNAL_MAP_DATA (iid, faces_map) {
    Face face(iid->value());
    Int32 nb_node = face.nbNode();
    nodes_uid.resize(nb_node);
    {
      Int32 index = 0;
      for (Node node : face.nodes()) {
        nodes_uid[index] = node.uniqueId();
        ++index;
      }
    }
    Int64 new_face_uid = MeshUtils::generateHashUniqueId(nodes_uid);
    face.mutableItemBase().setUniqueId(new_face_uid);
    // En parallèle, indique qu'il faudra positionner le owner de cette face
    // si elle est frontière.
    Int32 new_rank = my_rank;
    if (is_parallel && face.nbCell()==1)
      new_rank = A_NULL_RANK;
    face.mutableItemBase().setOwner(new_rank,my_rank);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Invalide les uid pour être certain qu'ils seront tous positionnés.
 */
void FaceUniqueIdBuilder2::
_unsetFacesUniqueId()
{
  ItemInternalMap& faces_map = m_mesh->facesMap();
  ENUMERATE_ITEM_INTERNAL_MAP_DATA(iid,faces_map){
    ItemInternal* face = iid->value();
    face->unsetUniqueId();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérifie que toutes les faces ont un uid valide.
 */
void FaceUniqueIdBuilder2::
_checkFacesUniqueId()
{
  ItemInternalMap& faces_map = m_mesh->facesMap();
  Integer nb_error = 0;
  ENUMERATE_ITEM_INTERNAL_MAP_DATA(iid,faces_map){
    Face face(iid->value());
    Int64 face_uid = face.uniqueId();
    if (face_uid==NULL_ITEM_UNIQUE_ID){
      info() << "Bad face uid cell0=" << face.cell(0).uniqueId();
      ++nb_error;
    }
  }
  if (nb_error!=0)
    ARCANE_FATAL("Internal error in face uniqueId computation: nb_invalid={0}", nb_error);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" void
_computeFaceUniqueIdVersion3(DynamicMesh* mesh)
{
  FaceUniqueIdBuilder2 f(mesh);
  f.computeFacesUniqueIdAndOwnerVersion3();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" void
_computeFaceUniqueIdVersion5(DynamicMesh* mesh)
{
  FaceUniqueIdBuilder2 f(mesh);
  f.computeFacesUniqueIdAndOwnerVersion5();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
