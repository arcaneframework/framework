// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshRefinement.h                                            (C) 2000-2024 */
/*                                                                           */
/* Management of unstructured mesh adaptation by refinement                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_MESHREFINEMENT_H
#define ARCANE_MESH_MESHREFINEMENT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/PerfCounterMng.h"
#include "arcane/utils/AMRCallBackMng.h"

#include "arcane/core/ItemRefinementPattern.h"

#include "arcane/mesh/MeshGlobal.h"
#include "arcane/mesh/MapCoordToUid.h"
#include "arcane/mesh/DynamicMeshKindInfos.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IMesh;
class Node;
class Cell;
class AMRCallBackMng;
class FaceFamily;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

class DynamicMesh;
class ItemRefinement;
class ParallelAMRConsistency;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implementation of unstructured mesh refinement adaptation algorithms.
 */
class MeshRefinement
:  public TraceAccessor
{
public:
#ifdef ACTIVATE_PERF_COUNTER
  struct PerfCounter
  {
      typedef enum {
        INIT,
        CLEAR,
        ENDUPDATE,
        UPDATEMAP,
        UPDATEMAP2,
        CONSIST,
        PCONSIST,
        PCONSIST2,
        PGCONSIST,
        CONTRACT,
        COARSEN,
        REFINE,
        INTERP,
        PGHOST,
        COMPACT,
        NbCounters
      }  eType ;

      static const std::string m_names[NbCounters] ;
  } ;
#endif
  /**
   * Constructor.
   */
  MeshRefinement(DynamicMesh* mesh);

private:
  // The copy constructor and assignment operator are
  // declared private but not implemented. This is the
  // standard technique to prevent them from being used.
  MeshRefinement(const MeshRefinement&);
  MeshRefinement& operator=(const MeshRefinement&);

public:

  /**
   * Destructor.
   */
  ~MeshRefinement();

  /**
   * Clears all currently stored data.
   */
  void clear();
  /**
   * Calculates the maximum UID.
   */
  void init();

  void update() ;
  bool needUpdate() const {
    return m_need_update ;
  }
  void invalidate() {
    m_need_update = true ;
  }
  void initMeshContainingBox() ;
  /**
   * Flags items for refinement/coarsening
   */
  void flagItems(const Int32Array& flag_per_cell,
                 const Integer max_level = -1);

  /*!
   * \brief Passing the error committed by the mesh to the refinement flag.
   *
   * This method could be implemented in different ways:
   * 1- current implementation: the user performs the transformation themselves
   * in this case, they modify the itemInternal object by setting the refinement flag
   * 2- the user performs the transformation themselves but stores and returns an array of flags
   * the MeshRefinement class, in this case, implement a setter from the returned array
   * 3- to avoid copying the array of flags, implement the converter directly in meshRefinement
   * and the user only provides the error array
   */
  virtual void flagCellToRefine(Int32ConstArrayView cells_lids);
  virtual void flagCellToCoarsen(Int32ConstArrayView cells_lids);

  /*!
   * Refines and coarsens the items requested by the user. It also
   * refines/coarsens complementary items to satisfy the level-one rule.
   * It is possible that for a given set of flags there is no actual change when calling this method. Consequently,
   * it returns \p true if the mesh has actually changed (in which case
   * the data must be projected) and \p false otherwise.

   * The argument \p maintain_level_one is deprecated; use face_level_mismatch_limit() instead.
   */
  bool refineAndCoarsenItems(const bool maintain_level_one=true);

  /*!
   * Coarsens only the items requested by the user. Some items
   * may not be coarsened to satisfy the level-one rule.
   * It is possible that for a given set of flags there is no actual change when calling this method. Consequently,
   * it returns \p true if the mesh has actually changed (in which case
   * the data must be projected) and \p false otherwise.

   * The argument \p maintain_level_one is deprecated; use face_level_mismatch_limit() instead.
   */
  bool coarsenItems(const bool maintain_level_one = true);

  /*!
   * \brief Method allowing the removal of meshes marked with the
   * flag "II_Coarsen".
   *
   * The owners of faces and nodes having marked meshes
   * and unmarked meshes are likely to be updated.
   *
   * \param update_parent_flag If true, the parent flags will be
   * updated. This includes activating parent meshes.
   *
   * \return true if the mesh has changed.
   */
  bool coarsenItemsV2(bool update_parent_flag);

  /*!
   * Refines only the items requested by the user.
   * It is possible that for a given set of flags there is no actual change when calling this method. Consequently,
   * it returns \p true if the mesh has actually changed (in which case
   * the data must be projected) and \p false otherwise.

   * The argument \p maintain_level_one is deprecated; use face_level_mismatch_limit() instead.
   */
  bool refineItems(const bool maintain_level_one=true);

  /*!
   * Uniformly refines the mesh \p n times.
   */
  void uniformlyRefine(Integer n=1);

  /*!
   * Uniformly coarsens the mesh \p n times.
   */
  void uniformlyCoarsen(Integer n=1);

  /*!
   * \p max_level is the highest refinement level
   * an item can reach.
   *
   * \p max_level is unlimited (-1) by default
   */
  Integer& maxLevel();

  //! Constant reference to the mesh.
  const IMesh* getMesh() const;

  //! Reference to the mesh.
  IMesh* getMesh();

  //!
  void registerCallBack(IAMRTransportFunctor* f);
  //!
  void unRegisterCallBack(IAMRTransportFunctor* f);
  /*!
   * Adds a new uid associated with point \p p.
   * if p already exists, the old uid is kept.
   * The tolerance \p tol gives the search perimeter around p.
   */
  Int64 findOrAddNodeUid(const Real3& p,const Real& tol);
  /*!
   * Adds a new uid associated with the face center \p face_center.
   * if p already exists, the old uid is kept.
   * The tolerance \p tol gives the search perimeter around face_center.
   */
  Int64 findOrAddFaceUid(const Real3& face_center,const Real& tol,bool& is_added);
  /*!
   * Generates a new uid for children.
   */
  Int64 getFirstChildNewUid();

  void _update(ArrayView<ItemInternal*> cells_to_refine);
  void _update(ArrayView<Int64> cells_to_refine);
  void _invalidate(ArrayView<ItemInternal*> cells_to_coarsen);
  void _updateMaxUid(ArrayView<ItemInternal*> cells_to_refine);

  /*!
   * Returns the refinement pattern associated with the mesh type.
   */
  template <int typeID> const ItemRefinementPatternT<typeID>& getRefinementPattern() const ;
  /*!
   * Determination of non-conforming connections of refined meshes.
   */
  void populateBackFrontCellsFromChildrenFaces(Cell parent_cell);
  void populateBackFrontCellsFromParentFaces(Cell parent_cell);

 private:

  /*!
   * Returns true if and only if the mesh satisfies the level-one rule.
   * Returns false otherwise.
   * Stops execution if arcane_assert_yes is true and if
   * the mesh does not satisfy the level-one rule
   */
  bool _checkLevelOne(bool arcane_assert_yes = false);

  /*!
   * Returns true if and only if the mesh has no items
   * flagged for coarsening or refinement
   * Returns false otherwise
   * Stops execution if \a arcane_assert_yes is true and if
   * the mesh has flagged items
   */
  bool _checkUnflagged(bool arcane_assert_yes = false);

  /*!
   * If \p coarsen_by_parents is true,
   * items with the same parent will be flagged for coarsening
   * This should produce a coarsening closer to what was requested.
   *
   * \p coarsen_by_parents is true by default.
   */
  bool& coarsenByParents();


  /*!
   * If Face_level_mismatch_limit is set to a non-zero value, then   * refinement and coarsening will produce meshes where
   * the refinement level of two neighboring meshes per face will not differ by more than
   * this limit. If Face_level_mismatch_limit is 0, then level differences
   * will be unlimited.
   *
   * \p face_level_mismatch_limit is 1 by default. Currently, the only
   * supported options are 0 and 1.
   */
  unsigned char& faceLevelMismatchLimit();

  /*!
   * Removes inactive children from the mesh
   * Contracts an active item, i.e. deletes the pointers to each
   * inactive child. This should only be called after variable restriction
   * on the parents
   */
  bool _contract();

  //! Interpolation of data on the child meshes
  void _interpolateData(const Int64Array& cells_to_refine);

  //! Upscaling of data on the parent meshes
  void _upscaleData(Array<ItemInternal*>& parent_cells);

 private:

  /**
   * Coarsens the items requested by the user. The two methods _coarsenItems()
   * and _refineItems() are not in the public interface of MeshRefinement. Because appropriate preparation (makeRefinementCompatible, makeCoarseningCompatible) is
   * necessary to execute _coarsenItems().
   *
   * It is possible that for a given set of flags there is no actual change when calling this function. Consequently,
   * it returns \p true if the mesh has actually changed (in which case
   * the data must be projected) \p false otherwise.
   */
  bool _coarsenItems();

  /**
   * Refines the items requested by the user.
   *
   * It is possible that for a given set of flags there is no actual change when calling this function. Consequently,
   * it returns \p true if the mesh has actually changed (in which case
   * the data must be projected) \p false otherwise.
   */
  bool _refineItems(Int64Array& cells_to_refine);

  // Updates the owners of items from the meshes
  void _updateItemOwner(Int32ArrayView cell_to_remove);
  void _updateItemOwner2();
  //!
  bool _removeGhostChildren();
  //---------------------------------------------
  // Utility algorithms


  /**
   * Updates m_nodes_finder and m_faces_finder
   */
  void _updateLocalityMap();
  void _updateLocalityMap2();
  /**
   * Sets the refinement flag to II_DoNothing
   * for every item in the mesh.
   */
  void _cleanRefinementFlags();

  /**
   * Acts on the coarsening flags so that the level-one rule is satisfied.
   */
  bool _makeCoarseningCompatible(const bool);


  /**
   * Acts on the refinement flags so that the level-one rule is satisfied.
   */
  bool _makeRefinementCompatible(const bool);

  /**
   * Copies refinement flags onto boundary items from their
   * owner processors. Returns true if a flag has changed.
   */
  bool _makeFlagParallelConsistent();
  bool _makeFlagParallelConsistent2();

  /**
   * Determination of non-conforming connections of refined meshes
   */
  template <int typeID>
  void _populateBackFrontCellsFromParentFaces(Cell parent_cell) ;
  template <int typeID>
  void _populateBackFrontCellsFromChildrenFaces(Face face,Cell parent_cell,
                                                Cell neighbor_cell);

  void _checkOwner(const String & msg); // To avoid owner desynchronization

 private:

  /**
   * Reference to the mesh.
   */
  DynamicMesh* m_mesh;
  FaceFamily* m_face_family;
  bool m_need_update;

  /**
   * Quick search for nodes and faces based on their coordinates.
   * For faces, the coordinates are those of the face center.
   */
  MapCoordToUid::Box m_mesh_containing_box ;
  NodeMapCoordToUid m_node_finder;
  FaceMapCoordToUid m_face_finder;

  /**
   * Reference to the item refiner
   */
  ItemRefinement * m_item_refinement;

  /**
   * Ensures UID consistency in parallel
   */
  ParallelAMRConsistency* m_parallel_amr_consistency;

  /**
   * Manager of data transport functors between meshes
   */
  AMRCallBackMng* m_call_back_mng;

  /**
   * Refinement parameters
   */

  bool m_coarsen_by_parents;
  Integer m_max_level;
  Integer m_nb_cell_target;
  Byte m_face_level_mismatch_limit;

  Int64 m_max_node_uid;
  Int64 m_next_node_uid;

  Int64 m_max_cell_uid;
  Int64 m_next_cell_uid;

  Int64 m_max_face_uid;
  Int64 m_next_face_uid;

  Integer m_max_nb_hChildren;

  /**
   * Refinement patterns
   */
  const Quad4RefinementPattern4Quad m_quad4_refinement_pattern;
  const TetraRefinementPattern2Hex_2Penta_2Py_2Tetra m_tetra_refinement_pattern;
  const PyramidRefinementPattern4Hex_4Py m_pyramid_refinement_pattern;
  const PrismRefinementPattern4Hex_4Pr m_prism_refinement_pattern;
  const HexRefinementPattern8Hex m_hex_refinement_pattern;
  const HemiHex7RefinementPattern6Hex_2HHex7 m_hemihexa7_refinement_pattern;
  const HemiHex6RefinementPattern4Hex_4HHex7 m_hemihexa6_refinement_pattern;
  const HemiHex5RefinementPattern2Hex_4Penta_2HHex5 m_hemihexa5_refinement_pattern;
  const AntiWedgeLeft6RefinementPattern4Hex_4HHex7 m_antiwedgeleft6_refinement_pattern;
  const AntiWedgeRight6RefinementPattern4Hex_4HHex7 m_antiwedgeright6_refinement_pattern;
  const DiTetra5RefinementPattern2Hex_6HHex7 m_ditetra5_refinement_pattern;

  // Node owner memory : to patch owner desynchronization
  VariableNodeInt32 m_node_owner_memory;

#ifdef ACTIVATE_PERF_COUNTER
  PerfCounterMng<PerfCounter> m_perf_counter ;
#endif
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// MeshRefinement class inline members

inline bool& MeshRefinement::coarsenByParents()
{
  return m_coarsen_by_parents;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline Integer& MeshRefinement::maxLevel()
{
  return m_max_level;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline unsigned char& MeshRefinement::faceLevelMismatchLimit()
{
  return m_face_level_mismatch_limit;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
inline const ItemRefinementPatternT<IT_Quad4>&
MeshRefinement::getRefinementPattern<IT_Quad4>() const
{
  return m_quad4_refinement_pattern;
}
template <>
inline const ItemRefinementPatternT<IT_Tetraedron4>&
MeshRefinement::getRefinementPattern<IT_Tetraedron4>() const
{
  return m_tetra_refinement_pattern;
}
template <>
inline const ItemRefinementPatternT<IT_Pyramid5>&
MeshRefinement::getRefinementPattern<IT_Pyramid5>() const
{
  return m_pyramid_refinement_pattern;
}
template <>
inline const ItemRefinementPatternT<IT_Pentaedron6>&
MeshRefinement::getRefinementPattern<IT_Pentaedron6>() const
{
  return m_prism_refinement_pattern;
}
template <>
inline const ItemRefinementPatternT<IT_Hexaedron8>&
MeshRefinement::getRefinementPattern<IT_Hexaedron8>() const
{
  return m_hex_refinement_pattern;
}
template <>
inline const ItemRefinementPatternT<IT_HemiHexa7>&
MeshRefinement::getRefinementPattern<IT_HemiHexa7>() const
{
  return m_hemihexa7_refinement_pattern;
}
template <>
inline const ItemRefinementPatternT<IT_HemiHexa6>&
MeshRefinement::getRefinementPattern<IT_HemiHexa6>() const
{
  return m_hemihexa6_refinement_pattern;
}
template <>
inline const ItemRefinementPatternT<IT_HemiHexa5>&
MeshRefinement::getRefinementPattern<IT_HemiHexa5>() const
{
  return m_hemihexa5_refinement_pattern;
}
template <>
inline const ItemRefinementPatternT<IT_AntiWedgeLeft6>&
MeshRefinement::getRefinementPattern<IT_AntiWedgeLeft6>() const
{
  return m_antiwedgeleft6_refinement_pattern;
}
template <>
inline const ItemRefinementPatternT<IT_AntiWedgeRight6>&
MeshRefinement::getRefinementPattern<IT_AntiWedgeRight6>() const
{
  return m_antiwedgeright6_refinement_pattern;
}
template <>
inline const ItemRefinementPatternT<IT_DiTetra5>&
MeshRefinement::getRefinementPattern<IT_DiTetra5>() const
{
  return m_ditetra5_refinement_pattern;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif // end ARCANE_MESH_MESHREFINEMENT_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
