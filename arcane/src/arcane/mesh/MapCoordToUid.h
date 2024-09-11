// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MapCoordToUid.cc                                            (C) 2000-2024 */
/*                                                                           */
/* Recherche d'entités à partir de ses coordonnées.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_MAPCOORDTOUID_H
#define ARCANE_MESH_MAPCOORDTOUID_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Math.h"

#include "arcane/core/SharedVariable.h"
#include "arcane/core/MathUtils.h"

#include "arcane/mesh/MeshGlobal.h"
#include "arcane/mesh/DynamicMeshKindInfos.h"

#include <map>
#include <unordered_map>
#include <unordered_set>
#include "arcane/utils/PerfCounterMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/**
   * \brief structure de recherche d'un noeud à partir de ses coords
   * La clef de hashage est la position geometrique du noeud.
   * \todo utiliser une hash multimap.
   */
class MapCoordToUid
{
 public:

  typedef std::unordered_multimap<Int32, std::pair<const Real3, Int64>> map_type;
  //  typedef std::unordered_multimap<Int64, std::pair<const Real3,Int64> > map_type; // This (needed !) correction introduces a bug in a IFPEN application...investigation in progress (Scarab Arc356@IFPEN)
  typedef std::unordered_set<Int64> set_type;
  static const Real TOLERANCE;

  class Box
  {
   public:

    Box();
    virtual ~Box() {}
    void init(IMesh* mesh);
    void init2(IMesh* mesh);
    Real3 m_lower_bound;
    Real3 m_upper_bound;
    Real3 m_size;
  };

#ifdef ACTIVATE_PERF_COUNTER
  struct PerfCounter
  {
    typedef enum
    {
      Clear,
      Fill,
      Fill2,
      Insert,
      Find,
      Key,
      NbCounters
    } eType;

    static const std::string m_names[NbCounters];
  };
#endif
  MapCoordToUid(IMesh* mesh);

  void setBox(Box* box)
  {
    m_box = box;
  }

  void clear() { m_map.clear(); }
  void _clear();

  void updateNodeData(ArrayView<ItemInternal*> coarsen_cells);
  void updateFaceData(ArrayView<ItemInternal*> coarsen_cells);

  void clearNodeData(ArrayView<ItemInternal*> coarsen_cells);
  void clearFaceData(ArrayView<ItemInternal*> coarsen_cells);

  Int64 insert(const Real3, const Int64, const Real tol = TOLERANCE);
  void erase(const Real3, const Real tol = TOLERANCE);

  bool empty() const { return m_map.empty(); }

  Int64 find(const Real3, const Real tol = TOLERANCE);

  bool areClose(Real3 const& p1, Real3 const& p2, Real tol)
  {
    return Arcane::math::abs(p1.x - p2.x) +
    Arcane::math::abs(p1.y - p2.y) +
    Arcane::math::abs(p1.z - p2.z) <=
    tol;
  }

#ifdef ACTIVATE_PERF_COUNTER
  PerfCounterMng<PerfCounter>& getPerfCounter()
  {
    return m_perf_counter;
  }
#endif
 protected:

  Int64 key(const Real3);

 protected:

  IMesh* m_mesh;
  map_type m_map;
  Box* m_box;
  VariableNodeReal3& m_nodes_coords;
#ifdef ACTIVATE_PERF_COUNTER
  PerfCounterMng<PerfCounter> m_perf_counter;
#endif
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class NodeMapCoordToUid
: public MapCoordToUid
{
 public:

  typedef MapCoordToUid BaseType;
  NodeMapCoordToUid(IMesh* mesh)
  : MapCoordToUid(mesh)
  {}
  void init();
  void check();

  bool insert(const Real3 center, const Int64 uid, const Real tol = TOLERANCE)
  {
    return BaseType::insert(center, uid, tol) != uid;
  }

  void init2();
  void check2();

  void clearData(ArrayView<ItemInternal*> coarsen_cells);
  void updateData(ArrayView<ItemInternal*> refine_cells);

  inline bool isItemToSuppress(Node node, const Int64 parent_uid) const;

 protected:

  void fill();
  void fill2();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class FaceMapCoordToUid : public MapCoordToUid
{
 public:

  typedef MapCoordToUid BaseType;
  FaceMapCoordToUid(IMesh* mesh)
  : MapCoordToUid(mesh)
  , m_face_center(VariableBuildInfo(mesh, "AMRFaceCenter"))
  {}
  void init();
  void check();

  void init2();
  void check2();

  bool insert(const Real3& center, const Int64& uid, const Real tol = TOLERANCE)
  {
    Int64 old_uid = BaseType::insert(center, uid, tol);
    if (uid != old_uid) {
      m_new_uids.insert(uid);
      return true;
    }
    return false;
  }

  void clearNewUids()
  {
    m_new_uids.clear();
  }

  bool isNewUid(Int64 uid)
  {
    return m_new_uids.find(uid) != m_new_uids.end();
  }

  void clearData(ArrayView<ItemInternal*> coarsen_cells);
  void updateData(ArrayView<ItemInternal*> refine_cells);

  void initFaceCenter();
  void updateFaceCenter(ArrayView<ItemInternal*> refine_cells);

 protected:

  void fill();
  void fill2();

  Real3 faceCenter(Face face) const;
  bool isItemToSuppress(Face face) const;

 private:

  VariableFaceReal3 m_face_center;
  set_type m_new_uids;
};
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* MapCoordToUid_H_ */
