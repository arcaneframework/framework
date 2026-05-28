// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TiedInterface.h                                             (C) 2000-2014 */
/*                                                                           */
/* Information on mesh semi-conformities.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_TIEDINTERFACE_H
#define ARCANE_MESH_TIEDINTERFACE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/MultiArray2.h"
#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/ITiedInterface.h"
#include "arcane/core/ItemTypes.h"
#include "arcane/core/IMeshPartitionConstraint.h"

#include "arcane/core/VariableTypes.h"

#include "arcane/mesh/MeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IMeshPartitionConstraint;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ITiedInterfaceRebuilder
{
 public:

  virtual ~ITiedInterfaceRebuilder() {}

 public:

  virtual void fillTiedInfos(Face face,
                             Int32ArrayView tied_nodes_lid,
                             Real2ArrayView tied_nodes_isos,
                             Int32ArrayView tied_faces_lid) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Information on mesh semi-conformities.
 */
class TiedInterface
: public TraceAccessor
, public ITiedInterface
{
 public:

  class PartitionConstraintBase
  : public IMeshPartitionConstraint
  {
   public:

    virtual void setInitialRepartition(bool is_initial) = 0;
  };

 public:

  TiedInterface(IMesh* mesh);
  virtual ~TiedInterface(); //<! Releases resources

 public:

  virtual FaceGroup masterInterface() const;
  virtual FaceGroup slaveInterface() const;
  virtual TiedInterfaceNodeList tiedNodes() const;
  virtual TiedInterfaceFaceList tiedFaces() const;
  virtual String masterInterfaceName() const;
  virtual String slaveInterfaceName() const;

 public:

  static PartitionConstraintBase* createConstraint(IMesh* mesh, ConstArrayView<FaceGroup> slave_interfaces);

  //! Builds the tied interface on the group \a interface
  virtual void build(const FaceGroup& interface, bool is_structured);

  //! Defines the relative acceptance threshold for a point projected onto a candidate face
  /*! The value 0 (or <0) describes unconditional acceptance, a positive value defines the relative bound based on the Euclidean distance separating two consecutive vertices.
   */
  virtual void setPlanarTolerance(Real tol);

  virtual void reload(IItemFamily* face_family,
                      const String& master_interface_name,
                      const String& slave_interface_name);

  void resizeNodes(IntegerConstArrayView new_sizes);

  void setNodes(Integer index, ConstArrayView<TiedNode> nodes);

  void resizeFaces(IntegerConstArrayView new_sizes);

  void setFaces(Integer index, ConstArrayView<TiedFace> faces);

  static bool isDebug()
  {
    return m_is_debug;
  }

  void rebuild(ITiedInterfaceRebuilder* rebuilder,
               IntegerConstArrayView nb_slave_node,
               IntegerConstArrayView nb_slave_face);

  void checkValid();

 private:

  static bool m_is_debug;

 private:

  IMesh* m_mesh;
  String m_master_interface_name;
  String m_slave_interface_name;
  FaceGroup m_master_interface;
  FaceGroup m_slave_interface;
  UniqueMultiArray2<TiedNode> m_tied_nodes;
  UniqueMultiArray2<TiedFace> m_tied_faces;
  Real m_planar_tolerance;

  void _checkValid(bool is_print);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
