// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* EmptyMeshModifier                                           (C) 2000-2024 */
/*                                                                           */
/* Brief code description                                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_EMPTYMESHMODIFIER_H
#define ARCANE_EMPTYMESHMODIFIER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IMeshModifier.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class EmptyMeshModifier
: public IMeshModifier
{

 public:

  explicit EmptyMeshModifier() = default;

  ~EmptyMeshModifier() override = default;

 private:

  [[noreturn]] void _error() const { ARCANE_FATAL("Using EmptyMeshModifier"); }

 public:

  void build() override { _error(); }

 public:

  //! Maillage associé
  IMesh* mesh() override { _error(); }

 public:

  void setDynamic(bool) override { _error(); }
  void addCells(Integer, Int64ConstArrayView,
                Int32ArrayView) override { _error(); }
  void addCells(const MeshModifierAddCellsArgs&) { _error(); }
  void addFaces(Integer, Int64ConstArrayView,
                Int32ArrayView) override { _error(); }
  void addFaces(const MeshModifierAddFacesArgs&) { _error(); }
  void addEdges(Integer, Int64ConstArrayView,
                Int32ArrayView) override { _error(); }
  void addNodes(Int64ConstArrayView,
                Int32ArrayView) override { _error(); }

  void removeCells(Int32ConstArrayView) override { _error(); }
  void removeCells(Int32ConstArrayView, bool) override { _error(); }
  void detachCells(Int32ConstArrayView) override { _error(); }

  void removeDetachedCells(Int32ConstArrayView) override { _error(); }

  void flagCellToRefine(Int32ConstArrayView) override { _error(); }
  void flagCellToCoarsen(Int32ConstArrayView) override { _error(); }
  void refineItems() override { _error(); }
  void coarsenItems() override { _error(); }
  void coarsenItemsV2(bool) override { _error(); }
  bool adapt() override { _error(); }
  void registerCallBack(IAMRTransportFunctor*) override { _error(); }
  void unRegisterCallBack(IAMRTransportFunctor*) override { _error(); }
  void addHChildrenCells(Cell, Integer,
                         Int64ConstArrayView, Int32ArrayView) override { _error(); }

  void addParentCellToCell(Cell, Cell) override { _error(); }
  void addChildCellToCell(Cell, Cell) override { _error(); }

  void addParentFaceToFace(Face, Face) override { _error(); }
  void addChildFaceToFace(Face, Face) override { _error(); }

  void addParentNodeToNode(Node, Node) override { _error(); }
  void addChildNodeToNode(Node, Node) override { _error(); }

  void clearItems() override { _error(); }

  ARCANE_DEPRECATED_240 void addCells(ISerializer*) override { _error(); }
  ARCANE_DEPRECATED_240 void addCells(ISerializer*, Int32Array&) override { _error(); }

  void endUpdate() override { _error(); }

  void endUpdate(bool, bool) override { _error(); } // SDC: this signature is needed @IFPEN.

 public:

  void updateGhostLayers() override { _error(); }

  //! AMR
  void updateGhostLayerFromParent(Array<Int64>&,
                                  Array<Int64>&,
                                  bool) override { _error(); }

  void addExtraGhostCellsBuilder(IExtraGhostCellsBuilder*) override { _error(); }

  void removeExtraGhostCellsBuilder(IExtraGhostCellsBuilder*) override { _error(); }

  void addExtraGhostParticlesBuilder(IExtraGhostParticlesBuilder*) override { _error(); }

  void removeExtraGhostParticlesBuilder(IExtraGhostParticlesBuilder*) override { _error(); }

 public:

  void mergeMeshes(ConstArrayView<IMesh*>) override { _error(); }

 public:

  IMeshModifierInternal* _modifierInternalApi() override
  {
    _error();
    return nullptr;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif //ARCANE_EMPTYMESHMODIFIER_H
