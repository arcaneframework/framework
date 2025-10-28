﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AMRPatchPositionSignature.cc                                       (C) 2000-2025 */
/*                                                                           */
/* Informations sur un patch AMR d'un maillage cartésien.                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/AMRPatchPositionSignature.h"

#include "ICartesianMesh.h"
#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/utils/FatalErrorException.h"
#include "arccore/trace/ITraceMng.h"
#include "internal/ICartesianMeshInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

namespace
{
constexpr Integer MIN_SIZE = 4;
constexpr Integer TARGET_SIZE = 8;
constexpr Real TARGET_SIZE_WEIGHT_IN_EFFICACITY = 0.5;
constexpr Integer MAX_NB_CUT = 5;
constexpr Real TARGET_EFFICACITY = 0.90;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AMRPatchPositionSignature::
AMRPatchPositionSignature()
: m_is_null(true)
, m_mesh(nullptr)
, m_nb_cut(0)
, m_stop_cut(false)
, m_numbering(nullptr)
, m_have_cells(false)
, m_is_computed(false)
, m_all_patches(nullptr)
{
}

AMRPatchPositionSignature::
AMRPatchPositionSignature(AMRPatchPosition patch, ICartesianMesh* cmesh, AMRPatchPositionLevelGroup* all_patches)
: m_is_null(false)
, m_patch(patch)
, m_mesh(cmesh)
, m_nb_cut(0)
, m_stop_cut(false)
, m_numbering(cmesh->_internalApi()->cartesianMeshNumberingMng().get())
, m_have_cells(false)
, m_is_computed(false)
, m_sig_x(patch.maxPoint().x - patch.minPoint().x, 0)
, m_sig_y(patch.maxPoint().y - patch.minPoint().y, 0)
, m_all_patches(all_patches)
{}

AMRPatchPositionSignature::
AMRPatchPositionSignature(AMRPatchPosition patch, ICartesianMesh* cmesh, AMRPatchPositionLevelGroup* all_patches, Integer nb_cut)
: m_is_null(false)
, m_patch(patch)
, m_mesh(cmesh)
, m_nb_cut(nb_cut)
, m_stop_cut(false)
, m_numbering(cmesh->_internalApi()->cartesianMeshNumberingMng().get())
, m_have_cells(false)
, m_is_computed(false)
, m_sig_x(patch.maxPoint().x - patch.minPoint().x, 0)
, m_sig_y(patch.maxPoint().y - patch.minPoint().y, 0)
, m_all_patches(all_patches)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AMRPatchPositionSignature::
~AMRPatchPositionSignature()
= default;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRPatchPositionSignature::
compress()
{
  if (!m_have_cells) {
    return;
  }

  Integer reduce_x_min = 0;
  if (m_sig_x[0] == 0) {
    for (; reduce_x_min < m_sig_x.size(); ++reduce_x_min) {
      if (m_sig_x[reduce_x_min] != 0) {
        break;
      }
    }
    if (reduce_x_min == m_sig_x.size()) {
      ARCANE_FATAL("aaa");
    }
  }
  Integer reduce_y_min = 0;
  if (m_sig_y[0] == 0) {
    for (; reduce_y_min < m_sig_y.size(); ++reduce_y_min) {
      if (m_sig_y[reduce_y_min] != 0) {
        break;
      }
    }
    if (reduce_y_min == m_sig_y.size()) {
      ARCANE_FATAL("bbb");
    }
  }

  Integer reduce_x_max = m_sig_x.size()-1;
  if (m_sig_x[reduce_x_max] == 0) {
    for (; reduce_x_max >= 0; --reduce_x_max) {
      if (m_sig_x[reduce_x_max] != 0) {
        break;
      }
    }
    if (reduce_x_max < reduce_x_min) {
      ARCANE_FATAL("ccc");
    }
  }
  Integer reduce_y_max = m_sig_y.size()-1;
  if (m_sig_y[reduce_y_max] == 0) {
    for (; reduce_y_max >= 0; --reduce_y_max) {
      if (m_sig_y[reduce_y_max] != 0) {
        break;
      }
    }
    if (reduce_y_max < reduce_y_min) {
      ARCANE_FATAL("ddd");
    }
  }

  if (reduce_x_min != 0 || reduce_x_max != m_sig_x.size()-1) {
    reduce_x_max++;
    UniqueArray tmp = m_sig_x.subView(reduce_x_min, reduce_x_max - reduce_x_min);
    m_sig_x = tmp;
    Int64x3 patch_min = m_patch.minPoint();
    Int64x3 patch_max = m_patch.maxPoint();
    patch_min.x += reduce_x_min;
    patch_max.x = patch_min.x + (reduce_x_max - reduce_x_min);
    m_patch.setMinPoint(patch_min);
    m_patch.setMaxPoint(patch_max);
  }
  if (reduce_y_min != 0 || reduce_y_max != m_sig_y.size()-1) {
    reduce_y_max++;
    UniqueArray tmp = m_sig_y.subView(reduce_y_min, reduce_y_max - reduce_y_min);
    m_sig_y = tmp;
    Int64x3 patch_min = m_patch.minPoint();
    Int64x3 patch_max = m_patch.maxPoint();
    patch_min.y += reduce_y_min;
    patch_max.y = patch_min.y + (reduce_y_max - reduce_y_min);
    m_patch.setMinPoint(patch_min);
    m_patch.setMaxPoint(patch_max);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRPatchPositionSignature::
fillSig()
{
  m_sig_x.fill(0);
  m_sig_y.fill(0);
  ENUMERATE_ (Cell, icell, m_mesh->mesh()->ownLevelCells(m_patch.level())) {
    if (!icell->hasFlags(ItemFlags::II_Refine)) {
      continue;
    }

    Integer pos_x = m_numbering->cellUniqueIdToCoordX(*icell);
    Integer pos_y = m_numbering->cellUniqueIdToCoordY(*icell);

    if (pos_x < m_patch.minPoint().x || pos_x >= m_patch.maxPoint().x || pos_y < m_patch.minPoint().y || pos_y >= m_patch.maxPoint().y ) {
      continue;
    }
    m_have_cells = true;
    m_sig_x[pos_x - m_patch.minPoint().x]++;
    m_sig_y[pos_y - m_patch.minPoint().y]++;
  }

  if (m_all_patches->maxLevel() > m_patch.level()) {
    Int64x3 min = m_patch.minPoint();
    Integer nb_proc = m_mesh->mesh()->parallelMng()->commSize();
    Integer my_proc = m_mesh->mesh()->parallelMng()->commRank();

    Integer base = m_sig_x.size() / nb_proc;
    Integer reste = m_sig_x.size() % nb_proc;
    Integer size = base + (my_proc < reste ? 1 : 0);
    Integer begin = my_proc * base + std::min(my_proc, reste);
    Integer end = begin + size;

    for (Integer j = 0; j < m_sig_y.size(); ++j) {
      Integer pos_y = min.y + j;
      for (Integer i = begin; i < end; ++i) {
        Integer pos_x = min.x + i;
        for (auto elem : m_all_patches->patches(m_patch.level()+1)) {
          if (elem.isInWithMargin(m_patch.level(), pos_x, pos_y)) {
            m_have_cells = true;
            m_sig_x[i]++;
            m_sig_y[j]++;
            break;
          }
        }
      }
    }
  }

  m_mesh->mesh()->parallelMng()->reduce(MessagePassing::ReduceSum, m_sig_x);
  m_mesh->mesh()->parallelMng()->reduce(MessagePassing::ReduceSum, m_sig_y);
  m_have_cells = m_mesh->mesh()->parallelMng()->reduce(MessagePassing::ReduceMax, m_have_cells);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool AMRPatchPositionSignature::
isValid() const
{
  if (m_is_null) {
    return false;
  }
  if (m_sig_x.size() < MIN_SIZE || m_sig_y.size() < MIN_SIZE) {
    return false;
  }
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool AMRPatchPositionSignature::
canBeCut() const
{
  m_mesh->traceMng()->info() << "canBeCut() -- m_sig_x.size : " << m_sig_x.size()
  << " -- m_sig_y.size : " << m_sig_y.size()
  << " -- min = " << m_patch.minPoint()
  << " -- max = " << m_patch.maxPoint()
  << " -- length = " << m_patch.length()
  << " -- isValid : " << isValid()
  << " -- efficacity : " << efficacity() << " / " << TARGET_EFFICACITY
  << " -- m_nb_cut : " << m_nb_cut << " / " << MAX_NB_CUT
  << " -- m_stop_cut : " << m_stop_cut
;

  if (!isValid()) {
    return false;
  }

  if (m_stop_cut) {
    return false;
  }

  if (efficacity() > TARGET_EFFICACITY) {
    return false;
  }
  if (MAX_NB_CUT != -1 && m_nb_cut >= MAX_NB_CUT) {
    return false;
  }
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRPatchPositionSignature::
compute()
{
  m_mesh->traceMng()->info() << "Compute() -- Patch before compute : min = " << m_patch.minPoint() << " -- max = " << m_patch.maxPoint() << " -- length = " << m_patch.length();
  fillSig();
  //m_mesh->traceMng()->info() << "Compute() -- Signature : x = " << m_sig_x << " -- y = " << m_sig_y ;
  compress();
  m_mesh->traceMng()->info() << "Compute() -- Compress : min = " << m_patch.minPoint() << " -- max = " << m_patch.maxPoint() << " -- x = " << m_sig_x << " -- y = " << m_sig_y ;
  m_mesh->traceMng()->info() << "Compute() -- Patch computed :       min = " << m_patch.minPoint() << " -- max = " << m_patch.maxPoint() << " -- length = " << m_patch.length();

  m_is_computed = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real AMRPatchPositionSignature::
efficacity() const
{
  if (!m_is_computed) {
    // Sans compression, pas terrible.
    m_mesh->traceMng()->warning() << "Need to be computed";
  }
  Integer sum = 0;
  for (const Integer elem : m_sig_x) {
    sum += elem;
  }

  Real eff = static_cast<Real>(sum) / (m_sig_x.size() * m_sig_y.size());

  if constexpr (TARGET_SIZE == -1 || TARGET_SIZE_WEIGHT_IN_EFFICACITY == 0) {
    return eff;
  }

  Real eff_xy = 0;
  if (m_sig_x.size() <= TARGET_SIZE) {
    eff_xy = static_cast<Real>(m_sig_x.size()) / TARGET_SIZE;
  }
  else if (m_sig_x.size() < TARGET_SIZE*2) {
    Real size_x = math::abs(m_sig_x.size() - TARGET_SIZE*2);
    eff_xy = size_x / TARGET_SIZE;
  }

  if (m_sig_y.size() <= TARGET_SIZE) {
    eff_xy = (eff_xy + (static_cast<Real>(m_sig_y.size()) / TARGET_SIZE)) / 2;
  }
  else if (m_sig_y.size() < TARGET_SIZE*2) {
    Real size_y = math::abs(m_sig_y.size() - TARGET_SIZE*2);
    eff_xy = (eff_xy + (size_y / TARGET_SIZE)) / 2;
  }
  else {
    eff_xy /= 2;
  }
  eff_xy *= TARGET_SIZE_WEIGHT_IN_EFFICACITY;

  return (eff+eff_xy)/(1+TARGET_SIZE_WEIGHT_IN_EFFICACITY);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::pair<AMRPatchPositionSignature, AMRPatchPositionSignature> AMRPatchPositionSignature::
cut(Integer dim, Integer cut_point) const
{
  auto [fst, snd] = m_patch.cut(cut_point, dim);
  return {AMRPatchPositionSignature(fst, m_mesh, m_all_patches, m_nb_cut+1), AMRPatchPositionSignature(snd, m_mesh, m_all_patches, m_nb_cut+1)};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool AMRPatchPositionSignature::
isIn(Integer x, Integer y) const
{
    return m_patch.isIn(x, y);
}
ConstArrayView<Integer> AMRPatchPositionSignature::sigX() const
{
  return m_sig_x;
}
ConstArrayView<Integer> AMRPatchPositionSignature::sigY()const
{
  return m_sig_y;
}
AMRPatchPosition AMRPatchPositionSignature::patch() const
{
  return m_patch;
}
ICartesianMesh* AMRPatchPositionSignature::mesh() const
{
  return m_mesh;
}
bool AMRPatchPositionSignature::stopCut() const
{
  return m_stop_cut;
}
void AMRPatchPositionSignature::setStopCut(bool stop_cut)
{
  m_stop_cut = stop_cut;
}
bool AMRPatchPositionSignature::isComputed() const
{
  return m_is_computed;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
