// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AMRPatchPositionSignature.cc                                (C) 2000-2025 */
/*                                                                           */
/* Calcul des signatures d'une position de patch.                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/internal/AMRPatchPositionSignature.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Math.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/ItemGroup.h"

#include "arcane/cartesianmesh/ICartesianMesh.h"

#include "arcane/cartesianmesh/internal/ICartesianMeshInternal.h"
#include "arcane/cartesianmesh/internal/AMRPatchPositionLevelGroup.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

// On peut jouer avec ces paramètres pour avoir un meilleur raffinement.
// TODO : Fixer ça n'est peut-être pas la meilleur façon de faire. Les patchs
//        créés aujourd'hui sont trop nombreux et trop petits mais conviennent
//        dans la majorité des cas de figures.
namespace
{
  constexpr Integer MIN_SIZE = 1;
  constexpr Integer TARGET_SIZE = 8;
  constexpr Real TARGET_SIZE_WEIGHT_IN_EFFICACITY = 1;
  constexpr Integer MAX_NB_CUT = 6;
  constexpr Real TARGET_EFFICACITY = 1.0;
} // namespace

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
{
}

AMRPatchPositionSignature::
AMRPatchPositionSignature(const AMRPatchPosition& patch, ICartesianMesh* cmesh)
: m_is_null(false)
, m_patch(patch)
, m_mesh(cmesh)
, m_nb_cut(0)
, m_stop_cut(false)
, m_numbering(cmesh->_internalApi()->cartesianMeshNumberingMngInternal().get())
, m_have_cells(false)
, m_is_computed(false)
, m_sig_x(patch.maxPoint().x - patch.minPoint().x, 0)
, m_sig_y(patch.maxPoint().y - patch.minPoint().y, 0)
, m_sig_z(patch.maxPoint().z - patch.minPoint().z, 0)
{}

AMRPatchPositionSignature::
AMRPatchPositionSignature(const AMRPatchPosition& patch, ICartesianMesh* cmesh, Integer nb_cut)
: m_is_null(false)
, m_patch(patch)
, m_mesh(cmesh)
, m_nb_cut(nb_cut)
, m_stop_cut(false)
, m_numbering(cmesh->_internalApi()->cartesianMeshNumberingMngInternal().get())
, m_have_cells(false)
, m_is_computed(false)
, m_sig_x(patch.maxPoint().x - patch.minPoint().x, 0)
, m_sig_y(patch.maxPoint().y - patch.minPoint().y, 0)
, m_sig_z(patch.maxPoint().z - patch.minPoint().z, 0)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRPatchPositionSignature::
compress()
{
  if (!m_have_cells) {
    return;
  }

  // On cherche la première position où il n'y a plus de zéro.
  CartCoord reduce_x_min = 0;
  if (m_sig_x[0] == 0) {
    for (; reduce_x_min < m_sig_x.size(); ++reduce_x_min) {
      if (m_sig_x[reduce_x_min] != 0) {
        break;
      }
    }
  }
  CartCoord reduce_y_min = 0;
  if (m_sig_y[0] == 0) {
    for (; reduce_y_min < m_sig_y.size(); ++reduce_y_min) {
      if (m_sig_y[reduce_y_min] != 0) {
        break;
      }
    }
  }
  CartCoord reduce_z_min = 0;
  if (m_sig_z[0] == 0) {
    for (; reduce_z_min < m_sig_z.size(); ++reduce_z_min) {
      if (m_sig_z[reduce_z_min] != 0) {
        break;
      }
    }
  }

  // On cherche la première position où il n'y a plus de zéro en partant de la
  // fin.
  CartCoord reduce_x_max = m_sig_x.size() - 1;
  if (m_sig_x[reduce_x_max] == 0) {
    for (; reduce_x_max >= 0; --reduce_x_max) {
      if (m_sig_x[reduce_x_max] != 0) {
        break;
      }
    }
  }
  CartCoord reduce_y_max = m_sig_y.size() - 1;
  if (m_sig_y[reduce_y_max] == 0) {
    for (; reduce_y_max >= 0; --reduce_y_max) {
      if (m_sig_y[reduce_y_max] != 0) {
        break;
      }
    }
  }
  CartCoord reduce_z_max = m_sig_z.size() - 1;
  if (m_sig_z[reduce_z_max] == 0) {
    for (; reduce_z_max >= 0; --reduce_z_max) {
      if (m_sig_z[reduce_z_max] != 0) {
        break;
      }
    }
  }

  // Si une réduction de taille doit avoir lieu en X.
  if (reduce_x_min != 0 || reduce_x_max != m_sig_x.size()-1) {
    // Le patch ne peut pas être "plat".
    if (reduce_x_max < reduce_x_min) {
      ARCANE_FATAL("Bad patch X : no refine cell");
    }
    reduce_x_max++;
    UniqueArray tmp = m_sig_x.subView(reduce_x_min, reduce_x_max - reduce_x_min);
    m_sig_x = tmp;
    CartCoord3 patch_min = m_patch.minPoint();
    CartCoord3 patch_max = m_patch.maxPoint();
    patch_min.x += reduce_x_min;
    patch_max.x = patch_min.x + (reduce_x_max - reduce_x_min);
    m_patch.setMinPoint(patch_min);
    m_patch.setMaxPoint(patch_max);
  }

  if (reduce_y_min != 0 || reduce_y_max != m_sig_y.size()-1) {
    if (reduce_y_max < reduce_y_min) {
      ARCANE_FATAL("Bad patch Y : no refine cell");
    }
    reduce_y_max++;
    UniqueArray tmp = m_sig_y.subView(reduce_y_min, reduce_y_max - reduce_y_min);
    m_sig_y = tmp;
    CartCoord3 patch_min = m_patch.minPoint();
    CartCoord3 patch_max = m_patch.maxPoint();
    patch_min.y += reduce_y_min;
    patch_max.y = patch_min.y + (reduce_y_max - reduce_y_min);
    m_patch.setMinPoint(patch_min);
    m_patch.setMaxPoint(patch_max);
  }

  if (m_mesh->mesh()->dimension() == 3 && (reduce_z_min != 0 || reduce_z_max != m_sig_z.size() - 1)) {
    if (reduce_z_max < reduce_z_min) {
      ARCANE_FATAL("Bad patch Z : no refine cell");
    }
    reduce_z_max++;
    UniqueArray tmp = m_sig_z.subView(reduce_z_min, reduce_z_max - reduce_z_min);
    m_sig_z = tmp;
    CartCoord3 patch_min = m_patch.minPoint();
    CartCoord3 patch_max = m_patch.maxPoint();
    patch_min.z += reduce_z_min;
    patch_max.z = patch_min.z + (reduce_z_max - reduce_z_min);
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
  m_sig_z.fill(0);

  // Le calcul de la signature se fait en deux fois.
  // D'abord, on calcule la signature avec les flags II_Refine.
  // Chaque sous-domaine calcule sa signature locale et une réduction est
  // faite à la fin.
  ENUMERATE_ (Cell, icell, m_mesh->mesh()->ownLevelCells(m_patch.level())) {
    if (!icell->hasFlags(ItemFlags::II_Refine)) {
      continue;
    }

    const CartCoord3 pos = m_numbering->cellUniqueIdToCoord(*icell);
    if (!m_patch.isIn(pos)) {
      continue;
    }
    m_have_cells = true;
    m_sig_x[pos.x - m_patch.minPoint().x]++;
    m_sig_y[pos.y - m_patch.minPoint().y]++;
    m_sig_z[pos.z - m_patch.minPoint().z]++;
  }

  // Compresser un patch vide ne fonctionnera pas.
  m_have_cells = m_mesh->mesh()->parallelMng()->reduce(MessagePassing::ReduceMax, m_have_cells);

  if (m_have_cells) {
    m_mesh->mesh()->parallelMng()->reduce(MessagePassing::ReduceSum, m_sig_x);
    m_mesh->mesh()->parallelMng()->reduce(MessagePassing::ReduceSum, m_sig_y);
    m_mesh->mesh()->parallelMng()->reduce(MessagePassing::ReduceSum, m_sig_z);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool AMRPatchPositionSignature::
isValid() const
{
  if (!m_have_cells) {
    return false;
  }
  if (m_is_null) {
    return false;
  }
  if (m_sig_x.size() < MIN_SIZE || m_sig_y.size() < MIN_SIZE || (m_mesh->mesh()->dimension() == 3 && m_sig_z.size() < MIN_SIZE)) {
    return false;
  }
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool AMRPatchPositionSignature::
canBeCut() const
{
  // m_mesh->traceMng()->info() << "canBeCut() -- m_sig_x.size : " << m_sig_x.size()
  // << " -- m_sig_y.size : " << m_sig_y.size()
  //                            << " -- m_sig_z.size : " << m_sig_z.size()
  //                            << " -- min = " << m_patch.minPoint()
  //                            << " -- max = " << m_patch.maxPoint()
  //                            << " -- length = " << m_patch.length()
  //                            << " -- isValid : " << isValid()
  //                            << " -- efficacity : " << efficacity() << " / " << TARGET_EFFICACITY
  //                            << " -- m_nb_cut : " << m_nb_cut << " / " << MAX_NB_CUT
  //                            << " -- m_stop_cut : " << m_stop_cut;

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
  // m_mesh->traceMng()->info() << "Compute() -- Patch before compute : min = " << m_patch.minPoint() << " -- max = " << m_patch.maxPoint() << " -- length = " << m_patch.length();
  fillSig();
  //m_mesh->traceMng()->info() << "Compute() -- Signature : x = " << m_sig_x << " -- y = " << m_sig_y ;
  compress();
  // m_mesh->traceMng()->info() << "Compute() -- Compress : min = " << m_patch.minPoint() << " -- max = " << m_patch.maxPoint() << " -- x = " << m_sig_x << " -- y = " << m_sig_y << " -- z = " << m_sig_z;
  // m_mesh->traceMng()->info() << "Compute() -- Patch computed :       min = " << m_patch.minPoint() << " -- max = " << m_patch.maxPoint() << " -- length = " << m_patch.length();

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

  // On base le calcul de l'efficacité sur deux choses :
  // - le ratio maille à raffiner dans le patch sur nombre de mailles du patch,
  // - la taille de chaque dimension par rapport à la taille cible.
  // TODO : À peaufiner/réécrire.

  Int32 sum = 0;
  for (const Int32 elem : m_sig_x) {
    sum += elem;
  }

  Real eff = static_cast<Real>(sum) / (m_sig_x.size() * m_sig_y.size() * m_sig_z.size());

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
    eff_xy += static_cast<Real>(m_sig_y.size()) / TARGET_SIZE;
  }
  else if (m_sig_y.size() < TARGET_SIZE * 2) {
    Real size_y = math::abs(m_sig_y.size() - TARGET_SIZE * 2);
    eff_xy += size_y / TARGET_SIZE;
  }

  if (m_mesh->mesh()->dimension() == 2) {
    eff_xy /= 2;
  }
  else {
    if (m_sig_z.size() <= TARGET_SIZE) {
      eff_xy = (eff_xy + (static_cast<Real>(m_sig_z.size()) / TARGET_SIZE)) / 3;
    }
    else if (m_sig_z.size() < TARGET_SIZE * 2) {
      Real size_z = math::abs(m_sig_z.size() - TARGET_SIZE * 2);
      eff_xy = (eff_xy + (size_z / TARGET_SIZE)) / 3;
    }
    else {
      eff_xy /= 3;
    }
  }
  eff_xy *= TARGET_SIZE_WEIGHT_IN_EFFICACITY;

  return (eff+eff_xy)/(1+TARGET_SIZE_WEIGHT_IN_EFFICACITY);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::pair<AMRPatchPositionSignature, AMRPatchPositionSignature> AMRPatchPositionSignature::
cut(Integer dim, CartCoord cut_point) const
{
  auto [fst, snd] = m_patch.cut(cut_point, dim);
  return { AMRPatchPositionSignature(fst, m_mesh, m_nb_cut + 1), AMRPatchPositionSignature(snd, m_mesh, m_nb_cut + 1) };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<CartCoord> AMRPatchPositionSignature::
sigX() const
{
  return m_sig_x;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<CartCoord> AMRPatchPositionSignature::
sigY() const
{
  return m_sig_y;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<CartCoord> AMRPatchPositionSignature::
sigZ() const
{
  return m_sig_z;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AMRPatchPosition AMRPatchPositionSignature::
patch() const
{
  return m_patch;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ICartesianMesh* AMRPatchPositionSignature::
mesh() const
{
  return m_mesh;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool AMRPatchPositionSignature::
stopCut() const
{
  return m_stop_cut;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRPatchPositionSignature::
setStopCut(bool stop_cut)
{
  m_stop_cut = stop_cut;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool AMRPatchPositionSignature::
isComputed() const
{
  return m_is_computed;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
