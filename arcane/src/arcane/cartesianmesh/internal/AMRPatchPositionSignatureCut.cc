// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AMRPatchPositionSignatureCut.cc                             (C) 2000-2026 */
/*                                                                           */
/* Patch cutting methods based on their signatures.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/internal/AMRPatchPositionSignatureCut.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/Math.h"

#include "arcane/core/IMesh.h"

#include "arcane/cartesianmesh/ICartesianMesh.h"

#include "arcane/cartesianmesh/internal/AMRPatchPositionSignature.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

namespace
{
  constexpr Integer MIN_SIZE = 1;
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AMRPatchPositionSignatureCut::
AMRPatchPositionSignatureCut() = default;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AMRPatchPositionSignatureCut::
~AMRPatchPositionSignatureCut() = default;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartCoord AMRPatchPositionSignatureCut::
_cutDim(ConstArrayView<CartCoord> sig)
{
  // If the cut produces patches that are too small, we return -1.
  if (sig.size() < MIN_SIZE * 2) {
    return -1;
  }

  CartCoord cut_point = -1;
  CartCoord mid = sig.size() / 2;

  // Hole part.
  // We search for a hole in the signature.
  // The signature must have been compressed beforehand
  // (AMRPatchPositionSignature::compress()).
  {
    for (CartCoord i = 0; i < sig.size(); ++i) {
      if (sig[i] == 0) {
        cut_point = i;
        break;
      }
    }

    if (cut_point == 0) {
      ARCANE_FATAL("Call AMRPatchPositionSignature::compress() before");
    }
    if (cut_point != -1 && cut_point >= MIN_SIZE && sig.size() - cut_point >= MIN_SIZE) {
      return cut_point;
    }
  }

#if 0
  // Second derivative part.
  // Does not necessarily produce better results compared to the middle cut part.
  {
    UniqueArray<CartCoord> dsec_sig(sig.size(), 0);

    CartCoord max = -1;
    for (CartCoord i = 1; i < dsec_sig.size() - 1; ++i) {
      dsec_sig[i] = sig[i + 1] - 2 * sig[i] + sig[i - 1];
      CartCoord dif = math::abs(dsec_sig[i - 1] - dsec_sig[i]);
      if (dif > max) {
        cut_point = i;
        max = dif;
      }
      else if (dif == max && math::abs(cut_point - mid) > math::abs(i - mid)) {
        cut_point = i;
      }
    }

    if (cut_point != -1 && cut_point >= MIN_SIZE && sig.size() - cut_point >= MIN_SIZE) {
      return cut_point;
    }
  }
#endif

  // Middle cut part.
  {
    cut_point = mid;

    if (cut_point != -1 && cut_point >= MIN_SIZE && sig.size() - cut_point >= MIN_SIZE) {
      return cut_point;
    }
  }

  return -1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::pair<AMRPatchPositionSignature, AMRPatchPositionSignature> AMRPatchPositionSignatureCut::
cut(const AMRPatchPositionSignature& sig)
{
  // We cut along the three dimensions.
  CartCoord cut_point_x = _cutDim(sig.sigX());
  CartCoord cut_point_y = _cutDim(sig.sigY());
  CartCoord cut_point_z = (sig.mesh()->mesh()->dimension() == 2 ? -1 : _cutDim(sig.sigZ()));

  // If it is impossible to cut along one of the dimensions.
  if (cut_point_x == -1 && cut_point_y == -1 && cut_point_z == -1) {
    return {};
  }

  // We adjust the cut points to pass them to the AMRPatchPositionSignature::cut() method.
  if (cut_point_x != -1) {
    cut_point_x += sig.patch().minPoint().x;
  }
  if (cut_point_y != -1) {
    cut_point_y += sig.patch().minPoint().y;
  }
  if (cut_point_z != -1) {
    cut_point_z += sig.patch().minPoint().z;
  }

  // We must choose the best cut point among the three.
  if (cut_point_x != -1 && cut_point_y != -1 && cut_point_z != -1) {

    // To choose, we cut, calculate the efficiency of the resulting
    // patches, and choose the most efficient cut.
    // TODO: Since the compute() methods are collective methods performing
    // several small reductions, this part should be optimized by combining
    // the reductions.
    Real x_efficacity = 0;
    auto [fst_x, snd_x] = sig.cut(MD_DirX, cut_point_x);
    {
      // sig.mesh()->traceMng()->info() << "Cut() -- Compute X -- Cut Point : " << cut_point_x;

      fst_x.compute();
      snd_x.compute();
      if (fst_x.isValid() && snd_x.isValid()) {

        // sig.mesh()->traceMng()->info() << "Cut() -- X.fst_x"
        //                                << " -- min = " << fst_x.patch().minPoint()
        //                                << " -- max = " << fst_x.patch().maxPoint()
        //                                << " -- efficacity : " << fst_x.efficacity();
        // sig.mesh()->traceMng()->info() << "Cut() -- X.snd_x"
        //                                << " -- min = " << snd_x.patch().minPoint()
        //                                << " -- max = " << snd_x.patch().maxPoint()
        //                                << " -- efficacity : " << snd_x.efficacity();

        x_efficacity = (fst_x.efficacity() + snd_x.efficacity()) / 2;
        // sig.mesh()->traceMng()->info() << "Cut() -- efficacity X : " << x_efficacity;
      }
      // else {
      //   sig.mesh()->traceMng()->info() << "Cut() -- Compute X invalid (too small) -- fst_x.length() : " << fst_x.patch().length() << " -- snd_x.length() : " << snd_x.patch().length();
      // }
    }

    Real y_efficacity = 0;
    auto [fst_y, snd_y] = sig.cut(MD_DirY, cut_point_y);
    {
      // sig.mesh()->traceMng()->info() << "Cut() -- Compute Y -- Cut Point : " << cut_point_y;

      fst_y.compute();
      snd_y.compute();
      if (fst_y.isValid() && snd_y.isValid()) {

        // sig.mesh()->traceMng()->info() << "Cut() -- Y.fst_y"
        //                                << " -- min = " << fst_y.patch().minPoint()
        //                                << " -- max = " << fst_y.patch().maxPoint()
        //                                << " -- efficacity : " << fst_y.efficacity();
        // sig.mesh()->traceMng()->info() << "Cut() -- Y.snd_y"
        //                                << " -- min = " << snd_y.patch().minPoint()
        //                                << " -- max = " << snd_y.patch().maxPoint()
        //                                << " -- efficacity : " << snd_y.efficacity();

        y_efficacity = (fst_y.efficacity() + snd_y.efficacity()) / 2;
        // sig.mesh()->traceMng()->info() << "Cut() -- efficacity Y : " << y_efficacity;
      }
      // else {
      //   sig.mesh()->traceMng()->info() << "Cut() -- Compute Y invalid (too small) -- fst_y.length() : " << fst_y.patch().length() << " -- snd_y.length() : " << snd_y.patch().length();
      // }
    }

    Real z_efficacity = 0;
    auto [fst_z, snd_z] = sig.cut(MD_DirZ, cut_point_z);
    {
      // sig.mesh()->traceMng()->info() << "Cut() -- Compute Z -- Cut Point : " << cut_point_z;

      fst_z.compute();
      snd_z.compute();
      if (fst_z.isValid() && snd_z.isValid()) {

        // sig.mesh()->traceMng()->info() << "Cut() -- Z.fst_z"
        //                                << " -- min = " << fst_z.patch().minPoint()
        //                                << " -- max = " << fst_z.patch().maxPoint()
        //                                << " -- efficacity : " << fst_z.efficacity();
        // sig.mesh()->traceMng()->info() << "Cut() -- Z.snd_z"
        //                                << " -- min = " << snd_z.patch().minPoint()
        //                                << " -- max = " << snd_z.patch().maxPoint()
        //                                << " -- efficacity : " << snd_z.efficacity();

        z_efficacity = (fst_z.efficacity() + snd_z.efficacity()) / 2;
        // sig.mesh()->traceMng()->info() << "Cut() -- efficacity Z : " << z_efficacity;
      }
      // else {
      //   sig.mesh()->traceMng()->info() << "Cut() -- Compute Z invalid (too small) -- fst_z.length() : " << fst_z.patch().length() << " -- snd_z.length() : " << snd_z.patch().length();
      // }
    }

    // If the cut does not improve the efficiency, we return.
    {
      Real sig_efficacity = sig.efficacity();
      if (sig_efficacity > x_efficacity && sig_efficacity > y_efficacity && sig_efficacity > z_efficacity) {
        return {};
      }
    }

    // We return the best efficiency.
    if (x_efficacity >= y_efficacity && x_efficacity >= z_efficacity && x_efficacity != 0) {
      return { fst_x, snd_x };
    }
    if (y_efficacity >= x_efficacity && y_efficacity >= z_efficacity && y_efficacity != 0) {
      return { fst_y, snd_y };
    }
    if (z_efficacity == 0) {
      ARCANE_FATAL("Invalid cut");
    }
    return { fst_z, snd_z };
  }

  // Same principle as above.
  if (cut_point_x != -1 && cut_point_y != -1) {
    Real x_efficacity = 0;
    auto [fst_x, snd_x] = sig.cut(MD_DirX, cut_point_x);
    {
      // sig.mesh()->traceMng()->info() << "Cut() -- Compute X -- Cut Point : " << cut_point_x;

      fst_x.compute();
      snd_x.compute();
      if (fst_x.isValid() && snd_x.isValid()) {

        // sig.mesh()->traceMng()->info() << "Cut() -- X.fst_x"
        //                                << " -- min = " << fst_x.patch().minPoint()
        //                                << " -- max = " << fst_x.patch().maxPoint()
        //                                << " -- efficacity : " << fst_x.efficacity();
        // sig.mesh()->traceMng()->info() << "Cut() -- X.snd_x"
        //                                << " -- min = " << snd_x.patch().minPoint()
        //                                << " -- max = " << snd_x.patch().maxPoint()
        //                                << " -- efficacity : " << snd_x.efficacity();

        x_efficacity = (fst_x.efficacity() + snd_x.efficacity()) / 2;
        // sig.mesh()->traceMng()->info() << "Cut() -- efficacity X : " << x_efficacity;
      }
      // else {
      //   sig.mesh()->traceMng()->info() << "Cut() -- Compute X invalid (too small) -- fst_x.length() : " << fst_x.patch().length() << " -- snd_x.length() : " << snd_x.patch().length();
      // }
    }

    Real y_efficacity = 0;
    auto [fst_y, snd_y] = sig.cut(MD_DirY, cut_point_y);
    {
      // sig.mesh()->traceMng()->info() << "Cut() -- Compute Y -- Cut Point : " << cut_point_y;

      fst_y.compute();
      snd_y.compute();
      if (fst_y.isValid() && snd_y.isValid()) {

        // sig.mesh()->traceMng()->info() << "Cut() -- Y.fst_y"
        //                                << " -- min = " << fst_y.patch().minPoint()
        //                                << " -- max = " << fst_y.patch().maxPoint()
        //                                << " -- efficacity : " << fst_y.efficacity();
        // sig.mesh()->traceMng()->info() << "Cut() -- Y.snd_y"
        //                                << " -- min = " << snd_y.patch().minPoint()
        //                                << " -- max = " << snd_y.patch().maxPoint()
        //                                << " -- efficacity : " << snd_y.efficacity();

        y_efficacity = (fst_y.efficacity() + snd_y.efficacity()) / 2;
        // sig.mesh()->traceMng()->info() << "Cut() -- efficacity Y : " << y_efficacity;
      }
      // else {
      //   sig.mesh()->traceMng()->info() << "Cut() -- Compute Y invalid (too small) -- fst_y.length() : " << fst_y.patch().length() << " -- snd_y.length() : " << snd_y.patch().length();
      // }
    }
    {
      Real sig_efficacity = sig.efficacity();
      if (sig_efficacity > x_efficacity && sig_efficacity > y_efficacity) {
        return {};
      }
    }

    if (x_efficacity >= y_efficacity && x_efficacity != 0) {
      return { fst_x, snd_x };
    }
    if (y_efficacity == 0) {
      ARCANE_FATAL("Invalid cut");
    }
    return { fst_y, snd_y };
  }

  if (cut_point_x != -1) {
    Real x_efficacity = 0;
    auto [fst_x, snd_x] = sig.cut(MD_DirX, cut_point_x);

    // sig.mesh()->traceMng()->info() << "Cut() -- Compute X -- Cut Point : " << cut_point_x;

    fst_x.compute();
    snd_x.compute();
    if (fst_x.isValid() && snd_x.isValid()) {

      // sig.mesh()->traceMng()->info() << "Cut() -- efficacity X.fst_x : " << fst_x.efficacity();
      // sig.mesh()->traceMng()->info() << "Cut() -- efficacity X.snd_x : " << snd_x.efficacity();
      x_efficacity = (fst_x.efficacity() + snd_x.efficacity()) / 2;
      // sig.mesh()->traceMng()->info() << "Cut() -- efficacity X : " << x_efficacity;
    }
    // else {
    //   sig.mesh()->traceMng()->info() << "Cut() -- Compute X invalid (too small) -- fst_x.length() : " << fst_x.patch().length() << " -- snd_x.length() : " << snd_x.patch().length();
    // }
    if (sig.efficacity() > x_efficacity) {
      return {};
    }
    return { fst_x, snd_x };
  }

  if (cut_point_y != -1) {
    Real y_efficacity = 0;
    auto [fst_y, snd_y] = sig.cut(MD_DirY, cut_point_y);

    // sig.mesh()->traceMng()->info() << "Cut() -- Compute Y -- Cut Point : " << cut_point_y;

    fst_y.compute();
    snd_y.compute();
    if (fst_y.isValid() && snd_y.isValid()) {

      // sig.mesh()->traceMng()->info() << "Cut() -- efficacity Y.fst_y : " << fst_y.efficacity();
      // sig.mesh()->traceMng()->info() << "Cut() -- efficacity Y.snd_y : " << snd_y.efficacity();
      y_efficacity = (fst_y.efficacity() + snd_y.efficacity()) / 2;
      // sig.mesh()->traceMng()->info() << "Cut() -- efficacity Y : " << y_efficacity;
    }
    // else {
    //   sig.mesh()->traceMng()->info() << "Cut() -- Compute Y invalid (too small) -- fst_y.length() : " << fst_y.patch().length() << " -- snd_y.length() : " << snd_y.patch().length();
    // }
    if (sig.efficacity() > y_efficacity) {
      return {};
    }
    return { fst_y, snd_y };
  }
  if (cut_point_z != -1) {
    Real z_efficacity = 0;
    auto [fst_z, snd_z] = sig.cut(MD_DirZ, cut_point_z);

    // sig.mesh()->traceMng()->info() << "Cut() -- Compute Z -- Cut Point : " << cut_point_z;

    fst_z.compute();
    snd_z.compute();
    if (fst_z.isValid() && snd_z.isValid()) {

      // sig.mesh()->traceMng()->info() << "Cut() -- efficacity Z.fst_z : " << fst_z.efficacity();
      // sig.mesh()->traceMng()->info() << "Cut() -- efficacity Z.snd_z : " << snd_z.efficacity();
      z_efficacity = (fst_z.efficacity() + snd_z.efficacity()) / 2;
      // sig.mesh()->traceMng()->info() << "Cut() -- efficacity Z : " << z_efficacity;
    }
    // else {
    //   sig.mesh()->traceMng()->info() << "Cut() -- Compute Z invalid (too small) -- fst_z.length() : " << fst_z.patch().length() << " -- snd_z.length() : " << snd_z.patch().length();
    // }
    if (sig.efficacity() > z_efficacity) {
      return {};
    }
    return { fst_z, snd_z };
  }
  return {};
}

void AMRPatchPositionSignatureCut::
cut(UniqueArray<AMRPatchPositionSignature>& sig_array_a)
{
  // We swap in and out at each iteration.
  UniqueArray<AMRPatchPositionSignature> sig_array_b;
  bool a_a_b_b = false;

  // As long as the cut is possible.
  bool need_cut = true;
  while (need_cut) {
    need_cut = false;
    a_a_b_b = !a_a_b_b;

    UniqueArray<AMRPatchPositionSignature>& array_in = a_a_b_b ? sig_array_a : sig_array_b;
    UniqueArray<AMRPatchPositionSignature>& array_out = a_a_b_b ? sig_array_b : sig_array_a;

    for (auto& sig : array_in) {
      // sig.mesh()->traceMng()->info() << "Cut() -- i : " << i;

      // If the cut is still possible.
      if (!sig.stopCut()) {
        if (!sig.isComputed()) {
          sig.compute();
        }
        if (sig.canBeCut()) {
          auto [fst, snd] = cut(sig);

          // If the cut is valid, we add the two new patches to the out array.
          if (fst.isValid()) {
            need_cut = true;
            array_out.add(fst);
            array_out.add(snd);
            // sig.mesh()->traceMng()->info() << "First Signature :";
            // sig.mesh()->traceMng()->info() << "\tmin = " << fst.patch().minPoint() << " -- max = " << fst.patch().maxPoint() << " -- length = " << fst.patch().length();
            // sig.mesh()->traceMng()->info() << "Second Signature :";
            // sig.mesh()->traceMng()->info() << "\tmin = " << snd.patch().minPoint() << " -- max = " << snd.patch().maxPoint() << " -- length = " << snd.patch().length();
            continue;
          }
          // If the cut does not produce a valid patch, we stop cutting for this patch.
          sig.setStopCut(true);
          // sig.mesh()->traceMng()->info() << "Invalid Signature";
        }
        // If the patch can no longer be cut, we stop cutting this patch.
        else {
          sig.setStopCut(true);
        }
      }
      // sig.mesh()->traceMng()->info() << "No Update";
      // sig.mesh()->traceMng()->info() << "\tmin = " << sig.patch().minPoint() << " -- max = " << sig.patch().maxPoint();
      // If the patch could not be cut, we keep it in the out array.
      // TODO: Meh
      if (sig.isValid()) {
        array_out.add(sig);
      }
    }
    array_in.clear();
  }

  if (a_a_b_b) {
    sig_array_a.clear();
    sig_array_a = sig_array_b;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
