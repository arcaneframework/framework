// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SodStandardGroupsBuilder.h                                  (C) 2000-2025 */
/*                                                                           */
/* Creation of groups for Sod shock tube test cases.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_INTERNAL_SODSTANDARDGROUPSBUILDER_H
#define ARCANE_STD_INTERNAL_SODSTANDARDGROUPSBUILDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class for creating standard groups for a Sod shock tube.
 *
 * The created groups are the face groups corresponding to the sides of the
 * meshes (XMIN,XMAX,YMIN,YMAX,ZMIN,ZMAX), the left cell groups (ZG)
 * and right (ZD) along the X-axis, and for the right group
 * the top part (ZD_HAUT) and the bottom part (ZD_BAS).
 *
 * \sa SodMeshGenerator
 */
class SodStandardGroupsBuilder
: public TraceAccessor
{
 public:

  explicit SodStandardGroupsBuilder(ITraceMng* tm)
  : TraceAccessor(tm)
  {}

 public:

  /*!
   * \brief Creates the groups for initializing a Sod shock tube.
   *
   * The groups corresponding to the boundaries ((X|Y|Z)(MIN|MAX) are always created.
   * The other groups corresponding to the left and right zones for
   * a Sod shock tube are created if \a do_zg_and_zd is true.
   */
  void generateGroups(IMesh* mesh, Real3 min_pos, Real3 max_pos,
                      Real middle_x, Real middle_height, bool do_zg_and_zd);

 private:

  void _createFaceGroup(IMesh* mesh, const String& name, Int32ConstArrayView faces_lid);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
