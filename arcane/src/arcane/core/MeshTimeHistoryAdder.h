// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshTimeHistoryAdder.h                                      (C) 2000-2024 */
/*                                                                           */
/* Class allowing the addition of a value history linked to a mesh.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MESHTIMEHISTORYADDER_H
#define ARCANE_CORE_MESHTIMEHISTORYADDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ITimeHistoryAdder.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/MeshHandle.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class allowing the addition of one or more values to a
 * value history.
 *
 * This class will record curves supported by a mesh.
 * That is, the curves will be linked to a mesh, in addition to being
 * linked to the entire domain or the requested subdomain.
 * If the link to the mesh is not desired, the GlobalTimeHistoryAdder
 * class may be more interesting.
 *
 * For a given history name, there may be only one curve of a
 * or several values per mesh and per subdomain (and one global one for
 * all subdomains).
 *
 * Example: several curves of pressure averages (let's call them
 * "avg_pressure"), two subdomains (0 and 1) and two meshes (mesh0 and mesh1).
 * One value per iteration.
 * - An "avg_pressure" curve linked to subdomain 0 and mesh 0. Each
 *   value is the average of the pressures of each element of mesh 0 and
 *   of subdomain 0.
 * - An "avg_pressure" curve linked to subdomain 0 and mesh 1. Each
 *   value is the average of the pressures of each element of mesh 1 and
 *   of subdomain 0.
 * - An "avg_pressure" curve linked to subdomain 1 and mesh 0. Each
 *   value is the average of the pressures of each element of mesh 0 and
 *   of subdomain 1.
 * - An "avg_pressure" curve linked to subdomain 1 and mesh 1. Each
 *   value is the average of the pressures of each element of mesh 1 and
 *   of subdomain 1.
 * - An "avg_pressure" curve linked to the entire domain and mesh 0.
 *   Each value is the average of the pressures of mesh 0 across each
 *   subdomain.
 * - An "avg_pressure" curve linked to the entire domain and mesh 1.
 *   Each value is the average of the pressures of mesh 1 across each
 *   subdomain.
 *
 * It can be noted that it is possible to have several independent curves
 * with the same name but linked to different meshes and
 * different subdomains (+1 global curve). It is important to
 * emphasize that this same name can also be used with the curves of
 * GlobalTimeHistoryAdder independently, so the example above can be
 * complementary to the one given in the description of GlobalTimeHistoryAdder!
 * (meaning potentially 9 independent curves but with the same name!)
 */
class ARCANE_CORE_EXPORT MeshTimeHistoryAdder
: public ITimeHistoryAdder
{
 public:

  /*!
   * \brief Constructor.
   *
   * \param time_history_mng A pointer to an instance of ITimeHistoryMng.
   * \param mesh_handle The mesh to link to the curves.
   */
  MeshTimeHistoryAdder(ITimeHistoryMng* time_history_mng, const MeshHandle& mesh_handle);
  ~MeshTimeHistoryAdder() override = default;

 public:

  void addValue(const TimeHistoryAddValueArg& thp, Real value) override;
  void addValue(const TimeHistoryAddValueArg& thp, Int32 value) override;
  void addValue(const TimeHistoryAddValueArg& thp, Int64 value) override;
  void addValue(const TimeHistoryAddValueArg& thp, RealConstArrayView values) override;
  void addValue(const TimeHistoryAddValueArg& thp, Int32ConstArrayView values) override;
  void addValue(const TimeHistoryAddValueArg& thp, Int64ConstArrayView values) override;

 private:

  ITimeHistoryMng* m_thm;
  MeshHandle m_mesh_handle;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
