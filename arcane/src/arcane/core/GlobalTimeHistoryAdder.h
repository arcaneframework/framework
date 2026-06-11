// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GlobalTimeHistoryAdder.h                                    (C) 2000-2024 */
/*                                                                           */
/* Class allowing the addition of a global value history.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_GLOBALTIMEHISTORYADDER_H
#define ARCANE_CORE_GLOBALTIMEHISTORYADDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ITimeHistoryAdder.h"

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
 * This class will record the curves globally, without support.
 * That is, the curves will only be linked to the complete domain or the
 * requested subdomain, unlike MeshTimeHistoryAdder which links the
 * curves to the desired mesh.
 *
 * For a given history name, there can only be one curve of one
 * or more values per subdomain (and one global one for all
 * subdomains).
 *
 * Example: several curves of pressure averages (let's call them
 * "avg_pressure") and two subdomains (0 and 1). One value per iteration.
 * - An "avg_pressure" curve linked to subdomain 0. Each value is the
 *   average of the pressures of each cell in subdomain 0.
 * - An "avg_pressure" curve linked to subdomain 1. Each value is the
 *   average of the pressures of each cell in subdomain 1.
 * - An "avg_pressure" curve linked to the complete domain. Each value is the
 *   average of the pressures of each subdomain.
 *
 * It can be noted that it is possible to have several curves
 * independent ones with the same name but linked to different subdomains
 * (+1 global curve).
 */
class ARCANE_CORE_EXPORT GlobalTimeHistoryAdder
: public ITimeHistoryAdder
{
 public:

  /*!
   * \brief Constructor.
   *
   * \param time_history_mng A pointer to an instance of ITimeHistoryMng.
   */
  explicit GlobalTimeHistoryAdder(ITimeHistoryMng* time_history_mng);
  ~GlobalTimeHistoryAdder() override = default;

 public:

  void addValue(const TimeHistoryAddValueArg& thp, Real value) override;
  void addValue(const TimeHistoryAddValueArg& thp, Int32 value) override;
  void addValue(const TimeHistoryAddValueArg& thp, Int64 value) override;
  void addValue(const TimeHistoryAddValueArg& thp, RealConstArrayView values) override;
  void addValue(const TimeHistoryAddValueArg& thp, Int32ConstArrayView values) override;
  void addValue(const TimeHistoryAddValueArg& thp, Int64ConstArrayView values) override;

 private:

  ITimeHistoryMng* m_thm;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
