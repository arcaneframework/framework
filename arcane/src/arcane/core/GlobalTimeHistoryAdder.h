// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GlobalTimeHistoryAdder.h                                    (C) 2000-2024 */
/*                                                                           */
/* Classe permettant d'ajouter un historique de valeur global.               */
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

class ARCANE_CORE_EXPORT GlobalTimeHistoryAdder
: public ITimeHistoryAdder
{
 public:
  explicit GlobalTimeHistoryAdder(ITimeHistoryMng* thm);
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
