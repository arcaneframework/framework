// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TODO                                          (C) 2000-2024 */
/*                                                                           */
/*                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ITimeHistoryAdder.h"
#include "arcane/core/ITimeHistoryMng.h"
#include "arcane/core/IMesh.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_IMPL_EXPORT MeshTimeHistoryAdder
: public ITimeHistoryAdder
{
 public:
  MeshTimeHistoryAdder(ITimeHistoryMng* thm, IMesh* mesh);
  ~MeshTimeHistoryAdder() override = default;

 public:
  void addValue(const String& name, Real value, bool end_time = true, bool is_local = false) override;

 private:
  ITimeHistoryMng* m_thm;
  IMesh* m_mesh;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
