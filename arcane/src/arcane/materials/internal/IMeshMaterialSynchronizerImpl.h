// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshMaterialSynchronizerImpl.h                                  (C) 2000-2023 */
/*                                                                           */
/* Synchronisation de la liste des matériaux/milieux des entités.            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_INTERNAL_IMESHMATERIALSYNCHRONIZERIMPL_H
#define ARCANE_MATERIALS_INTERNAL_IMESHMATERIALSYNCHRONIZERIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/ArrayView.h"

#include "arcane/VariableTypedef.h"

#include "arcane/materials/MaterialsGlobal.h"
#include "arcane/materials/MatItem.h"

#include "arcane/materials/CellToAllEnvCellConverter.h"
#include "arcane/accelerator/core/RunQueue.h"

#include "arcane/accelerator/Accelerator.h"
#include "arcane/accelerator/IndexSelecter.h"
#include "arcane/accelerator/SpanViews.h"
#include "arcane/IApplication.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/accelerator/Reduce.h"
#include "arcane/accelerator/Runner.h"
#include "arcane/accelerator/VariableViews.h"
#include "arcane/accelerator/Accelerator.h"
#include "arcane/accelerator/RunCommandLoop.h"
#include "arcane/accelerator/RunCommandEnumerate.h"
#include "arcane/accelerator/core/RunQueueBuildInfo.h"
#include "arcane/accelerator/core/Memory.h"
#include "arcane/accelerator/MaterialVariableViews.h"
#include "arcane/accelerator/RunCommandMaterialEnumerate.h"
#include "arcane/accelerator/core/IAcceleratorMng.h"
#include "arcane/accelerator/core/RunQueueEvent.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Stratégie de synchronisation de la liste des matériaux/milieux des entités.
 *
 * Cette classe abstraite determine la méthode de syncrhonisation entre les sous-domaines la liste
 * des matériaux/milieux auxquelles une maille appartient.
 *
 */
class IMeshMaterialSynchronizerImpl
{
protected:

explicit IMeshMaterialSynchronizerImpl(){};

public:
  virtual ~IMeshMaterialSynchronizerImpl(){};

  /*!
   * \brief Synchronisation de la liste des matériaux/milieux des entités.
   *
   * Cette opération est collective.
   *
   * Retourne \a true si des mailles ont été ajoutées ou supprimées d'un matériau
   * ou d'un milieu lors de cette opération pour ce sous-domaine.
   */
  virtual bool synchronizeMaterialsInCells() = 0;


};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
