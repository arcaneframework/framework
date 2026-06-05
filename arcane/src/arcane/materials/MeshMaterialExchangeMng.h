// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialExchangeMng.h                                   (C) 2000-2016 */
/*                                                                           */
/* Management of material exchange between subdomains.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MESHMATERIALEXCHANGEMNG_H
#define ARCANE_MATERIALS_MESHMATERIALEXCHANGEMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/materials/MaterialsGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IItemFamilySerializeStepFactory;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MeshMaterialMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Management of material exchange between subdomains.
 */
class MeshMaterialExchangeMng
: public TraceAccessor
{
 public:

  class ExchangeCellFactory;
  class ExchangeCellStep;

 public:

  MeshMaterialExchangeMng(MeshMaterialMng* material_mng);
  virtual ~MeshMaterialExchangeMng();

 public:

  void build();
  IMeshMaterialMng* materialMng() const;
  void registerFactory();
  bool isInMeshMaterialExchange() const
  {
    return m_is_in_mesh_material_exchange;
  }

 public:

  MeshMaterialMng* m_material_mng;
  IItemFamilySerializeStepFactory* m_serialize_cells_factory;
  bool m_is_in_mesh_material_exchange;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
