// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialExchangeMng.h                                   (C) 2000-2016 */
/*                                                                           */
/* Gestion de l'échange des matériaux entre sous-domaines.                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MESHMATERIALEXCHANGEMNG_H
#define ARCANE_MATERIALS_MESHMATERIALEXCHANGEMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/materials/MaterialsGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
class IItemFamilySerializeStepFactory;
MATERIALS_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MeshMaterialMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Gestion de l'échange des matériaux entre sous-domaines.
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
