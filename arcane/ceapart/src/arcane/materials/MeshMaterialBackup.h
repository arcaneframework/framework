// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialBackup.h                                        (C) 2000-2018 */
/*                                                                           */
/* Sauvegarde/restauration des valeurs des matériaux et milieux.             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MESHMATERIALBACKUP_H
#define ARCANE_MATERIALS_MESHMATERIALBACKUP_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/ArrayView.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/Ref.h"

#include "arcane/ItemUniqueId.h"

#include "arcane/materials/MaterialsGlobal.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMesh;
class IData;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMeshMaterialMng;
class IMeshMaterial;
class IMeshMaterialBackup;
class IMeshMaterialVariable;
class IMeshComponent;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Sauvegarde/restoration des valeurs des matériaux et milieux.
 */
class ARCANE_MATERIALS_EXPORT MeshMaterialBackup
: public TraceAccessor
{
  struct VarData
  {
    VarData() : data_index(0){}
    VarData(Ref<IData> d) : data(d), data_index(0){}
    ~VarData();
    Ref<IData> data;
    Integer data_index;
  };
 public:

  MeshMaterialBackup(IMeshMaterialMng* mm,bool use_unique_ids);
  ~MeshMaterialBackup();

 public:

  void saveValues();

  void restoreValues();

 private:

  IMeshMaterialMng* m_material_mng;
  bool m_use_unique_ids;
  std::map<IMeshMaterialVariable*,VarData*> m_saved_data;
  std::map<IMeshComponent*,SharedArray<Int32>> m_ids_array;
  std::map<IMeshComponent*,SharedArray<ItemUniqueId>> m_unique_ids_array;
  UniqueArray<IMeshMaterialVariable*> m_vars;

 private:
  
  void _save();
  void _restore();
  void _saveIds(IMeshComponent* component);
  bool _isValidComponent(IMeshMaterialVariable* var,IMeshComponent* component);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

