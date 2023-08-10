// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialBackup.h                                        (C) 2000-2023 */
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
#include "arcane/utils/String.h"

#include "arcane/core/ItemUniqueId.h"

#include "arcane/materials/MaterialsGlobal.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Sauvegarde/restoration des valeurs des matériaux et milieux.
 *
 * Les instances de cette classe ne peuvent être utilisées que pour une
 * seule sauvegarde et une seule restauration.
 *
 * Il est possible de spécifier le nom d'un service de compression
 * pour les données avant les sauvegardes via la méthode setCompressorServiceName().
 * Si cette méthode n'est pas appelée, la valeur par défaut est celle de
 * IMeshMaterialMng::dataCompressorServiceName().
 */
class ARCANE_MATERIALS_EXPORT MeshMaterialBackup
: public TraceAccessor
{
  struct VarData;

 public:

  MeshMaterialBackup(IMeshMaterialMng* mm, bool use_unique_ids);
  ~MeshMaterialBackup();

 public:

  //! Nom du service utilisé pour compresser les données.
  void setCompressorServiceName(const String& name);
  const String& compressorServiceName() const { return m_compressor_service_name; }

 public:

  void saveValues();
  void restoreValues();

 private:

  IMeshMaterialMng* m_material_mng;
  bool m_use_unique_ids;
  std::map<IMeshMaterialVariable*, VarData*> m_saved_data;
  std::map<IMeshComponent*, SharedArray<Int32>> m_ids_array;
  std::map<IMeshComponent*, SharedArray<ItemUniqueId>> m_unique_ids_array;
  UniqueArray<IMeshMaterialVariable*> m_vars;
  bool m_use_v2 = false;
  String m_compressor_service_name;

 private:

  void _save();
  void _restore();
  void _saveIds(IMeshComponent* component);
  bool _isValidComponent(IMeshMaterialVariable* var, IMeshComponent* component);
  void _saveV1();
  void _saveV2();
  void _restoreV1();
  void _restoreV2();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

