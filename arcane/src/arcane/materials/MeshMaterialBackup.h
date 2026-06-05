// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialBackup.h                                        (C) 2000-2023 */
/*                                                                           */
/* Saving/restoring material and medium values.                              */
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
 * \brief Saving/restoring material and medium values.
 *
 * Instances of this class can only be used for a single save and a single restore.
 *
 * It is possible to specify the name of a compression service
 * for the data before saving via the setCompressorServiceName() method.
 * If this method is not called, the default value is that of
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

  //! Name of the service used to compress the data.
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

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
