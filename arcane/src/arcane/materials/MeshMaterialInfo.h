// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialInfo.h                                          (C) 2000-2022 */
/*                                                                           */
/* Information about a material of a mesh.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MESHMATERIALINFO_H
#define ARCANE_MATERIALS_MESHMATERIALINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/Array.h"
#include "arcane/materials/MaterialsGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Info about a material of a mesh.
 *
 * This instance contains the information about a material.
 * This information is static. Instances of this class should not
 * be created directly. They are created via the call to
 * IMeshMaterialMng::registerMaterialInfo().
 */
class MeshMaterialInfo
{
  friend class MeshMaterialMng;

 private:

  MeshMaterialInfo(IMeshMaterialMng* mng, const String& name);
  virtual ~MeshMaterialInfo() = default;

 public:

  //! Associated manager.
  IMeshMaterialMng* materialMng() { return m_material_mng; }

  //! Material name.
  String name() const { return m_name; }

  //! Names of the environments in which this material is present
  ConstArrayView<String> environmentsName() const { return m_environments_name; }

 protected:

  void _addEnvironment(const String& env_name)
  {
    m_environments_name.add(env_name);
  }

 private:

  IMeshMaterialMng* m_material_mng;
  String m_name;
  //! List of environments to which the material belongs
  UniqueArray<String> m_environments_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
