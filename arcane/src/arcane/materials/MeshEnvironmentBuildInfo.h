// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshEnvironmentBuildInfo.h                                  (C) 2000-2023 */
/*                                                                           */
/* Information for creating an environment.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MESHENVIRONMENTBUILDINFO_H
#define ARCANE_MATERIALS_MESHENVIRONMENTBUILDINFO_H
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
 * \ingroup ArcaneMaterials
 * \brief Information for creating an environment.
 *
 * This instance contains the necessary information to create an environment.
 * Once the information is specified, the environment must be created
 * via IMeshMaterialMng::createEnvironment().
 *
 * For now, the only relevant information about an environment is its
 * name and the list of materials composing it.
 */
class ARCANE_MATERIALS_EXPORT MeshEnvironmentBuildInfo
{
 public:
  class MatInfo
  {
   public:
    MatInfo(const String& name) : m_name(name){}
   public:
    String m_name;
   public:
    // The default constructor should not be available but it crashes at
    // compilation with VS2010 if it is absent
    MatInfo() {}
  };
 public:

  MeshEnvironmentBuildInfo(const String& name);
  ~MeshEnvironmentBuildInfo();

 public:

  //! Name of the environment
  const String& name() const { return m_name; }

  /*!
   * \brief Adds the material named \a name to the environment
   *
   * The material must already have been registered via
   * IMeshMaterialMng::registerMaterialInfo().
   */
  void addMaterial(const String& name);

 public:

  /*!
   * \internal
   * List of materials.
   */
  ConstArrayView<MatInfo> materials() const
  {
    return m_materials;
  }

 private:

  String m_name;
  UniqueArray<MatInfo> m_materials;

  void _checkValid(const String& name);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
