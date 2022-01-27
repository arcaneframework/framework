// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialInfo.h                                          (C) 2000-2022 */
/*                                                                           */
/* Informations d'un matériau d'un maillage.                                 */
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
 * \brief Infos d'un matériau d'un maillage.
 *
 * Cette instance contient les infos d'un matériau.
 * Ces informations sont statiques. Les instances de cette classe ne
 * doivent pas être créées directement. Elles le sont via l'appel à
 * IMeshMaterialMng::registerMaterialInfo().
 */
class MeshMaterialInfo
{
  friend class MeshMaterialMng;

 private:

  MeshMaterialInfo(IMeshMaterialMng* mng,const String& name);
  virtual ~MeshMaterialInfo() = default;

 public:

  //! Gestionnaire associé.
  IMeshMaterialMng* materialMng() { return m_material_mng; }
  
  //! Nom du matériau.
  String name() const { return m_name; }

  //! Nom des milieux dans lequel ce matériau est présent
  ConstArrayView<String> environmentsName() const { return m_environments_name; }

 protected:

  void _addEnvironment(const String& env_name)
  {
    m_environments_name.add(env_name);
  }

 private:

  IMeshMaterialMng* m_material_mng;
  String m_name;
  //! Liste des milieux auquel le matériau appartient
  UniqueArray<String> m_environments_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

