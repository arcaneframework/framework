// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialInfo.h                                          (C) 2000-2012 */
/*                                                                           */
/* Informations d'un matériau d'un maillage.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MESHMATERIALINFO_H
#define ARCANE_MATERIALS_MESHMATERIALINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/materials/MaterialsGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMeshMaterialMng;

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
 public:

  MeshMaterialInfo(IMeshMaterialMng* mng,const String& name);
  virtual ~MeshMaterialInfo(){}

 public:

  //! Gestionnaire associé.
  virtual IMeshMaterialMng* materialMng() { return m_material_mng; }
  
  //! Nom du matériau.
  virtual const String& name() const { return m_name; }

 public:

  void build();

 private:

  IMeshMaterialMng* m_material_mng;
  String m_name;
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

