// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariableFactoryMng.cc                           (C) 2000-2022 */
/*                                                                           */
/* Gestionnaire des fabriques de variables matériaux.                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/materials/MaterialVariableTypeInfo.h"
#include "arcane/core/materials/IMeshMaterialVariableFactory.h"
#include "arcane/core/materials/IMeshMaterialVariableFactoryMng.h"
#include "arcane/core/materials/IMeshMaterialMng.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// TODO: Utiliser Ref<>
class MeshMaterialVariableFactoryMng
: public TraceAccessor
, public IMeshMaterialVariableFactoryMng
{
 public:

  MeshMaterialVariableFactoryMng(IMeshMaterialMng* mm);
  ~MeshMaterialVariableFactoryMng() override;

 public:

  void build() override;
  ITraceMng* traceMng() const override;
  void registerFactory(Ref<IMeshMaterialVariableFactory> factory) override;
  IMeshMaterialVariable* createVariable(const String& storage_type,
                                        const MaterialVariableBuildInfo& build_info) override;

 private:

  IMeshMaterialMng* m_material_mng;
  std::map<String, Ref<IMeshMaterialVariableFactory>> m_factories;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialVariableFactoryMng::
MeshMaterialVariableFactoryMng(IMeshMaterialMng* mm)
: TraceAccessor(mm->traceMng())
, m_material_mng(mm)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialVariableFactoryMng::
~MeshMaterialVariableFactoryMng()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshMaterialVariable* MeshMaterialVariableFactoryMng::
createVariable(const String& storage_type, const MaterialVariableBuildInfo& build_info)
{
  auto x = m_factories.find(storage_type);
  if (x == m_factories.end())
    ARCANE_FATAL("Can not find mesh IMeshMaterialVariableFactory named={0}", storage_type);

  return x->second->createVariable(build_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableFactoryMng::
build()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITraceMng* MeshMaterialVariableFactoryMng::
traceMng() const
{
  return TraceAccessor::traceMng();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableFactoryMng::
registerFactory(Ref<IMeshMaterialVariableFactory> factory)
{
  MaterialVariableTypeInfo t = factory->materialVariableTypeInfo();
  m_factories.insert(std::make_pair(t.fullName(), factory));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" IMeshMaterialVariableFactoryMng*
arcaneCreateMeshMaterialVariableFactoryMng(IMeshMaterialMng* mm)
{
  auto* x = new MeshMaterialVariableFactoryMng(mm);
  x->build();
  return x;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
