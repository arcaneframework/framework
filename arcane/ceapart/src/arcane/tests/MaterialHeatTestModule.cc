// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MaterialHeatTestModule.cc                                   (C) 2000-2023 */
/*                                                                           */
/* Module de test des matériaux.                                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCANE_TRACE_ENUMERATOR

#include "arcane/core/VariableTypes.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/Item.h"

#include "arcane/materials/MeshMaterialVariableRef.h"
#include "arcane/materials/MeshEnvironmentBuildInfo.h"
#include "arcane/materials/MeshMaterialInfo.h"

#include "arcane/tests/ArcaneTestGlobal.h"
#include "arcane/tests/MaterialHeatTest_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

using namespace Arcane;
using namespace Arcane::Materials;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module de test pour la gestion des matériaux et des milieux.
 */
class MaterialHeatTestModule
: public ArcaneMaterialHeatTestObject
{
 public:

  explicit MaterialHeatTestModule(const ModuleBuildInfo& mbi);
  ~MaterialHeatTestModule();

 public:

  void buildInit() override;
  void compute() override;
  void startInit() override;
  void continueInit() override;

 private:

  IMeshMaterialMng* m_material_mng;

 private:

  void _computeDensity();
  void _fillDensity();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MaterialHeatTestModule::
MaterialHeatTestModule(const ModuleBuildInfo& mbi)
: ArcaneMaterialHeatTestObject(mbi)
, m_material_mng(IMeshMaterialMng::getReference(mbi.meshHandle()))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MaterialHeatTestModule::
~MaterialHeatTestModule()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MaterialHeatTestModule::
buildInit()
{
  // La création des milieux et des matériaux doit se faire dans un point
  // d'entrée de type 'build' pour que la liste des variables créés par les
  // milieux et les matériaux soit accessible dans le post-traitement.
  info() << "MaterialHeatTestModule::buildInit()";

  Materials::IMeshMaterialMng* mm = IMeshMaterialMng::getReference(defaultMesh());

  int flags = (int)eModificationFlags::GenericOptimize | (int)eModificationFlags::OptimizeMultiAddRemove;
  m_material_mng->setModificationFlags(flags);
  m_material_mng->setMeshModificationNotified(true);

  if (subDomain()->isContinue()) {
    mm->recreateFromDump();
  }
  else {
    UniqueArray<MeshMaterialInfo*> materials_info;
    // Lit les infos des matériaux du JDD et les enregistre dans le gestionnaire
    for (Integer i = 0, n = options()->material().size(); i < n; ++i) {
      String mat_name = options()->material[i].name;
      info() << "Found material name=" << mat_name;
      materials_info.add(mm->registerMaterialInfo(mat_name));
    }

    UniqueArray<IMeshEnvironment*> saved_envs;

    // Créé les milieux
    for (Integer i = 0, n = options()->environment().size(); i < n; ++i) {
      String env_name = options()->environment[i].name;
      info() << "Found environment name=" << env_name;
      Materials::MeshEnvironmentBuildInfo env_build(env_name);
      for (Integer k = 0, kn = options()->environment[i].material.size(); k < kn; ++k) {
        String mat_name = options()->environment[i].material[k];
        info() << "Add material " << mat_name << " for environment " << env_name;
        env_build.addMaterial(mat_name);
      }
      IMeshEnvironment* env = mm->createEnvironment(env_build);
      saved_envs.add(env);
    }

    mm->endCreate(subDomain()->isContinue());

    info() << "List of materials:";
    for (MeshMaterialInfo* m : materials_info) {
      info() << "MAT=" << m->name();
      for (String s : m->environmentsName())
        info() << " In ENV=" << s;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MaterialHeatTestModule::
startInit()
{
  m_global_deltat.assign(1.0);
  m_mat_temperature.globalVariable().fill(0.0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MaterialHeatTestModule::
continueInit()
{
  info() << "MaterialHeatTestModule::continueInit()";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MaterialHeatTestModule::
compute()
{
  info() << "MaterialHeatTestModule::compute()";

  Real3 heat_center(0.3, 0.4, 0.0);
  Real3 heat_radius(0.1, 0.15, 0.0);
  Real heat_radius_norm = heat_radius.squareNormL2();
  VariableNodeReal3& node_coord = defaultMesh()->nodesCoordinates();
  ENUMERATE_ (Cell, icell, allCells()) {
    Cell cell = *icell;
    Real3 center;
    auto cell_nodes = cell.nodeIds();
    for (NodeLocalId nodeid : cell_nodes) {
      center += node_coord[nodeid];
    }
    center /= cell_nodes.size();
    Real distance2 = (center - heat_center).squareNormL2();
    if (distance2 < heat_radius_norm) {
      Real to_add = 1000.0;
      if (distance2 > 0.0)
        to_add = 1000 / distance2;
      m_mat_temperature[cell] += to_add;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_MATERIALHEATTEST(MaterialHeatTestModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
