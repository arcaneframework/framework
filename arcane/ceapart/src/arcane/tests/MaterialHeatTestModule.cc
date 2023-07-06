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
#include "arcane/materials/CellToAllEnvCellConverter.h"
#include "arcane/materials/MeshMaterialModifier.h"

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
 private:

  //! Caractéristiques de l'objet qui chauffe (disque ou sphère)
  struct HeatObject
  {
    //! Centre à t=0
    Real3 center;
    //! Rayon
    Real3 radius;
    //! Vitesse
    Real3 velocity;
    //! Matériaux qui sera chauffé par cet objet.
    IMeshMaterial* material = nullptr;
  };

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
  UniqueArray<HeatObject> m_heat_objects;

 private:

  void _computeCellsCenter();
  void _computeOneMaterial(const HeatObject& heat_object);
  void _buildHeatObjects();
  void _copyToGlobal(IMeshMaterial* mat, Int32 var_index);
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

    mm->endCreate(false);

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
  _buildHeatObjects();
  m_global_deltat.assign(1.0);
  m_mat_temperature.globalVariable().fill(0.0);
  m_material_mng->forceRecompute();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MaterialHeatTestModule::
continueInit()
{
  info() << "MaterialHeatTestModule::continueInit()";
  _buildHeatObjects();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MaterialHeatTestModule::
compute()
{
  info() << "MaterialHeatTestModule::compute()";

  _computeCellsCenter();

  for (const HeatObject& ho : m_heat_objects)
    _computeOneMaterial(ho);

  // Remplit la variable composante associée de la variable 'AllTemperature'
  // pour chaque matériau. Cela permet de voir leur valeur facilement dans
  // les outils de dépouillement
  ENUMERATE_ (Cell, icell, allCells()) {
    m_all_temperature[icell].fill(0.0);
  }
  {
    Int32 index = 0;
    for (const HeatObject& ho : m_heat_objects) {
      _copyToGlobal(ho.material, index);
      ++index;
    }
  }

  {
    // Calcule dans 'Temperature' la somme des températures des milieux et matériaux
    CellToAllEnvCellConverter all_env_cell_converter(m_material_mng);
    ENUMERATE_ (Cell, icell, allCells()) {
      Cell cell = *icell;
      AllEnvCell all_env_cell = all_env_cell_converter[cell];
      Real global_temperature = 0.0;
      ENUMERATE_CELL_ENVCELL (ienvcell, all_env_cell) {
        EnvCell env_cell = *ienvcell;
        Real env_temperature = 0.0;
        ENUMERATE_CELL_MATCELL (imatcell, env_cell) {
          MatCell mc = *imatcell;
          env_temperature += m_mat_temperature[mc];
        }
        m_mat_temperature[env_cell] = env_temperature;
        global_temperature += env_temperature;
      }
      m_mat_temperature[cell] = global_temperature;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MaterialHeatTestModule::
_computeOneMaterial(const HeatObject& heat_object)
{
  Real3 heat_center = heat_object.center;
  heat_center += heat_object.velocity * m_global_time();
  const Real heat_value = 1000.0;
  Real3 heat_radius = heat_object.radius;
  const Real heat_radius_norm = heat_radius.squareNormL2();

  IMeshMaterial* current_mat = heat_object.material;

  CellToAllEnvCellConverter all_env_cell_converter(m_material_mng);

  UniqueArray<Real> mat_cells_to_add_value;
  UniqueArray<Int32> mat_cells_to_add;
  ENUMERATE_ (Cell, icell, allCells()) {
    Cell cell = *icell;
    AllEnvCell all_env_cell = all_env_cell_converter[cell];
    Real3 center = m_cell_center[icell];
    Real distance2 = (center - heat_center).squareNormL2();
    if (distance2 < heat_radius_norm) {
      Real to_add = heat_value / (1.0 + distance2);
      MatCell mc = current_mat->findMatCell(all_env_cell);
      if (mc.null()) {
        // Si 'mc' est nul cela signifie que ce matériau n'est
        // pas présent dans la maille. Il faudra donc l'ajouter.
        // On conserve la valeur à ajouter pour ne pas la recalculer
        mat_cells_to_add_value.add(to_add);
        mat_cells_to_add.add(cell.localId());
      }
      else
        m_mat_temperature[mc] += to_add;
    }
  }

  // Ajoute les mailles matériaux nécessaires
  if (!mat_cells_to_add.empty()) {
    ConstArrayView<Int32> ids(mat_cells_to_add);
    {
      MeshMaterialModifier modifier(m_material_mng);
      info() << "MAT_MODIF: Add n=" << ids.size() << " cells to material=" << current_mat->name();
      modifier.addCells(current_mat, ids);
    }
    // Initialise les nouvelles valeurs partielles
    for (Int32 i = 0, n = ids.size(); i < n; ++i) {
      AllEnvCell all_env_cell = all_env_cell_converter[CellLocalId(ids[i])];
      MatCell mc = current_mat->findMatCell(all_env_cell);
      // Teste que la maille n'est pas nulle.
      // Ne devrait pas arriver car on l'a ajouté juste avant
      if (mc.null())
        ARCANE_FATAL("Internal invalid null mat cell");
      m_mat_temperature[mc] = mat_cells_to_add_value[i];
    }
  }

  // Refroidit chaque maille du matériau d'une quantité fixe.
  // Si la températeur devient inférieure à zéro on supprime la maille
  // de ce matériau.
  UniqueArray<Int32> mat_cells_to_remove;
  ENUMERATE_MATCELL (imatcell, current_mat) {
    MatCell mc = *imatcell;
    Real t = m_mat_temperature[mc];
    t -= 300.0;
    if (t < 0) {
      t = 0.0;
      mat_cells_to_remove.add(mc.globalCell().localId());
    }
    m_mat_temperature[mc] = t;
  }

  if (!mat_cells_to_remove.empty()) {
    MeshMaterialModifier modifier(m_material_mng);
    info() << "MAT_MODIF: Remove n=" << mat_cells_to_remove.size() << " cells to material=" << current_mat->name();
    modifier.addCells(current_mat, mat_cells_to_remove);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MaterialHeatTestModule::
_copyToGlobal(IMeshMaterial* mat, Int32 var_index)
{
  ENUMERATE_MATCELL (imatcell, mat) {
    MatCell mc = *imatcell;
    m_all_temperature[mc.globalCell()][var_index] = m_mat_temperature[mc];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MaterialHeatTestModule::
_computeCellsCenter()
{
  // Calcule le centre des mailles
  VariableNodeReal3& node_coord = defaultMesh()->nodesCoordinates();
  ENUMERATE_ (Cell, icell, allCells()) {
    Cell cell = *icell;
    Real3 center;
    auto cell_nodes = cell.nodeIds();
    for (NodeLocalId nodeid : cell_nodes) {
      center += node_coord[nodeid];
    }
    center /= cell_nodes.size();
    m_cell_center[icell] = center;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MaterialHeatTestModule::
_buildHeatObjects()
{
  info() << "MaterialHeatTestModule::_buildHeatObjects()";

  IMeshEnvironment* env = m_material_mng->findEnvironment("ENV1");
  const Int32 nb_env = env->nbMaterial();
  if (nb_env == 0)
    ARCANE_FATAL("Environment '{0}' has no materials", env->name());

  {
    HeatObject ho;
    ho.center = Real3(0.3, 0.4, 0.0);
    ho.velocity = Real3(0.02, 0.04, 0.0);
    ho.radius = Real3(0.1, 0.15, 0.0);
    ho.material = env->materials()[0];
    m_heat_objects.add(ho);
  }
  {
    HeatObject ho;
    ho.center = Real3(0.8, 0.4, 0.0);
    ho.velocity = Real3(-0.02, 0.04, 0.0);
    ho.radius = Real3(0.2, 0.15, 0.0);
    ho.material = env->materials()[1];
    m_heat_objects.add(ho);
  }

  m_all_temperature.resize(m_heat_objects.size());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_MATERIALHEATTEST(MaterialHeatTestModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
