// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MaterialHeatTestModule.cc                                   (C) 2000-2024 */
/*                                                                           */
/* Module de test des matériaux.                                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCANE_TRACE_ENUMERATOR

#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/IProfilingService.h"

#include "arcane/core/VariableTypes.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/Item.h"
#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/IParallelMng.h"

#include "arcane/materials/MeshMaterialVariableRef.h"
#include "arcane/materials/MeshEnvironmentBuildInfo.h"
#include "arcane/materials/MeshMaterialInfo.h"
#include "arcane/materials/CellToAllEnvCellConverter.h"
#include "arcane/materials/MeshMaterialModifier.h"
#include "arcane/materials/ComponentItemVectorView.h"

#include "arcane/accelerator/core/IAcceleratorMng.h"
#include "arcane/accelerator/core/RunCommand.h"
#include "arcane/accelerator/core/RunQueue.h"
#include "arcane/accelerator/VariableViews.h"
#include "arcane/accelerator/MaterialVariableViews.h"
#include "arcane/accelerator/RunCommandMaterialEnumerate.h"

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

  //! Caractéristiques de l'objet qui chauffe (disque ou sphère)
  struct HeatObject
  {
    //! Centre à t=0
    Real3 center;
    //! Rayon
    Real radius = 0.0;
    //! Vitesse
    Real3 velocity;
    //! Températeure de refroissement
    Real cold_value = 300.0;
    //! Températeure de chauffage
    Real heat_value = 1000.0;
    //! Matériaux qui sera chauffé par cet objet.
    IMeshMaterial* material = nullptr;
    //! Index de cet objet dans la liste globale
    Int32 index = 0;
    //! Température attendue au temps final
    Real expected_final_temperature = 0.0;
  };

  //! Tableau de travail pour la mise à jour des liste de matériaux
  struct MaterialWorkArray
  {
    void clear()
    {
      mat_cells_to_add_value.clear();
      mat_cells_to_add.clear();
      mat_cells_to_remove.clear();
    }

   public:

    //! Liste des valeurs de température dans les mailles à ajouter
    UniqueArray<Real> mat_cells_to_add_value;
    //! Liste des mailles à ajouter
    UniqueArray<Int32> mat_cells_to_add;
    //! Liste des mailles à supprimer
    UniqueArray<Int32> mat_cells_to_remove;
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

  IMeshMaterialMng* m_material_mng = nullptr;
  UniqueArray<HeatObject> m_heat_objects;
  IProfilingService* m_profiling_service = nullptr;

 private:

  void _computeCellsCenter();
  void _buildHeatObjects();
  void _copyToGlobal(const HeatObject& heat_object);
  void _computeTotalTemperature(const HeatObject& heat_object, bool do_check);
  IMeshMaterial* _findMaterial(const String& name);
  void _computeGlobalTemperature();
  void _computeCellsToAdd(const HeatObject& heat_object, MaterialWorkArray& wa);
  void _computeCellsToRemove(const HeatObject& heat_object, MaterialWorkArray& wa);
 public:
  void _addHeat(const HeatObject& heat_object);
 private:
  void _addCold(const HeatObject& heat_object);
  void _initNewCells(const HeatObject& heat_object, MaterialWorkArray& wa);
  void _compute();
  void _printCellsTemperature(Int32ConstArrayView ids);
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
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_PROFILE_HEATTEST", true))
    if (v.value() != 0)
      m_profiling_service = platform::getProfilingService();
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
  ProfilingRegistry::setProfilingLevel(2);

  // La création des milieux et des matériaux doit se faire dans un point
  // d'entrée de type 'build' pour que la liste des variables créés par les
  // milieux et les matériaux soit accessibles dans le post-traitement.
  info() << "MaterialHeatTestModule::buildInit()";

  Materials::IMeshMaterialMng* mm = IMeshMaterialMng::getReference(defaultMesh());

  int flags = options()->modificationFlags();

  info() << "MaterialHeatTestModule modification_flags=" << flags;

  m_material_mng->setModificationFlags(flags);
  m_material_mng->setMeshModificationNotified(true);
  m_material_mng->setUseMaterialValueWhenRemovingPartialValue(true);
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
  bool is_end = (m_global_iteration() >= options()->nbIteration());
  if (is_end)
    subDomain()->timeLoopMng()->stopComputeLoop(true);

  _compute();

  bool do_check = is_end && options()->checkNumericalResult();
  for (const HeatObject& ho : m_heat_objects) {
    _computeTotalTemperature(ho, do_check);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MaterialHeatTestModule::
_compute()
{
  _computeCellsCenter();

  UniqueArray<MaterialWorkArray> work_arrays(m_heat_objects.size());

  // Ajoute de la chaleur à chaque matériau
  for (const HeatObject& ho : m_heat_objects)
    _addHeat(ho);

  // Calcule les mailles à ajouter et supprimer
  for (const HeatObject& ho : m_heat_objects) {
    MaterialWorkArray& wa = work_arrays[ho.index];
    _computeCellsToRemove(ho, wa);
    _computeCellsToAdd(ho, wa);
  }

  // Effectue les modifications des matériaux
  {
    ProfilingSentryWithInitialize ps_sentry(m_profiling_service);
    MeshMaterialModifier modifier(m_material_mng);
    for (const HeatObject& ho : m_heat_objects) {
      MaterialWorkArray& wa = work_arrays[ho.index];
      IMeshMaterial* mat = ho.material;

      // Supprime les mailles matériaux nécessaires
      ConstArrayView<Int32> remove_ids = wa.mat_cells_to_remove;
      if (!remove_ids.empty()) {
        info() << "MAT_MODIF: Remove n=" << remove_ids.size() << " cells to material=" << mat->name();
        modifier.removeCells(mat, remove_ids);
      }

      // Ajoute les mailles matériaux nécessaires
      ConstArrayView<Int32> add_ids(wa.mat_cells_to_add.constView());
      if (!add_ids.empty()) {
        info() << "MAT_MODIF: Add n=" << add_ids.size() << " cells to material=" << mat->name();
        modifier.addCells(mat, add_ids);
      }
    }
  }

  // Affiche les valeurs pour les mailles modifiées
  if (options()->verbosityLevel() > 0) {
    for (const HeatObject& ho : m_heat_objects) {
      _printCellsTemperature(work_arrays[ho.index].mat_cells_to_add.constView());
      _printCellsTemperature(work_arrays[ho.index].mat_cells_to_remove.constView());
    }
  }

  // Initialise les nouvelles valeurs partielles
  for (const HeatObject& ho : m_heat_objects)
    _initNewCells(ho, work_arrays[ho.index]);

  // Refroidit chaque matériau
  for (const HeatObject& ho : m_heat_objects)
    _addCold(ho);

  // Remplit la variable composante associée de la variable 'AllTemperature'
  // pour chaque matériau. Cela permet de voir leur valeur facilement dans
  // les outils de dépouillement
  ENUMERATE_ (Cell, icell, allCells()) {
    m_all_temperature[icell].fill(0.0);
  }

  for (const HeatObject& ho : m_heat_objects) {
    _copyToGlobal(ho);
  }

  _computeGlobalTemperature();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MaterialHeatTestModule::
_initNewCells(const HeatObject& heat_object, MaterialWorkArray& wa)
{
  // Initialise les nouvelles valeurs partielles
  IMeshMaterial* current_mat = heat_object.material;
  CellToAllEnvCellConverter all_env_cell_converter(m_material_mng);
  ConstArrayView<Int32> ids(wa.mat_cells_to_add.constView());

  for (Int32 i = 0, n = ids.size(); i < n; ++i) {
    AllEnvCell all_env_cell = all_env_cell_converter[CellLocalId(ids[i])];
    MatCell mc = current_mat->findMatCell(all_env_cell);
    // Teste que la maille n'est pas nulle.
    // Ne devrait pas arriver car on l'a ajouté juste avant
    if (mc.null())
      ARCANE_FATAL("Internal invalid null mat cell");
    m_mat_temperature[mc] = wa.mat_cells_to_add_value[i];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MaterialHeatTestModule::
_addCold(const HeatObject& heat_object)
{
  IMeshMaterial* current_mat = heat_object.material;
  const Real cold_value = heat_object.cold_value;

  ENUMERATE_MATCELL (imatcell, current_mat) {
    Real t = m_mat_temperature[imatcell];
    t -= cold_value;
    if (t <= 0)
      ARCANE_FATAL("Invalid negative temperature '{0}' cell_lid={1}", t, (*imatcell).globalCell().localId());
    m_mat_temperature[imatcell] = t;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MaterialHeatTestModule::
_addHeat(const HeatObject& heat_object)
{
  Real3 heat_center = heat_object.center;
  heat_center += heat_object.velocity * m_global_time();
  const Real heat_value = heat_object.heat_value;
  const Real heat_radius_norm = heat_object.radius * heat_object.radius;

  IMeshMaterial* current_mat = heat_object.material;
  RunQueue* queue = this->acceleratorMng()->defaultQueue();
  auto command = makeCommand(queue);

  auto in_cell_center = viewIn(command, m_cell_center);
  auto inout_mat_temperature = viewInOut(command, m_mat_temperature);

  //! Chauffe les mailles déjà présentes dans le matériau
  command << RUNCOMMAND_MAT_ENUMERATE(MatAndGlobalCell, iter, current_mat)
  {
    auto [matcell, cell] = iter();
    Real3 center = in_cell_center[cell];
    Real distance2 = (center - heat_center).squareNormL2();
    if (distance2 < heat_radius_norm) {
      Real to_add = heat_value / (1.0 + distance2);
      inout_mat_temperature[matcell] += to_add;
    }
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MaterialHeatTestModule::
_computeCellsToAdd(const HeatObject& heat_object, MaterialWorkArray& wa)
{
  Real3 heat_center = heat_object.center;
  heat_center += heat_object.velocity * m_global_time();
  const Real heat_value = heat_object.heat_value;
  Real heat_radius = heat_object.radius;
  const Real heat_radius_norm = heat_radius * heat_radius;
  IMeshMaterial* current_mat = heat_object.material;
  CellToAllEnvCellConverter all_env_cell_converter(m_material_mng);

  const bool is_verbose = false;

  // Détermine les nouvelles à ajouter au matériau
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
        wa.mat_cells_to_add_value.add(to_add);
        wa.mat_cells_to_add.add(cell.localId());
        if (is_verbose)
          info() << "Add LID=" << cell.localId() << " T=" << to_add;
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MaterialHeatTestModule::
_computeCellsToRemove(const HeatObject& heat_object, MaterialWorkArray& wa)
{
  IMeshMaterial* current_mat = heat_object.material;
  const bool is_verbose = false;
  const Real cold_value = 300.0;

  // Refroidit chaque maille du matériau d'une quantité fixe.
  // Si la températeure devient inférieure à zéro on supprime la maille
  // de ce matériau.
  if (is_verbose)
    info() << "MAT_BEFORE: " << current_mat->matView()._internalLocalIds();
  {
    ENUMERATE_MATCELL (imatcell, current_mat) {
      MatCell mc = *imatcell;
      Real t = m_mat_temperature[mc];
      if (t < cold_value) {
        wa.mat_cells_to_remove.add(mc.globalCell().localId());
        if (is_verbose)
          info() << "Remove LID=" << mc.globalCell().localId() << " T=" << t;
      }
    }
  }
  if (is_verbose)
    info() << "MAT_AFTER: " << current_mat->matView()._internalLocalIds();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MaterialHeatTestModule::
_copyToGlobal(const HeatObject& heat_object)
{
  IMeshMaterial* mat = heat_object.material;
  Int32 var_index = heat_object.index;
  ENUMERATE_MATCELL (imatcell, mat) {
    MatCell mc = *imatcell;
    Real t = m_mat_temperature[mc];
    m_all_temperature[mc.globalCell()][var_index] = t;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MaterialHeatTestModule::
_computeTotalTemperature(const HeatObject& heat_object, bool do_check)
{
  IMeshMaterial* mat = heat_object.material;
  Real total_mat_temperature = 0.0;
  ENUMERATE_MATCELL (imatcell, mat) {
    MatCell mc = *imatcell;
    Cell cell = mc.globalCell();
    if (cell.isOwn())
      total_mat_temperature += m_mat_temperature[mc];
  }
  total_mat_temperature = parallelMng()->reduce(Parallel::ReduceSum, total_mat_temperature);
  info() << "TotalMatTemperature mat=" << mat->name() << " T=" << total_mat_temperature;
  if (do_check) {
    Real ref_value = heat_object.expected_final_temperature;
    Real current_value = total_mat_temperature;
    Real epsilon = 1.0e-12;
    if (!math::isNearlyEqualWithEpsilon(current_value, ref_value, epsilon)) {
      Real relative_diff = math::abs(ref_value - current_value);
      if (ref_value != 0.0)
        relative_diff /= ref_value;
      ARCANE_FATAL("Bad value for mat '{0}' ref={1} v={2} diff={3}",
                   mat->name(), ref_value, current_value, relative_diff);
    }
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

  {
    Int32 index = 0;
    for (const auto& opt : options()->heatObject()) {
      HeatObject ho;
      ho.center = opt->center;
      ho.velocity = opt->velocity;
      ho.radius = opt->radius;
      ho.material = _findMaterial(opt->material);
      ho.index = index;
      ho.expected_final_temperature = opt->expectedFinalTemperature;
      m_heat_objects.add(ho);
      ++index;
    }
  }

  m_all_temperature.resize(m_heat_objects.size());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshMaterial* MaterialHeatTestModule::
_findMaterial(const String& name)
{
  for (IMeshMaterial* mat : m_material_mng->materials())
    if (mat->name() == name)
      return mat;
  ARCANE_FATAL("No material in environment with name '{0}'", name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MaterialHeatTestModule::
_computeGlobalTemperature()
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MaterialHeatTestModule::
_printCellsTemperature(Int32ConstArrayView ids)
{
  CellToAllEnvCellConverter all_env_cell_converter(m_material_mng);
  for( Int32 lid : ids ){
    CellLocalId cell_id(lid);
    AllEnvCell all_env_cell = all_env_cell_converter[cell_id];
    Cell global_cell = all_env_cell.globalCell();
    info() << "Cell=" << global_cell.uniqueId() << " v=" << m_mat_temperature[global_cell];
    ENUMERATE_CELL_ENVCELL(ienvcell,all_env_cell){
      EnvCell ec = *ienvcell;
      info() << " EnvCell " << m_mat_temperature[ec]
             << " mv=" << ec._varIndex()
             << " env=" << ec.component()->name();
      ENUMERATE_CELL_MATCELL(imatcell,(*ienvcell)){
        MatCell mc = *imatcell;
        info() << "  MatCell " << m_mat_temperature[mc]
               << " mv=" << mc._varIndex()
               << " mat=" << mc.component()->name();
      }
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
