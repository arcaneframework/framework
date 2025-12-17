// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MaterialHeatTestModule.cc                                   (C) 2000-2025 */
/*                                                                           */
/* Module de test des matériaux.                                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCANE_TRACE_ENUMERATOR

#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/IProfilingService.h"
#include "arcane/utils/IMemoryRessourceMng.h"

#include "arcane/utils/Profiling.h"
#include "arccore/base/internal/ProfilingInternal.h"

#include "arcane/core/VariableTypes.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/Item.h"
#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/VariableUtils.h"
#include "arcane/core/MeshUtils.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/ItemInfoListView.h"
#include "arcane/core/internal/IVariableInternal.h"
#include "arcane/core/internal/ItemGroupImplInternal.h"
#include "arcane/core/materials/internal/IMeshMaterialVariableInternal.h"
#include "arcane/core/materials/internal/IMeshMaterialMngInternal.h"

#include "arcane/materials/MeshMaterialVariableRef.h"
#include "arcane/materials/MeshEnvironmentBuildInfo.h"
#include "arcane/materials/MeshMaterialInfo.h"
#include "arcane/materials/CellToAllEnvCellConverter.h"
#include "arcane/materials/MeshMaterialModifier.h"
#include "arcane/materials/ComponentItemVectorView.h"

#include "arcane/accelerator/core/IAcceleratorMng.h"
#include "arcane/accelerator/core/RunCommand.h"
#include "arcane/accelerator/core/RunQueue.h"
#include "arcane/accelerator/core/Runner.h"
#include "arccore/common/accelerator/internal/RunnerInternal.h"
#include "arcane/accelerator/VariableViews.h"
#include "arcane/accelerator/MaterialVariableViews.h"
#include "arcane/accelerator/RunCommandMaterialEnumerate.h"
#include "arcane/accelerator/RunCommandEnumerate.h"
#include "arcane/accelerator/RunCommandLoop.h"
#include "arcane/accelerator/Filter.h"
#include "arcane/accelerator/SpanViews.h"
#include "arcane/accelerator/NumArrayViews.h"
#include "arcane/accelerator/Reduce.h"

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
    MaterialWorkArray(eMemoryRessource mem)
    : mat_cells_to_add_value(_allocator(mem))
    , mat_cells_to_add(_allocator(mem))
    , mat_cells_to_remove(_allocator(mem))
    , mat_cells_remove_filter(mem)
    {
    }

   public:

    void clear()
    {
      mat_cells_to_add_value.clear();
      mat_cells_to_add.clear();
      mat_cells_to_remove.clear();
      mat_cells_remove_filter.resize(0);
    }

    void resizeNbAdd(Int32 new_size)
    {
      mat_cells_to_add_value.resize(new_size);
      mat_cells_to_add.resize(new_size);
    }

    void resizeNbRemove(Int32 new_size)
    {
      mat_cells_to_remove.resize(new_size);
      mat_cells_remove_filter.resize(new_size);
    }

   public:

    static MemoryAllocationOptions _allocator(eMemoryRessource mem)
    {
      return MemoryUtils::getAllocationOptions(mem);
    }

   public:

    //! Liste des valeurs de température dans les mailles à ajouter
    UniqueArray<Real> mat_cells_to_add_value;
    //! Liste des mailles à ajouter
    UniqueArray<Int32> mat_cells_to_add;
    //! Liste des mailles à supprimer
    UniqueArray<Int32> mat_cells_to_remove;
    //! Filtre des mailles à supprimer
    NumArray<bool, MDDim1> mat_cells_remove_filter;
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
  RunQueue m_queue;
  Runner m_sequential_runner;
  UniqueArray<MeshMaterialVariableRef*> m_additional_variables;
  bool m_is_init_with_zero = false;
  bool m_is_check_init_new_cells = false;

 private:

  void _computeCellsCenter();
  void _buildHeatObjects();
  IMeshMaterial* _findMaterial(const String& name);

 public:

  void _addHeat(const HeatObject& heat_object);
  void _addCold(const HeatObject& heat_object);
  void _computeGlobalTemperature();
  void _computeCellsToAdd(const HeatObject& heat_object, MaterialWorkArray& wa);
  void _computeCellsToRemove(const HeatObject& heat_object, MaterialWorkArray& wa);
  void _copyToGlobal(const HeatObject& heat_object);
  void _initNewCells(const HeatObject& heat_object, MaterialWorkArray& wa);
  void _computeTotalTemperature(const HeatObject& heat_object, bool do_check);

 private:

  void _compute();
  void _printCellsTemperature(Int32ConstArrayView ids);
  void _changeVariableAllocator();
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
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_MATERIAL_NEW_ITEM_INIT", true)) {
    Int32 vv = v.value();
    // 0 -> initialisation à partir de la maille globale et pas de vérification
    // 1 -> initialisation à zéro et pas de vérification
    // 2 -> initialisation à zéro et vérification
    // 3 -> initialisation à partir de la maille globale et vérification
    m_is_init_with_zero = (vv == 1 || vv == 2);
    m_is_check_init_new_cells = (vv == 2 || vv == 3);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MaterialHeatTestModule::
~MaterialHeatTestModule()
{
  for (MeshMaterialVariableRef* v : m_additional_variables)
    delete v;
  m_additional_variables.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MaterialHeatTestModule::
buildInit()
{
  m_sequential_runner.initialize(Accelerator::eExecutionPolicy::Sequential);
  m_queue = *acceleratorMng()->defaultQueue();
  ProfilingRegistry::setProfilingLevel(2);

  // La création des milieux et des matériaux doit se faire dans un point
  // d'entrée de type 'build' pour que la liste des variables créés par les
  // milieux et les matériaux soit accessibles dans le post-traitement.
  info() << "MaterialHeatTestModule::buildInit()";

  IMesh* mesh = defaultMesh();
  Materials::IMeshMaterialMng* mm = IMeshMaterialMng::getReference(mesh);

  int flags = options()->modificationFlags();

  info() << "MaterialHeatTestModule modification_flags=" << flags;

  m_material_mng->setModificationFlags(flags);
  m_material_mng->setMeshModificationNotified(true);
  m_material_mng->setUseMaterialValueWhenRemovingPartialValue(true);
  m_material_mng->setDataInitialisationWithZero(m_is_init_with_zero);
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

  // Créé les éventuelles variables scalaires additionnelles.
  {
    Int32 nb_var_to_add = options()->nbAdditionalVariable();
    info() << "NbVariableToAdd = " << nb_var_to_add;
    for (Int32 i = 0; i < nb_var_to_add; ++i) {
      String var_name = "MaterialAdditionalVar" + String::fromNumber(i);
      auto* v = new MaterialVariableCellInt32(VariableBuildInfo(mesh, var_name));
      m_additional_variables.add(v);
      v->fill(i + 2);
    }
  }

  // Créé les éventuelles variables tableaux additionnelles.
  {
    Int32 nb_var_to_add = options()->nbAdditionalArrayVariable();
    info() << "NbArrayVariableToAdd = " << nb_var_to_add;
    for (Int32 i = 0; i < nb_var_to_add; ++i) {
      String var_name = "MaterialAdditionalArrayVar" + String::fromNumber(i);
      auto* v = new MaterialVariableCellArrayInt32(VariableBuildInfo(mesh, var_name));
      v->resize(1 + (i % 3));
      m_additional_variables.add(v);
      v->globalVariable().fill(i + 5);
      v->fillPartialValuesWithSuperValues(LEVEL_ALLENVIRONMENT);
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
  _computeCellsCenter();
  MeshUtils::markMeshConnectivitiesAsMostlyReadOnly(defaultMesh(), &m_queue, true);
  VariableUtils::markVariableAsMostlyReadOnly(m_cell_center);
  VariableUtils::markVariableAsMostlyReadOnly(defaultMesh()->nodesCoordinates());

  eMemoryRessource mem_ressource = eMemoryRessource::UnifiedMemory;
  if (m_queue.isAcceleratorPolicy() && m_material_mng->_internalApi()->runQueue().isAcceleratorPolicy())
    mem_ressource = eMemoryRessource::Device;

  const bool do_change_allocator = true;
  if (do_change_allocator) {
    info() << "Changing allocator to use device memory to '" << mem_ressource << "'";
    VariableUtils::experimentalChangeAllocator(m_mat_device_temperature.materialVariable(), mem_ressource);
  }
  {
    eMemoryRessource group_mem_ressource = mem_ressource;
    // En mode check il ne faut pas utiliser la mémoire du device car il y a des tests
    // effectués uniquement sur CPU
    if (arcaneIsCheck())
      group_mem_ressource = eMemoryRessource::UnifiedMemory;
    ENUMERATE_MAT (imat, m_material_mng) {
      IMeshMaterial* mat = *imat;
      mat->cells()._internalApi()->setMemoryRessourceForItemLocalId(group_mem_ressource);
    }
    ENUMERATE_ENV (ienv, m_material_mng) {
      IMeshEnvironment* env = *ienv;
      env->cells()._internalApi()->setMemoryRessourceForItemLocalId(group_mem_ressource);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MaterialHeatTestModule::
_changeVariableAllocator()
{
  VariableCollection used_variables = subDomain()->variableMng()->usedVariables();
  MemoryAllocationOptions host_mem_opts(MemoryUtils::getAllocationOptions(eMemoryRessource::Host));
  for (VariableCollection::Enumerator ivar(used_variables); ++ivar;) {
    IVariable* var = *ivar;
    if (var->name().startsWith("TimeHistoryMng")) {
      var->_internalApi()->changeAllocator(host_mem_opts);
      info() << "Change allocator for '" << var->fullName() << "'";
    }
  }
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
  Int32 iteration = m_global_iteration();
  if (iteration <= 2)
    _changeVariableAllocator();

  if (iteration>=2){
    Runner runner = *acceleratorMng()->defaultRunner();
    ostringstream o;
    runner._internalApi()->printProfilingInfos(o);
    info() << o.str();
  }

  bool is_end = (iteration >= options()->nbIteration());
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
  RunQueue* queue = acceleratorMng()->defaultQueue();
  eMemoryRessource mem = eMemoryRessource::UnifiedMemory;
  if (queue->isAcceleratorPolicy())
    mem = eMemoryRessource::Device;

  const Int32 nb_heat = m_heat_objects.size();
  UniqueArray<MaterialWorkArray> work_arrays;
  work_arrays.reserve(nb_heat);
  for (Int32 i = 0; i < nb_heat; ++i)
    work_arrays.add(MaterialWorkArray(mem));

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

  for (const HeatObject& ho : m_heat_objects) {
    _copyToGlobal(ho);
  }

  _computeGlobalTemperature();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  ARCCORE_HOST_DEVICE MatCell _getMatCell(AllEnvCell c, Int32 mat_id)
  {
    for (EnvCell env_cell : c.subEnvItems()) {
      for (MatCell mc : env_cell.subMatItems()) {
        Int32 mid = mc.materialId();
        if (mid == mat_id)
          return mc;
      }
    }
    return {};
  }
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MaterialHeatTestModule::
_initNewCells(const HeatObject& heat_object, MaterialWorkArray& wa)
{
  RunQueue* queue = this->acceleratorMng()->defaultQueue();

  bool init_with_zero = m_material_mng->isDataInitialisationWithZero();

  // Initialise les nouvelles valeurs partielles
  IMeshMaterial* current_mat = heat_object.material;
  Int32 mat_id = current_mat->id();
  CellToAllEnvCellConverter all_env_cell_converter(m_material_mng);
  SmallSpan<const Int32> ids(wa.mat_cells_to_add.constView());
  const Int32 nb_id = ids.size();
  const bool do_check = m_is_check_init_new_cells;
  if (do_check) {
    // Vérifie que la nouvelle valeur est initialisée avec 0 (si init_with_zero
    // est vrai) où qu'elle est initialisée avec la valeur globale
    auto command = makeCommand(queue);
    auto out_mat_temperature = viewInOut(command, m_mat_temperature);
    Accelerator::ReducerSum2<Int32> sum_error(command);
    command << RUNCOMMAND_LOOP1(iter, nb_id, sum_error)
    {
      auto [i] = iter();
      AllEnvCell all_env_cell = all_env_cell_converter[CellLocalId(ids[i])];
      MatCell mc = _getMatCell(all_env_cell, mat_id);
      MatVarIndex mvi = mc._varIndex();
      if (mvi.arrayIndex() != 0) {
        Real v = out_mat_temperature[mc];
        if (init_with_zero) {
          if (v != 0.0)
            sum_error.combine(1);
          //ARCANE_FATAL("Bad mat temperature (should be 0) i={0} v={1} mc={2}", i, v, mc);
        }
        else {
          Real global_v = out_mat_temperature[mc.globalCellId()];
          if (v != global_v)
            sum_error.combine(1);
          //ARCANE_FATAL("Bad mat temperature i={0} v={1} mc={2} expected_v={3}", i, v, mc, global_v);
        }
      }
    };
    Int32 nb_error = sum_error.reducedValue();
    if (nb_error != 0)
      ARCANE_FATAL("Errors with new cells nb_error={0}", nb_error);
  }
  {
    auto command = makeCommand(queue);
    auto in_value_to_add = viewIn(command, wa.mat_cells_to_add_value);
    auto out_mat_temperature = viewInOut(command, m_mat_temperature);
    command << RUNCOMMAND_LOOP1(iter, nb_id)
    {
      auto [i] = iter();
      AllEnvCell all_env_cell = all_env_cell_converter[CellLocalId(ids[i])];
      MatCell mc = _getMatCell(all_env_cell, mat_id);
      out_mat_temperature[mc] = in_value_to_add[i];
    };
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MaterialHeatTestModule::
_addCold(const HeatObject& heat_object)
{
  IMeshMaterial* current_mat = heat_object.material;
  const Real cold_value = heat_object.cold_value;

  RunQueue* queue = this->acceleratorMng()->defaultQueue();

  {
    auto command = makeCommand(queue);
    auto inout_mat_temperature = viewInOut(command, m_mat_temperature);
    command << RUNCOMMAND_MAT_ENUMERATE(MatCell, matcell, current_mat)
    {
      Real t = inout_mat_temperature[matcell];
      t -= cold_value;
      inout_mat_temperature[matcell] = t;
    };
  }

  if (arcaneIsCheck()) {
    ENUMERATE_MATCELL (imatcell, current_mat) {
      Real t = m_mat_temperature[imatcell];
      if (t <= 0)
        ARCANE_FATAL("Invalid negative temperature '{0}' cell_lid={1}", t, (*imatcell).globalCell().localId());
    }
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
  //RunQueue* queue = this->acceleratorMng()->defaultQueue();
  //auto queue = makeQueue(m_sequential_runner);
  auto command = makeCommand(m_queue);

  auto in_cell_center = viewIn(command, m_cell_center);
  auto inout_mat_temperature = viewInOut(command, m_mat_temperature);
  auto out_mat_device_temperature = viewInOut(command, m_mat_device_temperature);

  //! Chauffe les mailles déjà présentes dans le matériau
  command << RUNCOMMAND_MAT_ENUMERATE(MatAndGlobalCell, iter, current_mat)
  {
    auto [matcell, cell] = iter();
    Real3 center = in_cell_center[cell];
    Real distance2 = (center - heat_center).squareNormL2();
    if (distance2 < heat_radius_norm) {
      Real to_add = heat_value / (1.0 + distance2);
      inout_mat_temperature[matcell] += to_add;
      out_mat_device_temperature[matcell] = inout_mat_temperature[matcell];
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

  const Int32 mat_id = current_mat->id();

  // Détermine les nouvelles à ajouter au matériau
  {
    CellVectorView all_cells(allCells().view());

    const Int32 nb_item = all_cells.size();
    wa.resizeNbAdd(nb_item);
    Accelerator::GenericFilterer filterer(m_queue);
    auto in_cell_center = viewIn(m_queue, m_cell_center);
    auto cells_ids = viewIn(m_queue, all_cells.localIds());

    auto select_functor = [=] ARCCORE_HOST_DEVICE(Int32 index) -> bool {
      CellLocalId cellid(cells_ids[index]);
      AllEnvCell all_env_cell = all_env_cell_converter[cellid];
      Real3 center = in_cell_center[cellid];
      Real distance2 = (center - heat_center).squareNormL2();
      if (distance2 < heat_radius_norm) {
        MatCell mc = _getMatCell(all_env_cell, mat_id);
        // Si 'mc' est nul cela signifie que ce matériau n'est
        // pas présent dans la maille. Il faudra donc l'ajouter.
        // On conserve la valeur à ajouter pour ne pas la recalculer
        return mc.null();
      }
      return false;
    };

    auto cells_to_add_view = viewOut(m_queue, wa.mat_cells_to_add);
    auto cells_to_add_value_view = viewOut(m_queue, wa.mat_cells_to_add_value);
    auto setter_functor = [=] ARCCORE_HOST_DEVICE(Int32 input_index, Int32 output_index) {
      CellLocalId cellid(cells_ids[input_index]);
      Real3 center = in_cell_center[cellid];
      Real distance2 = (center - heat_center).squareNormL2();
      Real to_add = heat_value / (1.0 + distance2);
      cells_to_add_view[output_index] = cellid;
      cells_to_add_value_view[output_index] = to_add;
    };
    filterer.applyWithIndex(nb_item, select_functor, setter_functor, A_FUNCINFO);
    Int32 nb_out = filterer.nbOutputElement();
    wa.resizeNbAdd(nb_out);
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
  const Int32 nb_cell = current_mat->cells().size();
  wa.resizeNbRemove(nb_cell);

  {
    auto command = makeCommand(m_queue);
    auto out_cells_remove_filter = viewOut(command, wa.mat_cells_remove_filter);
    auto out_cells_local_id = viewOut(command, wa.mat_cells_to_remove);
    auto in_mat_temperature = viewIn(command, m_mat_temperature);
    command << RUNCOMMAND_MAT_ENUMERATE(MatAndGlobalCell, iter, current_mat)
    {
      auto [mvi, cid] = iter();
      Int32 index = iter.index();
      out_cells_remove_filter[index] = in_mat_temperature[mvi] < cold_value;
      out_cells_local_id[index] = cid;
    };
  }

  {
    Accelerator::GenericFilterer filterer(m_queue);
    SmallSpan<const Int32> in_remove_view = wa.mat_cells_to_remove.view();
    SmallSpan<Int32> out_remove_view = wa.mat_cells_to_remove.view();
    SmallSpan<const bool> filter_view = wa.mat_cells_remove_filter.to1DSmallSpan();
    filterer.apply(in_remove_view, out_remove_view, filter_view);
    Int32 nb_out = filterer.nbOutputElement();
    wa.resizeNbRemove(nb_out);
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

  {
    auto command = makeCommand(m_queue);
    auto in_mat_temperature = viewIn(command, m_mat_temperature);
    auto out_all_temperature = viewOut(command, m_all_temperature);
    command << RUNCOMMAND_MAT_ENUMERATE(MatAndGlobalCell, iter, mat)
    {
      auto [mvi, cid] = iter();
      Real t = in_mat_temperature[mvi];
      out_all_temperature[cid][var_index] = t;
    };
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MaterialHeatTestModule::
_computeTotalTemperature(const HeatObject& heat_object, bool do_check)
{
  IMeshMaterial* mat = heat_object.material;
  Real total_mat_temperature = 0.0;

  {
    auto command = makeCommand(m_queue);
    auto in_mat_temperature = viewIn(command, m_mat_temperature);
    Accelerator::ReducerSum2<double> total_temperature_reducer(command);
    CellInfoListView cells_info(defaultMesh()->cellFamily());
    command << RUNCOMMAND_MAT_ENUMERATE(MatAndGlobalCell, iter, mat, total_temperature_reducer)
    {
      auto [mvi, cid] = iter();
      if (cells_info.isOwn(cid)) {
        Real t = in_mat_temperature[mvi];
        total_temperature_reducer.combine(t);
      }
    };
    total_mat_temperature = total_temperature_reducer.reducedValue();
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
  auto command = makeCommand(m_queue);
  auto inout_mat_temperature = viewInOut(command, m_mat_temperature);
  command << RUNCOMMAND_ENUMERATE (Cell, cellid, allCells())
  {
    //Cell cell = *icell;
    AllEnvCell all_env_cell = all_env_cell_converter[cellid];
    Real global_temperature = 0.0;
    for (EnvCell env_cell : all_env_cell.subEnvItems()) {
      Real env_temperature = 0.0;
      for (MatCell mc : env_cell.subMatItems()) {
        env_temperature += inout_mat_temperature[mc];
      }
      inout_mat_temperature[env_cell] = env_temperature;
      global_temperature += env_temperature;
    }
    inout_mat_temperature[cellid] = global_temperature;
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MaterialHeatTestModule::
_printCellsTemperature(Int32ConstArrayView ids)
{
  CellToAllEnvCellConverter all_env_cell_converter(m_material_mng);
  for (Int32 lid : ids) {
    CellLocalId cell_id(lid);
    AllEnvCell all_env_cell = all_env_cell_converter[cell_id];
    Cell global_cell = all_env_cell.globalCell();
    info() << "Cell=" << global_cell.uniqueId() << " v=" << m_mat_temperature[global_cell];
    for (EnvCell ec : all_env_cell.subEnvItems()) {
      info() << " EnvCell " << m_mat_temperature[ec]
             << " mv=" << ec._varIndex()
             << " env=" << ec.component()->name();
      for (MatCell mc : ec.subMatItems()) {
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
