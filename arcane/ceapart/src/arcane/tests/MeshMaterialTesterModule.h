// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialTesterModule.h                                  (C) 2000-2025 */
/*                                                                           */
/* Module de test du gestionnaire des matériaux.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_TESTS_MESHMATERIALTESTERMODULE_H
#define ARCANE_TESTS_MESHMATERIALTESTERMODULE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCANE_TRACE_ENUMERATOR

#include "arcane/core/IUnitTest.h"
#include "arcane/core/materials/MeshMaterialVariableRef.h"
#include "arcane/core/materials/MeshEnvironmentVariableRef.h"

#include "arcane/tests/IMaterialEquationOfState.h"
#include "arcane/tests/MeshMaterialTester_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;
using namespace Arcane::Materials;
using namespace Arcane::Accelerator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MeshMaterialVariableTraits
{
 public:

  typedef CellMaterialVariableScalarRef<Byte> VariableByteType;
  typedef CellMaterialVariableScalarRef<Real> VariableRealType;
  typedef CellMaterialVariableScalarRef<Int64> VariableInt64Type;
  typedef CellMaterialVariableScalarRef<Int32> VariableInt32Type;
  typedef CellMaterialVariableScalarRef<Int16> VariableInt16Type;
  typedef CellMaterialVariableScalarRef<Int8> VariableInt8Type;
  typedef CellMaterialVariableScalarRef<BFloat16> VariableBFloat16Type;
  typedef CellMaterialVariableScalarRef<Float16> VariableFloat16Type;
  typedef CellMaterialVariableScalarRef<Float32> VariableFloat32Type;
  typedef CellMaterialVariableScalarRef<Real3> VariableReal3Type;
  typedef CellMaterialVariableScalarRef<Real3x3> VariableReal3x3Type;
  typedef CellMaterialVariableScalarRef<Real2> VariableReal2Type;
  typedef CellMaterialVariableScalarRef<Real2x2> VariableReal2x2Type;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module de test pour la gestion des matériaux et des milieux.
 */
class MeshMaterialTesterModule
: public ArcaneMeshMaterialTesterObject
{
 public:

  explicit MeshMaterialTesterModule(const ModuleBuildInfo& mbi);
  ~MeshMaterialTesterModule();

 public:

  static void staticInitialize(ISubDomain* sd);

 public:

  void buildInit() override;
  void compute() override;
  void startInit() override;
  void continueInit() override;

 private:

  IMeshMaterialMng* m_material_mng = nullptr;
  VariableCellReal m_density;
  VariableCellReal m_pressure;
  MaterialVariableCellReal m_mat_density2;
  //! Variable pour tester la bonne prise en compte du PNoDump
  MaterialVariableCellReal m_mat_nodump_real;
  VariableCellInt32 m_present_material;
  MaterialVariableCellInt32 m_mat_int32;
  //! Variable pour tester la bonne prise en compte de setUsed(false)
  MaterialVariableCellReal m_mat_not_used_real;
  VariableScalarInt64 m_nb_starting_cell; //<! Nombre de mailles au démarrage
  IMeshMaterial* m_mat1 = nullptr;
  IMeshMaterial* m_mat2 = nullptr;
  IMeshBlock* m_block1 = nullptr;
  UniqueArray<VariableCellReal*> m_density_post_processing;
  //! Partitioner en cas d'équilibrage. Est géré par une option du JDD.
  IMeshPartitioner* m_mesh_partitioner = nullptr;
  // Si non nul, indique qu'il faut vérifier les valeurs spectrales,
  // car on a fait un re-partitionnement
  Integer m_check_spectral_values_iteration = 0;

 private:

  void _computeDensity();
  void _fillDensity(IMeshMaterial* mat, VariableCellReal& tmp_cell_mat_density,
                    VariableNodeReal& tmp_node_mat_density,
                    Int32Array& mat_to_add_array, Int32Array& mat_to_remove_array,
                    bool is_compute_mat);
  void _copyPartialToGlobal(IMeshMaterial* mat, VariableCellReal& global_density,
                            MaterialVariableCellReal& partial_variable);
  void _checkCreation();
  void _checkCreation2(Integer a, Integer n);
  void _checkTemporaryVectors(const CellGroup& test_group);
  void _checkSubViews(const CellGroup& test_group);
  void _dumpAverageValues();
  void _dumpNoDumpRealValues();
  void _checkRunQueues();

  void _computeMaterialDepend(IMeshMaterial* mat);
  void _setDependencies();
  void _doDependencies();
  Integer _checkParallelMatItem(MatItemVectorView view, MaterialVariableCellInt64& var);
  Integer _fillTestVar(ComponentItemVectorView view, MaterialVariableCellInt64& var);
  Integer _fillTestVar(IMeshMaterial* mat, MaterialVariableCellInt64& var);
  void _subViewFunctor(ComponentItemVectorView view);
  void _checkEqual(Integer expected_value, Integer value);
  template <typename ContainerType> void applyGeneric(const ContainerType& container, MaterialVariableCellReal& var, Real value);
  void _checkFillArrayFromTo(IMeshMaterial* mat, MaterialVariableCellReal& var);
  void _checkArrayVariableSynchronize();
  void _setOrCheckSpectralValues(Int64 iteration, bool is_check);
  template <typename VarType1, typename VarType2, typename VarType3> void
  _setOrCheckSpectralValues2(VarType1& var_real, VarType2& var_int32,
                             VarType3& var_scalar_int32, Int64 iteration, bool is_check);
  void _checkFillPartialValues();
  void _doSimd();
  template <typename VarType> void _checkFillPartialValuesHelper(VarType& mat_var);
  template <typename VarType>
  void _checkFillPartialValuesWithGlobal(const VarType& var_type, MeshComponentList components);
  template <typename VarType> void
  _checkFillPartialValuesWithSuper(const VarType& var_type, MeshComponentList components);
  template <typename VarType> void
  _fillVar(VarType& var_type, Real base_value);
  template <typename VectorType> void
  _checkVectorCopy(VectorType& var_type);
  void _testComponentPart(IMeshMaterial* mat, IMeshEnvironment* env);
  void _initUnitTest();
  void _applyEos(bool is_init);
  void _testDumpProperties();
  void _checkNullComponentItem();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
