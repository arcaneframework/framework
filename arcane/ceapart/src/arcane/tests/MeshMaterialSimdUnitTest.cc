// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialSimdUnitTest.cc                                 (C) 2000-2023 */
/*                                                                           */
/* Service de test unitaire de la vectorisation des matériaux/milieux.       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/ArithmeticException.h"
#include "arcane/utils/ValueChecker.h"

#include "arcane/utils/Simd.h"
#include "arcane/utils/SimdOperation.h"

#include "arcane/BasicUnitTest.h"
#include "arcane/ServiceBuilder.h"
#include "arcane/FactoryService.h"

#include "arcane/VariableView.h"

#include "arcane/materials/ComponentSimd.h"
#include "arcane/materials/IMeshMaterialMng.h"
#include "arcane/materials/IMeshMaterial.h"
#include "arcane/materials/IMeshEnvironment.h"
#include "arcane/materials/MeshEnvironmentBuildInfo.h"
#include "arcane/materials/MeshMaterialModifier.h"
#include "arcane/materials/MatCellVector.h"
#include "arcane/materials/EnvCellVector.h"
#include "arcane/materials/MatItemEnumerator.h"
#include "arcane/materials/MeshMaterialVariableRef.h"
#include "arcane/materials/MeshEnvironmentVariableRef.h"
#include "arcane/materials/EnvItemVector.h"

#include "arcane/tests/ArcaneTestGlobal.h"

// Pour les définitions, il faut finir par GCC car Clang et ICC définissent
// la macro __GNU__
// Pour CLANG, il n'y a pas encore d'équivalent au pragma ivdep.
// Celui qui s'en approche le plus est:
//   #pragma clang loop vectorize(enable)
// mais il ne force pas la vectorisation.
#ifdef __clang__
#  define PRAGMA_IVDEP_VALUE "clang loop vectorize(enable)"
#else
#  ifdef __INTEL_COMPILER
#    define PRAGMA_IVDEP_VALUE "ivdep"
#  else
#    ifdef __GNUC__
// S'assure qu'on compile avec la vectorisation même en '-O2'
// NOTE: à partir de GCC 12, le '-O2' implique aussi la vectorisation
#      pragma GCC optimize ("-ftree-vectorize")
#      define PRAGMA_IVDEP_VALUE "GCC ivdep"
#    endif
#  endif
#endif

//#undef PRAGMA_IVDEP_VALUE

#ifdef PRAGMA_IVDEP_VALUE
#define PRAGMA_IVDEP _Pragma(PRAGMA_IVDEP_VALUE)
#else
#define PRAGMA_IVDEP
#define PRAGMA_IVDEP_VALUE ""
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
MATERIALS_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
String getCompilerInfo()
{
  OStringStream ostr;
  String compiler_name = "Unknown";
  Integer version_major = 0;
  Integer version_minor = 0;
#ifdef __clang__
  compiler_name = "Clang";
  version_major = __clang_major__;
  version_minor = __clang_minor__;
#else
#ifdef __INTEL_COMPILER
  compiler_name = "ICC";
  version_major = __INTEL_COMPILER / 100;
  version_minor = __INTEL_COMPILER % 100;
#else
#ifdef __GNUC__
  compiler_name = "GCC";
  version_major = __GNUC__;
  version_minor = __GNUC_MINOR__;
#endif // __GNUC__
#endif // __INTEL_COMPILER
#endif // __clang__
  ostr() << compiler_name << " " << version_major << "." << version_minor;
  return ostr.str();
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Normalement pour l'EOS il faut utiliser la racine carrée mais GCC 6
// ne tente pas de vectorisation (au contraire d'ICC 17) si cette opération est présente.
// Pour vérifier qu'on vectorise bien, on supprime donc l'appel à la
// racine carrée. On définit la macro USE_SQRT_IN_EOS à 1 pour utiliser la
// racine carrée, à 0 sinon.
#define USE_SQRT_IN_EOS 1

#if USE_SQRT_IN_EOS == 1
#define DO_SQRT(v) math::sqrt( v )
#else
#define DO_SQRT(v) (v)
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Enumérateur sur une sous-partie (pure ou partielle) d'un
 * sous-ensemble des mailles d'un composant (matériau ou milieu)
 */
class FullComponentPartCellEnumerator
{
 protected:
  FullComponentPartCellEnumerator(IMeshComponent* component,
                                  ComponentPartItemVectorView pure_view,
                                  ComponentPartItemVectorView mixed_view)
  : m_index(0), m_component(component)
  {
    m_items_view[0] = pure_view;
    m_items_view[1] = mixed_view;
  }
 public:
  static FullComponentPartCellEnumerator create(IMeshComponent* c)
  {
    return FullComponentPartCellEnumerator(c,c->pureItems(),c->impureItems());
  }
 public:
  void operator++()
  {
    ++m_index;
  }
  bool hasNext() const { return m_index<2; }
  ComponentPartItemVectorView operator*() const
  {
    return m_items_view[m_index];
  }
 protected:

  Integer m_index;
  ComponentPartItemVectorView m_items_view[2];
  IMeshComponent* m_component;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ENUMERATE_ENVCELL2(iname,env) \
  for( FullComponentPartCellEnumerator iname##_part(Arcane::Materials::FullComponentPartCellEnumerator::create(env)); iname##_part . hasNext(); ++ iname##_part ) \
    PRAGMA_IVDEP \
    A_ENUMERATE_COMPONENTCELL(ComponentPartCellEnumerator,iname,*(iname##_part))

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;
using namespace Materials;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
class PerfTimer
{
  public:
  PerfTimer(const char* msg)
  : m_begin_time(0.0), m_msg(msg), m_has_error(false)
  {
    m_begin_time = platform::getRealTime();
  }
  ~PerfTimer()
  {
    Real end_time = platform::getRealTime();
    Real true_time_v = end_time - m_begin_time;
    double true_time = (double)(true_time_v);
    std::cout << " -- -- Time: ";
    std::cout.width(60);
    std::cout << m_msg << " = ";
    if (m_has_error)
      std::cout << "ERROR";
    else
      std::cout << (true_time);
    std::cout << '\n';
    std::cout.flush();
  }
  void setError(bool v) { m_has_error = v; }
 private:
  Real m_begin_time;
  const char* m_msg;
  bool m_has_error;
};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de test unitaire de la vectorisation des matériaux/milieux.
 */
class MeshMaterialSimdUnitTest
: public BasicUnitTest
{
 public:

  MeshMaterialSimdUnitTest(const ServiceBuildInfo& cb);
  ~MeshMaterialSimdUnitTest() override;

 public:

  void initializeTest() override;
  void executeTest() override;

 private:

  IMeshEnvironment* m_env1 = nullptr;
  MaterialVariableCellReal m_mat_a;
  MaterialVariableCellReal m_mat_b;
  MaterialVariableCellReal m_mat_c;
  MaterialVariableCellReal m_mat_d;
  MaterialVariableCellReal m_mat_e;

  MaterialVariableCellReal m_mat_adiabatic_cst;
  MaterialVariableCellReal m_mat_volume;
  MaterialVariableCellReal m_mat_density;
  MaterialVariableCellReal m_mat_old_volume;
  MaterialVariableCellReal m_mat_internal_energy;
  MaterialVariableCellReal m_mat_sound_speed;
  MaterialVariableCellReal m_mat_pressure;

  MaterialVariableCellReal m_mat_ref_internal_energy;
  MaterialVariableCellReal m_mat_ref_sound_speed;
  MaterialVariableCellReal m_mat_ref_pressure;

  UniqueArray<Int32> m_env1_pure_value_index;
  UniqueArray<Int32> m_env1_partial_value_index;
  EnvCellVector* m_env1_as_vector = nullptr;
  Int32 m_nb_z;

  void _initializeVariables();
  Real _executeDirect1(Integer nb_z);
  void _executeTest1(Integer nb_z);
  void _executeTest1_bis(Integer nb_z);
  void _executeTest2(Integer nb_z);
  void _executeTest2_bis(Integer nb_z);
  void _executeTest3(Integer nb_z);
  void _executeTest5(Integer nb_z);
  void _executeTest6(Integer nb_z);
  void _executeTest7(Integer nb_z);
  void _executeTest8(Integer nb_z);
  void _executeTest9(Integer nb_z);
  void _executeTest10(Integer nb_z);
  void _initForEquationOfState();
  void _compareValues();
  void _computeEquationOfStateReference();
  void _computeEquationOfStateDirect1();
  void _computeEquationOfStateIndirect1();
  void _computeEquationOfStateV1();
  void _computeEquationOfStateV1_bis();
  void _computeEquationOfStateV2();
  void _computeEquationOfStateV3();
  void _computeEquationOfStateV4();
  void _computeEquationOfStateV4_noview();

  template<typename Lambda>
  bool _apply(const char* message,Lambda&& lambda)
  {
    bool has_error = false;
    {
      PerfTimer pf(message);
      try{
        lambda();
      }
      catch(const ArithmeticException& ex){
        pf.setError(true);
        has_error = true;
        info() << "ArithmeticException for '" << message << "'";
      }
    }
    return has_error;
  }

  template<typename Lambda>
  void _applyCompare(const char* message,Lambda&& lambda)
  {
    _initForEquationOfState();
    bool has_error = _apply(message,lambda);
    if (!has_error)
      _compareValues();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY(MeshMaterialSimdUnitTest,
                                           IUnitTest,MeshMaterialSimdUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialSimdUnitTest::
MeshMaterialSimdUnitTest(const ServiceBuildInfo& sb)
: BasicUnitTest(sb)
, m_env1(nullptr)
, m_mat_a(VariableBuildInfo(mesh(),"MatA"))
, m_mat_b(VariableBuildInfo(mesh(),"MatB"))
, m_mat_c(VariableBuildInfo(mesh(),"MatC"))
, m_mat_d(VariableBuildInfo(mesh(),"MatD"))
, m_mat_e(VariableBuildInfo(mesh(),"MatE"))
, m_mat_adiabatic_cst(VariableBuildInfo(mesh(),"MatAdiabaticCst"))
, m_mat_volume(VariableBuildInfo(mesh(),"MatVolume"))
, m_mat_density(VariableBuildInfo(mesh(),"MatDensity"))
, m_mat_old_volume(VariableBuildInfo(mesh(),"MatOldVolume"))
, m_mat_internal_energy(VariableBuildInfo(mesh(),"MatInternalEnergy"))
, m_mat_sound_speed(VariableBuildInfo(mesh(),"MatSoundSpeed"))
, m_mat_pressure(VariableBuildInfo(mesh(),"MatPressure"))
, m_mat_ref_internal_energy(VariableBuildInfo(mesh(),"MatRefInternalEnergy"))
, m_mat_ref_sound_speed(VariableBuildInfo(mesh(),"MatRefSoundSpeed"))
, m_mat_ref_pressure(VariableBuildInfo(mesh(),"MatRefPressure"))
, m_nb_z(0)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialSimdUnitTest::
~MeshMaterialSimdUnitTest()
{
  delete m_env1_as_vector;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialSimdUnitTest::
initializeTest()
{
  Materials::IMeshMaterialMng* mm = IMeshMaterialMng::getReference(mesh());

  // Lit les infos des matériaux du JDD et les enregistre dans le gestionnaire
  UniqueArray<String> mat_names = { "MAT1", "MAT2", "MAT3" };
  for( String v : mat_names ){
    mm->registerMaterialInfo(v);
  }

  {
    Materials::MeshEnvironmentBuildInfo env_build("ENV1");
    env_build.addMaterial("MAT1");
    env_build.addMaterial("MAT2");
    mm->createEnvironment(env_build);
  }
  {
    Materials::MeshEnvironmentBuildInfo env_build("ENV2");
    env_build.addMaterial("MAT2");
    env_build.addMaterial("MAT3");
    mm->createEnvironment(env_build);
  }

  mm->endCreate(false);

  IMeshEnvironment* env1 = mm->environments()[0];
  IMeshEnvironment* env2 = mm->environments()[1];

  m_env1 = env1;

  IMeshMaterial* mat1 = env1->materials()[0];
  IMeshMaterial* mat2 = env2->materials()[1];

  {
    Int32UniqueArray env1_indexes;
    Int32UniqueArray mat2_indexes;
    Int32UniqueArray sub_group_indexes;
    Integer nb_cell = ownCells().size();
    Int64 total_nb_cell = nb_cell;
    ENUMERATE_CELL(icell,allCells()){
      Cell cell = *icell;
      Int64 cell_index = cell.uniqueId();
      if (cell_index<((2*total_nb_cell)/3)){
        env1_indexes.add(icell.itemLocalId());
      }
      if (cell_index<(total_nb_cell/2) && cell_index>(total_nb_cell/3)){
        mat2_indexes.add(icell.itemLocalId());
      }
      if ((cell_index%2)==0)
        sub_group_indexes.add(icell.itemLocalId());
    }

    // Ajoute les mailles du milieu 1
    {
      Materials::MeshMaterialModifier modifier(mm);
      Materials::IMeshEnvironment* env = mat1->environment();
      // Ajoute les mailles du milieu
      //modifier.addCells(env,env1_indexes);
      Int32UniqueArray mat1_indexes;
      Int32UniqueArray mat2_indexes;
      Integer nb_cell = env1_indexes.size();
      for( Integer z=0; z<nb_cell; ++z ){
        bool add_to_mat1 = (z<(nb_cell/2) && z>(nb_cell/4));
        bool add_to_mat2 = (z>=(nb_cell/2) || z<(nb_cell/3));
        if (add_to_mat1){
          mat1_indexes.add(env1_indexes[z]);
        }
        if (add_to_mat2)
          mat2_indexes.add(env1_indexes[z]);
      }
      // Ajoute les mailles du matériau 1
      modifier.addCells(mat1,mat1_indexes);
      Integer nb_mat = env->nbMaterial();
      if (nb_mat>1)
        // Ajoute les mailles du matériau 2
        modifier.addCells(mm->environments()[0]->materials()[1],mat2_indexes);
    }
    // Ajoute les mailles du milieu 2
    if (mat2){
      Materials::MeshMaterialModifier modifier(mm);
      //modifier.addCells(m_mat2->environment(),mat2_indexes);
      modifier.addCells(mat2,mat2_indexes);
    }
  }

  for( IMeshEnvironment* env : mm->environments() ){
    info() << "** ** ENV name=" << env->name() << " nb_item=" << env->view().nbItem();
    Integer nb_pure_env = 0;
    ENUMERATE_ENVCELL(ienvcell,env){
      if ( (*ienvcell).allEnvCell().nbEnvironment()==1 )
        ++nb_pure_env;
    }
    info() << "** ** NB_PURE=" << nb_pure_env;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialSimdUnitTest::
executeTest()
{
  m_env1_pure_value_index.clear();
  m_env1_partial_value_index.clear();
  Integer nb_unknown = 0;
  {
    ENUMERATE_ENVCELL(ienvcell,m_env1){
      EnvCell env_cell = *ienvcell;
      MatVarIndex mvi = env_cell._varIndex();
      Integer nb_env = env_cell.allEnvCell().nbEnvironment();
      if ( nb_env==1 )
        m_env1_pure_value_index.add(mvi.valueIndex());
      else if (nb_env>1)
        m_env1_partial_value_index.add(mvi.valueIndex());
      else
        ++nb_unknown;
    }
  }
  m_env1_as_vector = new EnvCellVector(m_env1->cells(),m_env1);

  Integer nb_z = 10000;
  if (arcaneIsDebug())
    nb_z /= 100;
  Integer nb_z2 = nb_z / 5;

  Int32 env_idx = m_env1->variableIndexer()->index() + 1;
  info() << "Using vectorisation name=" << SimdInfo::name()
         << " vector_size=" << SimdReal::Length << " index_size=" << SimdInfo::Int32IndexSize;
  info() << "Compiler=\"" << getCompilerInfo() << "\""
         << " machine=" << platform::getHostName();
  info() << "use_sqrt_in_eos?=" << USE_SQRT_IN_EOS
         << " PRAGMA_IVDEP=" << PRAGMA_IVDEP_VALUE;
  info() << "ENV_IDX=" << env_idx
         << " nb_pure=" << m_env1_pure_value_index.size()
         << " nb_partial=" << m_env1_partial_value_index.size()
         << " nb_unknown=" << nb_unknown
         << " nb_z=" << nb_z << " nb_z2=" << nb_z2;

  _initializeVariables();
  {
    PerfTimer pf("executeDirect1");
    _executeDirect1(nb_z);
  }
  {
    PerfTimer pf("executeTest1 (ENVCELL)");
    _executeTest1(nb_z);
  }
  {
    PerfTimer pf("executeTest1_bis (ENVCELL2)");
    _executeTest1_bis(nb_z);
  }
  {
    PerfTimer pf("executeTest2 (ENVCELL,IVDEP,view)");
    _executeTest2(nb_z);
  }
  {
    PerfTimer pf("executeTest2_bis (ENVCELL2,view)");
    _executeTest2_bis(nb_z);
  }
  {
    PerfTimer pf("executeTest3 (2 loops)");
    _executeTest3(nb_z);
  }
  {
    PerfTimer pf("executeTest4 (2 loops, ivdep)");
    _executeTest3(nb_z);
  }
  {
    PerfTimer pf("executeTest5 (2 loops, functor)");
    _executeTest5(nb_z);
  }
  {
    PerfTimer pf("executeTest6 (2 loops, functor)");
    _executeTest6(nb_z);
  }
  {
    PerfTimer pf("executeTest7 (2 loops, loop functor)");
    _executeTest7(nb_z);
  }
  {
    PerfTimer pf("executeTest8 (simple_env_loop)");
    _executeTest8(nb_z);
  }
  {
    PerfTimer pf("executeTest9 (SIMD_ENVCELL)");
    _executeTest9(nb_z);
  }
  {
    PerfTimer pf("executeTest10 (SIMD_ENVCELL, functor)");
    _executeTest10(nb_z);
  }
  m_nb_z = nb_z2;
  _initForEquationOfState();
  _computeEquationOfStateReference();
  _apply("_computeEquationOfStateDirect1",
         [&](){_computeEquationOfStateDirect1();});
  _apply("_computeEquationOfStateIndirect1",
         [&](){_computeEquationOfStateIndirect1();});
  _applyCompare("_computeEquationOfStateV1 (ENVCELL)",
                [&](){_computeEquationOfStateV1();});
  _applyCompare("_computeEquationOfStateV1_bis (ENVCELL2)",
                [&](){_computeEquationOfStateV1_bis();});

  _applyCompare("_computeEquationOfStateV2 (simple_simd_env_loop)",
                [&](){_computeEquationOfStateV2();});
  _applyCompare("_computeEquationOfStateV3 (simple_env_loop)",
                [&](){_computeEquationOfStateV3();});

  _applyCompare("_computeEquationOfStateV4 (lambda simple_env_loop2)",
                [&](){_computeEquationOfStateV4();});
  _applyCompare("_computeEquationOfStateV4_noview (lambda simple_env_loop2)",
                [&](){_computeEquationOfStateV4_noview();});

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialSimdUnitTest::
_initializeVariables()
{
  MaterialVariableCellReal& a(m_mat_a);
  MaterialVariableCellReal& b(m_mat_b);
  MaterialVariableCellReal& c(m_mat_c);
  MaterialVariableCellReal& d(m_mat_d);
  MaterialVariableCellReal& e(m_mat_e);

  ENUMERATE_ENVCELL(i,m_env1){
    Real z = (Real)i.index();
    b[i] = z*2.3;
    c[i] = z*3.1;
    d[i] = z*2.5;
    e[i] = z*4.2;
    a[i] = 0;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real MeshMaterialSimdUnitTest::
_executeDirect1(Integer nb_z)
{
  Integer TRUE_SIZE = m_env1->cells().size();
  Real ARCANE_RESTRICT *a = new Real[TRUE_SIZE];
  Real *b = new Real[TRUE_SIZE];
  Real *c = new Real[TRUE_SIZE];
  Real *d = new Real[TRUE_SIZE];
  Real *e = new Real[TRUE_SIZE];
#pragma omp parallel for
  for( Integer i=0, is=TRUE_SIZE; i<is; ++i ){
    Real z = (Real)i;
    a[i] = b[i] = c[i] = d[i] = e[i] = z;
  }
  Real s = 0.0;
  for( Integer z=0, iz=nb_z; z<iz; ++z ){
    for( Integer i=0, is=TRUE_SIZE; i<is; ++i ){
      a[i] = b[i] + c[i] * d[i] + e[i];
    }
    s += a[z%5];
  }
  delete[] a;
  delete[] b;
  delete[] c;
  delete[] d;
  delete[] e;
  return s;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialSimdUnitTest::
_executeTest1(Integer nb_z)
{
  MaterialVariableCellReal& a(m_mat_a);
  MaterialVariableCellReal& b(m_mat_b);
  MaterialVariableCellReal& c(m_mat_c);
  MaterialVariableCellReal& d(m_mat_d);
  MaterialVariableCellReal& e(m_mat_e);

  for( Integer z=0, iz=nb_z; z<iz; ++z ){
    ENUMERATE_ENVCELL(i,m_env1){
      a[i] = b[i] + c[i] * d[i] + e[i];
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialSimdUnitTest::
_executeTest1_bis(Integer nb_z)
{
  MaterialVariableCellReal& a(m_mat_a);
  MaterialVariableCellReal& b(m_mat_b);
  MaterialVariableCellReal& c(m_mat_c);
  MaterialVariableCellReal& d(m_mat_d);
  MaterialVariableCellReal& e(m_mat_e);

  for( Integer z=0, iz=nb_z; z<iz; ++z ){
    ENUMERATE_ENVCELL2(i,m_env1){
      a[i] = b[i] + c[i] * d[i] + e[i];
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialSimdUnitTest::
_executeTest2(Integer nb_z)
{
  auto a(viewOut(m_mat_a));
  auto b(viewIn(m_mat_b));
  auto c(viewIn(m_mat_c));
  auto d(viewIn(m_mat_d));
  auto e(viewIn(m_mat_e));

  for( Integer z=0, iz=nb_z; z<iz; ++z ){
    PRAGMA_IVDEP
    // Certaines versions du compilateur intel (au moins la version 20.0)
    // donnent une erreur sur le 'pragma omp simd' car le test de la boucle
    // est une fonction. Pour éviter une erreur de compilation on supprime
    // ce pragma avec le compilateur intel. A noter que cela semble fonctionner
    // avec les versions 2021+ de Intel (icpc, icpx et DPC++).
#ifndef __INTEL_COMPILER
#pragma omp simd
#endif
    ENUMERATE_ENVCELL(i,m_env1){
      a[i] = b[i] + c[i] * d[i] + e[i];
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialSimdUnitTest::
_executeTest2_bis(Integer nb_z)
{
  auto a(viewOut(m_mat_a));
  auto b(viewIn(m_mat_b));
  auto c(viewIn(m_mat_c));
  auto d(viewIn(m_mat_d));
  auto e(viewIn(m_mat_e));

  for( Integer z=0, iz=nb_z; z<iz; ++z ){
    ENUMERATE_ENVCELL2(i,m_env1){
      a[i] = b[i] + c[i] * d[i] + e[i];
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialSimdUnitTest::
_executeTest3(Integer nb_z)
{
  auto a(viewOut(m_mat_a));
  auto b(viewIn(m_mat_b));
  auto c(viewIn(m_mat_c));
  auto d(viewIn(m_mat_d));
  auto e(viewIn(m_mat_e));

  for( Integer z=0, iz=nb_z; z<iz; ++z ){
    {
      Int32ConstArrayView indexes = m_env1_pure_value_index;
      PRAGMA_IVDEP
      for( Integer i=0, n=indexes.size(); i<n; ++i ){
        Int32 xi = indexes[i];
        MatVarIndex mvi(0,xi);
        a[mvi] = b[mvi] + c[mvi] * d[mvi] + e[mvi];
      }
    }
    {
      Int32ConstArrayView indexes = m_env1_partial_value_index;
      Int32 env_idx = m_env1->variableIndexer()->index() + 1;
      PRAGMA_IVDEP
      for( Integer i=0, n=indexes.size(); i<n; ++i ){
        Int32 xi = indexes[i];
        MatVarIndex mvi(env_idx,xi);
        a[mvi] = b[mvi] + c[mvi] * d[mvi] + e[mvi];
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialSimdUnitTest::
_executeTest5(Integer nb_z)
{
  auto a(viewOut(m_mat_a));
  auto b(viewIn(m_mat_b));
  auto c(viewIn(m_mat_c));
  auto d(viewIn(m_mat_d));
  auto e(viewIn(m_mat_e));

  auto func = [=](Int32 env_idx,Int32ConstArrayView indexes)
  {
    for( Integer i=0, n=indexes.size(); i<n; ++i ){
      Int32 xi = indexes[i];
      MatVarIndex mvi(env_idx,xi);
      a[mvi] = b[mvi] + c[mvi] * d[mvi] + e[mvi];
    }
  };

  for( Integer z=0, iz=nb_z; z<iz; ++z ){
    Int32 env_idx = m_env1->variableIndexer()->index() + 1;
    func(0,m_env1_pure_value_index);
    func(env_idx,m_env1_partial_value_index);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialSimdUnitTest::
_executeTest6(Integer nb_z)
{
  auto a(viewOut(m_mat_a));
  auto b(viewIn(m_mat_b));
  auto c(viewIn(m_mat_c));
  auto d(viewIn(m_mat_d));
  auto e(viewIn(m_mat_e));

  auto func = [=](Int32 env_idx,Int32ConstArrayView indexes)
  {
    for( Integer i=0, n=indexes.size(); i<n; ++i ){
      Int32 xi = indexes[i];
      MatVarIndex mvi(env_idx,xi);
      a[mvi] = b[mvi] + c[mvi] * d[mvi] + e[mvi];
    }
  };

  for( Integer z=0, iz=nb_z; z<iz; ++z ){
    {
      ComponentPartItemVectorView pv = m_env1_as_vector->pureItems();
      func(pv.componentPartIndex(),pv.valueIndexes());
    }
    {
      ComponentPartItemVectorView pv = m_env1_as_vector->impureItems();
      func(pv.componentPartIndex(),pv.valueIndexes());
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename Lambda> void
env_loop(const EnvCellVector& env,const Lambda& lambda)
{
  lambda(env.pureItems());
  lambda(env.impureItems());
}

void MeshMaterialSimdUnitTest::
_executeTest7(Integer nb_z)
{
  auto a(viewOut(m_mat_a));
  auto b(viewIn(m_mat_b));
  auto c(viewIn(m_mat_c));
  auto d(viewIn(m_mat_d));
  auto e(viewIn(m_mat_e));

  for( Integer z=0, iz=nb_z; z<iz; ++z ){
    env_loop(*m_env1_as_vector,[=](ComponentPartItemVectorView view){
        ENUMERATE_COMPONENTITEM(ComponentPartCell,mvi,view){
          a[mvi] = b[mvi] + c[mvi] * d[mvi] + e[mvi];
        }});
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename Lambda> void
simple_env_loop(ComponentPartItemVectorView pure_items,
                ComponentPartItemVectorView impure_items,
                const Lambda& lambda)
{
  ComponentPartItemVectorView views[2];
  views[0] = pure_items;
  views[1] = impure_items;

  for( Integer iview=0; iview<2; ++iview ){
    auto xiter = views[iview];
    Int32 cpi = xiter.componentPartIndex();
    Int32ConstArrayView item_indexes = xiter.valueIndexes();
    Integer nb_item = xiter.nbItem();
    //std::cout << " cpi=" << cpi << " nb_item=" << nb_item << " nb_item2=" << item_indexes.size() << '\n';
    PRAGMA_IVDEP
    for( Integer i=0; i<nb_item; ++i ){
      MatVarIndex mvi(cpi,item_indexes[i]);
      //std::cout << " cpi=" << cpi << " I=" << i << " IDX=" << item_indexes[i] << '\n';
      lambda(mvi);
    }
  }
}

template<typename ContainerType,typename Lambda> void
simple_env_loop(ContainerType& ct,const Lambda& lambda)
{
  simple_env_loop(ct.pureItems(),ct.impureItems(),lambda);
}

template<typename Lambda> void
simple_env_loop2(ComponentPartItemVectorView pure_items,
                 ComponentPartItemVectorView impure_items,
                 const Lambda& lambda)
{
  ComponentPartItemVectorView views[2];
  views[0] = pure_items;
  views[1] = impure_items;

  {
    auto view = views[0];
    Int32ConstArrayView item_indexes = view.valueIndexes();
    Integer nb_item = view.nbItem();
    PRAGMA_IVDEP
    for( Integer i=0; i<nb_item; ++i ){
      lambda(PureMatVarIndex(item_indexes[i]));
      //lambda(PureMatVarIndex(i));
    }
  }

  {
    auto view = views[1];
    Int32 cpi = view.componentPartIndex();
    Integer nb_item = view.nbItem();
    Int32ConstArrayView item_indexes = view.valueIndexes();
    PRAGMA_IVDEP
    for( Integer i=0; i<nb_item; ++i ){
      // On sait pour ce test que les mailles partielles sont contigues
      // et item_indexes[i] == i.
      //MatVarIndex mvi(cpi,item_indexes[i]);
      ComponentItemLocalId mvi(MatVarIndex(cpi,item_indexes[i]));
      lambda(mvi);
    }
  }
}

template<typename ContainerType,typename Lambda> void
simple_env_loop2(const ContainerType& ct,const Lambda& lambda)
{
  simple_env_loop2(ct.pureItems(),ct.impureItems(),lambda);
}

void MeshMaterialSimdUnitTest::
_executeTest8(Integer nb_z)
{
  auto a(viewOut(m_mat_a));
  auto b(viewIn(m_mat_b));
  auto c(viewIn(m_mat_c));
  auto d(viewIn(m_mat_d));
  auto e(viewIn(m_mat_e));

  for( Integer z=0, iz=nb_z; z<iz; ++z ){
    simple_env_loop(*m_env1_as_vector,[=](MatVarIndex mvi){
        a[mvi] = b[mvi] + c[mvi] * d[mvi] + e[mvi];
      });
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialSimdUnitTest::
_executeTest9(Integer nb_z)
{
  auto a(viewOut(m_mat_a));
  auto b(viewIn(m_mat_b));
  auto c(viewIn(m_mat_c));
  auto d(viewIn(m_mat_d));
  auto e(viewIn(m_mat_e));

  for( Integer z=0, iz=nb_z; z<iz; ++z ){
    env_loop(*m_env1_as_vector,[=](ComponentPartItemVectorView view){
        ENUMERATE_SIMD_COMPONENTCELL(mvi,view){
          a[mvi] = b[mvi] + c[mvi] * d[mvi] + e[mvi];
        }});
  }

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define A_ENUMERATE_LAMBDA(iter_type,iter,container,...)     \
  Arcane::Materials:: LoopFunctor ## iter_type ( (container) ) << __VA_ARGS__ ( Arcane::Materials:: LoopFunctor ## iter_type :: IterType iter)

#define A_CAPTURE(a) a
#define A_ENUMERATE_LAMBDA2(iter_type,iter,container,lambda_capture)     \
  Arcane::Materials:: LoopFunctor ## iter_type ( (container) ) << A_CAPTURE(lambda_capture) ( Arcane::Materials:: LoopFunctor ## iter_type :: IterType iter)

#define A_ENUMERATE_LAMBDA3(iter_type,iter,container)     \
  Arcane::Materials:: LoopFunctor ## iter_type ( (container) ) <<

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// TODO: mettre dans .h
template<typename Lambda> void
simple_simd_env_loop(const EnvCellVector& env,const Lambda& lambda)
{
  simple_simd_env_loop(env.pureItems(),env.impureItems(),lambda);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialSimdUnitTest::
_executeTest10(Integer nb_z)
{
  auto a(viewOut(m_mat_a));
  auto b(viewIn(m_mat_b));
  auto c(viewIn(m_mat_c));
  auto d(viewIn(m_mat_d));
  auto e(viewIn(m_mat_e));

  for( Integer z=0, iz=nb_z; z<iz; ++z ){
    simple_simd_env_loop(*m_env1_as_vector,[=](const SimdMatVarIndex& mvi){
        a[mvi] = b[mvi] + c[mvi] * d[mvi] + e[mvi];
      });
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialSimdUnitTest::
_initForEquationOfState()
{
  ENUMERATE_ENVCELL(icell,m_env1){
    EnvCellEnumerator vi = icell;
    m_mat_adiabatic_cst[vi] = 1.4;
    m_mat_volume[vi] = 1.2;
    m_mat_old_volume[vi] = 1.1;

    m_mat_density[vi] = 2.0;
    m_mat_pressure[vi] = 1.1;

    m_mat_internal_energy[vi] = m_mat_pressure[icell] / ((m_mat_adiabatic_cst[icell]-1.0) * m_mat_density[icell]);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialSimdUnitTest::
_computeEquationOfStateDirect1()
{
  Int32 nb_z = m_nb_z;
  Integer TRUE_SIZE = m_env1->cells().size();

  auto in_adiabatic_cst = new Real[TRUE_SIZE];
  auto in_volume = new Real[TRUE_SIZE];
  auto in_density = new Real[TRUE_SIZE];
  auto in_old_volume = new Real[TRUE_SIZE];
  auto in_internal_energy = new Real[TRUE_SIZE];

  auto out_internal_energy = in_internal_energy;
  auto out_sound_speed = new Real[TRUE_SIZE];
  auto out_pressure = new Real[TRUE_SIZE];

  for( Integer i=0, is=TRUE_SIZE; i<is; ++i ){
    Integer  vi = i;
    in_adiabatic_cst[vi] = 1.4;
    in_volume[vi] = 1.2;
    in_old_volume[vi] = 1.1;

    in_density[vi] = 2.0;
    Real in_pressure = 1.1;

    in_internal_energy[vi] = in_pressure / ((in_adiabatic_cst[vi]-1.0) * in_density[vi]);
  }
  for( Int32 iloop=0; iloop<nb_z; ++iloop){

    PRAGMA_IVDEP
    for( Integer i=0, n=TRUE_SIZE; i<n; ++i ){
      Integer  vi = i;
      Real adiabatic_cst = in_adiabatic_cst[vi];
      Real volume_ratio = in_volume[vi] / in_old_volume[vi];
      Real x = 0.5 * adiabatic_cst - 1.0;
      Real numer_accrois_nrj = 1.0 + x*(1.0-volume_ratio);
      Real denom_accrois_nrj = 1.0 + x*(1.0-(1.0/volume_ratio));
      Real internal_energy = in_internal_energy[vi];
      internal_energy = internal_energy * (numer_accrois_nrj/denom_accrois_nrj);
      out_internal_energy[vi] = internal_energy;
      Real density = in_density[vi];
      Real pressure = (adiabatic_cst - 1.0) * density * internal_energy;
      Real sound_speed = DO_SQRT((adiabatic_cst*pressure/density));
      out_pressure[vi] = pressure;
      out_sound_speed[vi] = sound_speed;
    }
  }
  delete[] in_adiabatic_cst;
  delete[] in_volume;
  delete[] in_density;
  delete[] in_old_volume;
  delete[] in_internal_energy;
  delete[] out_sound_speed;
  delete[] out_pressure;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialSimdUnitTest::
_computeEquationOfStateIndirect1()
{
  Int32 nb_z = m_nb_z;
  Integer TRUE_SIZE = m_env1->cells().size();

  auto in_adiabatic_cst = new Real[TRUE_SIZE];
  auto in_volume = new Real[TRUE_SIZE];
  auto in_density = new Real[TRUE_SIZE];
  auto in_old_volume = new Real[TRUE_SIZE];
  auto in_internal_energy = new Real[TRUE_SIZE];

  auto out_internal_energy = in_internal_energy;
  auto out_sound_speed = new Real[TRUE_SIZE];
  auto out_pressure = new Real[TRUE_SIZE];

  ARCANE_RESTRICT Int32* idx = new Int32[TRUE_SIZE];

  for( Integer i=0, is=TRUE_SIZE; i<is; ++i ){
    idx[i] = i;
    Integer  vi = idx[i];

    in_adiabatic_cst[vi] = 1.4;
    in_volume[vi] = 1.2;
    in_old_volume[vi] = 1.1;

    in_density[vi] = 2.0;
    Real in_pressure = 1.1;

    in_internal_energy[vi] = in_pressure / ((in_adiabatic_cst[vi]-1.0) * in_density[vi]);
  }
  for( Int32 iloop=0; iloop<nb_z; ++iloop){

    PRAGMA_IVDEP
    for( Integer i=0, n=TRUE_SIZE; i<n; ++i ){
      Integer  vi = idx[i];
      Real adiabatic_cst = in_adiabatic_cst[vi];
      Real volume_ratio = in_volume[vi] / in_old_volume[vi];
      Real x = 0.5 * adiabatic_cst - 1.0;
      Real numer_accrois_nrj = 1.0 + x*(1.0-volume_ratio);
      Real denom_accrois_nrj = 1.0 + x*(1.0-(1.0/volume_ratio));
      Real internal_energy = in_internal_energy[vi];
      internal_energy = internal_energy * (numer_accrois_nrj/denom_accrois_nrj);
      out_internal_energy[vi] = internal_energy;
      Real density = in_density[vi];
      Real pressure = (adiabatic_cst - 1.0) * density * internal_energy;
      Real sound_speed = DO_SQRT((adiabatic_cst*pressure/density));
      out_pressure[vi] = pressure;
      out_sound_speed[vi] = sound_speed;
    }
  }
  delete[] idx;
  delete[] in_adiabatic_cst;
  delete[] in_volume;
  delete[] in_density;
  delete[] in_old_volume;
  delete[] in_internal_energy;
  delete[] out_sound_speed;
  delete[] out_pressure;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialSimdUnitTest::
_computeEquationOfStateReference()
{
  Int32 nb_z = m_nb_z;

  auto in_adiabatic_cst = viewIn(m_mat_adiabatic_cst);
  auto in_volume = viewIn(m_mat_volume);
  auto in_density = viewIn(m_mat_density);
  auto in_old_volume = viewIn(m_mat_old_volume);
  auto in_internal_energy = viewIn(m_mat_internal_energy);

  for( Int32 iloop=0; iloop<nb_z; ++iloop){
    ENUMERATE_ENVCELL(icell,m_env1){
      auto vi = icell;
      Real adiabatic_cst = in_adiabatic_cst[vi];
      Real volume_ratio = in_volume[vi] / in_old_volume[vi];
      Real x = 0.5 * adiabatic_cst - 1.0;
      Real numer_accrois_nrj = 1.0 + x*(1.0-volume_ratio);
      Real denom_accrois_nrj = 1.0 + x*(1.0-(1.0/volume_ratio));
      Real internal_energy = in_internal_energy[vi];
      internal_energy = internal_energy * (numer_accrois_nrj/denom_accrois_nrj);
      m_mat_internal_energy[vi] = internal_energy;
      Real density = in_density[vi];
      Real pressure = (adiabatic_cst - 1.0) * density * internal_energy;
      Real sound_speed = DO_SQRT(adiabatic_cst*pressure/density);
      m_mat_pressure[vi] = pressure;
      m_mat_sound_speed[vi] = sound_speed;
    }
  }
  ENUMERATE_ENVCELL(vi,m_env1){
    m_mat_ref_internal_energy[vi] = m_mat_internal_energy[vi];
    m_mat_ref_pressure[vi] = m_mat_pressure[vi];
    m_mat_ref_sound_speed[vi] = m_mat_sound_speed[vi];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialSimdUnitTest::
_compareValues()
{
  ValueChecker vc(A_FUNCINFO);
  ENUMERATE_ENVCELL(icell,m_env1){
    vc.areEqual(m_mat_internal_energy[icell],m_mat_ref_internal_energy[icell],"Energy");
    vc.areEqual(m_mat_pressure[icell],m_mat_ref_pressure[icell],"Pressure");
    vc.areEqual(m_mat_sound_speed[icell],m_mat_ref_sound_speed[icell],"SoundSpeed");
  }
}

/*----------------------a-----------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialSimdUnitTest::
_computeEquationOfStateV1()
{
  Int32 nb_z = m_nb_z;
  auto in_adiabatic_cst = viewIn(m_mat_adiabatic_cst);
  auto in_volume = viewIn(m_mat_volume);
  auto in_density = viewIn(m_mat_density);
  auto in_old_volume = viewIn(m_mat_old_volume);
  auto in_internal_energy = viewIn(m_mat_internal_energy);

  auto out_internal_energy = viewOut(m_mat_internal_energy);
  auto out_sound_speed = viewOut(m_mat_sound_speed);
  auto out_pressure = viewOut(m_mat_pressure);

  for( Int32 iloop=0; iloop<nb_z; ++iloop){

    PRAGMA_IVDEP
    ENUMERATE_ENVCELL(icell,m_env1){
      auto vi = icell;
      Real adiabatic_cst = in_adiabatic_cst[vi];
      Real volume_ratio = in_volume[vi] / in_old_volume[vi];
      Real x = 0.5 * adiabatic_cst - 1.0;
      Real numer_accrois_nrj = 1.0 + x*(1.0-volume_ratio);
      Real denom_accrois_nrj = 1.0 + x*(1.0-(1.0/volume_ratio));
      Real internal_energy = in_internal_energy[vi];
      internal_energy = internal_energy * (numer_accrois_nrj/denom_accrois_nrj);
      out_internal_energy[vi] = internal_energy;
      Real density = in_density[vi];
      Real pressure = (adiabatic_cst - 1.0) * density * internal_energy;
      Real sound_speed = DO_SQRT(adiabatic_cst*pressure/density);
      out_pressure[vi] = pressure;
      out_sound_speed[vi] = sound_speed;
    }
  }
}

/*----------------------a-----------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialSimdUnitTest::
_computeEquationOfStateV1_bis()
{
  Int32 nb_z = m_nb_z;
  auto in_adiabatic_cst = viewIn(m_mat_adiabatic_cst);
  auto in_volume = viewIn(m_mat_volume);
  auto in_density = viewIn(m_mat_density);
  auto in_old_volume = viewIn(m_mat_old_volume);
  auto in_internal_energy = viewIn(m_mat_internal_energy);

  auto out_internal_energy = viewOut(m_mat_internal_energy);
  auto out_sound_speed = viewOut(m_mat_sound_speed);
  auto out_pressure = viewOut(m_mat_pressure);

  for( Int32 iloop=0; iloop<nb_z; ++iloop){

    ENUMERATE_ENVCELL2(icell,m_env1){
      auto vi = icell;
      Real adiabatic_cst = in_adiabatic_cst[vi];
      Real volume_ratio = in_volume[vi] / in_old_volume[vi];
      Real x = 0.5 * adiabatic_cst - 1.0;
      Real numer_accrois_nrj = 1.0 + x*(1.0-volume_ratio);
      Real denom_accrois_nrj = 1.0 + x*(1.0-(1.0/volume_ratio));
      Real internal_energy = in_internal_energy[vi];
      internal_energy = internal_energy * (numer_accrois_nrj/denom_accrois_nrj);
      out_internal_energy[vi] = internal_energy;
      Real density = in_density[vi];
      Real pressure = (adiabatic_cst - 1.0) * density * internal_energy;
      Real sound_speed = DO_SQRT((adiabatic_cst*pressure/density));
      out_pressure[vi] = pressure;
      out_sound_speed[vi] = sound_speed;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialSimdUnitTest::
_computeEquationOfStateV3()
{
  Int32 nb_z = m_nb_z;
  auto in_adiabatic_cst = viewIn(m_mat_adiabatic_cst);
  auto in_volume = viewIn(m_mat_volume);
  auto in_density = viewIn(m_mat_density);
  auto in_old_volume = viewIn(m_mat_old_volume);
  auto in_internal_energy = viewIn(m_mat_internal_energy);

  auto out_internal_energy = viewOut(m_mat_internal_energy);
  auto out_sound_speed = viewOut(m_mat_sound_speed);
  auto out_pressure = viewOut(m_mat_pressure);

  for( Int32 iloop=0; iloop<nb_z; ++iloop){

    auto func = [=](MatVarIndex vi){
      Real adiabatic_cst = in_adiabatic_cst[vi];
      Real volume_ratio = in_volume[vi] / in_old_volume[vi];
      Real x = 0.5 * adiabatic_cst - 1.0;
      Real numer_accrois_nrj = 1.0 + x*(1.0-volume_ratio);
      Real denom_accrois_nrj = 1.0 + x*(1.0-(1.0/volume_ratio));
      Real internal_energy = in_internal_energy[vi];
      internal_energy = internal_energy * (numer_accrois_nrj/denom_accrois_nrj);
      out_internal_energy[vi] = internal_energy;
      Real density = in_density[vi];
      Real pressure = (adiabatic_cst - 1.0) * density * internal_energy;
      Real sound_speed = DO_SQRT((adiabatic_cst*pressure/density));
      out_pressure[vi] = pressure;
      out_sound_speed[vi] = sound_speed;
    };
    simple_env_loop(*m_env1,func);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialSimdUnitTest::
_computeEquationOfStateV2()
{
  Int32 nb_z = m_nb_z;
  auto in_adiabatic_cst = viewIn(m_mat_adiabatic_cst);
  auto in_volume = viewIn(m_mat_volume);
  auto in_density = viewIn(m_mat_density);
  auto in_old_volume = viewIn(m_mat_old_volume);
  auto in_internal_energy = viewIn(m_mat_internal_energy);

  auto out_internal_energy = viewOut(m_mat_internal_energy);
  auto out_sound_speed = viewOut(m_mat_sound_speed);
  auto out_pressure = viewOut(m_mat_pressure);

  for( Int32 iloop=0; iloop<nb_z; ++iloop){

    ENUMERATE_COMPONENTITEM_LAMBDA(EnvPartSimdCell,mvi,m_env1){
      auto vi = mvi;
      SimdReal adiabatic_cst = in_adiabatic_cst[vi];
      SimdReal volume_ratio = in_volume[vi] / in_old_volume[vi];
      SimdReal x = 0.5 * adiabatic_cst - 1.0;
      SimdReal numer_accrois_nrj = 1.0 + x*(1.0-volume_ratio);
      SimdReal denom_accrois_nrj = 1.0 + x*(1.0-(1.0/volume_ratio));
      SimdReal internal_energy = in_internal_energy[vi];
      internal_energy = internal_energy * (numer_accrois_nrj/denom_accrois_nrj);
      out_internal_energy[vi] = internal_energy;
      SimdReal density = in_density[vi];
      SimdReal pressure = (adiabatic_cst - 1.0) * density * internal_energy;
      SimdReal sound_speed = DO_SQRT((adiabatic_cst*pressure/density));
      out_pressure[vi] = pressure;
      out_sound_speed[vi] = sound_speed;
    };
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialSimdUnitTest::
_computeEquationOfStateV4()
{
  Int32 nb_z = m_nb_z;
  auto in_adiabatic_cst = viewIn(m_mat_adiabatic_cst);
  auto in_volume = viewIn(m_mat_volume);
  auto in_density = viewIn(m_mat_density);
  auto in_old_volume = viewIn(m_mat_old_volume);
  auto in_internal_energy = viewIn(m_mat_internal_energy);

  auto out_internal_energy = viewOut(m_mat_internal_energy);
  auto out_sound_speed = viewOut(m_mat_sound_speed);
  auto out_pressure = viewOut(m_mat_pressure);

  for( Int32 iloop=0; iloop<nb_z; ++iloop){

    auto func = [=](auto vi){
      Real adiabatic_cst = in_adiabatic_cst[vi];
      Real volume_ratio = in_volume[vi] / in_old_volume[vi];
      Real x = 0.5 * adiabatic_cst - 1.0;
      Real numer_accrois_nrj = 1.0 + x*(1.0-volume_ratio);
      Real denom_accrois_nrj = 1.0 + x*(1.0-(1.0/volume_ratio));
      Real internal_energy = in_internal_energy[vi];
      internal_energy = internal_energy * (numer_accrois_nrj/denom_accrois_nrj);
      out_internal_energy[vi] = internal_energy;
      Real density = in_density[vi];
      Real pressure = (adiabatic_cst - 1.0) * density * internal_energy;
      Real sound_speed = DO_SQRT((adiabatic_cst*pressure/density));
      out_pressure[vi] = pressure;
      out_sound_speed[vi] = sound_speed;
    };
    simple_env_loop2(*m_env1_as_vector,func);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialSimdUnitTest::
_computeEquationOfStateV4_noview()
{
  Int32 nb_z = m_nb_z;

  for( Int32 iloop=0; iloop<nb_z; ++iloop){

    auto func = [&](auto vi){
      Real adiabatic_cst = m_mat_adiabatic_cst[vi];
      Real volume_ratio = m_mat_volume[vi] / m_mat_old_volume[vi];
      Real x = 0.5 * adiabatic_cst - 1.0;
      Real numer_accrois_nrj = 1.0 + x*(1.0-volume_ratio);
      Real denom_accrois_nrj = 1.0 + x*(1.0-(1.0/volume_ratio));
      Real internal_energy = m_mat_internal_energy[vi];
      internal_energy = internal_energy * (numer_accrois_nrj/denom_accrois_nrj);
      m_mat_internal_energy[vi] = internal_energy;
      Real density = m_mat_density[vi];
      Real pressure = (adiabatic_cst - 1.0) * density * internal_energy;
      Real sound_speed = DO_SQRT((adiabatic_cst*pressure/density));
      m_mat_pressure[vi] = pressure;
      m_mat_sound_speed[vi] = sound_speed;
    };
    simple_env_loop2(*m_env1_as_vector,func);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
